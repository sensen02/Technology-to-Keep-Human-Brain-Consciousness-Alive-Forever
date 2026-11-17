"""Repeat the saved nominal candidate and measure archived D2/D6 fields in compact form."""
import argparse
import ctypes
from pathlib import Path

import numpy as np
from scipy.signal import lfilter
import torch

from residual import BASE, HERE, TURNS, load, read, stand
from verify_native import Record, SO
import walking_bench as bench


LIB = HERE / "native_force_z.so"


class VerticalRecord(Record):
    def __init__(self, body, duration):
        super().__init__(body, duration)
        self.z_force = ctypes.CDLL(str(LIB)).ground_force_z
        self.z_force.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.z_force.restype = ctypes.c_double
        self.vertical = np.zeros((duration * 80, 2))
        self.mass = float(body.plant.body_subtreemass[body.th])
        self.gravity = float(body.plant.opt.gravity[2])
        np.testing.assert_array_equal(~body.planes, body.plant.geom_type != 0)

    def before(self, body):
        super().before(body)
        self.v_before = body.com_velocity()[2]

    def solved(self, body, before):
        super().solved(body, before)
        force = self.z_force(body.plant._address, body.d._address)
        self.vertical[self.count, 0] = force
        if self.count % 4031 == 0:
            archived = stand.physics.NativeRecorder.solved(None, body, {"qvel_before": self.velocity})
            np.testing.assert_allclose(force, archived["ground_force_native"][2], atol=1e-9, rtol=1e-12)

    def after(self, body, before, solved):
        dt = float(body.d.time) - self.poses[self.count, 6]
        self.vertical[self.count, 1] = (body.com_velocity()[2] - self.v_before) / dt
        super().after(body, before, solved)


def run(reference, output):
    expected = read(reference / "RESULT.json")
    assert expected["start"] == "nominal" and expected["walk_ms"] == 9600 and expected["pulse"] == "none"
    assert stand.sha(reference / "TRACE.npz") == expected["trace_sha256"]
    checkpoint = Path(expected["checkpoint"])
    assert stand.sha(checkpoint) == expected["checkpoint_sha256"]
    paths = [checkpoint, HERE / "independent_nominal.py", LIB, HERE / "native_force_z.cpp", SO,
             TURNS / "verify_native.py", BASE / "walking_bench.py", BASE / "native_score.py"]
    pins = {str(p): stand.sha(p) for p in paths}
    rec = []
    old_loader = bench.load_initial

    def attach(standing_bench, *args):
        old_loader(standing_bench, *args)
        c = standing_bench.body
        record = VerticalRecord(c, expected["actual_ms"])
        c.recorder = record
        rec.append(record)

    try:
        bench.load_initial = attach
        result = bench.rollout(output, expected["start"], expected["walk_ms"], load(checkpoint))
    finally:
        bench.load_initial = old_loader
    [record] = rec
    assert result["decision_sha256"] == expected["decision_sha256"]
    assert record.count == expected["native_integrations"]
    assert result["joint_sampled_pass"] and not record.poses[:, 7].any()
    with np.load(output/"TRACE.npz", allow_pickle=False) as trace, np.load(reference/"TRACE.npz", allow_pickle=False) as old:
        for key in trace.files:
            assert trace[key].tobytes() == old[key].tobytes(), key
        np.testing.assert_array_equal(record.endpoints, trace["qpos"])
    with np.load(reference/"NATIVE.npz", allow_pickle=False) as reference_native:
        assert record.forces.tobytes() == reference_native["forces"].tobytes()
        assert record.poses.tobytes() == reference_native["pose"].tobytes()
    for path, digest in pins.items():
        assert stand.sha(Path(path)) == digest
    force, accel = record.vertical.T
    raw = record.mass * accel - (force + record.mass * record.gravity)
    rates = np.stack((accel / abs(record.gravity), raw / bench.W))
    alpha = float(-np.expm1(-stand.physics.NATIVE_DT/.001))
    filtered, _ = lfilter([alpha], [1., -(1.-alpha)], rates, axis=1, zi=(1.-alpha)*rates[:, :1])
    walk = slice(500*80, 10100*80)
    falls = int(np.sum((force[walk] < .2 * bench.W) & (filtered[0, walk] <= -.5)))
    residual = filtered[1, walk]
    p95 = float(np.quantile(residual, .95))
    np.savez_compressed(output/"VERTICAL_RAW.npz", native_measurements=record.vertical.copy(),
                        times=record.poses[:, 6].copy(), filtered=filtered)
    audit = dict(pins=pins, reference=str(reference), all_trace_fields_bitwise=True,
                 all_native_poses_and_contacts_bitwise=True, original_decision_sha256=expected["decision_sha256"],
                 native_steps=record.count, native_ground_cross_checks=record.cross_checks, force_exactly_zero=True,
                 D2=dict(passed=falls == 0, free_fall_samples=falls),
                 D6=dict(passed=p95 <= .15, signed_p95_w=p95,
                         absolute_p95_w=float(np.quantile(abs(residual), .95)), causal_filter_ms=1.),
                 full_D1_D8=False, vertical_raw_hash=stand.sha(output/"VERTICAL_RAW.npz"))
    stand.put(output/"INDEPENDENT_AUDIT.json", audit)
    print(audit)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    torch.set_num_threads(1)
    run(a.reference.resolve(), a.output.resolve())
