"""Transfer measured FlyGym swing shapes into bounded rates on the frozen body."""
import argparse
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parents[1] / "20261116T000536Z"
sys.path.insert(0, str(ARCHIVE))
import prototype
import walking_bench as bench

SOURCE = HERE / "FLYGYM_FLAT_NO_ADH_V3"
BASELINE = ARCHIVE / "PROTO_nominal"
PROFILE = HERE / "SWING_REFERENCE.npz"
WALK_START = 500
WALK_MS = 9600
CONTROL_MS = 5
U = np.linspace(0., 1., 101)


def off_ground(mask):
    bounds = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
    return zip(bounds[::2], bounds[1::2], strict=True)


def arc(values):
    chord = np.linspace(values[0], values[-1], len(values))
    return values - chord


def build():
    source = bench.read(SOURCE / "RESULT.json")
    baseline = bench.read(BASELINE / "RESULT.json")
    assert source["adhesion"] is False and baseline["joint_sampled_pass"]
    assert bench.stand.sha(SOURCE / "TRACE.npz") == source["trace_sha256"]
    assert bench.stand.sha(BASELINE / "TRACE.npz") == baseline["trace_sha256"]
    profiles, crops = {}, {}
    with np.load(SOURCE / "TRACE.npz", allow_pickle=False) as z:
        mask = z["time_s"] >= 1.
        for li, leg in enumerate(z["legs"].tolist()):
            world = z["foot_world_mm"][mask, li]
            local = z["foot_body_mm"][mask, li]
            cycles = []
            ranges = []
            for a, b in off_ground(z["contact_count"][mask, li] == 0):
                if b - a < 10 or a == 0 or b >= len(world):
                    continue
                # The trajectory includes the last contact and first recontact samples.
                vertical = arc(world[a - 1:b + 1, 2])
                advance = local[b, 0] - local[a - 1, 0]
                peak = vertical.max()
                if peak < .035 or advance < .03:
                    continue
                cycles.append(np.interp(U, np.linspace(0., 1., len(vertical)), vertical / peak))
                ranges.append((int(a), int(b), float(peak), float(advance)))
            if len(cycles) < 20:
                continue
            profile = gaussian_filter1d(np.median(cycles, axis=0), 1.5)
            profile = arc(profile).clip(min=0.)
            profiles[leg] = profile / profile.max()
            crops[leg] = ranges

    if not profiles:
        raise ValueError("No corroborated FlyGym swing arcs")
    event_rows = []
    delta = np.zeros((WALK_START + WALK_MS + 1201, 6))
    with np.load(BASELINE / "TRACE.npz", allow_pickle=False) as data:
        z = {key: data[key] for key in ("time", "loads", "padz", "foot")}
        _, events = bench.cycles(z, WALK_START, WALK_START + WALK_MS, arm_before=20)
        for event in events:
            leg = event["leg"]
            if event["type"] != "complete" or leg not in profiles:
                continue
            li = bench.LEGS.index(leg)
            a = int(round((event["start_time"] - z["time"][0]) * 1000.))
            b = int(round((event["time"] - event["reload_dwell"] - z["time"][0]) * 1000.))
            # Apply shape changes only inside the truly clear portion of the existing swing.
            qualified = (z["loads"][a:b + 1, li] < .05 * bench.W) & (z["padz"][a:b + 1, li] > .003)
            blocks = list(off_ground(qualified))
            if not blocks:
                continue
            p, q = max(blocks, key=lambda pair: pair[1] - pair[0])
            lo, hi = int(a + p), int(a + q)
            if hi - lo < 35:
                continue
            u = np.linspace(0., 1., hi - lo + 1)
            vertical = arc(z["padz"][lo:hi + 1, li])
            source_shape = np.interp(u, U, profiles[leg])
            fade = np.sin(np.pi * u) ** 2
            correction = fade * (vertical.max() * source_shape - vertical)
            delta[lo:hi + 1, li] = correction
            event_rows.append(dict(leg=leg, from_ms=lo, to_ms=hi))
    if not event_rows:
        raise ValueError("No overlapping validated project swing arcs")
    with PROFILE.open("xb") as f:
        np.savez_compressed(f, correction_native=delta, phase_u=U, **profiles)
    meta = dict(
        source_dir=str(SOURCE), source_trace_sha256=source["trace_sha256"],
        source_result_sha256=bench.stand.sha(SOURCE / "RESULT.json"),
        receiver_baseline_dir=str(BASELINE), receiver_trace_sha256=baseline["trace_sha256"],
        ref_sha256=bench.stand.sha(PROFILE), profile_sample_ms=1,
        body_leg_order=list(bench.LEGS), source_legs=list(profiles),
        suppressed_legs=[leg for leg in bench.LEGS if leg not in profiles],
        source_contact_rule="contiguous ground-contact sensor count zero for ≥10 ms after 1s, bracketed by ground contact",
        source_kinematic_rule="≥0.035mm vertical arc above endpoint chord and ≥0.03mm fore advance in thorax frame",
        source_aggregation="per-leg median of unit-peak 101-point swing arcs, Gaussian sigma 1.5 samples; at least 20 swings",
        source_crops_indices_from_settled_trace=crops,
        receiver_rule="complete frozen baseline cycles; longest continuous run of padz>0.003 and load<0.05W; ≥35ms",
        receiver_morph="sin² taper of (same-height FlyGym arc minus project padz arc), computed within each receiver swing",
        receiver_windows=event_rows,
    )
    bench.stand.put(HERE / "SWING_REFERENCE.json", meta)
    print({k: v for k, v in meta.items() if k != "source_crops_indices_from_settled_trace"}, flush=True)


class Policy:
    def __init__(self, gain, reference):
        self.base = prototype.policy()
        with np.load(PROFILE, allow_pickle=False) as z:
            self.offset = gain * z["correction_native"]
        self.tick = 0
        self.gain = gain
        self.reference = reference
        self.rows = []

    def attach(self, c):
        self.c = c
        self.indices = []
        for leg in bench.LEGS:
            ids = [
                i for i, joint in enumerate(c.actuator.joint_ids)
                if c.plant.joint(int(joint)).name.startswith(f"joint_{leg}")
            ]
            if len(ids) != 7:
                raise ValueError(f"Unexpected joints for {leg}")
            self.indices.append(ids)
        self.target_scale = np.maximum(.5 * np.ptp(c.plant.jnt_range[c.actuator.joint_ids], axis=1), .1)
        self.jacobian = np.zeros((3, c.plant.nv))

    def __call__(self, feature, walk):
        t = self.tick
        self.tick += CONTROL_MS
        nominal_action = self.base(feature, walk)
        correction = np.zeros(42)
        pos = np.zeros(42)
        if walk and self.gain:
            c = self.c
            this = self.offset[t]
            next_ = self.offset[t + CONTROL_MS]
            for li in np.flatnonzero((this != 0.) | (next_ != 0.)):
                leg = bench.LEGS[li]
                g = min(c.pads[leg], key=lambda j: c.d.geom_xpos[j, 2] - c.plant.geom_size[j, 0])
                mj.mj_jacGeom(c.plant, c.d, self.jacobian, None, g)
                ids = self.indices[li]
                jac = self.jacobian[:, c.a_dof[ids]]
                dz = np.linalg.solve(jac @ jac.T + .008 * np.eye(3), np.array([0., 0., 1.]))
                direction = jac.T @ dz
                pos[ids] = np.clip(direction * this[li], -.045, .045)
                correction[ids] = direction * (next_[li] - this[li]) / .005 / bench.stand.physics.MAX_RATE
            corrected_feature = feature.copy()
            corrected_feature[126:168] -= pos / self.target_scale
            nominal_action = self.base(corrected_feature, walk)
            correction = np.clip(correction, -.20, .20)
        self.rows.append((t, pos, correction, self.offset[t].copy()))
        return np.clip(nominal_action + correction, -1., 1.)


def run(args):
    ref = bench.read(HERE / "SWING_REFERENCE.json")
    if bench.stand.sha(PROFILE) != ref["ref_sha256"]:
        raise ValueError("Frozen swing reference changed")
    if args.gain <= 0.:
        raise ValueError("A teacher candidate requires a positive source gain")
    p = Policy(args.gain, ref)
    load_initial = bench.load_initial

    def attach(standing_bench, *rest):
        load_initial(standing_bench, *rest)
        p.attach(standing_bench.body)

    bench.load_initial = attach
    result = bench.rollout(args.output, args.start, WALK_MS, p, native=args.native)
    cols = dict(zip(("ms", "joint_offset_rad", "feedforward_rate_normalized", "clearance_offset_native"),
                    (np.asarray(col) for col in zip(*p.rows))))
    np.savez_compressed(args.output / "TRANSFER.npz", **cols)
    result.update(
        control="frozen prototype with bounded measured FlyGym vertical-shape correction",
        source_ref=str(PROFILE), source_ref_sha256=ref["ref_sha256"], source_trace_sha256=ref["source_trace_sha256"],
        source_gain=args.gain, source_trace_dir=str(SOURCE),
        source_profile_legs=ref["source_legs"], transfer_sha256=bench.stand.sha(args.output / "TRANSFER.npz"),
        physically_qualified_teacher=False,
        provenance_sha256={str(path):bench.stand.sha(path) for path in (
            Path(__file__), ARCHIVE / "walking_bench.py", ARCHIVE / "prototype.py", prototype.BANK,
            prototype.CHECKPOINT, HERE / "SWING_REFERENCE.json")},
    )
    bench.stand.put(args.output / "RESULT.json", result)


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("output", type=Path, nargs="?")
    a.add_argument("--build", action="store_true")
    a.add_argument("--start", default="nominal")
    a.add_argument("--gain", type=float, default=.15)
    a.add_argument("--native", action="store_true")
    args = a.parse_args()
    torch.set_num_threads(1)
    if args.build:
        build()
    else:
        args.output = args.output.resolve()
        run(args)
