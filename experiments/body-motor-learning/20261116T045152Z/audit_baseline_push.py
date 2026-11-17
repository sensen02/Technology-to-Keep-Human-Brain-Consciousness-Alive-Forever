"""Re-run the original push baseline with an independent read-only native recorder."""
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "20261116T000536Z"
sys.path.insert(0, str(ORIGINAL))
sys.path.insert(0, str(HERE.parent / "20261116T004235Z"))

import push_walk
from verify_native import Record


def run():
    torch.set_num_threads(1)
    bench_root = HERE / "baseline"
    for name in ("PUSH_PREREG.zh-CN.md", "PROTOTYPES.npz", "PROTO_lean_right"):
        path = bench_root / name
        path.symlink_to(ORIGINAL / name)

    loader = push_walk.load_initial
    originals = push_walk.HERE
    traces = []

    def instrument(bench, *args):
        loader(bench, *args)
        rec = Record(bench.body, push_walk.TOTAL)
        bench.body.recorder = rec
        traces.append(rec)

    push_walk.HERE = bench_root
    push_walk.load_initial = instrument
    try:
        push_walk.run("walking")
    finally:
        push_walk.HERE = originals
        push_walk.load_initial = loader

    [native] = traces
    out = bench_root / "PUSH_walking"
    result = push_walk.read(out / "RESULT.json")
    old = push_walk.read(ORIGINAL / "PUSH_walking/RESULT.json")
    assert result["trace_sha256"] == old["trace_sha256"]
    assert result["decision_sha256"] == old["decision_sha256"]
    assert native.count == push_walk.TOTAL * 80
    with np.load(out / "TRACE.npz", allow_pickle=False) as recorded:
        np.testing.assert_array_equal(native.endpoints, recorded["qpos"])
        com = recorded["com"].copy()
    onset = result["onset_ms"] * 80
    end = onset + result["duration_ms"] * 80
    expected = np.zeros(native.count)
    expected[onset:end] = result["force_w"] * push_walk.W
    np.testing.assert_array_equal(native.poses[:,7], expected)
    window = native.forces[500*80:10100*80]
    total = window[:, :6].sum(axis=0)
    slip = window[:, 6:12].sum(axis=0) / total
    speed = float(np.linalg.norm((com[10100]-com[500])[:2])/9.6)
    archive = out / "NATIVE.npz"
    np.savez_compressed(archive, forces=native.forces, pose=native.poses)
    summary = dict(source_data_bitwise=True, verified_applied_force=True, native_steps=native.count,
                   cross_checks=native.cross_checks, peak_nonfoot_native=float(window[:,12].max()),
                   foot_load_w=float(total.sum()/len(window)/push_walk.W), slip_ratio_max=float(slip.max()/speed),
                   absolute_contact_slip_native_per_s=slip.tolist(), native_sha256=push_walk.stand.sha(archive))
    push_walk.stand.put(out / "NATIVE_PUSH_AUDIT.json", summary)
    print(summary)


if __name__ == "__main__":
    run()
