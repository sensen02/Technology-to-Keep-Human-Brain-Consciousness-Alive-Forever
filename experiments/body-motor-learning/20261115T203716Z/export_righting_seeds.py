"""Save exact simulator snapshots of the two observed fallen configurations."""
import numpy as np
import torch

from matrix import FACTORS, HERE, arrays, read
from snapshots import Neural, capture, load, model, stand
from aftermath import supplement
from bench import run

CASES = (("side_fallen", "nominal"), ("near_supine", "lean_right"))


def main():
    torch.set_num_threads(1)
    root = HERE / "RUN"
    starts = root / "INITIAL"
    matrix = root / "MATRIX"
    output = root / "RIGHTING_SEEDS_VERIFIED"
    output.mkdir()
    results = read(matrix / "RESULT.json")["cases"]
    prior_bank = read(starts / "BANK.json")
    push = FACTORS[-1]
    bench = stand.Bench(output)
    control = Neural(model())
    bank = {}
    for name, initial in CASES:
        key = f"{initial}__{push.name}"
        prior = results[key]["model"]
        source = matrix / prior["file"]
        native = arrays(source.with_suffix(".native.npz"), prior["native_sha256"])
        ms = arrays(source.with_suffix(".npz"), prior["legacy"]["trace_sha256"])
        event = supplement(native, prior["event"], ms)
        pose = event["terminal_fallen_pose"]
        if pose is None or max(pose["root_speed_native_s"], pose["root_angular_speed_rad_s"]) > .001:
            raise ValueError("Expected a near-static observed fall endpoint")
        load(bench, starts / f"{initial}.npz", prior_bank[initial]["sha256"])
        trace = output / f"{name}_source.npz"
        result = run(bench, push, control, trace)
        repeated = arrays(trace, result["trace_sha256"])
        for k in ms:
            np.testing.assert_array_equal(repeated[k], ms[k])
        snap = capture(bench)
        np.testing.assert_array_equal(snap["qpos"], native["qpos_ms"][-1])
        np.testing.assert_array_equal(snap["qvel"], native["qvel_ms"][-1])
        np.testing.assert_array_equal(snap["act_target_rad"], ms["target"][-1])
        if np.any(bench.body.d.xfrc_applied) or np.any(bench.body.d.qfrc_applied):
            raise ValueError("Fallen seed cannot have an active external wrench")
        path = output / f"{name}.npz"
        np.savez_compressed(path, **snap)
        digest = stand.sha(path)
        load(bench, path, digest)
        restored = capture(bench)
        for field in snap:
            np.testing.assert_array_equal(snap[field], restored[field])
        bank[name] = dict(**pose, initial=initial, case=key, source_replay_sha256=result["trace_sha256"],
                          snapshot_sha256=digest, native_steps=result["native_steps"])
        print(name, pose, flush=True)
    stand.identity.verify_runtime(bench.manifest, bench.body)
    stand.put(output / "BANK.json", dict(seeds=bank, restore_exact=True,
        native_steps=80000 + sum(item["native_steps"] for item in bank.values())))


if __name__ == "__main__":
    main()
