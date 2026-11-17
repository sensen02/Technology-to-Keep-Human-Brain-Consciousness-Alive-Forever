"""Check the archive, including equality before the first received-command difference."""
import argparse
import json
from pathlib import Path

import numpy as np

from approach_runtime import sha, turn


def read_commands(folder):
    result = turn.read(folder / "RESULT.json")
    assert all(sha(folder / name) == digest for name, digest in result["files"].items())
    with np.load(folder / "OBS.npz", allow_pickle=False) as archive:
        return archive["decisions"], archive["stamps"], archive["eyes"], archive["goal"]


def verify(control, intervention):
    old, old_views, old_eyes, old_goal = read_commands(control)
    new, new_views, new_eyes, new_goal = read_commands(intervention)
    length = min(len(old), len(new))
    np.testing.assert_array_equal(old[:length, 0], new[:length, 0])
    changed = np.flatnonzero(np.any(old[:length, 3:5] != new[:length, 3:5], axis=1))
    assert len(changed), "No command treatment was delivered"
    stamp = int(new[changed[0], 0])
    early = int(np.searchsorted(old_views, stamp - 20, side="right"))
    np.testing.assert_array_equal(old_views[:early], new_views[:early])
    np.testing.assert_array_equal(old_eyes[:early], new_eyes[:early])
    np.testing.assert_array_equal(old_goal, new_goal)
    with np.load(control / "TRACE.npz", allow_pickle=False) as before, np.load(
        intervention / "TRACE.npz", allow_pickle=False
    ) as after:
        for name in ("qpos", "qvel", "com", "loads", "ctrl"):
            np.testing.assert_array_equal(before[name][:stamp+1], after[name][:stamp+1])
    with np.load(control / "BLOCK_FORCE.npz", allow_pickle=False) as before, np.load(
        intervention / "BLOCK_FORCE.npz", allow_pickle=False
    ) as after:
        np.testing.assert_array_equal(before["force"][:80*stamp], after["force"][:80*stamp])
    return dict(passed=True, first_decoded_difference_ms=stamp, same_eyes_frames=early,
                native_prefix_steps=80*stamp, control_after_neck=old[changed[0], 3:5].tolist(),
                repaired_after_neck=new[changed[0], 3:5].tolist())


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("control", type=Path)
    cli.add_argument("intervention", type=Path)
    cli.add_argument("out", type=Path)
    args = cli.parse_args()
    result = verify(args.control, args.intervention)
    turn.stand.put(args.out, result)
    print(json.dumps(result), flush=True)
