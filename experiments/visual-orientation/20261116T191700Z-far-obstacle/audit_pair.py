"""Compare the far-box runs before the first descending intervention."""
import argparse
from pathlib import Path

import numpy as np

from approach_runtime import turn, sha


def audit(course, blind):
    results = [turn.read(folder / "RESULT.json") for folder in (course, blind)]
    for folder, record in zip((course,blind), results):
        for name, expected in {**record["files"], **record["supplemental_hashes"]}.items():
            assert sha(folder / name) == expected, name
        for name, expected in record["inputs"].items():
            assert sha(Path(name)) == expected, name
    assert (course / "OBSTACLE.json").read_bytes() == (blind / "OBSTACLE.json").read_bytes()
    with np.load(course / "OBS.npz") as c, np.load(blind / "OBS.npz") as b:
        np.testing.assert_array_equal(c["goal"], b["goal"])
        first_divergence = np.flatnonzero(np.any(c["decisions"][:len(b["decisions"]),3:5] != b["decisions"][:,3:5], axis=1))[0]
        tick = int(c["decisions"][first_divergence,0])
        i = int(np.searchsorted(c["stamps"], tick-20, side="right"))
        np.testing.assert_array_equal(c["stamps"][:i], b["stamps"][:i])
        np.testing.assert_array_equal(c["eyes"][:i], b["eyes"][:i])
    with np.load(course / "TRACE.npz") as c, np.load(blind / "TRACE.npz") as b:
        assert c.files == b.files
        for k in c.files:
            np.testing.assert_array_equal(c[k][:tick+1], b[k][:tick+1], err_msg=k)
    with np.load(course / "BLOCK_FORCE.npz") as c, np.load(blind / "BLOCK_FORCE.npz") as b:
        np.testing.assert_array_equal(c["force"][:tick*80], b["force"][:tick*80])
        assert not c["force"][:tick*80].any()
    return dict(same_obstacle_and_goal=True, same_initial_trace_and_retinal_inputs_through_ms=tick,
                identical_eye_frames=i, same_no_force_prefix_native=tick*80,
                detour_gate=results[0]["passed"], disabled_has_native_impact=results[1]["passed"],
                result_sha256=[sha(folder / "RESULT.json") for folder in (course,blind)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("course", type=Path)
    parser.add_argument("blind", type=Path)
    a = parser.parse_args()
    outcome = audit(a.course, a.blind)
    turn.stand.put(a.out, outcome)
    print(outcome, flush=True)
