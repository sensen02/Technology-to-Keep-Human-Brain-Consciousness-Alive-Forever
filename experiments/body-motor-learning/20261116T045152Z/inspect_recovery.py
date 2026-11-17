"""Extract synchronized observations around the held-out lateral pulse."""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "20261116T000536Z"))
from walking_bench import LEGS, W, read, stand


def pair():
    condition = (HERE / "student/PUSH_FIRST_FIT", HERE / "student/LEAN_RIGHT_FIRST_FIT")
    first, second = map(lambda p: read(p/"RESULT.json"), condition)
    assert first["checkpoint_sha256"] == second["checkpoint_sha256"]
    assert first["start"] == second["start"] == "lean_right"
    assert first["walk_ms"] == second["walk_ms"] == 9600
    assert first["pulse"] == "push" and second["pulse"] == "none"
    for run, result in zip(condition, (first, second), strict=True):
        assert stand.sha(run/"TRACE.npz") == result["trace_sha256"]
        assert stand.sha(run/"DECISIONS.npz") == result["decision_sha256"]
    heldout = HERE / "RECOVERY_HELDOUT"
    heldout.mkdir()
    onset, off = 3700, 3720
    end = off + 200
    grid = np.arange(onset, end+1, 5)
    inputs = {}
    differences = []
    with np.load(condition[0]/"TRACE.npz", allow_pickle=False) as push, \
            np.load(condition[1]/"TRACE.npz", allow_pickle=False) as plain, \
            np.load(condition[0]/"DECISIONS.npz", allow_pickle=False) as dp, \
            np.load(condition[1]/"DECISIONS.npz", allow_pickle=False) as dc:
        np.testing.assert_array_equal(dp["ms"],dc["ms"])
        w = np.zeros((2*first["actual_ms"],6))
        w[2*onset:2*off,1] = .2 * W
        np.testing.assert_array_equal(push["actual_wrench"],w)
        assert not plain["actual_wrench"].any()
        for name in ("qpos", "qvel", "com", "loads", "speed", "target", "applied"):
            np.testing.assert_array_equal(push[name][:onset+1], plain[name][:onset+1])
            inputs[name+"_push"], inputs[name+"_plain"] = push[name][grid], plain[name][grid]
        d = np.searchsorted(dp["ms"],grid)
        assert (dp["intent"][d] == 1).all() and (dc["intent"][d] == 1).all()
        inputs.update(ms=grid, x_push=dp["x"][d], x_plain=dc["x"][d],
                      actual_policy_action_push=dp["action"][d], actual_policy_action_plain=dc["action"][d])
        for a, b, event in ((onset,off,"applied_pulse"),(off,end,"early_recovery"),
                            (end,4700,"later_recovery"),(4700,10100,"stable_walk_tail")):
            s = slice(a+1,b+1)
            kinematics = [z["com"][b,1] for z in (push,plain)]
            sliding = []
            for z in (push,plain):
                q = z["loads"][s]
                sliding.append((q*z["speed"][s]).sum(0)/q.sum(0))
            differences.append(dict(event=event, from_ms=a, to_ms=b,
                 lateral_position_difference_native=kinematics[0]-kinematics[1],
                 slide_increase_native_s=dict(zip(LEGS, (sliding[0]-sliding[1]).tolist(), strict=True))))
    artifact = heldout / "OBSERVATIONS.npz"
    with artifact.open("xb") as f:
        np.savez_compressed(f,**inputs)
    stand.put(heldout/"RESULT.json",dict(
        observation_sha256=stand.sha(artifact), checkpoint_sha256=first["checkpoint_sha256"],
        cases=[dict(path=str(path),trace_sha256=result["trace_sha256"],decision_sha256=result["decision_sha256"])
               for path,result in zip(condition,(first,second),strict=True)],
        samples=len(grid), prefix_identical_through_ms=onset, windows_1ms=differences,
        usage="Held-out diagnostic observations; no demonstrated recovery labels. Do not include in training or tune on this trial.",
    ))


if __name__ == "__main__":
    pair()
