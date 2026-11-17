"""Check the final visual stop from the collision-free detour's recorded endpoint."""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np

import approach
from approach_runtime import Eyes, sha, turn
from far_fixture import prepare
from far_navigation import ProgressRecord
from geometry import Envelope
from obstacle import score as obstacle_score
from rejoin import Rejoin


def finish(out):
    source = Path(__file__).resolve().parent / "wider"
    result = turn.read(source / "RESULT.json")
    verified = {}
    for name in ("OBS.npz", "TRACE.npz"):
        verified[str(source / name)] = sha(source / name)
        assert verified[str(source / name)] == result["files"][name]
    with np.load(source / "TRACE.npz", allow_pickle=False) as trace:
        pose, velocity, ctrl, target = (trace[key][-1].copy() for key in ("qpos", "qvel", "ctrl", "target"))
        background = (trace["head"][0].copy(), .12, .255)
        heights = trace["padz"][-2:].copy()
    with np.load(source / "OBS.npz", allow_pickle=False) as saved:
        destination = saved["goal"].copy()
    envelope = None
    brain = Rejoin()
    # The previously archived controller was in the slow pursuit phase.
    brain.vision.phase = "walk"
    brain.vision.near = True

    def setup(path, start):
        nonlocal envelope
        body, motor = prepare(path, start, .43)
        body.d.qpos[:], body.d.qvel[:], body.d.ctrl[:] = pose, velocity, ctrl
        body.actuator.target_rad[:] = target
        body._padz_prev = heights[-1].copy()
        body._pad_rate = (heights[-1] - heights[-2])*1000.
        mj.mj_forward(body.plant, body.d)
        envelope = Envelope(body)
        return body, motor

    class SameSceneEyes(Eyes):
        def place(self, angle, distance):
            return destination.copy()

        def image(self, target, width, color, _background, cover=None):
            return super().image(target, width, color, background, cover)

    def score(trace, decisions, goal, actual, angle, mode, record):
        scored = obstacle_score(trace, decisions, goal, actual, angle, mode, record)
        resumed = decisions[:,3]>.06
        bound = envelope.path(trace["qpos"], 0, actual, stride=20)
        margin = float(bound[:,1].min()-envelope.far)
        np.savez_compressed(out / "BLOCK_FORCE.npz", force=record.block[:record.count])
        checks = dict(zero_obstacle=not bool(np.any(record.block[:record.count])),
                      no_floor_nonfoot=scored["nonfoot_native"]==0.,
                      whole_fly_stays_beyond=margin>=.03,
                      progressed=bool(resumed.any() and scored["initial_range"]-scored["final_range"] >= .28),
                      stops_on_vision=brain.phase=="done" and 1.18<scored["final_range"]<1.72,
                      stable=scored["standing"]["passed"])
        scored.update(passed=all(checks.values()), checks_finish=checks, spec=dict(blind=False),
                      minimum_whole_fly_beyond=margin, end_policy_phase=brain.phase,
                      verified_endpoint_files=verified)
        return scored

    approach.prepare, approach.Policy, approach.Eyes = setup, lambda: brain, SameSceneEyes
    approach.Record, approach.score, approach.START_MS = ProgressRecord, score, 0
    return approach.trial(out.resolve(),0.,6.,"intact",.12,16500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    result = finish(parser.parse_args().out)
    raise SystemExit(0 if result["passed"] else 1)
