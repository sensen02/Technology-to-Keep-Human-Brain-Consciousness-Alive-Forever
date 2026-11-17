"""Shared frozen turn bench for the first artificial-brain visual experiment."""
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
BODY = PROJECT / "experiments/body-motor-learning/20261116T004235Z"
sys.path.insert(0, str(BODY))
import turning as turn
from control_body import load_body_policy


def prepare(path, start):
    path.mkdir()
    torch.set_num_threads(1)
    bench = turn.stand.Bench(path)
    body = bench.body
    body.physics_order = turn.stand.physics.rc.SENS_IDX
    snapshot = turn.STARTS / f"{start}.npz"
    turn.load_initial(bench, snapshot, turn.read(turn.STARTS / "BANK.json")[start]["sha256"])
    return body, load_body_policy()


def sample(body, measurements):
    return turn.sample(body, measurements)


def arrays(rows):
    return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}


def stand_window(trace, end):
    if end < 200:
        return dict(passed=False)
    result = turn.standing(trace, end-200, end)
    net = float(abs(np.rad2deg(trace["yaw"][end] - trace["yaw"][end-200])))
    result["yaw_change_deg"] = net
    result["passed"] = bool(result["passed"] and net <= .5)
    return result


def check_forces(body):
    if np.any(body.d.xfrc_applied) or np.any(body.d.qfrc_applied):
        raise RuntimeError("Unexpected external wrench")
