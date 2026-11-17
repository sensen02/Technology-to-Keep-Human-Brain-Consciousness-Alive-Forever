"""Run and independently check the visual propulsion pilot."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from walk_pilot import HERE, fit, trial, turn

CHECKPOINT = HERE / "WALK_HEAD.pt"
SETTINGS = dict(onset=780, walk_ms=9600, start="nominal", width=.161, distance=3.52,
                bearing=-1.5, shift=.133, gray=.27)


def exercise(mode, name):
    trial(SimpleNamespace(mode=mode, output=HERE/name, checkpoint=CHECKPOINT, **SETTINGS))


def qualify(normal, cut, restore, off):
    modes = {"normal": normal, "cut": cut, "restore": restore, "off": off}
    outcomes = {mode: turn.read(folder/"WALK_RESULT.json") for mode, folder in modes.items()}
    for mode, folder in modes.items():
        result = outcomes[mode]
        assert result["case"]["mode"] == mode
        for name, fingerprint in result["files"].items():
            assert turn.stand.sha(folder/name) == fingerprint
    cases = [{k:v for k,v in r["case"].items() if k not in {"mode", "output"}} for r in outcomes.values()]
    assert all(c == cases[0] for c in cases[1:])
    verified = []
    for mode in ("normal", "cut", "restore"):
        with np.load(modes[mode]/"WALK_COMMANDS.npz", allow_pickle=False) as data:
            verified.append(data["commands"].copy())
    whole, disconnected, returned = verified
    motion = np.flatnonzero(whole[:, 2])
    first_ms = int(whole[motion[0], 0]) if len(motion) else None
    if first_ms is None:
        raise ValueError("No received propulsion in the complete case")
    with np.load(normal/"TRACE.npz", allow_pickle=False) as a, np.load(cut/"TRACE.npz", allow_pickle=False) as b, np.load(restore/"TRACE.npz", allow_pickle=False) as c:
        pre = all(np.array_equal(a[k][:first_ms+1], b[k][:first_ms+1]) and
                  np.array_equal(b[k][:first_ms+1], c[k][:first_ms+1]) for k in ("qpos", "qvel", "com", "loads"))
        endpoint_displacement = [float(trace["com"][-1, 0]-trace["com"][0, 0]) for trace in (a,b,c)]
    limit = SETTINGS["onset"]+400
    brain_already_requested = bool(disconnected[disconnected[:,0] < limit, 1].any())
    resumed = (returned[:,0] >= limit) & (returned[:,2] > 0)
    signed_source = {mode: r["graph_sha256"] for mode,r in outcomes.items()}
    checks = dict(all_physical_pass=all(r["passed"] for r in outcomes.values()),
                  exact_shared_state_until_first_motor_change=pre,
                  intact_starts_then_stops=bool(whole[-1,2] == 0),
                  cut_brain_active_motor_silent=brain_already_requested and not disconnected[:,2].any(),
                  restore_from_new_delivered_activity=bool(resumed.any() and
                      not returned[returned[:,0] < limit,2].any() and
                      returned[np.flatnonzero(resumed)[0],4] >= limit),
                  same_pinned_source=len(set(signed_source.values())) == 1)
    summary = dict(passed=all(checks.values()), checks=checks, first_intact_motion_ms=first_ms,
                   forward_displacement_intact_cut_restore=endpoint_displacement,
                   inputs={mode:dict(path=str(folder.resolve()), sha256=turn.stand.sha(folder/"WALK_RESULT.json")) for mode,folder in modes.items()},
                   scope="causal single-speed visual locomotion pilot on flat terrain")
    turn.stand.put(HERE/"WALK_ACCEPTANCE.json", summary)
    print(summary, flush=True)
    return summary["passed"]


if __name__ == "__main__":
    torch.set_num_threads(1)
    p = argparse.ArgumentParser(description=__doc__)
    actions = p.add_subparsers(dest="action", required=True)
    f = actions.add_parser("fit")
    f.add_argument("teachers", nargs="+", type=Path)
    r = actions.add_parser("run")
    r.add_argument("mode", choices=("normal", "cut", "restore", "off"))
    r.add_argument("name")
    a = actions.add_parser("audit")
    for option in ("normal","cut","restore","off"):
        a.add_argument(option, type=Path)
    args = p.parse_args()
    if args.action == "fit":
        fit(SimpleNamespace(output=CHECKPOINT, teachers=args.teachers))
    elif args.action == "run":
        exercise(args.mode, args.name)
    else:
        raise SystemExit(0 if qualify(args.normal, args.cut, args.restore, args.off) else 1)
