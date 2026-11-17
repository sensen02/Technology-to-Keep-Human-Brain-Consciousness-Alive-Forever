"""Audit state-only pace control against its stored per-decision actuator commands."""
import argparse
from pathlib import Path

import numpy as np
import torch

from pace_lookup import ObservedPhase
from speed_teacher import WARM, stand
from walking_bench import read, legacy


def audit(output,run,replay):
    result=read(run/"RESULT.json")
    assert result["native_joint_pass"]
    for f,key in (("TRACE.npz","trace_sha256"),("DECISIONS.npz","decision_sha256"),("NATIVE.npz","native_sha256")):
        assert stand.sha(run/f)==result[key]
    with np.load(run/"DECISIONS.npz",allow_pickle=False) as q:
        x,command,label=q["x"],q["command"],q["action"]
    axes=np.arange(len(x))
    order=np.random.default_rng(3108).permutation(axes)
    a,b,c=ObservedPhase(),ObservedPhase(),ObservedPhase()
    generate=lambda actor,i:legacy.bounded_action(actor(x[i],command[i]))
    direct=np.array([generate(a,i) for i in axes])
    reverse=np.array([generate(b,i) for i in axes[::-1]])[::-1]
    shuffled=np.array([generate(c,i) for i in order])[np.argsort(order)]
    decisions=dict(direct_bitwise=bool(np.array_equal(label,direct)),reverse_bitwise=bool(np.array_equal(label,reverse)),
                   shuffled_bitwise=bool(np.array_equal(label,shuffled)),tested_decisions=len(label),
                   absolute_max_action_difference=float(max(abs(label-direct).max(),abs(label-reverse).max(),
                                                           abs(label-shuffled).max())),
                   old_reference_phase_state_unused=bool(a.phase==b.phase==c.phase==WARM))
    output_obj=dict(scope="physically tested stateless engineering decoder; no learned model",
                    source=str(Path(__file__).parent/"pace_lookup.py"),controller="ObservedPhase(x, current_command)",
                    case=str(run),checked_native_trial_sha256=stand.sha(run/"RESULT.json"),
                    decisions=decisions,valid_native=True,walk=result["walk"],measurement_windows=result["windows"])
    if replay is not None:
        r=read(replay/"RESULT.json")
        exact={f:stand.sha(run/f)==stand.sha(replay/f) for f in ("TRACE.npz","DECISIONS.npz","NATIVE.npz")}
        assert r["native_joint_pass"]
        output_obj["native_repeat"]=dict(path=str(replay),exact=exact,passed=all(exact.values()))
    stand.put(output,output_obj)
    print(decisions,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("run",type=Path)
    p.add_argument("--replay",type=Path)
    a=p.parse_args()
    torch.set_num_threads(1)
    audit(a.output.resolve(),a.run.resolve(),a.replay.resolve() if a.replay else None)
