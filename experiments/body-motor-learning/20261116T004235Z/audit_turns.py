"""Reload the fixed controller, reproduce actions, and score the exit heading."""
import argparse
from pathlib import Path

import numpy as np
import torch

from turning import HERE, PARENT, W, read, score, stand
from control_body import load_body_policy


def audit(path,policy):
    result=read(path/"RESULT.json")
    assert result["kind"]=="model"
    assert stand.identity.verify(read(path/"TURN_IDENTITY.json"))
    trace=path/"TRACE.npz"
    decisions=path/"DECISIONS.npz"
    assert stand.sha(trace)==result["trace_sha256"]
    assert stand.sha(decisions)==result["decisions_sha256"]
    with np.load(decisions,allow_pickle=False) as a,np.load(trace,allow_pickle=False) as z:
        steps, x, command, action=(a[k] for k in ("ms","x","command","action"))
        data={k:z[k] for k in ("yaw","qpos","com","loads","nonfoot","padz","speed","foot","upright","time")}
    np.testing.assert_array_equal(steps,np.arange(len(steps))*5)
    # Reverse execution order to catch hidden timing or previous-action dependence.
    advance=int(result["mode"].startswith("arc"))
    yaw_request=1 if result["mode"].endswith("left") else -1
    replay=np.asarray([policy(v,advance,yaw_request) if c==2 else policy(v,int(c),0)
                       for v,c in zip(x[::-1],command[::-1],strict=True)])[::-1]
    np.testing.assert_allclose(action,replay,atol=1e-12,rtol=0.)
    recomputed=score(data,result["mode"],result["bounds"])
    assert all(result[k]==v for k,v in recomputed.items())
    yaw=np.unwrap(data["yaw"])
    begin,end,after,finish=result["bounds"]
    terminal = min(after+1200,len(yaw)-1)
    exit_phase=None
    if end<after<len(yaw):
        direction=np.array([np.cos(yaw[end]),np.sin(yaw[end])])
        delta=data["com"][after,:2]-data["com"][end,:2]
        forward=float(np.dot(delta,direction)/((after-end)*.001))
        yaw_change=float(np.rad2deg(yaw[after]-yaw[end]))
        wt=data["loads"][end+1:after+1]
        support=float(wt.sum(1).mean()/W)
        ok=bool(result["continue_walk"]["passed"] and forward>=.05 and abs(yaw_change)<=10 and support>=.9)
        exit_phase=dict(speed_on_new_heading=forward,yaw_change_deg=yaw_change,support_w=support,passed=ok)
    tail_yaw=float(abs(np.rad2deg(yaw[terminal]-yaw[terminal-200]))) if terminal>=200 else None
    return dict(case=str(path),signed_turn=result.get("signed_turn_deg"),engineering_result=result["joint_pass"],
                action_max_error=float(np.abs(replay-action).max()),resume=exit_phase,
                terminal_pose_change_deg=float(np.rad2deg(yaw[terminal]-yaw[min(begin,terminal)])),
                tail_yaw_drift_deg=tail_yaw)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("runs",type=Path,nargs="+")
    a=p.parse_args()
    torch.set_num_threads(1)
    bank=read(HERE/"BUILD.json")
    for mode,r in bank.items():
        path=Path(r["source"])
        assert stand.sha(path/"RESULT.json")==r["result_sha"]
        assert stand.sha(path/"DECISIONS.npz")==r["decisions_sha"]
        assert stand.sha(HERE/f"BANK_{mode}.npz")==r["sha"]
    policy=load_body_policy()
    parent=PARENT/"PROTO_nominal"
    parent_record=read(parent/"RESULT.json")
    items=parent/"DECISIONS.npz"
    assert stand.sha(items)==parent_record["decision_sha256"]
    with np.load(items,allow_pickle=False) as data:
        x,c,y=(data[k] for k in ("x","intent","action"))
    baseline=np.asarray([policy(v,int(command),0) for v,command in zip(x,c,strict=True)])
    np.testing.assert_array_equal(baseline,y)
    stand.put(a.output,dict(cases=[audit(p.resolve(),policy) for p in a.runs],source_verified=True,
                            straight_and_standing_actions_bitwise=len(x)))
