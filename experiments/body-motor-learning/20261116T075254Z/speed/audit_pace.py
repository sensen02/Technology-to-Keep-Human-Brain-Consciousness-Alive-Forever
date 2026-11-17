"""Qualify physically measured forward pace, separately from command smoothness."""
import argparse
from pathlib import Path

import numpy as np

from speed_teacher import NOMINAL_V, REFERENCE, SOURCE, WARM, stand
from walking_bench import read


def arm(run):
    names={"TRACE.npz":"trace_sha256","NATIVE.npz":"native_sha256","DECISIONS.npz":"decision_sha256"}
    result = read(run/"RESULT.json")
    for f,key in names.items():
        if stand.sha(run/f) != result[key]:
            raise ValueError(f"Native record or trajectory checksum mismatch {run/f}")
    with np.load(run/"NATIVE.npz",allow_pickle=False) as n, np.load(run/"DECISIONS.npz",allow_pickle=False) as d:
        assert len(n["pose"]) == len(n["forces"]) == result["native_steps"] == 80*result["actual_ms"]
        assert (n["pose"][:,7]==0.).all() and result["zero_extra_force"]
        np.testing.assert_array_equal(d["ms"],np.arange(len(d["ms"]))*5)
        assert d["x"].shape==(len(d["ms"]),213) and d["action"].shape==(len(d["ms"]),42)
        assert d["command"].shape==(len(d["ms"]),2) and not d["command"][:,1].any()
        assert np.isfinite(d["action"]).all() and (abs(d["action"])<=1.).all()
        active = d["command"][:,0]>0
        np.testing.assert_array_equal(active,(d["ms"]>=WARM)&(d["ms"]<WARM+result["walk_ms"]))
        command = d["command"][active,0]
    return dict(path=str(run),native_joint_pass=bool(result["native_joint_pass"]),
                native_steps=result["native_steps"],source_result_sha256=stand.sha(run/"RESULT.json"),
                command_min_native_s=float(command.min()),command_max_native_s=float(command.max()),
                command_max_per_decision_increment_native_s=float(abs(np.diff(command)).max()),
                actual_forward_native_s=result["walk"]["signed_forward_native_s"],
                actual_max_normalized_slip=result["walk"]["max_ground_normalized_slip"],
                movement_gates=result["native_gates"],measurement_windows=result["windows"],
                gait_cycles=result["gait_cycles"],initial=result["initial_stand"]["passed"],
                terminal=result["terminal_stand"]["passed"])


def stock_compare(stock):
    q=[stock/"TRACE.npz",REFERENCE]
    archive_action=SOURCE/"PROTO_nominal"/"DECISIONS.npz"
    with np.load(q[0]) as a, np.load(q[1]) as b:
        common=sorted(set(a.files)&set(b.files))
        exact={key:bool(np.array_equal(a[key],b[key])) for key in common}
    with np.load(stock/"DECISIONS.npz") as a, np.load(archive_action) as b:
        for key in ("ms","x","action"):
            exact["decision_"+key]=bool(np.array_equal(a[key],b[key]))
        exact["old_0_1_intent_semantics"]=bool(np.array_equal(a["command"][:,0]>0,b["intent"]!=0))
    return dict(source=str(REFERENCE),archive_decisions=str(archive_action),comparison=exact,
                passed=all(exact.values()))


def audit(output,stock,cases,pair):
    statuses = [arm(run) for run in cases]
    parents=[read(run/"RUN_IDENTITY.json")["config"]["initial_sha256"] for run in [stock,*cases]]
    if len(set(parents))!=1:
        raise ValueError("Pacing comparison starts from different frozen snapshots")
    constants=[r for r in statuses if r["command_min_native_s"]==r["command_max_native_s"]]
    constant_qualified=sorted([r for r in constants if r["native_joint_pass"]],key=lambda r:r["command_min_native_s"])
    speeds=[r["actual_forward_native_s"] for r in constant_qualified]
    monotonic=bool(len(speeds)>=3 and (np.diff(speeds)>0.001).all())
    result=dict(scope="same-body pacing; teachers only; full D1–D8 remains separate",
                common_command_columns=["requested forward native length/s","requested signed yaw rad/s"],
                reference_speed_native_s=NOMINAL_V,legacy_stock=stock_compare(stock),cases=statuses,
                physically_qualified_teachers=[r["path"] for r in statuses if r["native_joint_pass"]],
                distinct_ordered_three_qualified_speeds=monotonic,qualified_constant_count=len(constant_qualified))
    if pair is not None:
        parent, replay = pair
        check=arm(replay)
        result["native_restart"]=dict(original=str(parent),retry=str(replay),native_joint_pass=check["native_joint_pass"],
                             exact_matches={path:stand.sha(parent/path)==stand.sha(replay/path)
                                            for path in ("TRACE.npz","DECISIONS.npz","NATIVE.npz")})
    stand.put(output,result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--stock",type=Path,required=True)
    p.add_argument("--cases",type=Path,nargs="+",required=True)
    p.add_argument("--replay-pair",type=Path,nargs=2)
    a=p.parse_args()
    audit(a.output.resolve(),a.stock.resolve(),[q.resolve() for q in a.cases],
          [q.resolve() for q in a.replay_pair] if a.replay_pair else None)
