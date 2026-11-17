"""Qualify the native unloaded-to-bearing transition with a helper-action ablation."""
import argparse
from pathlib import Path

import numpy as np

from support_experiment import PER_MS, STEP, W, bench, group, read, stan

LOAD = .020
HOLD_MS = 20


def open_measured(place):
    result = read(place/"RESULT.json")
    for name,key in (("TRACE.npz","trace_sha256"),("DECISIONS.npz","decision_sha256"),("NATIVE.npz","native_sha256")):
        if stan.sha(place/name) != result[key]:
            raise ValueError(f"Archive checksum differs for {place}/{name}")
    return (result,np.load(place/"TRACE.npz",allow_pickle=False),
            np.load(place/"DECISIONS.npz",allow_pickle=False),np.load(place/"NATIVE.npz",allow_pickle=False))


def spans(mask,start=0):
    change = np.diff(np.r_[False, mask[start:], False].astype(int))
    return list(zip(start+np.flatnonzero(change==1),start+np.flatnonzero(change==-1),strict=True))


def first_sustained(mask,start):
    return next(((int(a),int(b)) for a,b in spans(mask,start) if b-a >= HOLD_MS * PER_MS),None)


def layout_clear(points,opening,margin=.010):
    x0,x1,y0,y1 = opening
    return ((points[:,0] < x0-margin) | (points[:,0] > x1+margin)
            | (points[:,1] < y0-margin) | (points[:,1] > y1+margin))


def stand_final(raw):
    force = raw["forces"][-200*PER_MS:]/W
    pose = raw["pose"][-200*PER_MS:]
    return dict(support_mean_W=float(force[:,:6].sum(1).mean()),
                fraction_three=float(((force[:,:6]>=LOAD).sum(1)>=3).mean()),
                COM_min=float(pose[:,2].min()),nonfoot_max_W=float(force[:,12].max()),
                upright_dot_min=float(pose[:,5].min()))


def audit(on,off,filled,out,replay=None):
    a,at,ad,an = open_measured(on)
    b,bt,bd,bn = open_measured(off)
    g,gt,gd,gn = open_measured(filled)
    try:
        assert a["mode"]=="on" and b["mode"]=="off" and g["mode"]=="control"
        for key in ("scenario","target","helper","initial_snapshot_sha256","first_trigger_ms"):
            if a[key]!=b[key]:
                raise ValueError(f"Arms differ in the declared matching identity {key}")
        assert dict(a["scenario"],filled=True)==g["scenario"]
        assert a["initial_snapshot_sha256"] == g["initial_snapshot_sha256"]
        assert all(arm["native_count"] == arm["actual_ms"] * PER_MS for arm in (a,b,g))
        if a["first_helper_ms"] is None:
            raise ValueError("No active helper command in the measured episode")
        helper = bench.LEGS.index(a["helper"])
        missing = bench.LEGS.index(a["target"])
        hs = group(helper)
        act_a,act_b = ad["action"],bd["action"]
        shared = min(len(act_a),len(act_b))
        np.testing.assert_array_equal(ad["ms"][:shared],bd["ms"][:shared])
        differ = np.flatnonzero(np.any(act_a[:shared]!=act_b[:shared],axis=1))
        h = int(ad["ms"][differ[0]])
        n = h*PER_MS
        decision_idx = differ[0]
        action_deltas = np.flatnonzero(act_a[decision_idx]!=act_b[decision_idx])
        assert len(action_deltas) and np.all(action_deltas//7==helper)
        event = a["first_trigger_ms"]*PER_MS
        assert min(arm["native_count"] for arm in (a,b,g)) > max(event,n) + HOLD_MS*PER_MS
        exact = (all(np.array_equal(at[key][:h+1],bt[key][:h+1])
                     for key in ("qpos","qvel","com","loads","target","padz","pair_pad_xyz"))
                 and all(np.array_equal(an[key][:n],bn[key][:n])
                         for key in ("forces","pose","pair_pad","missing_floor")))
        shape = a["scenario"]["opening"]
        x0,x1,y0,y1 = shape
        ap, bp = an["pair_pad"],bn["pair_pad"]
        apad = ap[event,0]
        inside = x0 < apad[0] < x1 and y0 < apad[1] < y1
        load_a, load_b, load_f = [raw["forces"][:,:6]/W for raw in (an,bn,gn)]
        pre = slice(max(0,n-20*PER_MS),n)
        prem = load_a[pre,helper]
        aclr = layout_clear(ap[:,1],shape)
        bclr = layout_clear(bp[:,1],shape)
        start_a = first_sustained(aclr & (load_a[:,helper] >= LOAD),n)
        start_b = first_sustained(bclr & (load_b[:,helper] >= LOAD),n)
        if start_a is None:
            aa = bb = None
        else:
            aa,bb = start_a
        checks = dict(paired_pre_helper_bitwise=bool(exact), acts_at_or_after_matched_miss=bool(a["first_trigger_ms"]<=h),
                      loss_inside_cut_and_clearance=bool(inside and apad[3]<-.008),
                      losing_foot_unloaded=bool(load_a[event,missing]<=.001),
                      filled_reference_takes_load_now=bool(load_f[event,missing]>.001),
                      helper_zero_contact_before=bool(len(prem) and not prem.any()),
                      helper_positive_clearance_before=bool(ap[pre,1,3].min()>0.),
                      helper_sustained_real_upper_ground=bool(start_a is not None),
                      helper_contacts_earlier_than_off=bool(start_a is not None and
                                                          (start_b is None or start_a[0]<start_b[0])),
                      active_arm_recovers_stable_terminal=bool(a["terminal_stand"]["passed"]),
                      all_native_command_wrench_zero=bool(not an["pose"][:,7].any() and not bn["pose"][:,7].any()))
        times = dict(trigger_ms=event/PER_MS,first_helper_action_ms=h, helper_on_load_ms=aa/PER_MS if aa is not None else None,
                     helper_off_load_ms=start_b[0]/PER_MS if start_b else None,helper_on_contact_duration_ms=(bb-aa)/PER_MS if aa is not None else None,
                     accelerated_by_ms=(start_b[0]-aa)/PER_MS if start_b and aa is not None else None)
        critical = slice(n,min(n+500*PER_MS,len(load_a)))
        body = {}
        for label,raw,load in (("on",an,load_a),("off",bn,load_b)):
            body[label] = dict(minimum_total_W=float(load[critical].sum(1).min()),
                               minimum_COM=float(raw["pose"][critical,2].min()),
                               maximum_nonfoot_500ms_W=float(raw["forces"][critical,12].max()/W),
                               sampled_terminal_pass=bool((a if label=="on" else b)["terminal_stand"]["passed"]),
                               actual_ms=(a if label=="on" else b)["actual_ms"],terminal_200ms=stand_final(raw))
        report = dict(scope="actual single alternative landing and stop",qualified=all(checks.values()),checks=checks,
                      source_arms={k:str(v) for k,v in (("on",on),("off",off),("filled",filled))},
                      target=a["target"],helper=a["helper"],actual_ms=a["actual_ms"],native_steps=a["native_count"],
                      only_helper_differing_dofs=action_deltas.tolist(), times=times, native_step_s=STEP,
                      target_force_at_trigger_W=float(load_a[event,missing]),filled_target_force_at_trigger_W=float(load_f[event,missing]),
                      target_pad_bottom_at_trigger_native=float(apad[3]),helper_pad_lowest_pre_action_native=float(ap[pre,1,3].min()),
                      helper_on_placement_xyz_native=ap[aa,1,:3].tolist() if aa is not None else None,
                      helper_loaded_min_20ms_W=float(load_a[aa:aa+HOLD_MS*PER_MS,helper].min()) if aa is not None else None,
                      off_load_peak_during_first_20ms_on_load_W=float(load_b[aa:aa+HOLD_MS*PER_MS,helper].max()) if aa is not None else None,
                      load_threshold_W=LOAD,required_duration_ms=HOLD_MS,body=body)
        if replay is not None:
            other,ot,od,onative = open_measured(replay)
            try:
                report["independent_replay"] = dict(path=str(replay),exact={
                    k:other[k]==a[k] for k in ("trace_sha256","decision_sha256","native_sha256")})
                report["independent_replay"]["passed"] = all(report["independent_replay"]["exact"].values())
            finally:
                ot.close()
                od.close()
                onative.close()
        stan.put(out,report)
        print(report,flush=True)
    finally:
        for f in (at,ad,an,bt,bd,bn,gt,gd,gn):
            f.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("on",type=Path)
    p.add_argument("off",type=Path)
    p.add_argument("filled",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--replay",type=Path)
    a=p.parse_args()
    audit(a.on.resolve(),a.off.resolve(),a.filled.resolve(),a.output.resolve(),a.replay.resolve() if a.replay else None)
