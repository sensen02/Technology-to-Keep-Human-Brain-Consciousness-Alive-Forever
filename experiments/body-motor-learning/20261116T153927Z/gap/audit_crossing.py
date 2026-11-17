"""Bind actual named toe motion, far ground force and paired raw trajectory artifacts."""
import argparse
import json
from pathlib import Path

import numpy as np

from crossing import END, PER_MS, RELEASE, RESUMED_WALK, RF, W, bench, read, stan


def sustained(mask, ms):
    edges = np.flatnonzero(np.diff(np.r_[False,mask,False]))
    starts, stops = edges[::2],edges[1::2]
    selected = stops-starts >= int(round(ms*PER_MS))
    return [(int(a),int(b)) for a,b in zip(starts[selected],stops[selected],strict=True)]


def view(run):
    result = read(run/"RESULT.json")
    for name, field in (("TRACE.npz","trace_sha256"),("DECISIONS.npz","decisions_sha256"),
                        ("NATIVE.npz","native_sha256"),("GEOMETRY_NATIVE.npz","geometry_native_sha256"),
                        ("NATIVE_AUDIT.json","native_audit_sha256")):
        if field in result:
            assert stan.sha(run/name) == result[field],name
    with (
        np.load(run/"GEOMETRY_NATIVE.npz",allow_pickle=False) as geometry,
        np.load(run/"NATIVE.npz",allow_pickle=False) as raw,
        np.load(run/"TRACE.npz",allow_pickle=False) as table,
        np.load(run/"DECISIONS.npz",allow_pickle=False) as decisions,
    ):
        rf = geometry["rf_spatial"]
        plate = raw["rf_floor"]/W
        forces = raw["forces"]/W
        other = np.arange(6) != RF
        ground = plate[:,1]
        t = raw["pose"]
        z = raw["rf_pad_bottom_z"]
        xmin,xmax,ymin,ymax = result["scenario"]["opening"]
        inside_y = (rf[:,1]>ymin)&(rf[:,1]<ymax)
        opening = (rf[:,0]>xmin)&(rf[:,0]<xmax)&inside_y
        miss = opening & (z < -.012) & (forces[:,RF] <= .004) & (ground<=.001)
        missed = np.flatnonzero(miss)
        i = np.arange(len(z))
        start = int(missed[0]) if len(missed) else None
        near = sustained((rf[:,4]<xmin)&(ground>=.02)&(i<start),20) if start is not None else []
        far_pose = (rf[:,3]>xmax+.002) & (ground>=.02) & (plate[:,0]<.002) & (plate[:,3]<.002)
        in_lane = sustained(far_pose&inside_y&(i>start),30) if start is not None else []
        any_far = sustained(far_pose&(i>start),30) if start is not None else []
        lift = sustained((z>.005)&(i>start)&(rf[:,0]>xmin)&inside_y,5) if start is not None else []
        far = in_lane[0] if in_lane else None
        crossed_air = sustained((rf[:,3]>xmax+.002)&(z>.005)&inside_y&(i>start),5) if start is not None else []
        filled_loaded = sustained(opening&(ground>=.02),30) if result["scenario"]["filled"] else []
        dr = slice(lift[0][0],crossed_air[0][0]+1) if lift and crossed_air and crossed_air[0][0]>=lift[0][0] else None
        other_window = slice(start,min(len(rf),start+200*PER_MS)) if start is not None else slice(0,0)
        others = dict(min_total_W=float(forces[other_window,:6][:,other].sum(1).min()),
                      min_feet_above_02W=int((forces[other_window,:6][:,other]>=.02).sum(1).min()),
                      helper_RM_at_loss_W=float(forces[start,bench.LEGS.index("RM")])) if start is not None else None
        def event(k):
            return dict(native_step_after_t0=int(k),ms=k/PER_MS,rf_center_xyz=rf[k,:3].tolist(),
                        rf_pad_extent_x=rf[k,3:].tolist(),rf_pad_lowest_z=float(z[k]),
                        COM=t[k,:3].tolist(),top_ground_load_W=float(ground[k]),
                        bottom_ground_load_W=float(plate[k,0]),edge_normal_W=float(plate[k,3]),
                        leg_load_W=dict(zip(bench.LEGS,forces[k,:6].tolist(),strict=True)))
        center_end_x=float(table["com"][-1,0])
        end_pad_x=dict(zip(bench.LEGS,table["all_pad_xyz"][-1,:,0].tolist(),strict=True))
        after_start = min(len(forces),RELEASE*PER_MS)
        to_end = min(len(forces),(RELEASE+RESUMED_WALK)*PER_MS)
        renewed = plate[after_start:to_end]
        posture = dict(min_com=float(t[:,2].min()),peak_nonfoot_W=float(forces[:,12].max()),
                       total_support_mean_W=float(forces[:,:6].sum(1).mean()),
                       zero_extra_force=bool(not t[:,7].any()))
        assert len(t) == len(rf) == len(forces) == result["native_iterations"] == result["actual_ms"]*PER_MS
        assert np.all(np.isfinite(rf))
        return dict(path=str(run.resolve()),mode=result["mode"],steps=len(rf),actual_ms=result["actual_ms"],
            controller_trigger_ms=result["onset_ms"],
            fixed_motion_branch=dict(active_5ms_steps=int(np.sum(np.any(decisions["action"]!=decisions["base"],axis=1))),
                                     command_arrow_native_and_rad_per_s=True),
            prior_near_support=event(near[-1][0]) if near else None,
            prior_near_load_last_ms=(near[-1][1]-1)/PER_MS if near else None,
            first_slot_miss=event(start) if start is not None else None,
            peak_miss_depth_native=float(-z[miss].min()) if len(missed) else None,
            first_lift_after_loss=event(lift[0][0]) if lift else None,
            first_airborne_far_rim_clearance=event(crossed_air[0][0]) if crossed_air else None,
            all_mean_pad_y_inside_slot_until_airborne_crossing=bool(dr and inside_y[dr].all()),
            filled_support_at_same_locus=event(filled_loaded[0][0]) if filled_loaded else None,
            first_in_lane_far_support=event(far[0]) if far else None,
            far_support_sustained_ms=(far[1]-far[0])/PER_MS if far else None,
            first_far_support_any_y=event(any_far[0][0]) if any_far else None,
            helpers=others,posture=posture,before_renewal_stand=result["before_renewal_stand"],
            no_reentry_after_renewal=bool(len(renewed) and np.max(renewed[:,[0,3]]) < .002),
            center_end_x=center_end_x,end_pad_x=end_pad_x,whole_body_past_far_rim=bool(
                center_end_x>xmax and min(end_pad_x.values())>xmax),
            native_audit=read(run/"NATIVE_AUDIT.json") if (run/"NATIVE_AUDIT.json").is_file() else None)


def paired_prefix(a,b,until):
    for path,index in (("TRACE.npz",until+1),("DECISIONS.npz",until//5),
                       ("NATIVE.npz",until*PER_MS),("GEOMETRY_NATIVE.npz",until*PER_MS)):
        with np.load(a/path,allow_pickle=False) as x,np.load(b/path,allow_pickle=False) as y:
            for key in x.files:
                np.testing.assert_array_equal(x[key][:index],y[key][:index],err_msg=key)
    return True


def completed(candidate, repeat, off, flat, output, brake_only):
    case, rep, bare, floor = map(view,(candidate,repeat,off,flat))
    assert case["mode"]=="far" and rep["mode"]=="far" and bare["mode"]=="off" and floor["mode"]=="flat"
    exact = all(stan.sha(candidate/k)==stan.sha(repeat/k) for k in
             ("TRACE.npz","DECISIONS.npz","NATIVE.npz","GEOMETRY_NATIVE.npz","NATIVE_AUDIT.json"))
    prefix = paired_prefix(candidate,off,case["controller_trigger_ms"])
    minimal = case["first_slot_miss"]
    support = case["first_in_lane_far_support"]
    prior = case["prior_near_support"]
    lift = case["first_lift_after_loss"]
    rim = case["first_airborne_far_rim_clearance"]
    bare_support = bare["first_in_lane_far_support"]
    brake = view(brake_only) if brake_only is not None else None
    if brake:
        assert brake["mode"] == "brake_only"
        prefix_brake = paired_prefix(candidate,brake_only,case["controller_trigger_ms"])
    else:
        prefix_brake = False
    checks = dict(sequence=bool(prior and minimal and lift and rim and support and
                          prior["ms"] < minimal["ms"] <= case["controller_trigger_ms"] < lift["ms"] < rim["ms"] < support["ms"] < RELEASE
                          and case["all_mean_pad_y_inside_slot_until_airborne_crossing"]),
                  same_unmodified_prefix=prefix, independent_exact=exact,
                  same_gap_off_misses=bare["first_slot_miss"] is not None,
                  same_locus_filled_never_misses=(floor["first_slot_miss"] is None and
                                                floor["fixed_motion_branch"]["active_5ms_steps"]==0 and
                                                floor["filled_support_at_same_locus"] is not None),
                  earlier_far_support_than_off=bool(support and (not bare_support or support["ms"]<bare_support["ms"])),
                  first_200ms_other_support=bool(case["helpers"] and case["helpers"]["min_total_W"]>=.8 and
                                               case["helpers"]["min_feet_above_02W"]>=3),
                  continuous_recovery_zero_force=bool(case["steps"]==END*PER_MS and case["posture"]["zero_extra_force"] and
                                                case["posture"]["min_com"]>=1.2 and case["posture"]["peak_nonfoot_W"]<=.20),
                  recovered_stance=bool(case["before_renewal_stand"] and case["before_renewal_stand"]["passed"]),
                  ordinary_9_6s_stand_and_walk=bool(case["native_audit"] and case["native_audit"]["native_joint_pass"]),
                  no_bottom_or_edge_after_renewal=case["no_reentry_after_renewal"])
    foothold_keys=("sequence","same_unmodified_prefix","independent_exact","same_gap_off_misses",
                   "same_locus_filled_never_misses","earlier_far_support_than_off","first_200ms_other_support")
    brake_attribution=dict(matched_prefix=prefix_brake,earlier_far=bool(support and brake and
                (not brake["first_in_lane_far_support"] or support["ms"]<brake["first_in_lane_far_support"]["ms"])))
    out=dict(scope="one local RF-lane hole, far-rim support, commanded resumed walking",
             passed=all(checks.values()),local_far_foothold_pass=all(checks[k] for k in foothold_keys),
             checks=checks,brake_only_attribution=brake_attribution,
             candidate=case,independent=rep,no_reflex=bare,filled=floor,
             same_brake_rf_off=brake,full_D1_D8=False,weights_trained=False)
    stan.put(output,out)
    print(json.dumps(dict(passed=out["passed"],checks=checks,far_contact=support),ensure_ascii=False))
    return out["passed"]


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs",nargs="+",type=Path)
    p.add_argument("--repeat",type=Path)
    p.add_argument("--off",type=Path)
    p.add_argument("--filled",type=Path)
    p.add_argument("--brake-only",type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    if a.repeat:
        if len(a.runs)!=1 or not a.off or not a.filled or not a.output:
            p.error("Paired audit also requires exactly one candidate, --off, --filled and --output")
        raise SystemExit(0 if completed(a.runs[0],a.repeat,a.off,a.filled,a.output,a.brake_only) else 1)
    viewed=list(map(view,a.runs))
    if a.output:
        stan.put(a.output,viewed)
    else:
        print(json.dumps(viewed,ensure_ascii=False,indent=2))
