"""Measure crossing of the far x station by the COM and six actual support pads."""
import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from audit_crossing import paired_prefix, sustained
from crossing import PER_MS, RELEASE, W, bench, build_scene, read, stan
from geometry import make_gap_model
from hybrid_floor import applied


def main(travel, short, output, repeat=None, local=None):
    result=read(travel/"RESULT.json")
    short_summary=read(short.parent/"CLEAR_ACCEPTANCE.json")
    assert short_summary["passed"] and short_summary["candidate"]["path"]==str(short.resolve())
    for file,field in (("TRACE.npz","trace_sha256"),("DECISIONS.npz","decisions_sha256"),
                       ("NATIVE.npz","native_sha256"),("GEOMETRY_NATIVE.npz","geometry_native_sha256"),
                       ("NATIVE_AUDIT.json","native_audit_sha256")):
        if field in result:
            assert stan.sha(travel/file)==result[field],file
    scenario=result["scenario"]
    x0,x1,y0,y1=scenario["opening"]
    long_duration=result["actual_ms"]
    anchored=result["mode"]=="far_anchored_course"
    full_walk_ms=result["walk_ms"] if anchored else 45_600
    expected_duration=RELEASE+full_walk_ms+1_200
    assert long_duration>RELEASE and long_duration<=expected_duration
    declared_identity=read(travel/"RUN_IDENTITY.json")["config"]
    if local is None:
        common_ms=min(long_duration,RELEASE if anchored else RELEASE+9_600)
        local_proof=paired_prefix(travel,short,common_ms)
        matched_history=str(short.resolve())
        original_far=short_summary["candidate"]["first_in_lane_far_support"]
    else:
        revised=read(local)
        assert revised["candidate"]["path"]==str(travel.resolve())
        same_start=(declared_identity["initial_snapshot_sha256"]==
                     read(short/"RUN_IDENTITY.json")["config"]["initial_snapshot_sha256"])
        same_opening=(result["scenario"]["opening"]==read(short/"RESULT.json")["scenario"]["opening"]
                      and result["scenario"]["depth"]==read(short/"RESULT.json")["scenario"]["depth"])
        common_ms=revised["matched_geometry_prefix_ms"]
        local_proof=bool(revised["passed"] and same_start and same_opening)
        matched_history=str(local.resolve())
        original_far=revised["candidate"]["first_in_lane_far_support"]
    mandatory=("TRACE.npz","DECISIONS.npz","NATIVE.npz","GEOMETRY_NATIVE.npz")
    artifacts=(*mandatory,"NATIVE_AUDIT.json") if (travel/"NATIVE_AUDIT.json").is_file() else mandatory
    exact_repeat=None
    if repeat is not None:
        exact_repeat=all(stan.sha(travel/k)==stan.sha(repeat/k) for k in artifacts)
        assert exact_repeat
    with np.load(travel/"TRACE.npz",allow_pickle=False) as table, np.load(travel/"NATIVE.npz",allow_pickle=False) as native:
        toe=table["all_pad_xyz"]
        pos=native["pose"]
        no_extra=not pos[:,7].any()
        center_native=np.flatnonzero((pos[:,0]>x1)&(np.arange(len(pos))>=RELEASE*PER_MS))
        first_COM=float(center_native[0]/PER_MS) if len(center_native) else None
        del pos
        load=native["forces"][:,:6]/W
        assert len(load)==result["native_iterations"]==long_duration*PER_MS
        positive_slots=native["rf_floor"][RELEASE*PER_MS:]/W
        bottom_steps=int(np.sum(positive_slots[:,0]>=.002))
        edge_steps=int(np.sum(positive_slots[:,3]>=.002))
        del positive_slots

        # Replay the terminal state once to recover all three pads' true extents.
        with applied():
            model=make_gap_model(scenario)
            c, _ = build_scene(scenario,bench.STARTS/"nominal.npz",
                               read(bench.STARTS/"BANK.json")["nominal"]["sha256"])
        data=mj.MjData(model)
        data.qpos[:]=table["qpos"][-1]
        data.qvel[:]=table["qvel"][-1]
        mj.mj_forward(model,data)
        pads=np.asarray([c.pads[name] for name in bench.LEGS])
        assert all(c.plant.geom(i).name==model.geom(i).name for i in pads.ravel())
        final_min=(data.geom_xpos[pads,0]-model.geom_size[pads,0]).min(1)
        leg_data={}
        start_index=np.arange(len(load))
        for j,name in enumerate(bench.LEGS):
            # Conservative radius bound from each pad triplet; body coordinates alone
            # do not classify a sole under load on the far side.
            assert len(np.unique(model.geom_bodyid[pads[j]]))==1
            p=data.geom_xpos[pads[j]]
            offsets=np.linalg.norm(p-p.mean(0),axis=1)
            margin=float(max(.035,np.max(offsets+model.geom_size[pads[j],0])))
            ahead=(toe[:,j,0]>x1+margin)&(np.arange(len(toe))>=RELEASE)
            sample_native=ahead[np.minimum(start_index//PER_MS,len(ahead)-1)]
            available=sustained(sample_native&(load[:,j]>=.02),30)
            stable_tail=load[-200*PER_MS:,j]
            leg_data[name]=dict(first_far_support_ms=available[0][0]/PER_MS if available else None,
                                final_center_xyz=toe[-1,j].tolist(),final_pad_min_x=float(final_min[j]),
                                final_200ms_mean_W=float(stable_tail.mean()),
                                final_200ms_min_W=float(stable_tail.min()))
        terminal_com=table["com"][-1].tolist()
        last_active=min(RELEASE+full_walk_ms,long_duration)
        post_release_dx=float(table["com"][last_active,0]-table["com"][RELEASE,0])
    native=read(travel/"NATIVE_AUDIT.json") if (travel/"NATIVE_AUDIT.json").is_file() else None
    checks=dict(full_scheduled_run=(long_duration==expected_duration),
                independent_selected_early_rail=local_proof,
                original_far_slot_contact_retained=original_far is not None,
                center_and_all_foot_pads_beyond_far_x=bool(first_COM and terminal_com[0]>x1 and
                    all(v["first_far_support_ms"] and v["final_pad_min_x"]>x1
                        for v in leg_data.values())),
                extended_stock_gates=bool(native and native["native_joint_pass"]),
                final_standing=bool(native and native["standing"]["terminal"]["passed"]),
                no_external_force=bool(no_extra and (native is None or native["zero_extra_force"])))
    final=dict(scope="single literal RF-lane slot and subsequent passage of its x station by COM and all six pads",
               whole_body_path_support_crossed_x_station=checks["center_and_all_foot_pads_beyond_far_x"],
               transverse_gap_under_whole_footprint=False, passed=all(checks.values()),
               prior_selected_short=str(short.resolve()),extended=str(travel.resolve()),
               same_physical_prefix_through_ms=common_ms,matched_aligned_history=matched_history,
               renewed_at_ms=RELEASE,actual_duration_ms=long_duration,
               mode=result["mode"], independent_repeat=str(repeat.resolve()) if repeat else None,
               independent_exact_bytes=exact_repeat,
               actual_native_steps=result["native_iterations"],expected_duration_ms=expected_duration,
               x_tile_clearance_plan=declared_identity.get("tile_seam_contact_management"),
               post_renewal_RF_bottom_steps=bottom_steps,post_renewal_RF_edge_steps=edge_steps,
               gap_opening_xyz=dict(x=[x0,x1],y=[y0,y1],bottom=-scenario["depth"]),
               first_COM_beyond_far_x_ms=first_COM, final_COM_xyz=terminal_com, feet=leg_data,
               post_release_x_displacement_native=post_release_dx,checks=checks,raw_native_audit=native)
    stan.put(output,final)
    print(json.dumps(dict(passed=final["passed"],checks=checks,native_steps=final["actual_native_steps"],
                          final_COM=terminal_com,first_COM_ms=first_COM,feet=leg_data),ensure_ascii=False))
    return final["passed"]


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("long",type=Path)
    p.add_argument("short",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--repeat",type=Path)
    p.add_argument("--local",type=Path)
    a=p.parse_args()
    raise SystemExit(0 if main(a.long,a.short,a.output,a.repeat,a.local) else 1)
