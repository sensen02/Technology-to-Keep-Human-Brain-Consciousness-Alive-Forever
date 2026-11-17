"""Qualify the local RF reaction from matched real collision and own replay."""
import argparse
import json
from pathlib import Path

import numpy as np

from rollout import stan, W, LEGS, build_scene


def read(path):
    return json.loads(path.read_text())


def retained(path):
    r=read(path/'RESULT.json')
    for artifact,digest in r['files'].items():
        assert stan.sha(path/artifact)==digest
    return r


def audit(output, on, off, repeat):
    a,b,c = [retained(path) for path in (on,off,repeat)]
    assert a['mode']==c['mode']=='tactile' and b['mode']=='blind'
    assert a['plan']==b['plan']==c['plan'] and a['files']==c['files']
    assert a['actual_ms']==b['actual_ms']==c['actual_ms']==5500
    assert a['triggered']==c['triggered'] and len(a['triggered'])==1 and not b['triggered']
    trigger=a['triggered'][0]
    leg=trigger['leg']
    i=LEGS.index(leg)
    ms=int(round(trigger['relative_s']*1000))+a['plan']['warm_ms']
    ap,bp=a['block_passage'],b['block_passage']
    source_top=ap['first_box_top_native_step'][leg]
    same_face=ap['first_box_face_native_step'][leg]
    assert same_face==bp['first_box_face_native_step'][leg]
    with np.load(on/'DECISIONS.npz',allow_pickle=False) as x, np.load(off/'DECISIONS.npz',allow_pickle=False) as y:
        diff=np.flatnonzero(np.any(x['action']!=y['action'],axis=1))
        first_difference=int(x['ms'][diff[0]])
        signal=float(x['swing_resistance_per_W'][ms//5])
        assert signal==float(y['swing_resistance_per_W'][ms//5])==trigger['resistance_W']
        change_joints=np.flatnonzero(x['action'][diff[0]]!=y['action'][diff[0]]).tolist()
        assert all(i*7<=j<(i+1)*7 for j in change_joints)
    with np.load(on/'TRACE.npz',allow_pickle=False) as x, np.load(off/'TRACE.npz',allow_pickle=False) as y:
        for key in ('qpos','qvel','com','pad_xyz','loads','block_native'):
            np.testing.assert_array_equal(x[key][:first_difference+1],y[key][:first_difference+1])
        model,_,_=build_scene(a['plan'])
        radius=model.plant.geom_size[model.pads[leg],0]
        def pad_bottom(trace):
            return np.min(trace['pad_xyz'][ms:ms+500,i,:,2]-radius,axis=1)
        xa, xb=pad_bottom(x),pad_bottom(y)
        other=np.delete(np.arange(6),i)
        centers=[z['pad_xyz'][ms:ms+170,i,:,0].mean(1) for z in (x,y)]
        backward=[float(p[0]-p.min()) for p in centers]
        recovery=dict(actual_early_posterior_RF_pad_shift_native=backward,
                      actual_max_RF_pad_height_vs_floor_native=[float(v.max()) for v in (xa,xb)],
                      maximum_same_time_RF_lift_native=float((xa-xb).max()),
                      minimum_other_feet_W=[float(z['loads'][ms:ms+500][:,other].sum(1).min()/W) for z in (x,y)])
    impulse=[z['box_face_impulse_W_s'][leg] for z in (ap,bp)]
    checks=dict(local_force_precedes_correction=bool(same_face/80<=ms<first_difference and signal>=a['plan']['reflex_detect_W']),
                action_prefix_bitwise=True, independent_repeat_bitwise=True,
                corrected_leg_sustained_load=bool(source_top is not None and source_top/80>first_difference),
                no_corresponding_control_top_load=bp['first_box_top_native_step'][leg] is None,
                positive_physical_retraction_and_raising=bool(recovery['maximum_same_time_RF_lift_native']>.02
                    and recovery['actual_early_posterior_RF_pad_shift_native'][0]>.02
                    and recovery['actual_early_posterior_RF_pad_shift_native'][1]<.01),
                same_leg_lower_face_impulse=impulse[0]<=.25*impulse[1],
                safety_margin_all_probe_still_upright=bool(min(z['native_engineering']['minimum_com'] for z in (a,b))>=1.2))
    result=dict(kind='Physically qualified local closed-loop teacher only.',passed=all(checks.values()),checks=checks,
                trial_ms=a['actual_ms'],terminal_stand_validation=False,whole_body_crossing_validation=False,
                actual_first_face_ms=same_face/80, first_cue_gate_ms=ms,
                first_different_motor_ms=first_difference,first_corrected_joints=change_joints,
                actual_first_sustained_top_load_ms=source_top/80,
                max_active_top_load_W=ap['box_top_peak_W'][leg], max_control_top_load_W=bp['box_top_peak_W'][leg],
                on_and_off_side_impulse_W_s=impulse,side_impulse_reduction=float(1.-impulse[0]/impulse[1]),
                raw_slip_on_off=[z['native_engineering']['raw_slip_ratio'] for z in (a,b)],
                local_response_dynamics=recovery,
                artifact_hashes=[z['files'] for z in (a,b,c)], evidence_paths=[str(p) for p in (on,off,repeat)])
    stan.put(output,result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('output','on','off','repeat'):
        p.add_argument(field,type=Path)
    a=p.parse_args()
    audit(a.output.resolve(),a.on.resolve(),a.off.resolve(),a.repeat.resolve())
