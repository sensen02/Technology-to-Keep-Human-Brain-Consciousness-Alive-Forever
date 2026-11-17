"""Hash-bound concise output for every completed paired gap family."""
from pathlib import Path

import numpy as np

from physical import read,stan,W,RF,RM,RF_JOINTS,RM_JOINTS


def case(path):
    r=read(path/'RESULT.json')
    a=read(path/'NATIVE_AUDIT.json')
    for name,key in [('RESULT.json',None),('NATIVE_AUDIT.json','native_audit_sha256'),
                     ('TRACE.npz','trace_sha256'),('DECISIONS.npz','decision_sha256')]:
        if key is not None and stan.sha(path/name)!=r[key]:
            raise ValueError(f'Run identity changed: {path/name}')
    fingerprint=stan.sha(path/'RESULT.json')
    return dict(path=str(path.resolve()),result_sha256=fingerprint,identity=stan.sha(path/'RUN_IDENTITY.json'),
                 onset_ms=r.get('teacher_engagement_ms'),steps=a['native_integrations'],max_slip=a['max_ground_normalized_slip'],
                 forward=a['signed_forward_native_s'],ground_nonfoot_peak_W=a['ground_nonfoot_peak_W'],
                 native_joint=a['native_joint_pass'],stand=a['standing'],gates=a['native_gates'],
                 mode=r['mode'],drop_ms=r['event'].get('first_pad_drop_ms'),
                 upper_reload_ms=r['event'].get('first_upper_floor_reload_ms'),
                 depth_native=r['event'].get('peak_cavity_penetration_native'))


def contrast(full,off,duration):
    a={key:read(full/'RESULT.json').get(key) for key in ('scenario','initial_snapshot_sha256','teacher_engagement_ms')}
    b={key:read(off/'RESULT.json').get(key) for key in a}
    assert a==b
    start=a['teacher_engagement_ms']
    if start is None:
        return dict(all_inputs_paired=True,intervention_detected=False)
    k0=start*80
    k1=(start+duration)*80
    with np.load(full/'NATIVE.npz',allow_pickle=False) as n, np.load(off/'NATIVE.npz',allow_pickle=False) as m:
        whole=n['forces']
        ab=m['forces']
        if min(len(whole),len(ab))<k1:
            raise ValueError('Shorter trace than support comparison')
        np.testing.assert_array_equal(whole[:k0],ab[:k0])
        active_support=whole[k0:k1,RM]/W
        ordinary_support=ab[k0:k1,RM]/W
        other=whole[k0:k1,:6].sum(1)/W-whole[k0:k1,RF]/W
    with np.load(full/'DECISIONS.npz',allow_pickle=False) as z, np.load(off/'DECISIONS.npz',allow_pickle=False) as y:
        mask=(z['ms']>=start)&(z['ms']<start+duration)
        actual=z['action'][mask,RM_JOINTS]
        before=y['action'][mask,RM_JOINTS]
        rm_rows=int(np.count_nonzero(np.max(np.abs(actual-before),axis=1)>1e-10))
    return dict(window_ms=[start,start+duration],same_run_identity=a,prefix_raw_load_bitwise_exact=True,
                right_mid_leg_command_difference_rows=rm_rows,other_five_minimum_W=float(other.min()),
                mean_full_right_mid_W=float(active_support.mean()),
                mean_retraction_only_right_mid_W=float(ordinary_support.mean()),
                mean_right_mid_increase_W=float((active_support-ordinary_support).mean()),
                right_mid_bearing_at_first_step_full_W=float(active_support[0]),
                right_mid_bearing_at_first_step_retraction_only_W=float(ordinary_support[0]),
                right_mid_not_newly_planted=bool(active_support[0]>.02 and ordinary_support[0]>.02))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    families=('DEV_TEACH_AX','DEV_RF_ONLY_AX','CONTROL_AN_FILLED','CONTROL_AN','TEACH_SERVO_AN','RF_ONLY_SERVO_AN',
              'CONTROL_AE_FILLED','CONTROL_AE','TEACH_EDGE_EARLY','TEACH_HOLD_AE',
              'CONTROL_AL_FILLED','CONTROL_AL','TEACH_HOLD_AL','CONTROL_AS_FILLED','CONTROL_AS',
              'TEACH_BIAS_AS','TEACH_BIAS_AS_RF_ONLY')
    cases={name:case(Path(name)) for name in families if (Path(name)/'NATIVE_AUDIT.json').exists()}
    pairs={}
    for name,full,off,ms in (
            ('AX_200ms','DEV_TEACH_AX','DEV_RF_ONLY_AX',200),
            ('AN_200ms','TEACH_SERVO_AN','RF_ONLY_SERVO_AN',200),
            ('AN_435ms','TEACH_SERVO_AN','RF_ONLY_SERVO_AN',435),
            ('AS_200ms','TEACH_BIAS_AS','TEACH_BIAS_AS_RF_ONLY',200),
            ('AS_435ms','TEACH_BIAS_AS','TEACH_BIAS_AS_RF_ONLY',435)):
        if full in cases and off in cases:
            pairs[name]=contrast(Path(full),Path(off),ms)
    stan.put(args.output,dict(cases=cases,paired_causal_support=pairs,full_D1_D8=False))
