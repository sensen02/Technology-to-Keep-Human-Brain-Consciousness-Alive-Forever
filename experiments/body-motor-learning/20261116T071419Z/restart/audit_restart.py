"""Validate the new same-run physical restart, independent repeat and stationary control."""
import argparse
from pathlib import Path

import numpy as np

from run_restart import RELEASE,WALK_MS,TERMINAL,TRACE_KEYS,DECISION_KEYS,stan,read,bench,W


def check_digest(run):
    x=read(run/'RESULT.json')
    n=read(run/'NATIVE_AUDIT.json')
    assert stan.sha(run/'TRACE.npz')==x['trace_sha256']
    assert stan.sha(run/'DECISIONS.npz')==x['decision_sha256']
    assert stan.sha(run/'NATIVE_AUDIT.json')==x['native_audit_sha256']
    assert stan.sha(run/'NATIVE.npz')==n['native_sha256']
    return x,n


def audit(candidate,repeat,control,output):
    a,b,c=map(check_digest,(candidate,repeat,control))
    r,n=a
    assert r['mode']==b[0]['mode']=='immediate' and c[0]['mode']=='no_restart'
    assert r['actual_ms']==b[0]['actual_ms']==c[0]['actual_ms']==RELEASE+WALK_MS+TERMINAL
    for leaf in ('TRACE.npz','DECISIONS.npz','NATIVE.npz'):
        if stan.sha(candidate/leaf)!=stan.sha(repeat/leaf):
            raise ValueError(f'Independent reintegration differs in {leaf}')
    paired_exact={}
    with np.load(candidate/'TRACE.npz',allow_pickle=False) as x, np.load(control/'TRACE.npz',allow_pickle=False) as y:
        for k in TRACE_KEYS:
            np.testing.assert_array_equal(x[k][:RELEASE+1],y[k][:RELEASE+1],err_msg=k)
        paired_exact['trace_before_renewal']=list(TRACE_KEYS)
        sx={k:x[k] for k in x.files}
        sy={k:y[k] for k in y.files}
        hold=bench.standing(sy,RELEASE,RELEASE+WALK_MS+TERMINAL)
        active=np.flatnonzero(x['renewed_propulsion'])
        np.testing.assert_array_equal(active,np.arange(RELEASE,RELEASE+WALK_MS))
        assert not y['renewed_propulsion'].any()
        xadvance=float(sx['com'][RELEASE+WALK_MS,0]-sx['com'][RELEASE,0])
        hold_dx=float(sy['com'][RELEASE+WALK_MS,0]-sy['com'][RELEASE,0])
    with np.load(candidate/'DECISIONS.npz',allow_pickle=False) as x, np.load(control/'DECISIONS.npz',allow_pickle=False) as y:
        for k in DECISION_KEYS:
            np.testing.assert_array_equal(x[k][:RELEASE//5],y[k][:RELEASE//5],err_msg=k)
        assert np.array_equal(x['base'][RELEASE//5:],x['action'][RELEASE//5:])
        assert np.array_equal(y['base'][RELEASE//5:],y['action'][RELEASE//5:])
        paired_exact['decisions_before_renewal']=list(DECISION_KEYS)
    with np.load(candidate/'NATIVE.npz',allow_pickle=False) as x, np.load(control/'NATIVE.npz',allow_pickle=False) as y:
        for k in ('forces','pose','rf_floor','rf_pad_bottom_z'):
            np.testing.assert_array_equal(x[k][:RELEASE*80],y[k][:RELEASE*80],err_msg=k)
        paired_exact['native_before_renewal']=['forces','pose','rf_floor','rf_pad_bottom_z']
        post=x['rf_floor'][RELEASE*80:(RELEASE+WALK_MS)*80]
        n_floor=dict(supported_bottom_steps=int(np.sum(post[:,0]>.002*W)),
                     loaded_edge_steps=int(np.sum(post[:,3]>.002*W)))
        q=y['forces'][RELEASE*80:]
        raw_hold_min=float(q[:,:6].sum(1).min()/W)
        raw_hold_nonfoot=float(q[:,12].max()/W)
    checks=dict(pre_restart_original_stance=bool(r['prefix_stance_before_release']['passed']),
                independent_exact=True,stationary_control_accepted=bool(hold['passed'] and raw_hold_min>=.9 and raw_hold_nonfoot<.01),
                causal_renewal_small_control_drift=bool(xadvance>.40 and abs(hold_dx)<.01),
                first_strict_step=r['first_step'] is not None,resumed_six_foot_cycles=all(r['cycles'].values()),
                resumed_and_standing_joint=bool(r['joint_sampled_pass']),
                resumed_native_threshold=bool(n['native_joint_pass']),
                same_physical_opening_no_reentry=all(g==0 for g in n_floor.values()),
                native_zero_force=bool(n['zero_extra_force'] and c[1]['zero_extra_force']))
    summary=dict(scope='same continuous physical AS brake, full unassisted restart and final stand',
        passed=all(checks.values()),checks=checks,pairs=paired_exact,comparators={'renewal':str(candidate.resolve()),
               'independent_repeat':str(repeat.resolve()),'no_renewal':str(control.resolve())},
        renewed_at_ms=RELEASE,ending_ms=RELEASE+WALK_MS+TERMINAL,
        independent_native_steps=n['native_integrations'],continued_walk_native_steps=WALK_MS*80,
        matched_old_prefix_ms=RELEASE,first_strict_step=r['first_step'],
        first_restarted_step_latency_ms=r['restart']['first_resume_latency_ms'],cycles=r['cycles'],
        raw_slip=n['max_ground_normalized_slip'],raw_forward_speed_native_s=n['signed_forward_native_s'],
        raw_floor_evidence=n_floor,terminal_native_stand=n['standing']['terminal'],
        post_restart_signed_COM_dx=xadvance,no_renewal_signed_COM_dx=hold_dx,
        original_terminal_standing_control=hold,full_D1_D8=False)
    stan.put(output,summary)
    print(summary)
    if not summary['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate',type=Path)
    p.add_argument('repeat',type=Path)
    p.add_argument('control',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    audit(a.candidate,a.repeat,a.control,a.output)
