"""Independent saved-array LEFT audit. No Core, MuJoCo, training or rendering.

Metrics follow the frozen physics.summarise endpoint formulas exactly; native
peaks remain captured summaries, never a claim of independent native D1-D8.
No saved result means NOT_RUN, not a successful trajectory audit.
"""
import json
import os
import sys
import numpy as np
from source_identity import HERE, verify_sources, sha256


def verify():
    verify_sources()
    allowed=os.sched_getaffinity(0)
    os.sched_setaffinity(0,{1 if 1 in allowed else min(allowed)})
    folder=HERE/'left_yaw_p060'
    if not (folder/'RESULT.json').exists():
        print(json.dumps(dict(status='NOT_RUN',reason='No new physical RESULT.json; no trajectory audited',root=str(HERE))))
        return None
    # Explicit absolute baseline, never infer it from the output directory depth.
    from turning_teacher import BASELINE
    sys.path.insert(0,str(BASELINE))
    import identity
    from cycle_reward import CycleTracker,CycleConfig,LEG_ORDER
    r=json.loads((folder/'RESULT.json').read_text())
    manifest=json.loads((folder/'IDENTITY.json').read_text())
    identity.verify(manifest)
    with np.load(folder/'TRACE_1MS.npz',allow_pickle=False) as z:t={k:z[k] for k in z.files}
    with np.load(folder/'SAMPLES_5MS.npz',allow_pickle=False) as z:s={k:z[k] for k in z.files}
    with np.load(folder/'YAW_1MS.npz',allow_pickle=False) as z:y={k:z[k] for k in z.files}
    checks={}
    def check(k,v):checks[k]=bool(v)
    def close(a,b,tol=1e-10):return bool(np.shape(a)==np.shape(b) and np.max(np.abs(np.asarray(a)-np.asarray(b)))<tol)
    check('single_left_condition',r['condition']==dict(name='left_yaw_p060',speed_scale=1.,yaw_rate=.06))
    check('identity_condition_root',manifest['config']['condition']==r['condition'] and manifest['config']['prereg']==str(HERE/'PREREG.json'))
    check('source_freeze',r['sourcehash_sha256']==sha256(HERE/'SOURCEHASH.json'))
    check('extra_artifact_hashes',all(sha256(folder/name)==h for name,h in r['artifact_hashes'].items()))
    check('trace_hash',sha256(folder/'TRACE_1MS.npz')==r['trace']['sha256'])
    check('samples_hash',sha256(folder/'SAMPLES_5MS.npz')==r['samples']['sha256'])
    check('finite_all',all(np.isfinite(a).all() for group in (t,s,y) for a in group.values()))
    check('full15400',len(t['time'])==15400)
    check('full2880samples',s['X238'].shape==(2880,238) and s['Y42'].shape==(2880,42) and s['command2'].shape==(2880,2))
    # Shortened/early-stop runs cannot qualify, and must remain visibly failed.
    if len(t['time'])!=15400 or s['Y42'].shape!=(2880,42):
        result=dict(status='INCOMPLETE_TRAJECTORY_FAIL',integrity_all_pass=False,checks=checks,teacher_eligible=False,
                    early_stop=r.get('early_stop'),scope='Saved incomplete arrays retained; full-window audit unavailable')
        (folder/'VERIFY.json').write_text(json.dumps(result,indent=2)+'\n')
        raise RuntimeError('Incomplete candidate trajectory retained as FAIL')
    check('sources',np.array_equal(t['control_source'],np.r_[np.zeros(1000),np.full(4800,2),np.full(9600,5)]))
    check('clock1ms',close(t['time'],np.arange(1,15401)*.001,1e-8))
    check('clock5ms',close(s['actual_time'],1+np.arange(2880)*.005,1e-8))
    check('source_phase',close(s['source_phase'],np.arange(2880)*.005,1e-12))
    check('post_prime',np.array_equal(s['post_prime'],np.arange(2880)>=960))
    check('actions_equal',np.array_equal(t['action'],t['actual_requested_action']))
    check('bounded_actions',np.max(np.abs(t['action']))<=1.)
    check('samples_actions_held',np.array_equal(t['action'][1000:],np.repeat(s['Y42'],5,axis=0)))
    check('no_extras',all(np.max(np.abs(t[k]))==0 for k in ('external_body_force_max','external_generalized_force_max')))
    check('no_adhesion',manifest['adhesion']['enabled'] is False)
    check('native_timestep',manifest['physics']['timestep']==.0000125)
    check('normalization',close(s['command2'][:,1],s['command_speed_yaw'][:,1]/.06,1e-14) and np.max(np.abs(s['command2'][:,0]))<1e-14)
    u=np.clip((np.arange(2880)-960)*.005,0,1);blend=u*u*(3-2*u)
    check('ramp',close(s['command2'][:,1],blend,1e-12) and close(s['command_speed_yaw'][:,0],np.full(2880,.075),1e-14))
    check('yaw_clock',close(y['time'],t['time'][1000:],1e-12))
    heading=np.unwrap(np.r_[float(y['boundary_yaw']),y['yaw_world'][4800:]])
    actual=float(heading[-1]-heading[0])
    check('yaw_metric',abs(actual-r['actual_signed_yaw_rad'])<1e-12)
    check('yaw_boundary_geometry',close(y['boundary_com'],t['com'][5799],1e-10))
    # Rebuild heading from saved free-root quaternion [w,x,y,z], no FK.
    q=t['qpos'][1000:,3:7]
    yaw=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
    check('yaw_from_saved_root_quaternion',np.max(np.abs(np.arctan2(np.sin(yaw-y['yaw_world']),np.cos(yaw-y['yaw_world']))))<1e-8)
    post=slice(5800,15400);com=t['com'][post];loads=t['loads'][post];speed=t['speed'][post];nf=t['nonfoot'][post]
    duration=t['time'][-1]-t['time'][5800]
    delta=com[-1,:2]-com[0,:2];xy_speed=np.linalg.norm(delta)/duration
    den=loads.sum(0);ratios=(loads*speed).sum(0)/np.maximum(den,1e-12)/max(xy_speed,1e-9)
    weight=CycleConfig().weight
    metrics=dict(slip_ratio_max=float(ratios.max()),com_height_min=float(com[:,2].min()),foot_support_mean=float((loads.sum(1)/weight).mean()),forward_signed_native_per_s=float(delta[0]/duration),nonfoot_force_max_native=float(nf.max()))
    for k,v in metrics.items():check('metric_'+k,abs(v-r[k])<1e-10)
    check('per_leg_slip',close(ratios,[r['per_leg_slip_ratio'][leg] for leg in LEG_ORDER]))
    check('slip_denominators',close(den,r['slip_load_denominator_native']))
    gates=dict(support=metrics['foot_support_mean']>=.90,nonfoot=metrics['nonfoot_force_max_native']<=.20*weight,com=metrics['com_height_min']>=1.20,forward=metrics['forward_signed_native_per_s']>=.05,slip=metrics['slip_ratio_max']<=.5)
    check('original_all5_gates',gates==r['gates'] and all(gates.values())==r['all_gates'])
    observer=CycleTracker(CycleConfig(timeout=4.)).reset(start_time=5.8)
    counts={leg:0 for leg in LEG_ORDER}
    for i in range(5800,15400):
        step=observer.update(dict(time=float(t['time'][i]),loads=t['loads'][i],padz=t['padz'][i],foot=t['relativefoot'][i]))
        for event in step['events']:
            if event['type']=='complete':counts[event['leg']]+=1
    for event in observer.finalize()['events']:
        if event['type']=='complete':counts[event['leg']]+=1
    all6=all(n>=1 for n in counts.values())
    check('all_cycle_counts_recomputed',counts==r['cycles']['complete_per_leg'])
    check('all6_flag_recomputed',all6==r['cycles']['all_legs_completed'])
    native=json.loads((folder/'NATIVE_MANIFEST.json').read_text())
    check('native_manifest',native==r['native'])
    check('native_count',native['actual_native_steps']==1232000 and native['post_native_steps']==768000)
    check('native_extras_zero',native['extra_force_max']==0.)
    check('native_peak_not_less_than_endpoints',native['post_nonfoot_peak_native']>=metrics['nonfoot_force_max_native']-1e-12)
    ref=json.loads((folder/'REFERENCE.json').read_text())
    check('actual_rest_exact',np.array_equal(ref['rest_body_feet'],t['relativefoot'][999]))
    cfg=ref['reference']['cfg']
    check('unchanged_reference_cfg',cfg['stride_native']==.36 and cfg['lift_native']==.12 and cfg['step_s']==.8 and cfg['cycles']==100)
    eligible=bool(all(gates.values()) and all6 and native['post_nonfoot_peak_native']==0. and actual>=np.pi/6 and r['early_stop'] is None)
    check('eligibility_recomputed',eligible==r['teacher_eligible'])
    result=dict(status='SAVED_ARRAY_AUDIT_COMPLETE',integrity_all_pass=all(checks.values()),checks=checks,
        teacher_eligible=eligible,yaw_deg=float(np.degrees(actual)),gates=gates,metrics=metrics,
        per_leg_slip=dict(zip(LEG_ORDER,map(float,ratios))),cycles=counts,native_peak=native['post_nonfoot_peak_native'],
        sourcehash_sha256=sha256(HERE/'SOURCEHASH.json'),result_sha256=sha256(folder/'RESULT.json'),
        scope='Independent saved arrays and recorded native summaries; no Core/integration; native peak cannot be independently reconstructed from endpoint arrays; not full D1-D8')
    (folder/'VERIFY.json').write_text(json.dumps(result,indent=2)+'\n')
    (HERE/'VERIFIED_SUMMARY.json').write_text(json.dumps(dict(left_yaw_p060=result),indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not all(checks.values()):raise RuntimeError(checks)
    return result


if __name__=='__main__':verify()
