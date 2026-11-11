"""Independent saved-array audit, writes new unique names only in new ROOT.

No old verification writer imports/calls. Native peaks remain captured summaries;
this is not independent reconstruction of all native steps or full D1-D8.
"""
import json
import os
import sys
import numpy as np
from source_identity_wider import HERE,FREEZE,verify_sources,sha256,write_unique_audit


def main():
    verify_sources()
    allowed=os.sched_getaffinity(0);os.sched_setaffinity(0,{1 if 1 in allowed else min(allowed)})
    folder=HERE/'left_yaw_p060'
    if not (folder/'RESULT.json').exists():
        print(json.dumps(dict(status='NOT_RUN',new_trajectory_audited=False,root=str(HERE))))
        return
    from turning_teacher import BASELINE,WiderRadiusReference
    sys.path.insert(0,str(BASELINE))
    import identity
    from cycle_reward import CycleTracker,CycleConfig,LEG_ORDER
    from wave_reference import WaveReference
    r=json.loads((folder/'RESULT.json').read_text());manifest=json.loads((folder/'IDENTITY.json').read_text());identity.verify(manifest)
    arrays=[]
    for name in ('TRACE_1MS.npz','SAMPLES_5MS.npz','YAW_1MS.npz'):
        with np.load(folder/name,allow_pickle=False) as z:arrays.append({k:z[k] for k in z.files})
    t,s,y=arrays;checks={}
    def check(name,value):checks[name]=bool(value)
    def close(a,b,tol=1e-10):return bool(np.shape(a)==np.shape(b) and np.max(np.abs(np.asarray(a)-np.asarray(b)))<tol)
    check('condition',r['condition']==dict(name='left_yaw_p060',speed_scale=1.,yaw_rate=.06))
    check('identity_root_condition',manifest['config']['prereg']==str(HERE/'PREREG.json') and manifest['config']['condition']==r['condition'])
    check('source_freeze',r['sourcehash_sha256']==sha256(FREEZE))
    check('artifact_hashes',all(sha256(folder/name)==h for name,h in r['artifact_hashes'].items()))
    check('trace_hash',sha256(folder/'TRACE_1MS.npz')==r['trace']['sha256'])
    check('samples_hash',sha256(folder/'SAMPLES_5MS.npz')==r['samples']['sha256'])
    check('finite_all',all(np.isfinite(a).all() for group in arrays for a in group.values()))
    check('full15400',len(t['time'])==15400)
    check('full2880',s['X238'].shape==(2880,238) and s['Y42'].shape==(2880,42) and s['command2'].shape==(2880,2))
    if not checks['full15400'] or not checks['full2880']:
        out=write_unique_audit(dict(status='INCOMPLETE_TRAJECTORY_FAIL',integrity_all_pass=False,checks=checks,teacher_eligible=False,early_stop=r.get('early_stop')))
        print(str(out));raise RuntimeError('Full-window qualification unavailable; incomplete FAIL retained')
    check('clock1ms',close(t['time'],np.arange(1,15401)*.001,1e-8))
    check('clock5ms',close(s['actual_time'],1+np.arange(2880)*.005,1e-8))
    check('phase_no_warp',close(s['source_phase'],np.arange(2880)*.005,1e-12))
    check('post_prime',np.array_equal(s['post_prime'],np.arange(2880)>=960))
    check('sources',np.array_equal(t['control_source'],np.r_[np.zeros(1000),np.full(4800,2),np.full(9600,5)]))
    check('actual_actions',np.array_equal(t['action'],t['actual_requested_action']))
    check('held_actions',np.array_equal(t['action'][1000:],np.repeat(s['Y42'],5,axis=0)))
    check('bounded_actions',np.max(np.abs(t['action']))<=1.)
    check('no_extras',all(np.max(np.abs(t[k]))==0 for k in ('external_body_force_max','external_generalized_force_max')))
    check('no_adhesion',manifest['adhesion']['enabled'] is False)
    check('native_timestep',manifest['physics']['timestep']==.0000125)
    u=np.clip((np.arange(2880)*.005-4.8),0,1);b=u*u*(3-2*u)
    v=.075+.025*b;omega=.06*b
    check('physical_speed_yaw',close(s['command_speed_yaw'],np.column_stack([v,omega]),1e-13))
    check('normalized_forward_yaw',close(s['command2'],np.column_stack([v/.075-1,omega/.06]),1e-13))
    check('observation_feature_label',r['base_obs_target_speed_constant']==.075 and 'Historical' in r['base_obs_note'])
    # Read-only baseline prefix: full settle +4.8s straight teacher must match.
    old=HERE.parent/'left_finite_twist'/'left_yaw_p060'/'TRACE_1MS.npz'
    with np.load(old,allow_pickle=False) as z:
        check('physical_prefix_all_fields_bitexact',all(k in z.files and np.array_equal(t[k][:5800],z[k][:5800]) for k in t))
    ref=json.loads((folder/'REFERENCE_WIDER_RADIUS.json').read_text());rest=np.asarray(ref['rest_body_feet'])
    check('rest_exact',np.array_equal(rest,t['relativefoot'][999]))
    cfg=ref['original_reference']['cfg']
    check('original_base_cfg',cfg['stride_native']==.36 and cfg['step_s']==.8 and cfg['lift_native']==.12 and cfg['cycles']==100)
    wr=ref['wider_radius']
    check('declared_final_cfg',wr['final_stride']==.48 and wr['nominal_v_final']==.1 and wr['curvature_final']==.6 and wr['requested_yaw_final']==.06)
    check('yaw_clock',close(y['time'],t['time'][1000:],1e-12))
    q=t['qpos'][1000:,3:7];yaw=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
    check('yaw_quaternion',np.max(np.abs(np.arctan2(np.sin(yaw-y['yaw_world']),np.cos(yaw-y['yaw_world']))))<1e-8)
    check('yaw_boundary_com',close(y['boundary_com'],t['com'][5799],1e-10))
    heading=np.unwrap(np.r_[float(y['boundary_yaw']),y['yaw_world'][4800:]]);actual=float(heading[-1]-heading[0])
    check('actual_yaw',abs(actual-r['actual_signed_yaw_rad'])<1e-12)
    duration=t['time'][-1]-t['time'][5800];com=t['com'][5800:];loads=t['loads'][5800:];speed=t['speed'][5800:];nf=t['nonfoot'][5800:]
    delta=com[-1,:2]-com[0,:2];com_speed=float(np.linalg.norm(delta)/duration);load_den=loads.sum(0)
    vt=(loads*speed).sum(0)/np.maximum(load_den,1e-12);ratios=vt/max(com_speed,1e-9);weight=CycleConfig().weight
    metrics=dict(slip_ratio_max=float(ratios.max()),com_height_min=float(com[:,2].min()),foot_support_mean=float((loads.sum(1)/weight).mean()),forward_signed_native_per_s=float(delta[0]/duration),nonfoot_force_max_native=float(nf.max()))
    for k,val in metrics.items():check('metric_'+k,abs(val-r[k])<1e-10)
    check('per_leg_slip',close(ratios,[r['per_leg_slip_ratio'][leg] for leg in LEG_ORDER]))
    check('slip_load_den',close(load_den,r['slip_load_denominator_native']))
    check('slip_numerator',close(vt,r['slip_numerator_loadweighted_foot_speed']))
    check('slip_COM_denominator',abs(com_speed-r['slip_COM_speed_denominator'])<1e-10)
    check('slip_validity',bool((load_den>1e-12).all() and com_speed>1e-9))
    gates=dict(support=metrics['foot_support_mean']>=.90,nonfoot=metrics['nonfoot_force_max_native']<=.20*weight,com=metrics['com_height_min']>=1.20,forward=metrics['forward_signed_native_per_s']>=.05,slip=metrics['slip_ratio_max']<=.5)
    check('original_five_gates',gates==r['gates'] and all(gates.values())==r['all_gates'])
    observer=CycleTracker(CycleConfig(timeout=4.)).reset(start_time=5.8);counts={leg:0 for leg in LEG_ORDER}
    for i in range(5800,15400):
        result=observer.update(dict(time=float(t['time'][i]),loads=t['loads'][i],padz=t['padz'][i],foot=t['relativefoot'][i]))
        for e in result['events']:
            if e['type']=='complete':counts[e['leg']]+=1
    for e in observer.finalize()['events']:
        if e['type']=='complete':counts[e['leg']]+=1
    all6=all(n>=1 for n in counts.values())
    check('original_X_cycles_counts',counts==r['cycles']['complete_per_leg'])
    check('all6_flag',all6==r['cycles']['all_legs_completed'])
    native=json.loads((folder/'NATIVE_MANIFEST.json').read_text())
    check('native_manifest',native==r['native'])
    check('native_counts',native['actual_native_steps']==1232000 and native['post_native_steps']==768000)
    check('native_no_extras',native['extra_force_max']==0.)
    check('native_peak_at_least_endpoint',native['post_nonfoot_peak_native']>=metrics['nonfoot_force_max_native']-1e-12)
    attribution=native['nonfoot_contact_pairs']
    check('native_attribution_bounded',attribution['max_geom_pairs']==64 and len(attribution['pairs'])<=64 and attribution['full_native_arrays'] is False)
    check('native_attribution_zero_consistent',native['nonfoot_peak_native']!=0. or (attribution['pairs']=={} and attribution['overflow_contact_observations']==0 and attribution['positive_native_steps']==0))
    check('native_attribution_post_consistent',(native['post_nonfoot_peak_native']==0.)==(attribution['post_positive_native_steps']==0))
    eligible=bool(all(gates.values()) and all6 and native['post_nonfoot_peak_native']==0. and actual>=np.pi/6 and r['early_stop'] is None)
    check('eligibility',eligible==r['teacher_eligible'])
    audit=dict(status='SAVED_ARRAY_AUDIT_COMPLETE',integrity_all_pass=all(checks.values()),checks=checks,teacher_eligible=eligible,
        yaw_deg=float(np.degrees(actual)),gates=gates,metrics=metrics,cycles=counts,native_peak=native['post_nonfoot_peak_native'],
        per_leg_slip=dict(zip(LEG_ORDER,map(float,ratios))),slip_numerator_foot_speed=dict(zip(LEG_ORDER,map(float,vt))),
        slip_load_denominators=load_den.tolist(),slip_COM_speed_denominator=com_speed,
        sourcehash_sha256=sha256(FREEZE),result_sha256=sha256(folder/'RESULT.json'),
        scope='Independent saved arrays; original X cycles and gates; native peak is captured summary, not independently reconstructed all-native D1-D8; no Core/physics integration')
    out=write_unique_audit(audit)
    print(json.dumps(dict(audit_path=str(out),integrity_all_pass=audit['integrity_all_pass'],teacher_eligible=eligible,checks=len(checks))))
    if not audit['integrity_all_pass']:raise RuntimeError(checks)


if __name__=='__main__':main()
