"""Independent new saved-array/feedback/IK audit; no old writers, no Core/FK."""
import json
import os
import sys
from pathlib import Path
from guard_identity import HERE,FREEZE,verify_sources,sha256,exclusive


def main():
    verify_sources();folder=HERE/'left_yaw_p060'
    if not (folder/'RESULT.json').exists():print(json.dumps(dict(status='NOT_RUN',new_trajectory_audited=False)));return
    allowed=os.sched_getaffinity(0);os.sched_setaffinity(0,{1 if 1 in allowed else min(allowed)})
    import numpy as np
    baseline=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
    sys.path.insert(0,str(baseline));import identity
    from cycle_reward import CycleTracker,CycleConfig,LEG_ORDER
    r=json.loads((folder/'RESULT.json').read_text());manifest=json.loads((folder/'IDENTITY.json').read_text());identity.verify(manifest)
    groups=[]
    for name in ('TRACE_1MS.npz','SAMPLES_5MS.npz','YAW_1MS.npz'):
        with np.load(folder/name,allow_pickle=False) as z:groups.append({k:z[k] for k in z.files})
    t,s,y=groups;checks={}
    def check(k,v):checks[k]=bool(v)
    def close(a,b,tol=1e-10):return bool(np.shape(a)==np.shape(b) and np.max(np.abs(np.asarray(a)-np.asarray(b)))<=tol)
    check('source_freeze',r['sourcehash_sha256']==sha256(FREEZE))
    check('identity_root_condition',manifest['config']['prereg']==str(HERE/'PREREG.json') and manifest['config']['condition']==dict(name='left_yaw_p060',speed_scale=1.,yaw_rate=.06))
    check('artifacts',all(sha256(folder/name)==h for name,h in r['artifact_hashes'].items()))
    check('trace_hash',sha256(folder/'TRACE_1MS.npz')==r['trace']['sha256']);check('samples_hash',sha256(folder/'SAMPLES_5MS.npz')==r['samples']['sha256'])
    check('finite_all',all(np.isfinite(a).all() for group in groups for a in group.values()))
    check('full15400_2880',len(t['time'])==15400 and s['X238'].shape==(2880,238) and s['Y42'].shape==(2880,42))
    if not checks['full15400_2880']:
        out=exclusive('AUDIT_ACTUAL_GUARD',dict(status='INCOMPLETE_FULL_WINDOW_FAIL',checks=checks,integrity_all_pass=False,teacher_eligible=False,early_stop=r.get('early_stop')))
        print(str(out));raise RuntimeError('Incomplete candidate trajectory retained as FAIL')
    check('clock1ms',close(t['time'],np.arange(1,15401)*.001,1e-8));check('clock5ms',close(s['actual_time'],1+np.arange(2880)*.005,1e-8))
    check('phase',close(s['source_phase'],np.arange(2880)*.005,1e-12));check('post_prime',np.array_equal(s['post_prime'],np.arange(2880)>=960))
    check('sources',np.array_equal(t['control_source'],np.r_[np.zeros(1000),np.full(4800,2),np.full(9600,5)]))
    check('actual_actions',np.array_equal(t['action'],t['actual_requested_action']));check('held_actions',np.array_equal(t['action'][1000:],np.repeat(s['Y42'],5,axis=0)))
    check('bounded_actions',np.max(np.abs(t['action']))<=1.)
    check('no_extra',all(np.max(np.abs(t[k]))==0 for k in ('external_body_force_max','external_generalized_force_max')))
    check('physical_mode',manifest['adhesion']['enabled'] is False and manifest['physics']['timestep']==.0000125)
    b=np.clip(np.arange(2880)*.005-4.8,0,1);b=b*b*(3-2*b);v=.075+.025*b;omega=.06*b
    check('commands',close(s['command_speed_yaw'],np.column_stack([v,omega]),1e-13) and close(s['command2'],np.column_stack([v/.075-1,omega/.06]),1e-13))
    check('historical_obs',r['base_obs_target_speed_constant']==.075 and 'Historical' in r['base_obs_note'])
    with np.load(HERE.parent/'left_tarsal_clearance_teacher'/'left_yaw_p060'/'TRACE_1MS.npz',allow_pickle=False) as z:
        check('physical_prime_prefix_all_fields_exact',all(k in z.files and np.array_equal(t[k][:5800],z[k][:5800]) for k in t))
    ref=json.loads((folder/'REFERENCE_ACTUAL_GUARD.json').read_text());cfg=ref['original_reference']['cfg'];sec=ref['RH_secondary']
    check('actual_rest',np.array_equal(ref['rest_body_feet'],t['relativefoot'][999]))
    check('unchanged_reference',cfg['stride_native']==.36 and cfg['step_s']==.8 and cfg['lift_native']==.12 and cfg['cycles']==100 and ref['wider_radius']['final_stride']==.48)
    check('fixed_guard',sec['actual_guard']==.012 and sec['observed_pair_margin']==.002 and sec['tracking_reserve']==.010 and sec['config']['per_decision_trust_rad']==.005 and sec['config']['local_updates_max']==4)
    check('yaw_clock',close(y['time'],t['time'][1000:],1e-12));check('boundary_com',close(y['boundary_com'],t['com'][5799]))
    q=t['qpos'][1000:,3:7];qyaw=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
    check('yaw_geometry',np.max(np.abs(np.arctan2(np.sin(qyaw-y['yaw_world']),np.cos(qyaw-y['yaw_world']))))<1e-8)
    heading=np.unwrap(np.r_[float(y['boundary_yaw']),y['yaw_world'][4800:]]);actual=float(heading[-1]-heading[0]);check('yaw_metric',abs(actual-r['actual_signed_yaw_rad'])<1e-12)
    com=t['com'][5800:];loads=t['loads'][5800:];speed=t['speed'][5800:];nf=t['nonfoot'][5800:];W=CycleConfig().weight
    duration=t['time'][-1]-t['time'][5800];delta=com[-1,:2]-com[0,:2];comspeed=float(np.linalg.norm(delta)/duration)
    den=loads.sum(0);vt=(loads*speed).sum(0)/np.maximum(den,1e-12);ratios=vt/max(comspeed,1e-9)
    metrics=dict(slip_ratio_max=float(ratios.max()),foot_support_mean=float((loads.sum(1)/W).mean()),com_height_min=float(com[:,2].min()),forward_signed_native_per_s=float(delta[0]/duration),nonfoot_force_max_native=float(nf.max()))
    for k,val in metrics.items():check('metric_'+k,abs(val-r[k])<1e-10)
    check('per_leg_slip',close(ratios,[r['per_leg_slip_ratio'][leg] for leg in LEG_ORDER]));check('slip_denominators',close(den,r['slip_load_denominator_native']))
    check('slip_numerator_COMden',close(vt,r['slip_numerator_loadweighted_foot_speed']) and abs(comspeed-r['slip_COM_speed_denominator'])<1e-10)
    gates=dict(support=metrics['foot_support_mean']>=.9,nonfoot=metrics['nonfoot_force_max_native']<=.2*W,com=metrics['com_height_min']>=1.2,forward=metrics['forward_signed_native_per_s']>=.05,slip=metrics['slip_ratio_max']<=.5)
    check('original_five_gates',gates==r['gates'] and all(gates.values())==r['all_gates'])
    tracker=CycleTracker(CycleConfig(timeout=4.)).reset(start_time=5.8);counts={leg:0 for leg in LEG_ORDER}
    for i in range(5800,15400):
        for event in tracker.update(dict(time=float(t['time'][i]),loads=t['loads'][i],padz=t['padz'][i],foot=t['relativefoot'][i]))['events']:
            if event['type']=='complete':counts[event['leg']]+=1
    for event in tracker.finalize()['events']:
        if event['type']=='complete':counts[event['leg']]+=1
    all6=all(n>=1 for n in counts.values());check('original_X_cycles',counts==r['cycles']['complete_per_leg'] and all6==r['cycles']['all_legs_completed'])
    native=json.loads((folder/'NATIVE_MANIFEST.json').read_text())
    check('native_manifest',native==r['native']);check('native_counts',native['actual_native_steps']==1232000 and native['post_native_steps']==768000)
    check('native_zero_extras',native['extra_force_max']==0.);check('native_pairs_bound',len(native['nonfoot_contact_pairs']['pairs'])<=64)
    check('native_peak_vs_endpoints',native['post_nonfoot_peak_native']>=metrics['nonfoot_force_max_native']-1e-12)
    rh=np.asarray(ref['RH_actuator_indices']);other=np.ones(42,dtype=bool);other[rh]=False;previous=np.asarray(ref['initial_q_des_act'])
    ik={k:True for k in ('sequence_time','retained_q_des_chain','whole_qdot_clip2','RH_only_trust005','pad_error1e5','joint_bounds','actual_trigger_deficit_target','prime_exact','reject_revert','accepted_ascent','private_invariants','helper_updates4_target','actual_support_trace','unsupported_flags','finite')}
    summary=dict(records=0,triggers=0,accepted=0,actual_unsafe_q_des_safe=0,actual_below_guard_shadow_above_guard=0,post_actual_RH_unsupported=0,status_counts={},max_secondary_delta=0.,max_pad_error=0.,max_actual_deficit=0.)
    with (folder/'ACTUAL_GUARD_DIAGNOSTICS.jsonl').open() as f:
        for i,line in enumerate(f):
            d=json.loads(line)
            if i>=2880:ik['sequence_time']=False;continue
            p=np.asarray(d['previous_q_des']);primary=np.asarray(d['primary_q_des']);final=np.asarray(d['final_q_des']);raw=(final-p)/.005
            ik['sequence_time']&=d['decision']==i and abs(d['elapsed_s']-i*.005)<1e-12 and abs(d['native_time_s']-s['actual_time'][i])<1e-10
            ik['retained_q_des_chain']&=np.array_equal(p,previous);previous=final
            ik['whole_qdot_clip2']&=np.array_equal(raw,d['qdot_raw']) and np.array_equal(np.clip(raw,-2,2),d['qdot_clipped'])
            maxdelta=float(np.max(np.abs(final[rh]-primary[rh])));paderr=float(np.linalg.norm(np.asarray(d['final_RH_pad'])-d['primary_RH_pad']))
            ik['RH_only_trust005']&=np.array_equal(final[other],primary[other]) and maxdelta<=.005+1e-12 and close(final[rh]-primary[rh],d['RH_delta'],1e-12) and abs(maxdelta-d['max_abs_secondary_delta'])<1e-12
            ik['pad_error1e5']&=paderr<=1e-5 and abs(paderr-d['pad_error_norm'])<1e-12
            bounds=np.asarray(d['RH_bounds']);ik['joint_bounds']&=bool(np.all(final[rh]>=bounds[:,0]-1e-12) and np.all(final[rh]<=bounds[:,1]+1e-12))
            deficit=max(0.,.012-d['actual_current_height']);active=i>960 and d['actual_current_height']<.012;applied=deficit if active else 0.
            desired=d['q_des_height_before']+applied
            ik['actual_trigger_deficit_target']&=d['actual_guard']==.012 and d['trigger']==active and abs(deficit-d['actual_deficit'])<1e-12 and abs(applied-d['applied_deficit'])<1e-12 and abs(desired-d['desired_shadow_height'])<1e-12
            if i<=960:ik['prime_exact']&=not d['accepted'] and not d['trigger'] and np.array_equal(final,primary) and d['status']=='PRIME_INACTIVE'
            ik['reject_revert']&=d['accepted'] or np.array_equal(final,primary)
            if d['accepted']:ik['accepted_ascent']&=active and d['q_des_height_after']-d['q_des_height_before']>=1e-8
            ik['private_invariants']&=d['root_unchanged'] and d['other_q_des_unchanged']
            outcome=d['helper_outcome']
            if outcome is not None:ik['helper_updates4_target']&=len(outcome.get('attempts',[]))<=4 and (outcome.get('safe_height') is None or abs(outcome['safe_height']-desired)<1e-12)
            ti=999+i*5;load=float(t['loads'][ti,4]/W);pad=float(t['padz'][ti,4]);total=float(t['loads'][ti].sum()/W)
            ik['actual_support_trace']&=abs(load-d['actual_RH_load_W'])<1e-10 and abs(pad-d['actual_RH_padz'])<1e-10 and abs(total-d['actual_total_foot_support_W'])<1e-10 and abs(t['com'][ti,2]-d['actual_COM_z'])<1e-10
            loaded=load>=.1 and pad<=.004;ik['unsupported_flags']&=d['actual_RH_loaded_ground']==loaded and d['actual_RH_unloaded']==(load<=.05) and d['actual_RH_unsupported']==(not loaded)
            ik['finite']&=bool(np.isfinite(np.r_[p,primary,final,raw,d['actual_current_height'],d['q_des_height_before'],d['q_des_height_after'],desired]).all())
            summary['records']+=1;summary['triggers']+=int(d['trigger']);summary['accepted']+=int(d['accepted'])
            for key in ('actual_unsafe_q_des_safe','actual_below_guard_shadow_above_guard'):summary[key]+=int(d[key])
            summary['post_actual_RH_unsupported']+=int(i>960 and d['actual_RH_unsupported']);summary['status_counts'][d['status']]=summary['status_counts'].get(d['status'],0)+1
            summary['max_secondary_delta']=max(summary['max_secondary_delta'],maxdelta);summary['max_pad_error']=max(summary['max_pad_error'],paderr);summary['max_actual_deficit']=max(summary['max_actual_deficit'],deficit)
    for k,val in ik.items():check('IK_'+k,val)
    check('IK_full2880',summary['records']==2880);check('IK_summary',summary==r['actual_guard_summary'])
    eligible=bool(all(gates.values()) and all6 and native['post_nonfoot_peak_native']==0. and actual>=np.pi/6 and r['early_stop'] is None);check('eligibility',eligible==r['teacher_eligible'])
    audit=dict(status='SAVED_ARRAY_FEEDBACK_IK_AUDIT_COMPLETE',integrity_all_pass=all(checks.values()),checks=checks,teacher_eligible=eligible,gates=gates,yaw_deg=float(np.degrees(actual)),cycles=counts,native_peak=native['post_nonfoot_peak_native'],
        per_leg_slip=dict(zip(LEG_ORDER,map(float,ratios))),foot_speed_numerators=vt.tolist(),COM_speed_denominator=comspeed,actual_guard_summary=summary,
        sourcehash_sha256=sha256(FREEZE),result_sha256=sha256(folder/'RESULT.json'),scope='Independent saved arrays/feedback/IK arithmetic; geometric heights logged under frozen source identity, not FK-reconstructed; captured native summaries, not full D1-D8')
    out=exclusive('AUDIT_ACTUAL_GUARD',audit);print(json.dumps(dict(audit=str(out),integrity_all_pass=audit['integrity_all_pass'],teacher_eligible=eligible,checks=len(checks))))
    if not audit['integrity_all_pass']:raise RuntimeError('Integrity failure, exclusive audit retained')


if __name__=='__main__':main()
