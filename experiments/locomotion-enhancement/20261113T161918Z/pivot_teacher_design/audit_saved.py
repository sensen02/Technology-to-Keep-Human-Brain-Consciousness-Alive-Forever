"""Saved-only independent audit: NumPy after parent flag; no Core/physics import."""
import argparse
import math
import sys
from common import ROOT, SOURCE_CANDIDATE, read, save, sha, verify


def independent_counts(loads, heights, weight):
    # Independent per-leg interval scan, no common.PivotCycles calls.
    counts = []
    for leg in range(6):
        phase = 0; loaded_run = 0; low_run = 0; peak = 0.; begin = None; reload = 0; count = 0
        for i in range(len(loads)):
            grounded = loads[i, leg]/weight >= .1 and heights[i, leg] <= .004
            low = loads[i, leg]/weight <= .05
            if phase == 0:
                loaded_run = loaded_run+1 if grounded else 0
                if loaded_run >= 10: phase = 1
                continue
            if phase == 1:
                low_run = low_run+1 if low else 0
                peak = max(peak, float(heights[i, leg])) if low else 0.
                if low_run < 5: continue
                begin = i-4; phase = 2
            elif low:
                peak = max(peak, float(heights[i, leg]))
            reload = reload+1 if grounded else 0
            if i-begin+1 > 4000 or (grounded and peak <= .015):
                phase=0; loaded_run=low_run=reload=0; peak=0.; begin=None
            elif peak > .015 and reload >= 10:
                count += 1; phase=1; loaded_run=10; low_run=reload=0; peak=0.; begin=None
        counts.append(count)
    return counts


def main():
    p=argparse.ArgumentParser(); p.add_argument('--condition',choices=('stop','left_pivot','right_pivot'),required=True); p.add_argument('--execute-parent-saved-audit',action='store_true'); a=p.parse_args()
    if not a.execute_parent_saved_audit: p.error('Parent saved-only audit flag required')
    import numpy as np
    folder=ROOT/'trials'/a.condition; out=folder/'VERIFY.json'
    if out.exists(): raise FileExistsError(out)
    r=read(folder/'RESULT.json'); checks={}; verify(r['source_hashes'])
    for key in ('trace','teacher_samples','geometry','diagnostics'):
        item=r[key]; checks[key+'_hash']=sha(item['path'])==item['sha256']
    with np.load(folder/'trace_1ms.npz',allow_pickle=False) as z: t={k:z[k].copy() for k in z.files}
    with np.load(folder/'teacher_samples_5ms.npz',allow_pickle=False) as z: s={k:z[k].copy() for k in z.files}
    with np.load(folder/'geometry_1ms.npz',allow_pickle=False) as z: g={k:z[k].copy() for k in z.files}
    checks['all_saved_arrays_finite']=all(np.isfinite(v).all() for arrays in (t,s,g) for v in arrays.values())
    checks['trace_clock_1ms']=len(t['time'])==r['actual_ms'] and np.max(np.abs(np.diff(t['time'])-.001))<1e-8
    checks['geometry_clock_after_settle']=np.array_equal(g['time'],t['time'][1000:])
    checks['teacher_clock_5ms']=len(s['time'])==r['teacher_decisions'] and (len(s['time'])<2 or np.max(np.abs(np.diff(s['time'])-.005))<1e-8)
    checks['trace_full42']=t['action'].shape==(len(t['time']),42) and t['qpos'].shape[0]==len(t['time']) and t['qvel'].shape[0]==len(t['time'])
    checks['bounded42_no_extra_forces']=bool(np.max(np.abs(t['action']))<=1.+1e-12 and np.max(t['external_body_force_max'])==0. and np.max(t['external_generalized_force_max'])==0.)
    checks['candidate_source_window']=np.array_equal(t['control_source']>=SOURCE_CANDIDATE,t['time']>5.8+1e-8)
    # qpos freejoint quaternion is a saved actual-state cross-check, not FK replay.
    q=t['qpos'][1000:,3:7]; q=q/np.linalg.norm(q,axis=1,keepdims=True); w,x,y,z=q.T
    qyaw=np.arctan2(2*(w*z+x*y),1.-2*(y*y+z*z))
    rotation_yaw=np.arctan2(g['thorax_rotation'][:,1,0],g['thorax_rotation'][:,0,0])
    checks['yaw_from_saved_matrix']=bool(np.max(np.abs(np.angle(np.exp(1j*(rotation_yaw-g['thorax_yaw'])))))<1e-10)
    checks['yaw_from_saved_actual_root_quaternion']=bool(np.max(np.abs(np.angle(np.exp(1j*(qyaw-g['thorax_yaw'])))))<1e-8)
    # Each decision's actually applied actions must span its saved native1ms rows.
    aligned=True
    for i, action in enumerate(s['Y42']):
        block=t['action'][1000+i*5:min(1000+(i+1)*5,len(t['time']))]
        if len(block)==0 or not np.array_equal(block,np.repeat(action[None],len(block),axis=0)): aligned=False; break
    checks['decision_to_applied_action_exact']=aligned
    ident=read(folder/'IDENTITY.json'); canonical={k:v for k,v in ident.items() if k!='identity_sha256'}
    import hashlib,json
    checks['identity_canonical_digest']=hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest()==ident['identity_sha256']==r['identity_sha256']
    checks['identity_files_unchanged']=all(sha(path)==info['sha256'] for path,info in ident['files'].items())
    limit=np.asarray(ident['actuator']['torque_limit'])
    checks['actual_finitePD_torque_limits']=bool(np.all(np.abs(t['actualctrl'])<=limit[None]+1e-9))
    checks['native_steps_consistent']=r['native']['actual_native_steps']==len(t['time'])*80<=1232000
    m=t['control_source']==SOURCE_CANDIDATE; gm=g['control_source']==SOURCE_CANDIDATE
    measured=None
    if np.any(m):
        b=r['candidate_start']; W=r['weight_native']; L=r['body_length']['length_native']; com=t['com'][m]; loads=t['loads'][m]; speed=t['speed'][m]; times=t['time'][m]
        yaw_seq=np.unwrap(np.r_[b['yaw'],g['thorax_yaw'][gm]]); actual_yaw=float(yaw_seq[-1]-yaw_seq[0]); duration=float(times[-1]-b['time'])
        dist=np.linalg.norm(com[:,:2]-np.asarray(b['com'])[:2],axis=1); drift=float(dist.max()/L)
        cycles=independent_counts(np.vstack([t['loads'][5790:5800],loads]),np.vstack([t['padz'][5790:5800],t['padz'][m]]),W)
        old=r['original_five']; legacy_gates=None
        if len(times) >= 2:
            old_dt=float(times[-1]-times[0]); delta=com[-1,:2]-com[0,:2]; xy=float(np.linalg.norm(delta)/old_dt); load_den=loads.sum(0); vt=(loads*speed).sum(0)/np.maximum(load_den,1e-12); ratios=vt/max(xy,1e-9)
            legacy_gates=dict(support=float(loads.sum(1).mean()/W)>=.9,nonfoot=float(t['nonfoot'][m].max())<=.2*W,com=float(com[:,2].min())>=1.2,forward=float(delta[0]/old_dt)>=.05,slip=float(ratios.max())<=.5)
            checks['legacy_all5_unmodified']=legacy_gates==old['gates'] and bool(all(legacy_gates.values()))==old['all_gates']
            checks['legacy_scores_independent']=bool(np.isclose(float(delta[0]/old_dt),old['forward_signed_native_per_s'],atol=1e-12,rtol=0) and np.isclose(float(ratios.max()),old['slip_ratio_max'],atol=1e-10,rtol=0))
        else:
            checks['legacy_short_window_explicit']=old==dict(all_gates=False,sample_count=len(times),reason='less than two samples')
        native=r['native']; signed=actual_yaw if a.condition=='left_pivot' else -actual_yaw
        gates=dict(support=float(loads.sum(1).mean()/W)>=.9, nonfoot_1ms=float(t['nonfoot'][m].max())==0., com=float(com[:,2].min())>=1.2, drift=drift<=.5)
        if a.condition!='stop': gates.update(yaw=math.degrees(signed)>=45.,all6_actual_pivot_cycles=all(n>=1 for n in cycles))
        gates.update(native_nonfoot=native['post_nonfoot_peak_native']==0.,native_extra_force=native['extra_force_max']==0.,native_COM=native['post_COM_min']>=1.2,full_window=native['post_native_steps']==768000 and native['actual_native_steps']==1232000 and r['early_stop'] is None)
        rm=r['candidate_metrics']
        checks['candidate_yaw_drift_duration_independent']=abs(actual_yaw-rm['yaw_change_rad'])<1e-10 and abs(drift-rm['max_COM_xy_drift_bodylength'])<1e-10 and abs(duration-rm['duration_s'])<1e-10
        world_v=np.diff(np.vstack([b['com'],com]),axis=0)/.001
        body_v=np.stack([rot.T @ vel for rot,vel in zip(g['thorax_rotation'][gm],world_v)])/L
        checks['body_velocity_independent']=bool(np.allclose(body_v.mean(0),rm['body_velocity_mean_bodylength_per_s'],atol=1e-10,rtol=0) and np.allclose(np.sqrt((body_v**2).mean(0)),rm['body_velocity_RMS_bodylength_per_s'],atol=1e-10,rtol=0))
        checks['actual_cycle_counts_independent']=cycles==rm['pivot_cycle_counts']
        checks['candidate_gates_independent']=gates==rm['gates']
        checks['no_stop_pass_fabricated']=rm['task_pass_candidate'] is None if a.condition=='stop' else rm['task_pass_candidate']==bool(all(gates.values()))
        measured=dict(yaw_change_rad=actual_yaw,max_COM_xy_drift_bodylength=drift,duration_s=duration,pivot_cycle_counts=cycles,gates=gates,task_pass_candidate=None if a.condition=='stop' else bool(all(gates.values())),old_five=legacy_gates)
    else:
        checks['early_warm_stop_no_candidate_claim']=r['early_stop'] is not None and r['candidate_metrics'] is None
    checks['no_physical_or_torch_import']=not any(k in sys.modules for k in ('physics','mujoco','torch','static_teacher'))
    save(out,dict(kind='SAVED_ONLY_PIVOT_TEACHER_AUDIT_V1',integrity_all_pass=bool(all(checks.values())),checks=checks,result_sha256=sha(folder/'RESULT.json'),measured=measured,physical_steps=0,neural_forwards=0,scope='Saved-state numerical audit; native extrema accepted from every-substep recorder summary, not independently re-solved. Body length uses captured compiledgeometry report; no Core/FK reconstruction.'))
    if not all(checks.values()): raise ValueError('Saved audit failed; evidence retained')
    print(json.dumps(dict(condition=a.condition,integrity_all_pass=True,measured=measured)))


if __name__=='__main__': main()
