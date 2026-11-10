"""Independent13-trial saved search audit. No runtime/score imports or physics/FK.
Exit3 before BEST exists. Preserve partial AUDIT_SEARCH; write only final auditJSON.
"""
from pathlib import Path
import hashlib,json
import numpy as np
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T112003Z')
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T112003Z/optimize_stance.py')
BASE=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T101753Z/neural/teacher_gain128_4s')
W=24.446354755878005

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(a,b,tol=1e-10):return np.asarray(a).shape==np.asarray(b).shape and bool(np.allclose(a,b,atol=tol,rtol=tol))
def calculate(data,termination):
    failed=termination is not None
    if data.ndim!=2 or not len(data):return dict(objective=1e6,failed=True,reason='NO_TRACE')
    release=data[data[:,0]>=1-1e-9]
    if len(release)<40:return dict(objective=1e6+1e4*failed,failed=True,reason='INSUFFICIENT_RELEASE')
    width=min(20,len(release)//4);first,last=release[:width],release[-width:]
    initial,final=first[:,14:17].mean(0),last[:,14:17].mean(0)
    yaw=np.unwrap(release[:,17]);heading=float(yaw[:width].mean());f=np.array([np.cos(heading),np.sin(heading)])
    duration=max(float(last[:,0].mean()-first[:,0].mean()),1e-12);delta=final[:2]-initial[:2]
    speed=float(delta@f/duration);lateral=float(abs(delta@[-f[1],f[0]])/duration)
    force=release[:,1:7].sum(0);numerator=release[:,7:13].sum(0)
    slip=np.divide(numerator,force,out=np.zeros(6),where=force>0);ratios=slip/max(speed,.02)
    support=release[:,1:7].sum(1)/W;nf=release[:,13]/W
    drop=max(float(first[:,16].mean()-release[:,16].min())-.1,0);turn=float(yaw[-width:].mean()-yaw[:width].mean())
    penalty=1000*failed+100*float(nf.mean())+100*float(nf.max())+100*max(.05-speed,0)+20*max(.8-float(support.mean()),0)
    penalty+=20*drop+2*lateral/max(speed,.02)+abs(turn)+10*int((force<=0).any())
    return dict(objective=float(ratios.mean()+penalty),failed=bool(failed),termination=termination,
        foot_normal_weighted_material_speed=slip.tolist(),slip_to_progress_ratios=ratios.tolist(),forward_speed=speed,lateral_speed=lateral,
        heading_change=turn,foot_support_mean=float(support.mean()),nonfoot_force_mean_W=float(nf.mean()),nonfoot_force_max_W=float(nf.max()),
        minimum_COM=float(release[:,16].min()),maximum_tilt_rad=float(release[:,18].max()),sampled_torque_cap_fraction=float(release[:,19].mean()),
        material_velocity_crosscheck_max=float(release[:,20].max()),score_window=[float(release[0,0]),float(release[-1,0])],
        completed_full_duration=bool(data[-1,0]>=3-1e-8))

def main():
    if not (OUT/'BEST_PARAMETERS.json').exists():print(json.dumps(dict(status='SEARCH_NOT_FINISHED',new_physics_steps=0)));return 3
    top=read(OUT/'PREREGISTRATION.json');aggregate=read(OUT/'SEARCH_RESULTS.json');best=read(OUT/'BEST_PARAMETERS.json')
    candidates=top['candidates'];results=aggregate['results'];checks={};summary=[];inputs={};total=0
    checks['declared13budget']=len(candidates)==13 and top['parameters']['maximum_trials']==13
    checks['exactly13attempted']=len(results)==aggregate['trials_started']==best['trials_attempted']==13
    checks['order']=([r['tag'] for r in results]==[c['tag'] for c in candidates])
    refs=None
    for index,(candidate,result) in enumerate(zip(candidates,results),1):
        folder=OUT/candidate['tag'];trial=read(folder/'TRIAL_PREREGISTRATION.json');stored=read(folder/'RESULT.json')
        paths=[folder/name for name in ('TRIAL_PREREGISTRATION.json','RESULT.json','SEARCH_TRACE.npz')];inputs.update({str(p):sha(p) for p in paths})
        with np.load(paths[2],allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
        data=trace['rows'];cmd=trace['commands'];computed=calculate(data,result['termination']);local={}
        local['result_matches_aggregate']=stored==result
        local['candidate_parameters']=same(candidate['parameters'],result['parameters_rad']) and same(trial['parameters_rad'],candidate['parameters']) and same(trace['parameters'],candidate['parameters'])
        vector=np.asarray(candidate['parameters']);local['single_coordinate_only']=bool((index==1 and not vector.any()) or (index>1 and np.count_nonzero(vector)==1 and abs(vector).max()==.02))
        local['indices']=trial['index']==result['trial_index']==index and trial['actual_step_count_before']==total
        local['trial_source_hash']=trial['source_sha256']==top['source_hashes'][str(SOURCE)]
        errors=[]
        for key,value in computed.items():
            actual=result['score'].get(key)
            equal=(actual==value) if isinstance(value,(str,bool,type(None))) else same(actual,value,1e-11)
            if not equal:errors.append(key)
        local['all_score_fields_recomputed']=not errors
        steps=result['actual_physics_steps'];local['steps_within_trialbudget']=isinstance(steps,int) and 0<=steps<=240000 and trial['maximum_steps']==240000
        local['physics_time_count']=abs(result['actual_seconds']-steps*.0000125)<1e-8
        local['command_vs_native_steps']=bool(not cmd.size and steps==0) or bool(cmd.size and (len(cmd)-1)*80<steps<=len(cmd)*80)
        total+=steps;local['cumulative_steps']=result['actual_physics_step_calls_total']==total
        local['trace_columns']=data.size==0 or(data.ndim==2 and data.shape[1]==21)
        local['command_columns']=cmd.size==0 or(cmd.ndim==2 and cmd.shape[1]==85)
        local['finite_trace']=bool(all(np.isfinite(a).all() for a in trace.values()))
        limits,kp,kd=trace['limits'],trace['kp'],trace['kd']
        if refs is None:refs=(limits.copy(),kp.copy(),kd.copy())
        local['shared_caps_gains']=all(np.array_equal(x,y) for x,y in zip((limits,kp,kd),refs))
        local['declared_gain128']=same(kp,limits/.2*128,1e-12) and same(kd,limits*.005*np.sqrt(128),1e-12)
        local['gain_limits_positive_finite']=limits.shape==(42,) and bool((limits>0).all())
        local['requested_rate_cap']=not cmd.size or bool(abs(cmd[:,43:85]).max()<=30+1e-10)
        local['sampled_actual_torque_caps']=not data.size or ('ctrl' in trace and bool(np.all(abs(trace['ctrl'])<=limits+1e-9)))
        local['clip_metadata']=np.asarray(result['target_native_angle_clip_fraction']).shape==(42,) and bool(np.all((np.asarray(result['target_native_angle_clip_fraction'])>=0)&(np.asarray(result['target_native_angle_clip_fraction'])<=1)))
        local['sampled_velocity_crosscheck']=not data.size or float(data[:,20].max())<=1e-8
        if data.size:
            local['body_sample_alignment']=len(trace['time'])==len(data) and same(trace['time'],data[:,0]) and same(trace['com'],data[:,14:17])
            local['trace_time_within_simulation']=data[-1,0]<=result['actual_seconds']+1e-8
        valid=not result['score']['failed'] and result['score'].get('completed_full_duration') and result['warnings']==0 and result['finite']
        summary.append(dict(tag=candidate['tag'],checks=local,score_field_mismatches=errors,eligible=bool(valid),objective=computed['objective'],forward_speed=computed.get('forward_speed'),
            nonpositive_progress=bool(computed.get('forward_speed',0)<=0),termination=result['termination'],actual_steps=steps,max_target_angle_clip_fraction=float(np.max(result['target_native_angle_clip_fraction'])),
            selected=bool(best['selected'] and best['selected']['tag']==candidate['tag'])))
    checks['all_trial_checks']=all(all(row['checks'].values()) for row in summary)
    checks['actual_step_totals']=aggregate['actual_physics_steps']==best['actual_physics_steps']==total and total<=13*240000
    eligible=[r for r in results if not r['score']['failed'] and r['score'].get('completed_full_duration') and r['warnings']==0 and r['finite']]
    expected=min(eligible,key=lambda r:r['score']['objective']) if eligible else None
    checks['BEST_exact_actual_minimum']=best['selected']==expected
    checks['BEST_baseline_matches']=best['baseline']==results[0]
    checks['no_combined_candidate']=expected is None or any(expected==r for r in results)
    checks['selected_positive_progress']=expected is None or expected['score']['forward_speed']>0
    source={path:Path(path).exists() and sha(path)==h for path,h in top['source_hashes'].items()};checks['all_frozen_sources_unchanged']=all(source.values())
    for name in ('PREREGISTRATION.json','SEARCH_RESULTS.json','BEST_PARAMETERS.json','BASELINE_COMPARISON.json'):
        path=OUT/name
        if path.exists():inputs[str(path)]=sha(path)
    comparison=read(OUT/'BASELINE_COMPARISON.json') if (OUT/'BASELINE_COMPARISON.json').exists() else None
    report=dict(status='SEARCH_ACCOUNTING_VERIFIED_NOT_ACCEPTANCE' if all(checks.values()) else 'SEARCH_ACCOUNTING_AUDIT_FAILED',checks=checks,
        trials=summary,total_actual_physics_steps_recorded=total,maximum_physics_steps=13*240000,
        selected_tag=expected['tag'] if expected else None,selected_objective=expected['score']['objective'] if expected else None,
        baseline_comparison=comparison,baseline_identical_claim=False,
        source_hashes_unchanged=source,input_hashes=inputs,analysis_counts=dict(new_physics_steps=0,new_neural_steps=0,FK_calls=0),
        limitations=['Scores use5mscontactsampling andmean6slipratio, notstrictnativeD/heading/alllegslip acceptance.',
            'Baselineoriginal3sphysicaltraceisnear-reproducednotbitwise; explicitreferencejointclipping yieldsrecordedtinyfloatingpointdifferences.',
            'Eligibilitymatchesdeclaredsearchrule; selectedpositiveprogressischeckedseparately. No failed/incomplete/nonfinite/warningcandidate canwin.',
            'Nativecapidentityischeckedagainstthefrozenoriginalsourceprovenanceandsharedtracecaps; thisscriptdoesnotcompileamodel.',
            'Selectedbestisonecompletedmeasuredcandidate, notcombinedcoordinateeffects; requiresnewstrictnative4s/replay andlaterANN>=10s independently.'])
    output=OUT/'AUDIT_SEARCH_RESULTS.json'
    if output.exists():output.read_bytes();raise FileExistsError('preserve prior final audit')
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report),flush=True)
    return 0 if all(checks.values()) else 2
if __name__=='__main__':raise SystemExit(main())
