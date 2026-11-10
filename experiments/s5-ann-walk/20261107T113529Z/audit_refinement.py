"""Independent bounded second-stage accounting; parent executes after BEST exists.
No search/runtime imports, model/FK/physics/replay. Returns3 while incomplete.
"""
from pathlib import Path
import hashlib,json
import numpy as np
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T113529Z')
PREVIOUS=OUT.parent/'20261107T112003Z'
SEARCH_SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T112003Z/optimize_stance.py')
W=24.446354755878005

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def encode(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,np.ndarray):return x.tolist()
    raise TypeError(type(x).__name__)
def same(a,b,tol=1e-11):return np.asarray(a).shape==np.asarray(b).shape and bool(np.allclose(a,b,atol=tol,rtol=tol))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def score(data,reason):
    failed=reason is not None
    if data.ndim!=2 or not len(data):return dict(objective=1e6,failed=True,reason='NO_TRACE')
    r=data[data[:,0]>=1-1e-9]
    if len(r)<40:return dict(objective=1e6+1e4*failed,failed=True,reason='INSUFFICIENT_RELEASE')
    width=min(20,len(r)//4);start,finish=r[:width],r[-width:]
    yaw=np.unwrap(r[:,17]);heading=float(yaw[:width].mean());f=np.array([np.cos(heading),np.sin(heading)])
    delta=finish[:,14:16].mean(0)-start[:,14:16].mean(0);duration=max(float(finish[:,0].mean()-start[:,0].mean()),1e-12)
    speed=float(delta@f/duration);lateral=float(abs(delta@[-f[1],f[0]])/duration)
    force=r[:,1:7].sum(0);slip=np.divide(r[:,7:13].sum(0),force,out=np.zeros(6),where=force>0);ratios=slip/max(speed,.02)
    support=r[:,1:7].sum(1)/W;nf=r[:,13]/W;turn=float(yaw[-width:].mean()-yaw[:width].mean())
    heightdrop=max(float(start[:,16].mean()-r[:,16].min())-.1,0.)
    penalty=1000*failed+100*float(nf.mean())+100*float(nf.max())+100*max(.05-speed,0)+20*max(.8-float(support.mean()),0)
    penalty+=20*heightdrop+2*lateral/max(speed,.02)+abs(turn)+10*int((force<=0).any())
    return dict(objective=float(ratios.max()+.1*ratios.mean()+penalty),prior_mean_objective=float(ratios.mean()+penalty),worst_leg_ratio=float(ratios.max()),
        failed=bool(failed),termination=reason,foot_normal_weighted_material_speed=slip,slip_to_progress_ratios=ratios,
        forward_speed=speed,lateral_speed=lateral,heading_change=turn,foot_support_mean=float(support.mean()),
        nonfoot_force_mean_W=float(nf.mean()),nonfoot_force_max_W=float(nf.max()),minimum_COM=float(r[:,16].min()),
        maximum_tilt_rad=float(r[:,18].max()),sampled_torque_cap_fraction=float(r[:,19].mean()),material_velocity_crosscheck_max=float(r[:,20].max()),
        score_window=[float(r[0,0]),float(r[-1,0])],completed_full_duration=bool(data[-1,0]>=3-1e-8))

def main():
    if not (OUT/'BEST_PARAMETERS.json').exists():print(json.dumps(dict(status='REFINEMENT_NOT_FINISHED',physics_steps=0)));return 3
    protocol=read(OUT/'PREREGISTRATION.json');aggregate=read(OUT/'SEARCH_RESULTS.json');best=read(OUT/'BEST_PARAMETERS.json')
    previous=read(PREVIOUS/'BEST_PARAMETERS.json');previous_audit=read(PREVIOUS/'AUDIT_SEARCH_RESULTS.json')
    prior=previous['selected'];center=np.asarray(protocol['center']);candidates=protocol['candidates'];results=aggregate['results']
    checks=dict(center_is_measured_previous_best=same(center,prior['parameters_rad']) and same(center,[0,.02,0,0,0,0]),
        declared_budget=protocol['budget']['new_trials']==13 and protocol['budget']['prior_search_trials']==13 and protocol['budget']['aggregate_declared_trials']==26,
        previous13_accounting_verified=bool(all(previous_audit['checks'].values()) and previous['trials_attempted']==13),
        all13_completed_accounted=len(candidates)==len(results)==aggregate['trials_started']==best['trials_attempted']==13,
        result_order=[r['tag'] for r in results]==[c['tag'] for c in candidates])
    total=0;trials=[];inputs={};shared=None
    for i,(candidate,result) in enumerate(zip(candidates,results)):
        folder=OUT/candidate['tag'];paths=[folder/n for n in ('TRIAL_PREREGISTRATION.json','RESULT.json','SEARCH_TRACE.npz')]
        inputs.update({str(p):sha(p) for p in paths});pr=read(paths[0]);stored=read(paths[1]);trace=load(paths[2]);data=trace['rows'];commands=trace['commands']
        vector=np.asarray(candidate['parameters']);expected=center.copy()
        if i:expected[(i-1)//2]+=-.01 if (i-1)%2==0 else .01
        local=dict(candidate_center_coordinate=same(vector,expected) and bool(abs(vector).max()<=.03+1e-15),
            all_parameter_records=same(vector,result['parameters_rad']) and same(vector,pr['parameters_rad']) and same(vector,trace['parameters']),
            stored_result_matches=stored==result,indices=pr['index']==result['trial_index']==i+1 and pr['actual_step_count_before']==total,
            original_physics_source_hash=pr['source_sha256']==sha(SEARCH_SOURCE))
        computed=score(data,result['termination']);mismatch=[]
        for key,value in computed.items():
            equal=(result['score'].get(key)==value) if isinstance(value,(str,bool,type(None))) else same(result['score'].get(key),value)
            if not equal:mismatch.append(key)
        local['all_score_fields_recomputed']=not mismatch
        steps=result['actual_physics_steps'];local['trial_step_budget']=isinstance(steps,int) and 0<=steps<=240000 and pr['maximum_steps']==240000
        local['time_from_steps']=abs(result['actual_seconds']-steps*.0000125)<1e-8
        local['command_step_accounting']=bool(commands.size and (len(commands)-1)*80<steps<=len(commands)*80) or bool(not commands.size and steps==0)
        total+=steps;local['cumulative_step_accounting']=result['actual_physics_step_calls_total']==total
        gains=[trace[k] for k in ('limits','kp','kd')]
        if shared is None:shared=[x.copy() for x in gains]
        local['same_caps_gains']=all(np.array_equal(x,y) for x,y in zip(gains,shared))
        limits,kp,kd=gains;local['declared_gain128']=same(kp,limits/.2*128,1e-12) and same(kd,limits*.005*np.sqrt(128),1e-12)
        local['finite_trace']=all(np.isfinite(x).all() for x in trace.values())
        local['command_rate_cap']=not commands.size or bool(abs(commands[:,43:85]).max()<=30+1e-10)
        local['sampled_native_torque_caps']=not data.size or ('ctrl' in trace and bool(np.all(abs(trace['ctrl'])<=limits+1e-9)))
        clips=np.asarray(result['target_native_angle_clip_fraction']);local['clip_metadata']=clips.shape==(42,) and bool(np.all((clips>=0)&(clips<=1)))
        if data.size:
            local['trace_body_alignment']=same(trace['time'],data[:,0]) and same(trace['com'],data[:,14:17])
            local['velocity_crosscheck']=float(data[:,20].max())<=1e-8
        eligible=not result['score']['failed'] and result['score'].get('completed_full_duration') and result['finite'] and not result['warnings']
        trials.append(dict(tag=candidate['tag'],parameters=vector,checks=local,score_mismatches=mismatch,eligible=bool(eligible),objective=computed['objective'],
            worst_ratio=computed.get('worst_leg_ratio'),forward_speed=computed.get('forward_speed'),actual_steps=steps,termination=result['termination'],max_target_clip_fraction=float(clips.max())))
    checks['every_trial_checks']=all(all(row['checks'].values()) for row in trials)
    checks['new_step_totals']=total==aggregate['actual_physics_steps']==best['actual_physics_steps'] and total<=3120000
    combined=total+previous['actual_physics_steps'];checks['aggregate26_step_budget']=combined<=6240000 and previous['actual_physics_steps']==3120000
    valid=[r for r in results if not r['score']['failed'] and r['score'].get('completed_full_duration') and r['finite'] and not r['warnings']]
    winner=min(valid,key=lambda r:r['score']['objective']) if valid else None
    checks['BEST_exact_minimum_valid']=best['selected']==winner;checks['BEST_center_record']=best['baseline']==results[0]
    checks['selected_measured_not_combined']=winner is None or winner in results
    checks['selected_positive_progress']=winner is None or winner['score']['forward_speed']>0
    center_trace=load(OUT/'measured_center/SEARCH_TRACE.npz');prior_trace=load(PREVIOUS/prior['tag']/'SEARCH_TRACE.npz')
    comparison={key:dict(bitwise_equal=bool(np.array_equal(center_trace[key],prior_trace[key])),max_abs_error=float(abs(center_trace[key]-prior_trace[key]).max())) for key in ('rows','commands','qpos','qvel','ctrl','com','feet','time')}
    sourcechecks={path:Path(path).exists() and sha(path)==h for path,h in protocol['source_hashes'].items()};checks['registered_sources_unchanged']=all(sourcechecks.values())
    for path in [OUT/'PREREGISTRATION.json',OUT/'SEARCH_RESULTS.json',OUT/'BEST_PARAMETERS.json',PREVIOUS/'BEST_PARAMETERS.json',PREVIOUS/'AUDIT_SEARCH_RESULTS.json',Path(__file__)]:inputs[str(path)]=sha(path)
    report=dict(status='REFINEMENT_ACCOUNTING_VERIFIED_NOT_ACCEPTANCE' if all(checks.values()) else 'REFINEMENT_ACCOUNTING_FAILED',checks=checks,
        trials=trials,selected_tag=winner['tag'] if winner else None,selected_parameters=winner['parameters_rad'] if winner else None,
        selected_search_objective=winner['score']['objective'] if winner else None,prior_trials=13,new_trials=len(results),aggregate_declared_trials=26,
        prior_physics_steps=previous['actual_physics_steps'],new_physics_steps_recorded=total,total_search_physics_steps_recorded=combined,max_search_steps=6240000,
        center_vs_previous_measured_best=comparison,source_hashes_unchanged=sourcechecks,input_hashes=inputs,
        analysis_counts=dict(new_physics_steps=0,model_compile=0,FK_calls=0,replay=0),
        limitations=['Searchobjectivechangedtoworst6+.1mean6+unchangedsoftpenalties; physicalacceptancecriteriaareunchanged.',
            'All13newtrialsaremeasuredcenterplusonecoordinate.01; noneareinferredcombinationsofseparatebenefits. Total26searchtrialsarebounded, notunlimited.',
            '5msmaterial/100msmeanprogresssearchscoreisnotnativeD/alllegslip<=.5/headingacceptance. Newstrict4s/replay mustjudge selectedcandidatebefore anyneuralclaim.',
            'PreviousselectedstrictteacherpassedD/steps/headingbutfailedLH/RHmaterialslip; it wasnotacceptedwalking.',
            'Same-modelcaps/gains arecheckedfromsavedmetadata/frozensourceids; no modelorphysicalexecutionbythisauditor.'])
    output=OUT/'AUDIT_REFINEMENT.json'
    if output.exists():output.read_bytes();raise FileExistsError('preserve prior refinement audit')
    output.write_text(json.dumps(report,indent=2,allow_nan=False,default=encode)+'\n');print(json.dumps(report,default=encode),flush=True)
    return 0 if all(checks.values()) else 2
if __name__=='__main__':raise SystemExit(main())
