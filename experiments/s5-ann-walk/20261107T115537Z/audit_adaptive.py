"""Independent adaptive25 saved-search audit. Parent executes after BEST ready.
No search/controller imports, model/FK/physics/replay. Scalar-safe JSON from outset.
"""
from pathlib import Path
import hashlib,json
import numpy as np
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T115537Z')
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
SOURCE=ROOT/'experiments/s5-ann-walk/20261107T115537Z/optimize_adaptive.py'
SOURCE_SHA='d4a620ece0e026cdc28e82172153427bf255c60ebd9779dcd974a678f23fbb60'
W=24.446354755878005

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def encode(v):
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,np.ndarray):return v.tolist()
    raise TypeError(type(v).__name__)
def same(a,b,tol=1e-11):return np.asarray(a).shape==np.asarray(b).shape and bool(np.allclose(a,b,atol=tol,rtol=tol))
def eligible(r):return bool(not r['score']['failed'] and r['score'].get('completed_full_duration',False) and r['finite'] and r['warnings']==0 and np.isfinite(r['score']['objective']))
def best(prior,rows):
    valid=([prior] if prior is not None and eligible(prior) else [])+[r for r in rows if eligible(r)]
    return min(valid,key=lambda r:r['score']['objective']) if valid else None

def score(data,reason):
    failed=reason is not None
    if data.ndim!=2 or not len(data):return dict(objective=1e6,failed=True,reason='NO_TRACE')
    r=data[data[:,0]>=1-1e-9]
    if len(r)<40:return dict(objective=1e6+1e4*failed,failed=True,reason='INSUFFICIENT_RELEASE')
    width=min(20,len(r)//4);first,last=r[:width],r[-width:];yaw=np.unwrap(r[:,17]);heading=float(yaw[:width].mean())
    forward=np.array([np.cos(heading),np.sin(heading)]);delta=last[:,14:16].mean(0)-first[:,14:16].mean(0)
    duration=max(float(last[:,0].mean()-first[:,0].mean()),1e-12);speed=float(delta@forward/duration);lateral=float(abs(delta@[-forward[1],forward[0]])/duration)
    load=r[:,1:7].sum(0);slip=np.divide(r[:,7:13].sum(0),load,out=np.zeros(6),where=load>0);ratios=slip/max(speed,.02)
    mean=float(ratios.mean());support=r[:,1:7].sum(1)/W;nf=r[:,13]/W;turn=float(yaw[-width:].mean()-yaw[:width].mean())
    drop=max(float(first[:,16].mean()-r[:,16].min())-.1,0.)
    penalty=1000*failed+100*float(nf.mean())+100*float(nf.max())+100*max(.05-speed,0)+20*max(.8-float(support.mean()),0)
    penalty+=20*drop+2*lateral/max(speed,.02)+abs(turn)+10*int((load<=0).any())
    # Source subtracts old mean to recover penalty; reproduce its arithmetic order.
    old=mean+penalty;unchanged=float(old-mean);objective=float(ratios.max()+.1*mean+unchanged);unloaded=load<=0
    output=dict(objective=objective,failed=bool(failed),termination=reason,foot_normal_weighted_material_speed=slip,
        slip_to_progress_ratios=ratios,forward_speed=speed,lateral_speed=lateral,heading_change=turn,
        foot_support_mean=float(support.mean()),nonfoot_force_mean_W=float(nf.mean()),nonfoot_force_max_W=float(nf.max()),minimum_COM=float(r[:,16].min()),
        maximum_tilt_rad=float(r[:,18].max()),sampled_torque_cap_fraction=float(r[:,19].mean()),material_velocity_crosscheck_max=float(r[:,20].max()),
        score_window=[float(r[0,0]),float(r[-1,0])],completed_full_duration=bool(data[-1,0]>=3-1e-8),
        maximum_leg_ratio=float(ratios.max()),mean_leg_ratio=mean,unchanged_penalty=unchanged,never_loaded_leg_mask=unloaded.tolist())
    if unloaded.any():output.update(failed=True,selection_failure='NEVER_LOADED_LEG')
    if not np.isfinite(objective):output.update(objective=1e6,failed=True,selection_failure='NONFINITE_OBJECTIVE')
    return output

def main():
    if not (OUT/'BEST_PARAMETERS.json').exists():print(json.dumps(dict(status='ADAPTIVE_SEARCH_NOT_FINISHED',physics_steps=0)));return 3
    require_source=sha(SOURCE)==SOURCE_SHA
    protocol=read(OUT/'PREREGISTRATION.json');aggregate=read(OUT/'SEARCH_RESULTS.json');final=read(OUT/'BEST_PARAMETERS.json');trace=read(OUT/'ADAPTATION_TRACE.json')
    results=aggregate['results'];bytag={r['tag']:r for r in results};center=np.array([0.,.02,0.,0.,0.,.01]);prior_evidence=read(protocol['center_evidence'])
    checks=dict(frozen_source=require_source,declared25=len(results)==aggregate['trials_started']==final['trials_attempted']==25,
        no_duplicate_trials=len(bytag)==25,declared6batches=len(trace)==6,initial_center=same(protocol['center'],center) and same(prior_evidence['selected']['parameters_rad'],center),
        budget=protocol['budget']['new_trials']==25 and protocol['budget']['max_native_steps']==6000000 and protocol['budget']['combined_declared_offset_trials']==51)
    assert results,'No saved trials';expected_tags=['measured_center'];expected_parameters={'measured_center':center}
    rng=np.random.default_rng(115537);measured=best(None,[results[0]]);current=center.copy();batch_checks=[]
    for b in range(6):
        record=trace[b];prior=measured;current=np.asarray(prior['parameters_rad']).copy() if prior is not None else current
        state_before=rng.bit_generator.state;directions=rng.normal(size=(2,6));directions/=np.linalg.norm(directions,axis=1)[:,None]
        perturb=np.array([directions[0],-directions[0],directions[1],-directions[1]])*.025;parameters=np.clip(current[None]+perturb,-.08,.08);state_after=rng.bit_generator.state
        tags=['batch'+str(b+1)+'_candidate'+str(i+1) for i in range(4)];expected_tags.extend(tags)
        for tag,x in zip(tags,parameters):expected_parameters[tag]=x
        rows=[bytag[tag] for tag in tags];chosen=best(prior,rows)
        ranked=sorted(([prior] if prior is not None else [])+rows,key=lambda r:r['score']['objective'])
        ranking=[dict(tag=r['tag'],objective=r['score']['objective'],eligible=eligible(r)) for r in ranked]
        local=dict(batch_number=record['batch']==b,rng_before=record['rng_state_before']==state_before,rng_after=record['rng_state_after']==state_after,
            prior_best=record['prior_best_tag']==(prior['tag'] if prior else None) and record['prior_best_objective']==(prior['score']['objective'] if prior else None),
            measured_center=same(record['center'],current),unit_direction=same(record['directions'],directions) and bool(np.allclose(np.linalg.norm(record['directions'],axis=1),1.,atol=1e-14)),
            antithetic_perturb=same(record['perturbations'],perturb),candidate_clip=same(record['candidates'],parameters),
            clipmask=bool(np.array_equal(record['parameter_clip_mask'],current[None]+perturb!=parameters)),all4_measured=all(same(row['parameters_rad'],x) for row,x in zip(rows,parameters)),
            status_measured=record['status']=='BATCH_MEASURED',stable_ranking=record['ranking']==ranking,
            selected_actual_eligible=record['selected_tag']==(chosen['tag'] if chosen else None) and record['selected_parameters']==(chosen['parameters_rad'] if chosen else None),
            selected_objective=record['selected_objective']==(chosen['score']['objective'] if chosen else None),measured_selection_only=record['measured_selection_only'] is True)
        batch_checks.append(dict(batch=b,checks=local,selected_tag=chosen['tag'] if chosen else None));measured=chosen
    checks['all_batch_rng_and_selection']=all(all(b['checks'].values()) for b in batch_checks)
    checks['trial_order']=[r['tag'] for r in results]==expected_tags
    inputs={};summaries=[];total=0;shared=None
    for i,result in enumerate(results,1):
        folder=OUT/result['tag'];files=[folder/name for name in ('TRIAL_PREREGISTRATION.json','RESULT.json','SEARCH_TRACE.npz')]
        inputs.update({str(p):sha(p) for p in files});registration=read(files[0]);stored=read(files[1]);arrays=load(files[2]);commands=arrays['commands'];data=arrays['rows']
        computed=score(data,result['termination']);wrong=[]
        for key,value in computed.items():
            actual=result['score'].get(key);equal=(actual==value) if isinstance(value,(str,bool,type(None))) else same(actual,value)
            if not equal:wrong.append(key)
        local=dict(saved_result_matches=stored==result,score_all_fields=not wrong,
            deterministic_parameters=same(result['parameters_rad'],expected_parameters[result['tag']]) and same(registration['parameters_rad'],result['parameters_rad']) and same(arrays['parameters'],result['parameters_rad']),
            global_parameter_bounds=bool(abs(np.asarray(result['parameters_rad'])).max()<=.08),
            index_and_source=registration['index']==result['trial_index']==i and registration['source_sha256']==SOURCE_SHA,
            prior_cumulative_steps=registration['actual_step_count_before']==total)
        steps=result['actual_physics_steps'];local['step_budget']=isinstance(steps,int) and 0<=steps<=240000 and registration['maximum_steps']==240000
        local['time_count']=abs(result['actual_seconds']-steps*.0000125)<1e-8
        local['command_count']=(bool(commands.size and (len(commands)-1)*80<steps<=len(commands)*80) or bool(not commands.size and steps==0))
        total+=steps;local['cumulative_steps']=result['actual_physics_step_calls_total']==total
        limits,kp,kd=[arrays[key] for key in ('limits','kp','kd')]
        if shared is None:shared=[x.copy() for x in (limits,kp,kd)]
        local['same_caps_and_gains']=all(np.array_equal(a,b) for a,b in zip(shared,(limits,kp,kd)))
        local['gain128']=same(kp,limits/.2*128,1e-12) and same(kd,limits*.005*np.sqrt(128),1e-12)
        local['positive_limits']=limits.shape==(42,) and bool((limits>0).all())
        local['finite_or_explicit_failure']=bool(all(np.isfinite(a).all() for a in arrays.values())) or bool(result['score']['failed'] and not eligible(result))
        local['requested_rate_cap']=not commands.size or bool(abs(commands[:,43:85]).max()<=30+1e-10)
        local['sampled_ctrl_cap']=not data.size or bool(np.all(abs(arrays['ctrl'])<=limits+1e-9))
        clips=np.asarray(result['target_native_angle_clip_fraction']);local['clip_metadata']=clips.shape==(42,) and bool(np.all((clips>=0)&(clips<=1)))
        if data.size:local['body_times_and_COM']=same(arrays['time'],data[:,0]) and same(arrays['com'],data[:,14:17])
        summaries.append(dict(tag=result['tag'],checks=local,score_mismatches=wrong,eligible=eligible(result),objective=computed['objective'],maximum_leg_ratio=computed.get('maximum_leg_ratio'),
            forward_speed=computed.get('forward_speed'),termination=result['termination'],steps=steps,clip_max=float(clips.max())))
    checks['every_trial']=all(all(r['checks'].values()) for r in summaries)
    global_min=best(None,results);checks['final_global_actual_minimum']=final['selected']==measured==global_min
    checks['final_baseline_retest']=final['baseline_retest']==results[0]
    checks['selected_positive_progress']=measured is None or measured['score']['forward_speed']>0
    checks['new_actual_step_budget']=total==aggregate['actual_physics_steps']==final['actual_physics_steps'] and total<=6000000
    previous=[read(OUT.parent/stamp/'BEST_PARAMETERS.json') for stamp in ('20261107T112003Z','20261107T113529Z')];prior_steps=sum(x['actual_physics_steps'] for x in previous)
    checks['prior26_and_combined51']=sum(x['trials_attempted'] for x in previous)==26 and prior_steps<=6240000 and final['total_declared_offset_budget']==51 and prior_steps+total<=12240000
    checks['recorded_declared_counts']=aggregate['previous_offset_trials']==final['previous_offset_trials']==26 and aggregate['combined_declared_budget']==51
    frozen={path:Path(path).exists() and sha(path)==digest for path,digest in protocol['source_hashes'].items()};checks['all_sources_unchanged']=all(frozen.values())
    checks['effective_source_unchanged']=sha(OUT/'EFFECTIVE_SEARCH.py')==protocol['effective_sha256']
    selected_origin=OUT.parent/'20261107T113529Z'/prior_evidence['selected']['tag']/'SEARCH_TRACE.npz';old=load(selected_origin);fresh=load(OUT/'measured_center/SEARCH_TRACE.npz')
    center_difference={key:dict(bitwise=bool(np.array_equal(old[key],fresh[key])),max_abs=float(abs(old[key]-fresh[key]).max())) for key in ('rows','commands','qpos','qvel','ctrl','com','feet','time')}
    for path in [OUT/name for name in ('PREREGISTRATION.json','SEARCH_RESULTS.json','ADAPTATION_TRACE.json','BEST_PARAMETERS.json','EFFECTIVE_SEARCH.py')]+[Path(__file__),selected_origin]:inputs[str(path)]=sha(path)
    report=dict(status='ADAPTIVE_ACCOUNTING_VERIFIED_NOT_ACCEPTANCE' if all(checks.values()) else 'ADAPTIVE_ACCOUNTING_FAILED',checks=checks,batches=batch_checks,trials=summaries,
        selected=final['selected'],new_trials=25,prior_offset_trials=26,combined_declared_offset_trials=51,new_recorded_steps=total,prior_recorded_steps=prior_steps,total_recorded_search_steps=prior_steps+total,
        maximum_new_steps=6000000,maximum_total_steps=12240000,center_retest_vs_prior=center_difference,source_hashes_unchanged=frozen,input_hashes=inputs,
        analysis_counts=dict(new_physics_steps=0,model_compile=0,FK_calls=0,replay=0),
        limitations=['Source writes eachbatchcandidate/rngregistration beforeits4trialloop; savedfilesystemartifactsalone cannotprove walltimeordering ornonbackdating.',
            'Antithetic rawdirectionsbecomenonantithetic whenclippedatglobalbounds; thisisdeclaredandexactclippingisreconstructed.',
            'Everycenterisactualvalidmeasuredbest; stabletieskeeppriorbest. Failedcandidatesremaininrankingsbutcannotreplaceeligiblebest.',
            'These51countscoveroffsetsearchtrialsonly, not unrelatedANN/strict/controlphysicalruns. Theydonotclaimallprojectsimulationusage.',
            'PreviouscenterstrictteacherpassedD/steps/headingbutmaterialfailed. Newwinner5msscoreisnotstrictnativeacceptance orneuralwalking.',
            'Winnerrequiresseparate4sstrictnative/.5msmaterialreplay andlaterANN>=10s/controls. No model/FK/physics/replayexecutedbythisauditor.'])
    output=OUT/'AUDIT_ADAPTIVE.json'
    if output.exists():output.read_bytes();raise FileExistsError('preserve existing adaptive audit')
    output.write_text(json.dumps(report,indent=2,allow_nan=False,default=encode)+'\n');print(json.dumps(report,default=encode),flush=True)
    return 0 if all(checks.values()) else 2
if __name__=='__main__':raise SystemExit(main())
