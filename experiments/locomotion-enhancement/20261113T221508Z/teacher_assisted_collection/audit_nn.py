"""Fresh strict007d replay of saved collection; parent numerical execution only.
Import is stdlib-only. This audit never creates a Core or a physical trajectory.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import sys
from common import ROOT,OLD,BASE,FORMAT,CASES,CONDITIONS,read,save,sha,verify,cpu1,load_motor,frozen

CHECKPOINT_SHA='007d0c74118715eacd03d49c7e6847d8349e55db2793dd9d461a4b9c311e0fcd'
MASK_NAMES=('actual_COM','actual_NF_1ms','IK','unbacked_plannedload','teacher_supported','qdot_clip','rate_clip','target_bounds','tau_ff_clip','finite42','actual_restore')
FORBIDDEN=('mujoco','physics','flygym','student_wave')


def _run(case,report):
    if any(name in sys.modules for name in FORBIDDEN):raise ValueError('Fresh NN process required before any physical backend')
    cpu1();sources=frozen();verify(sources)
    folder=ROOT/'trials'/case;r=read(folder/'RESULT.json');binding=read(ROOT/'INPUT_BINDINGS.json');verify(r['source_hashes'])
    report.update(result_sha256=sha(folder/'RESULT.json'),source_hashes=sources,checkpoint_sha256=binding['checkpoint_sha256'])
    condition,command=CASES[case]
    if r['kind']!=FORMAT or r.get('case',case)!=case or r['condition'] not in (case,condition) or r['checkpoint_sha256']!=CHECKPOINT_SHA or binding['checkpoint_sha256']!=CHECKPOINT_SHA or r['checkpoint']!=binding['checkpoint'] or r['state_dims']!=[128,64,64,96] or r['state_total']!=352 or r['final_step']!=600 or r['command']!=command:raise ValueError('Actual collection/fixed007d binding identity required')
    # Strict checkpoint load occurs before numerical metrics or physical modules.
    m,payload=load_motor()
    if payload['step']!=600 or payload['format_version']!='INDEPENDENT_SPEED_YAW_MOTOR_FASTPATH_REPAIR_V1':raise ValueError('Strict fastpath final600 required')
    import torch
    import numpy as np
    checks=report['checks']
    def weight_digest():
        h=hashlib.sha256()
        for name,value in sorted(m.state_dict().items()):
            h.update(name.encode());h.update(str(value.dtype).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    weights_before=weight_digest();artifacts={}
    required={'trace':'trace_1ms.npz','neural_decisions':'neural_decisions.npz','initial_state':'INITIAL_STATE.json','runtime_geometry':'runtime_geometry.npz','rng_initial':'RNG_INITIAL.json'}
    for key,filename in required.items():
        item=r[key];path=folder/filename
        checks[key+'_SHA']=item['path']==str(path) and sha(path)==item['sha256'];artifacts[str(path)]=sha(path)
    warm_artifact=folder/'WARM_BOUNDARIES.json';item=r['warm_boundaries']
    checks['warm_boundaries_SHA']=item['path']==str(warm_artifact) and sha(warm_artifact)==item['sha256'];artifacts[str(warm_artifact)]=sha(warm_artifact)
    for key,filename in (('geometry','geometry_1ms.npz'),('handover','HANDOVER.json'),('body_length_artifact','BODY_LENGTH.json'),('teacher_diagnostics','teacher_diagnostics.jsonl')):
        path=folder/filename
        if key in r:
            checks[key+'_SHA']=r[key]['path']==str(path) and sha(path)==r[key]['sha256'];artifacts[str(path)]=sha(path)
    if not all(checks.values()):raise ValueError('Saved artifact identity must pass before array replay')
    def arrays(path):
        with np.load(path,allow_pickle=False) as z:return {key:z[key].copy() for key in z.files}
    t=arrays(folder/'trace_1ms.npz');d=arrays(folder/'neural_decisions.npz');compiled=arrays(folder/'runtime_geometry.npz')
    g=arrays(folder/'geometry_1ms.npz') if 'geometry' in r else {'time':np.empty(0),'control_source':np.empty(0,dtype=int)}
    n=len(d['time']);ms=np.arange(n)*5;candidate=ms>=5800;warm=(ms>=1000)&(ms<5800);settle=ms<1000;assisted=case!='old_walk_control'
    fields=('mean','rawmean','oldmean','oldmean_direct','residual','policy_action','student_action','actually_applied_action','mixed_applied_action','teacher_action','supervision_target')
    if any(key not in d for key in fields+('teacher_available','label_mask','label_qualified','hidden_before','hidden_after','X238','command','physical_step_ms','control_source')):raise ValueError('Complete collection neural schema required')
    checks['all_saved_arrays_finite']=all(bool(np.isfinite(v).all()) for data in (t,d,g,compiled) for v in data.values())
    checks['neural_clock']=np.array_equal(d['physical_step_ms'],ms) and bool(np.all(np.abs(d['time']-ms*.001)<1e-8))
    checks['trace_clock']=bool(np.all(np.abs(t['time']-np.arange(1,len(t['time'])+1)*.001)<1e-8))
    checks['neural_shapes']=n>0 and d['X238'].shape==(n,238) and d['command'].shape==(n,2) and d['hidden_before'].shape==d['hidden_after'].shape==(n,352) and all(d[key].shape==(n,42) for key in fields)
    checks['label_schema']=d['label_mask'].shape==(n,11) and d['label_mask'].dtype==np.bool_ and d['label_qualified'].dtype==np.bool_ and d['teacher_available'].dtype==np.bool_ and tuple(r['label_mask_names'])==MASK_NAMES
    checks['state_chain']=np.array_equal(d['hidden_before'][0],np.zeros(352)) and np.array_equal(d['hidden_before'][1:],d['hidden_after'][:-1])
    checks['warm_new96_zero']=bool(np.all(d['hidden_before'][ms<5800,256:]==0.) and np.all(d['hidden_after'][ms<5800,256:]==0.))
    checks['phase_sources']=np.array_equal(d['control_source'],np.where(settle,0,np.where(warm,2,7 if assisted else 8)))
    checks['actual_action_5ms_hold']=np.array_equal(t['action'],np.repeat(d['actually_applied_action'],5,axis=0)[:len(t['time'])])
    checks['actual_source_5ms_hold']=np.array_equal(t['control_source'],np.repeat(d['control_source'],5)[:len(t['time'])])
    checks['bounded_actions']=all(bool(np.all(np.abs(d[key])<=1.+1e-12)) for key in ('policy_action','student_action','actually_applied_action','mixed_applied_action','teacher_action','supervision_target'))
    checks['no_extra_forces']=bool(np.all(t['external_body_force_max']==0.) and np.all(t['external_generalized_force_max']==0.))
    checks['geometry_clock']=np.array_equal(g['time'],t['time'][1000:])
    # Independently reconstruct each recorded pre-action boundary from saved
    # endpoints/history, never instantiate a physical Core for this comparison.
    initial=read(folder/'INITIAL_STATE.json');boundaries=read(warm_artifact)['rows']
    present=set(int(v) for v in d['physical_step_ms']);expected_events=[]
    if 1000 in present:expected_events.append(('first_warm_factory',1000))
    expected_events.extend(('neural_and_actual_boundary',v) for v in (1000,5795,5800) if v in present)
    checks['warm_boundary_exact_event_set']=sorted((row['event'],row['physical_ms']) for row in boundaries)==sorted(expected_events)
    mechanical_keys=('qpos','qvel','ctrl','xfrc_applied','qfrc_applied','target','buffer');boundary_checks=[]
    for row in boundaries:
        physical_ms=int(row['physical_ms']);k=physical_ms//5
        if physical_ms not in present or physical_ms<40:raise ValueError('Boundary must refer to an actual recorded preaction time')
        values=dict(qpos=t['qpos'][physical_ms-1],qvel=t['qvel'][physical_ms-1],ctrl=t['actualctrl'][physical_ms-1],target=t['target'][physical_ms-1],buffer=t['target'][physical_ms-40:physical_ms])
        for name,max_name in (('xfrc_applied','external_body_force_max'),('qfrc_applied','external_generalized_force_max')):
            if name in t:values[name]=t[name][physical_ms-1]
            elif t[max_name][physical_ms-1]==0.:values[name]=np.zeros_like(np.asarray(initial[name],dtype=np.float64))
            else:raise ValueError('Cannot reconstruct nonzero unsaved boundary force')
        exact_hashes={name:hashlib.sha256(np.ascontiguousarray(values[name]).tobytes()).hexdigest() for name in mechanical_keys}
        same=np.array_equal(np.asarray(row['target']),values['target']) and np.array_equal(np.asarray(row['buffer']),values['buffer'])
        if row['event']=='first_warm_factory':
            same=same and row['before_sha256']==row['after_sha256']==exact_hashes and np.array_equal(np.asarray(row['hidden']),d['hidden_before'][k])
        elif row['event']=='neural_and_actual_boundary':
            same=same and row['physical_sha256']==exact_hashes and np.array_equal(np.asarray(row['hidden_before']),d['hidden_before'][k]) and np.array_equal(np.asarray(row['hidden_after']),d['hidden_after'][k]) and np.array_equal(np.asarray(row['actually_applied_action']),d['actually_applied_action'][k])
        else:same=False
        boundary_checks.append(bool(same))
    checks['warm_boundary_trace_history_hidden_exact']=all(boundary_checks)
    checks['all71opaque']=int(np.sum(compiled['geom_type']==7))==71 and bool(np.all(compiled['geom_rgba'][compiled['geom_type']==7,3]==1.))
    replay={key:[] for key in ('mean','rawmean','oldmean','oldmean_direct','residual','hidden_before','hidden_after','policy_action')};state=m.initial_state(1);counts=report['freshNN_calls'];errors={}
    def count_modules(module,inputs):counts['module_forward_calls']+=1
    handle=torch.nn.modules.module.register_module_forward_pre_hook(count_modules)
    try:
        with torch.no_grad():
            for k,x in enumerate(d['X238']):
                physical_ms=k*5;hb=torch.cat(state,-1)[0].detach().numpy().copy();xt=torch.tensor(x[None],dtype=torch.float64);u=torch.tensor(d['command'][k:k+1],dtype=torch.float64)
                if physical_ms<1000:
                    if not np.array_equal(d['command'][k],np.zeros(2)):raise ValueError('Settle consumes no NN/task')
                    mean=torch.zeros((1,42),dtype=torch.float64);direct=mean.clone();residual=torch.zeros_like(mean)
                else:
                    expected=[1.,0.] if physical_ms<5800 or not assisted else command
                    if not np.array_equal(d['command'][k],np.asarray(expected)):raise ValueError('Saved task command changed')
                    projected=m.legacy_projection(u)
                    if not assisted and physical_ms>=5800 and not torch.equal(projected,torch.zeros_like(projected)):raise ValueError('Old control task[1,0] must project to actual old command[0,0]')
                    direct,oldafter=m.old.step(xt,projected,state[:3]);counts['old_step_direct']+=1
                    if physical_ms<5800:
                        mean,state=m.prime_old(xt,u,state);residual=torch.zeros_like(mean);counts['prime_old']+=1
                    else:mean,state,residual=m.step_details(xt,u,state);counts['step_details']+=1
                    if any(not torch.equal(a,b) for a,b in zip(oldafter,state[:3])):raise ValueError('Independent direct old256 chain differs')
                row=dict(mean=mean[0].numpy().copy(),rawmean=mean[0].numpy().copy(),oldmean=(mean-residual)[0].numpy().copy(),oldmean_direct=direct[0].numpy().copy(),residual=residual[0].numpy().copy(),hidden_before=hb,hidden_after=torch.cat(state,-1)[0].numpy().copy(),policy_action=mean[0].tanh().numpy().copy())
                for key,value in row.items():replay[key].append(value)
    finally:handle.remove()
    for key,rows in replay.items():
        got=np.asarray(rows);err=float(np.max(np.abs(got-d[key]))) if got.size else 0.;errors[key]=err
        checks['fresh_replay_'+key]=np.array_equal(got,d[key]) if key=='oldmean_direct' else err<=1e-12
    if 'oldmean_compat' in d:checks['oldmean_compat_alias']=np.array_equal(d['oldmean_compat'],d['oldmean'])
    checks['student_full007d_policy']=np.array_equal(d['student_action'],d['policy_action'])
    checks['actual_mixed_alias']=np.array_equal(d['mixed_applied_action'],d['actually_applied_action'])
    checks['settle_zero_actions']=bool(np.all(d['actually_applied_action'][settle]==0.) and np.all(d['supervision_target'][settle]==0.))
    checks['warm_supervision_actual']=np.array_equal(d['supervision_target'][warm],d['actually_applied_action'][warm])
    checks['teacher_availability_phase']=np.array_equal(d['teacher_available'],warm|(candidate if assisted else np.zeros(n,dtype=bool)))
    checks['warm_teacheraction_actual']=np.array_equal(d['teacher_action'][warm],d['actually_applied_action'][warm])
    checks['absent_teacher_zero']=bool(np.all(d['teacher_action'][~d['teacher_available']]==0.))
    mask_not_applicable=~candidate if assisted else np.ones(n,dtype=bool)
    checks['inapplicable_mask_allfalse']=not bool(np.any(d['label_mask'][mask_not_applicable])) and not bool(np.any(d['label_qualified'][mask_not_applicable]))
    checks['no_qualified_labels_beforecandidate']=not bool(np.any(d['label_qualified'][~candidate]))
    checks['explicit_no_neural_acceptance']=r['neural_acceptance'] is False and r['training_updates']==0 and r['data_qualified_for_training'] is False
    if assisted:
        checks['fixed_mixture_weights']=r['student_weight']==.25 and r['teacher_weight']==.75 and r['teaching_label_target']=='unmixed_full_teacher_action'
        expected=.75*d['teacher_action'][candidate]+.25*d['student_action'][candidate]
        checks['assisted_fixed_mixture']=bool(np.all(np.abs(expected-d['actually_applied_action'][candidate])<=1e-12))
        checks['unmixed_supervision_target']=np.array_equal(d['supervision_target'][candidate],d['teacher_action'][candidate])
        checks['candidate_teacher_available']=bool(np.all(d['teacher_available'][candidate]))
        checks['candidate_qualified_conjunction']=np.array_equal(d['label_qualified'][candidate],np.all(d['label_mask'][candidate],axis=1))
    else:
        direct_action=np.tanh(np.asarray(replay['oldmean_direct']))
        checks['old_control_applied_direct']=bool(np.all(np.abs(d['actually_applied_action'][candidate]-direct_action[candidate])<=1e-12))
        checks['old_control_target_direct']=bool(np.all(np.abs(d['supervision_target'][candidate]-direct_action[candidate])<=1e-12))
        checks['control_no_candidate_teacher']=not bool(np.any(d['teacher_available'][candidate])) and bool(np.all(d['teacher_action'][candidate]==0.))
    checks['actual_teacher_call_count']=r['teacher_calls']==int(warm.sum())+(int(candidate.sum()) if assisted else 0) and r['teacher_calls_after_handover']==(int(candidate.sum()) if assisted else 0)
    checks['native_counts_bound']=r['actual_ms']==len(t['time']) and r['actual_native_steps']==len(t['time'])*80==r['native']['actual_native_steps']<=1232000 and n==int(math.ceil(len(t['time'])/5))
    checks['freshNN_counts_honest']=(counts['module_forward_calls']>0 if bool(warm.any() or candidate.any()) else counts['module_forward_calls']==0) and counts['prime_old']==int(warm.sum()) and counts['step_details']==int(candidate.sum()) and counts['old_step_direct']==int(warm.sum()+candidate.sum())
    if (folder/'HANDOVER.json').exists():
        hand=read(folder/'HANDOVER.json');checks['handover_noreset']=hand['no_state_reset'] and hand['new96_zero'] and hand['actual_state_exact'] and hand['neural_state_exact'] and hand['physical_before_sha256']==hand['physical_after_sha256'] and hand['hidden_before_sha256']==hand['hidden_after_sha256']
    # Reuse only the four ORIGINAL independent saved-metric functions. This is
    # a separate implementation from collector common.score, with no new gates.
    audit_path=OLD/'independent_motor_body_component/audit.py'
    if sources.get(str(audit_path))!=sha(audit_path) or sha(audit_path)!='92cd050ae850952cdcfa38d5242d457fe031532d2177760f0a565c41e8febc77':raise ValueError('Frozen original independent audit source required')
    selected_names=('independent_pivot_counts','legacy_walking_counts','independent_metrics','compare_value')
    selected=[node for node in ast.parse(audit_path.read_text()).body if isinstance(node,ast.FunctionDef) and node.name in selected_names]
    if tuple(node.name for node in selected)!=selected_names:raise ValueError('Original independent audit functions missing or reordered')
    tree=ast.Module(body=selected,type_ignores=[]);metric_ns={'__name__':'_collection_original_independent_saved_metrics','BASE':BASE,'SOURCE_CANDIDATE':7 if assisted else 8,'math':math,'importlib':importlib,'sys':sys}
    exec(compile(tree,str(audit_path)+'::single_layer_independent_metrics','exec'),metric_ns)
    independent=None
    if r['candidate_start'] is not None:
        independent,legacy,legacy_scores=metric_ns['independent_metrics'](t,g,r['candidate_start'],r['body_length']['length_native'],r['weight_native'],condition,r['native'])
        checks['independent_engineering_metrics_and_cycles']=metric_ns['compare_value'](independent,r['candidate_metrics'])
        original=r['original_five']
        actual_candidate_samples=int(np.sum(t['control_source']==(7 if assisted else 8)))
        if actual_candidate_samples>=2:checks['independent_legacy_five']=original['gates']==legacy and original['all_gates']==bool(all(legacy.values())) and metric_ns['compare_value'](legacy_scores,original)
        else:checks['independent_legacy_shortwindow']=original==dict(all_gates=False,sample_count=actual_candidate_samples,reason='less than two samples')
    else:checks['no_candidate_engineering_claim']=r['candidate_metrics'] is None and r['early_stop'] is not None
    report.update(independent_engineering_metrics=independent,selected_independent_metric_AST_sha256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),engineering_interpretation='Saved physical engineering metrics only; assisted is teacher75/student25 and old control uses frozen-old direct action. Neither is autonomous007d success.')
    ident=read(folder/'IDENTITY.json');canonical={k:v for k,v in ident.items() if k!='identity_sha256'}
    checks['identity_digest']=hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest()==ident['identity_sha256']==r['identity_sha256']
    checks['identity_files_unchanged']=all(sha(path)==item['sha256'] for path,item in ident['files'].items())
    checks['finite_PD_torque']=bool(np.all(np.abs(t['actualctrl'])<=np.asarray(ident['actuator']['torque_limit'])[None]+1e-9))
    checks['all_weights_unchanged']=weight_digest()==weights_before;m.verify_old()
    checks['no_physical_backend']=not any(name in sys.modules for name in FORBIDDEN)
    verify(artifacts);verify(sources);verify(r['source_hashes'])
    report.update(replay_max_abs_error=errors,neural_records=n,assisted=assisted,candidate_rows=int(candidate.sum()),full_window_recorded=len(t['time'])==15400,artifact_hashes=artifacts,descriptive_saved_data=dict(label_qualified_count=int(d['label_qualified'][candidate].sum()),label_mask_pass_counts=d['label_mask'][candidate].sum(0).tolist(),actual_native_steps=r['native']['actual_native_steps']),behavior_pass=None,neural_acceptance=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--case','--condition',dest='case',choices=tuple(CASES),required=True);p.add_argument('--execute-parent-saved-audit',action='store_true');args=p.parse_args()
    if not args.execute_parent_saved_audit:p.error('Parent fresh saved-NN audit flag required')
    out=ROOT/'trials'/args.case/'NN_VERIFY.json'
    if out.exists():raise FileExistsError(out)
    report=dict(kind=FORMAT+'_FRESH_NN_COLLECTION_AUDIT',case=args.case,checks={},freshNN_calls=dict(module_forward_calls=0,prime_old=0,step_details=0,old_step_direct=0),Core_constructions=0,physical_steps=0,training_updates=0,scope='TEACHER_ASSISTED_DATA_COLLECTION_NOT_NEURAL_BEHAVIOR_ACCEPTANCE_OR_BRAIN_NECK_DEPLOYMENT',integrity_all_pass=False,no_automatic_retry=True)
    error=None
    try:
        _run(args.case,report);report['integrity_all_pass']=bool(report['checks']) and bool(all(report['checks'].values()))
        if not report['integrity_all_pass']:raise ValueError('Fresh collection replay audit failed; evidence retained')
    except BaseException as exc:error=exc;report.update(error_type=type(exc).__name__,error=repr(exc),integrity_all_pass=False)
    save(out,report)
    if error is not None:raise error
    print(json.dumps(dict(case=args.case,integrity_all_pass=True,checks=len(report['checks']),freshNN_calls=report['freshNN_calls'])),flush=True)


if __name__=='__main__':main()
