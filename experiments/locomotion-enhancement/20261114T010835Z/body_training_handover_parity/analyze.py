"""Parent saved-array handover comparison only: no NN, Core or physical backend."""
import argparse
import hashlib
import json
import math
import os
import resource
import signal
import sys
import time
from pathlib import Path
from metadata import ROOT,CAL,BODY,read,save,sha,verify,qualify

FORBIDDEN=('torch','mujoco','physics','flygym','student_wave','static_teacher')
PHYSICAL_ARRAYS=('body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat','body_gravcomp','jnt_type','jnt_pos','jnt_axis','jnt_limited','jnt_range','jnt_qposadr','jnt_dofadr','jnt_stiffness','dof_damping','dof_frictionloss','dof_armature','geom_type','geom_size','geom_pos','geom_quat','geom_friction','geom_solref','geom_solimp','geom_contype','geom_conaffinity','geom_margin','geom_gap','actuator_trntype','actuator_trnid','actuator_gainprm','actuator_biasprm','actuator_forcerange','actuator_ctrlrange')


def resource_gate(start):
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
    if rss>1500 or time.monotonic()-start>90:raise RuntimeError('Saved handover analysis resource cap exceeded; no retry/NN/physics')
    return rss


def comparison(a,b):
    import numpy as np
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape:return dict(status='SHAPE_MISMATCH',a_shape=list(a.shape),b_shape=list(b.shape),exact=False,max_abs_error=None)
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite saved input comparison')
    delta=np.abs(a.astype(np.float64)-b.astype(np.float64));different=a!=b
    first=np.argwhere(different)[0].tolist() if bool(different.any()) else None
    return dict(status='COMPARED',shape=list(a.shape),exact=bool(np.array_equal(a,b)),within_1e12=bool(np.max(delta,initial=0.)<=1e-12),max_abs_error=float(np.max(delta,initial=0.)),RMS_error=float(np.sqrt(np.mean(delta*delta))) if delta.size else 0.,different_elements=int(different.sum()),first_different_index=first)


def take_arrays(path,names):
    import numpy as np
    with np.load(path,allow_pickle=False) as z:return {name:z[name].copy() for name in names if name in z.files}


def identity_comparison(source,body):
    a=source['compiled_arrays'];b=body['compiled_arrays'];shared=[key for key in PHYSICAL_ARRAYS if key in a and key in b]
    fields={key:dict(equal=a[key]==b[key],source=a[key],body=b[key]) for key in shared}
    all_shared=set(a)&set(b);other_diffs=[key for key in sorted(all_shared) if a[key]!=b[key] and key not in PHYSICAL_ARRAYS]
    source_files=source.get('files',{});body_files=body.get('files',{});shared_files={path:dict(source_sha256=source_files[path]['sha256'],body_sha256=body_files[path]['sha256'],equal=source_files[path]['sha256']==body_files[path]['sha256']) for path in sorted(set(source_files)&set(body_files)) if Path(path).name in ('physics.py','repair_core.py','incremental_actuator.py','proprioception.py') or path.endswith('.xml')}
    return dict(shared_physical_source_and_XML_hashes=shared_files,physical_array_SHA_comparisons=fields,physical_array_mismatches=[key for key in shared if a[key]!=b[key]],not_recorded_in_both=[key for key in PHYSICAL_ARRAYS if key not in a or key not in b],source_actuator=source.get('actuator'),body_actuator=body.get('actuator'),actuator_metadata_status='COMPARED' if 'actuator' in source and 'actuator' in body else 'NOT_RECORDED_FOR_AT_LEAST_ONE_SIDE',actuator_metadata_equal=source['actuator']==body['actuator'] if 'actuator' in source and 'actuator' in body else None,source_physics=source.get('physics'),body_physics=body.get('physics'),physics_metadata_status='COMPARED' if 'physics' in source and 'physics' in body else 'NOT_RECORDED_FOR_AT_LEAST_ONE_SIDE',physics_metadata_equal=source['physics']==body['physics'] if 'physics' in source and 'physics' in body else None,source_adhesion=source.get('adhesion'),body_adhesion=body.get('adhesion'),other_compiled_mismatches_descriptive=other_diffs,group_render_note='Other fields such as geom_group/rgba or scene names are metadata differences; they are not automatically classified as physical causes.',no_Core_or_XML_compilation=True)


def source_task(entry,samples,index):
    import numpy as np
    if entry.get('collection'):return samples['command'][index].copy()
    if entry['teacher']:return np.asarray([0.,float(entry['turn'])])
    return np.asarray([1.,float(samples['raw_turn'][index])])


def case_report(case,evaluation,conditional,mapping,start):
    import numpy as np
    entry=case['entry'];condition=case['condition'];i=case['index_in_training'];s=case['sourcepaths'];b=case['bodypaths'];offset=int(entry['start']);warm_start=int(entry['warm_start'])
    sample_names=('X238','time','physical_ms','physical_step_ms','command','raw_turn','hidden_before','hidden_after','mean','policy_action','actually_applied_action','Y42','supervision_target','candidate_phase')
    src=take_arrays(s['samples'],sample_names);body=take_arrays(b['samples'],sample_names+('residual','control_source'))
    if len(body['time'])<1161 or body['hidden_before'].shape[1]!=352:raise ValueError('Actual first352 NN handover missing')
    if abs(float(src['time'][offset])-5.8)>1e-8 or abs(float(body['time'][1160])-5.8)>1e-8 or int(body['physical_step_ms'][1160])!=5800:raise ValueError('Actual sample preaction clocks must align at5800ms')
    trace_names=('time','qpos','qvel','target','actualctrl','action','control_source','com','external_body_force_max','external_generalized_force_max')
    source_trace=take_arrays(s['trace'],trace_names);body_trace=take_arrays(b['trace'],trace_names)
    if min(len(source_trace['time']),len(body_trace['time']))<5805:raise ValueError('First5ms actual target window missing')
    for trace in (source_trace,body_trace):
        if not np.allclose(trace['time'][:5805],np.arange(1,5806)*.001,atol=1e-8,rtol=0):raise ValueError('Trace endpoint clock mismatch; use ms-1')
    warm={name:comparison(body_trace[name][:5800],source_trace[name][:5800]) for name in ('qpos','qvel','target','actualctrl','action') if name in source_trace and name in body_trace}
    for value in warm.values():value['first_different_endpoint_ms']=None if value['first_different_index'] is None else value['first_different_index'][0]+1
    source_initial=read(s['initial_state']);body_initial=read(b['initial_state'])
    initial_mechanics={name:comparison(body_initial[name],source_initial[name]) if name in body_initial and name in source_initial else dict(status='NOT_RECORDED') for name in ('qpos','qvel','ctrl','xfrc_applied','qfrc_applied','target','buffer')}
    pre={name:comparison(body_trace[name][5799],source_trace[name][5799]) for name in ('qpos','qvel','target','actualctrl')}
    X=body['X238'][1160];sx=src['X238'][offset];task=source_task(entry,src,offset);bodytask=body['command'][1160]
    x_columns=np.abs(X-sx);x_report=comparison(X,sx);x_report.update(per_column_abs_error=x_columns.tolist(),different_column_indices=np.flatnonzero(X!=sx).tolist(),target_error_columns_42_84=comparison(X[42:84],sx[42:84]),target_offset_columns_126_168=comparison(X[126:168],sx[126:168]))
    warm_x=comparison(body['X238'][200:1160],src['X238'][warm_start:offset])
    new96zero=bool(np.all(body['hidden_before'][1160,256:]==0.));whole_newwarmzero=bool(np.all(body['hidden_before'][:1161,256:]==0.) and np.all(body['hidden_after'][:1160,256:]==0.))
    hidden=dict(status='NOT_RECORDED',reason='Teacher source has no saved old256 state; no NN reconstruction is performed.')
    if 'hidden_before' in src:
        hidden=dict(status='RECORDED_OLD256',first_before=comparison(body['hidden_before'][1160,:256],src['hidden_before'][offset,:256]),warm_before=comparison(body['hidden_before'][200:1160,:256],src['hidden_before'][warm_start:offset,:256]),warm_after=comparison(body['hidden_after'][200:1160,:256],src['hidden_after'][warm_start:offset,:256]))
    tag=case['conditional_tag'];cm=conditional[tag+'_mean'];ca=conditional[tag+'_action'];cp=conditional[tag+'_pretarget'];cq=conditional[tag+'_posttarget'];ch=conditional[tag+'_new96']
    first_outputs=dict(body_vs_EVALUATION_conditional_mean=comparison(body['mean'][1160],evaluation['conditional_mean'][0,i]),body_vs_EVALUATION_conditional_action=comparison(body['policy_action'][1160],evaluation['conditional_action'][0,i]),body_vs_CONDITIONAL600_mean=comparison(body['mean'][1160],cm[0]),body_vs_CONDITIONAL600_action=comparison(body['policy_action'][1160],ca[0]),body_first_new96after_vs_CONDITIONAL600=comparison(body['hidden_after'][1160,256:],ch[0]),EVALUATION_vs_CONDITIONAL600_mean=comparison(evaluation['conditional_mean'][0,i],cm[0]),EVALUATION_vs_CONDITIONAL600_action=comparison(evaluation['conditional_action'][0,i],ca[0]),body_policy_vs_applied=comparison(body['policy_action'][1160],body['actually_applied_action'][1160]))
    targets=dict(body_pre5799_vs_training_conditional_pre=comparison(body_trace['target'][5799],cp[0]),body_post5804_vs_training_conditional_post=comparison(body_trace['target'][5804],cq[0]),body_post5804_vs_EVALUATION_conditional_post=comparison(body_trace['target'][5804],evaluation['conditional_post_targets'][0,i]),source_actual_post5804_vs_new_body_post=comparison(source_trace['target'][5804],body_trace['target'][5804]),body_firstfive_applied_hold=comparison(body_trace['action'][5800:5805],np.repeat(body['actually_applied_action'][1160:1161],5,axis=0)),source_actual_post_note='Teacher/source realized first action is its archived controller label, not the new model policy; divergence after5800 may be expected.')
    qa=np.asarray(mapping['qpos_indices'],dtype=int);half=np.asarray(mapping['halfspan42'],dtype=np.float64);q0=np.asarray(mapping['q0_42'],dtype=np.float64)
    if qa.shape!=(42,) or half.shape!=(42,) or q0.shape!=(42,):raise ValueError('Actual source-bound42 qpos mapping/normalizers required')
    normalization={}
    for label,trace,observed in (('source',source_trace,sx),('body',body_trace,X)):
        actualq=trace['qpos'][5799,qa];target=trace['target'][5799]
        err=np.clip((target-actualq)/half,-10.,10.).astype(np.float32).astype(np.float64);off=np.clip((target-q0)/half,-10.,10.).astype(np.float32).astype(np.float64)
        normalization[label]=dict(target_error_float32_encoding=comparison(err,observed[42:84]),target_offset_float32_encoding=comparison(off,observed[126:168]))
    source_label=src['Y42'][offset] if entry['teacher'] and not entry.get('collection') else src['supervision_target'][offset] if entry.get('collection') else src['actually_applied_action'][offset]
    identity=identity_comparison(read(s['identity']),read(b['identity']))
    interface_inputs_match=warm_x['exact'] and x_report['exact'] and comparison(bodytask,task)['exact'] and all(row['exact'] for row in pre.values()) and new96zero and (hidden['status']=='NOT_RECORDED' or hidden['first_before']['exact'])
    output_match=first_outputs['body_vs_CONDITIONAL600_mean']['within_1e12'] and first_outputs['body_vs_CONDITIONAL600_action']['within_1e12']
    expectation_status='NOT_CONTROLLED_FOR_COMMAND' if not comparison(bodytask,task)['exact'] else 'EXPECTED_EXACT_GIVEN_MATCHED_SAVED_WARM_HISTORY_AND_CALIBRATION_PRIME_PROOFS' if interface_inputs_match else 'NOT_CONTROLLED_FOR_SAVED_INPUT_OR_STATE'
    interpretation='Initial saved interface consistent at declared tolerance; does not establish later physics/closed-loop generalization or behavior PASS.' if interface_inputs_match and output_match else 'Saved initial input/command/state or output mismatch recorded; inspect first differences before attributing policy capacity or changing training.'
    resource_gate(start)
    return dict(condition=condition,source_id=entry['id'],training_index=i,source_kind='TEACHER_ALGORITHM_LABELS' if entry['teacher'] else 'ARCHIVED_NEURAL_LABELS',source_start=offset,source_warm_start=warm_start,source_warm_rows=offset-warm_start,body_first_index=1160,preaction_ms=5800,pre_trace_index=5799,post_trace_index=5804,first_five_applied_trace_indices=[5800,5801,5802,5803,5804],source_task=task.tolist(),body_task=bodytask.tolist(),task_comparison=comparison(bodytask,task),initial_saved_mechanics=initial_mechanics,warm_physical_prefix_0to5800ms=warm,warm_X238=warm_x,handover_actual_mechanics=pre,handover_X238=x_report,source_hidden256=hidden,new96_beforefirstNN_actualzero=new96zero,new96_entirewarmzero=whole_newwarmzero,first_outputs=first_outputs,first_targets=targets,actual_source_normalizer_mapping=normalization,qpos_indices_from_bound_source=qa.tolist(),training_label_vs_newNNaction_descriptive=comparison(source_label,body['policy_action'][1160]),training_label_note='Archived teacher action/oldNN target is supervision, not an expected newNN action. Only saved model conditional outputs are parity references.',physical_identity=identity,initial_saved_inputs_match=bool(interface_inputs_match),first_newNN_output_matches_training_saved_reference_le1e12=bool(output_match),first_action_expectation_status=expectation_status,first_output_parity_as_interface_test=bool(output_match) if interface_inputs_match else None,command_different=not comparison(bodytask,task)['exact'],X_different=not x_report['exact'],interpretation=interpretation,body_behavior_pass=read(b['result'])['candidate_metrics']['task_pass_candidate'],no_behavior_gate_changes=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute-parent-saved-handover-analysis',action='store_true');args=parser.parse_args()
    if not args.execute_parent_saved_handover_analysis:parser.error('Explicit parent saved-array analysis flag required')
    if any(name in sys.modules for name in FORBIDDEN):raise ValueError('Fresh noNN/noCore process required')
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ValueError(name+'=1 required')
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});started=time.monotonic()
    def deadline(*unused):raise RuntimeError('Saved-array analysis90second deadline exceeded')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(90)
    if (ROOT/'RESULT.json').exists():raise FileExistsError('No overwrite/no repeated analysis')
    try:
        meta=qualify();resource_gate(started)
        evaluation=take_arrays(meta['evaluation_arrays']['path'],('conditional_mean','conditional_action','conditional_post_targets'))
        tags=[case['conditional_tag'] for case in meta['cases']];conditional=take_arrays(meta['conditional_arrays']['path'],tuple(tag+'_'+field for tag in tags for field in ('mean','action','pretarget','posttarget','new96')))
        rows=[case_report(case,evaluation,conditional,meta['normalizer_mapping'],started) for case in meta['cases']]
        for key in ('source_hashes','data_hashes','proof_hashes'):verify(meta[key])
        if any(name in sys.modules for name in FORBIDDEN):raise ValueError('Forbidden neural/physical module loaded')
        rss=resource_gate(started);result=dict(kind='BODY_TRAINING_HANDOVER_SAVED_PARITY_V1',rows=rows,checkpoint_sha256=meta['checkpoint_sha256'],source_hashes=meta['source_hashes'],data_hashes=meta['data_hashes'],proof_hashes=meta['proof_hashes'],wall_s=time.monotonic()-started,peak_RSS_MiB=rss,NN_forwards=0,checkpoint_loads=0,Core_constructions=0,physical_steps=0,training_updates=0,scope='Saved initial handover and training source parity only; BodyANNcomponent_NOTnewbrainneckdeployment',no_behavior_or_policy_capacity_inference=True)
        save(ROOT/'RESULT.json',result)
        lines=['# 身体交接与训练源保存数据对齐诊断','','只读取实际保存数组，0NN／0Core／0物理。全部95九阶段已完成后才允许运行；本报告不改任何行为门、不证明泛化或policy容量。','','|case|首输入/命令/保存状态一致|首输出实测差≤1e−12|是否受控可期待精确|源old256记录|原行为PASS|','|---|---|---|---|---|---|']
        for row in rows:lines.append('|'+row['condition']+'|'+str(row['initial_saved_inputs_match'])+'|'+str(row['first_newNN_output_matches_training_saved_reference_le1e12'])+'|'+row['first_action_expectation_status']+'|'+row['source_hidden256']['status']+'|'+str(row['body_behavior_pass'])+'|')
        lines.extend(['','时间：首preaction5800ms，trace5799是前一毫秒末端；第一个动作实际积分trace5800..5804，首posttarget取5804。教师source样本start960／warm0，完整NNsource start1160／warm200。walk训练命令按真实raw_turn恢复，不假设正好0。','缺失教师old256为NOT_RECORDED，本分析未运行神经补算。源教师首动作是训练标签，不是新模型预测；只以EVALUATION和CONDITIONAL600的新模型保存输出作首NN参考。','若首接口一致，只支持初始对齐，不支持后续真实闭环或行为成功；若不同，应先核warm/input/command/state差异，不能据此判policy容量。','具体首差列／预热qpos/qvel/target全前缀／physicalcompiledSHA／normalizer字段核对见RESULT。'])
        with (ROOT/'REPORT.zh-CN.md').open('x') as stream:stream.write('\n'.join(lines)+'\n')
        checks=dict(all4_completed_cases=len(rows)==4,all_inputs_unchanged=True,zero_NN_Core_physics=True,resource_caps=rss<=1500 and time.monotonic()-started<=90)
        save(ROOT/'VERIFY.json',dict(integrity_all_pass=all(checks.values()),checks=checks,result_sha256=sha(ROOT/'RESULT.json'),report_sha256=sha(ROOT/'REPORT.zh-CN.md'),wall_s=time.monotonic()-started,peak_RSS_MiB=rss,NN_forwards=0,Core_constructions=0,physical_steps=0,parity_match_is_not_integrity_gate=True))
        print(json.dumps(dict(status='SAVED_HANDOVER_ANALYSIS_COMPLETED',result=str(ROOT/'RESULT.json'),NN_forwards=0,physical_steps=0)),flush=True)
    finally:signal.alarm(0)


if __name__=='__main__':main()
