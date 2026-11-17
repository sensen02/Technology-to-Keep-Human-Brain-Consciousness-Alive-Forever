"""Fresh NN-free teacher replay at saved pre-action states; no integration.
Numerical imports are lazy and belong exclusively to the parent runtime.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from common import ROOT,FORMAT,CASES,read,save,sha,verify,frozen,physical_stack,make_teacher,restore_api

MASK_NAMES=('actual_COM','actual_NF_1ms','IK','unbacked_plannedload','teacher_supported','qdot_clip','rate_clip','target_bounds','tau_ff_clip','finite42','actual_restore')
MECHANICAL=('qpos','qvel','ctrl','xfrc_applied','qfrc_applied','target','buffer')
CHECKPOINT_SHA='007d0c74118715eacd03d49c7e6847d8349e55db2793dd9d461a4b9c311e0fcd'


def _cpu1_stdlib():
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(key)!='1':raise ValueError(key+'=1 required')
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})


def independent_mask(action,diag,actual_com,actual_nf,source_exact):
    """Frozen11 definitions reimplemented here; never call common.teacher_mask."""
    import numpy as np
    unbacked=sum(float(diag['per_leg_fz_over_W'][i]) for i in diag['unbacked_support_indices'])
    mask=dict(actual_COM=float(actual_com[2])>=1.2,actual_NF_1ms=float(actual_nf)==0.,IK=float(max(diag['ik_error_norm_per_leg']))<=.01,unbacked_plannedload=unbacked<=.01,teacher_supported=not diag['unsupported'],qdot_clip=diag['qdot_des_clip_count']==0,rate_clip=diag['rate_clip_count']==0,target_bounds=len(diag['motor_target_bound_indices'])==0,tau_ff_clip=len(diag['tau_ff_native_clip_indices'])==0,finite42=action.shape==(42,) and bool(np.isfinite(action).all()),actual_restore=bool(source_exact))
    if tuple(mask)!=MASK_NAMES:raise ValueError('Independent11 mask schema changed')
    return mask,unbacked


def compare_diagnostic(actual,expected,tolerance=1e-6,path='diagnostic',errors=None):
    """Compare every saved diagnostic field; discrete contacts/indices exact."""
    if errors is None:errors=[]
    if hasattr(actual,'tolist'):actual=actual.tolist()
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual)!=set(expected):errors.append(path+':keys');return errors
        for key in expected:compare_diagnostic(actual[key],expected[key],tolerance,path+'.'+key,errors)
    elif isinstance(expected,list):
        if not isinstance(actual,(list,tuple)) or len(actual)!=len(expected):errors.append(path+':shape');return errors
        for i,(a,b) in enumerate(zip(actual,expected)):compare_diagnostic(a,b,tolerance,path+'['+str(i)+']',errors)
    elif expected is None or isinstance(expected,(str,bool)):
        if type(actual) is not type(expected) or actual!=expected:errors.append(path+':discrete')
    elif isinstance(expected,(int,float)):
        if isinstance(actual,bool) or not isinstance(actual,(int,float)) or not math.isfinite(float(actual)) or not math.isfinite(float(expected)):
            errors.append(path+':nonfinite_or_type')
        elif 'actual_contact_counts' in path or '_indices' in path or path.endswith('_count'):
            if actual!=expected:errors.append(path+':integer_or_contact')
        elif abs(float(actual)-float(expected))>tolerance:errors.append(path+':numeric')
    elif actual!=expected:errors.append(path+':value')
    return errors


def _run(case,report):
    _cpu1_stdlib();sources=frozen();verify(sources);folder=ROOT/'trials'/case;r=read(folder/'RESULT.json');verify(r['source_hashes'])
    condition,command=CASES[case]
    if case=='old_walk_control':raise ValueError('Teacher replay has exactly three assisted cases; control has no teacher labels')
    if r['kind']!=FORMAT or r.get('case',case)!=case or r['condition'] not in (case,condition) or r['checkpoint_sha256']!=CHECKPOINT_SHA:raise ValueError('Actual collection identity required')
    report.update(source_hashes=sources,result_sha256=sha(folder/'RESULT.json'),checkpoint_sha256=r['checkpoint_sha256'])
    checks=report['checks'];artifact_hashes={}
    for key,filename in (('trace','trace_1ms.npz'),('neural_decisions','neural_decisions.npz'),('initial_state','INITIAL_STATE.json'),('teacher_diagnostics','teacher_diagnostics.jsonl')):
        path=folder/filename;item=r[key];checks[key+'_SHA']=item['path']==str(path) and sha(path)==item['sha256'];artifact_hashes[str(path)]=sha(path)
    if not all(checks.values()):raise ValueError('Saved artifacts must pass SHA before replay')
    # Standalone original restore is admitted by frozen source hashes. Never
    # import the old query common/metadata namespace or invoke query.main.
    restore=restore_api()
    if 'query' in sys.modules:raise ValueError('Teacher audit must not enter old query main')
    import numpy as np
    def arrays(path):
        with np.load(path,allow_pickle=False) as z:return {key:z[key].copy() for key in z.files}
    t=arrays(folder/'trace_1ms.npz');d=arrays(folder/'neural_decisions.npz');initial=read(folder/'INITIAL_STATE.json')
    candidate=np.flatnonzero(d['physical_step_ms']>=5800)
    with (folder/'teacher_diagnostics.jsonl').open() as stream:rows=[json.loads(line) for line in stream if line.strip()]
    checks['exact_candidate_row_count']=len(rows)==len(candidate) and len(rows)<=1920
    checks['mask_names']=tuple(r['label_mask_names'])==MASK_NAMES
    checks['all_saved_neural_fields_finite']=all(bool(np.isfinite(v).all()) for v in d.values())
    if not all(checks.values()):raise ValueError('Teacher replay saved schema mismatch')
    if not len(candidate):
        checks['early_stop_without_candidate_honest']=r['early_stop'] is not None and r['candidate_start'] is None and not len(rows) and not bool(np.any(d['label_mask'])) and not bool(np.any(d['label_qualified']))
        verify(sources);verify(r['source_hashes']);verify(artifact_hashes)
        report.update(artifact_hashes=artifact_hashes,physical_integration_steps=0,NN_forward_calls=0,torch_load_calls=0,teacher_labels_available=False,data_qualified_for_training=False,behavior_pass=None,neural_acceptance=False,status='NO_CANDIDATE_LABELS_EARLY_STOP_NOT_DATA_QUALIFIED',zero_call_evidence='No physical stack, Core, teacher, neural loader or replay entered in empty-label branch')
        return
    hand=read(folder/'HANDOVER.json');ident=read(folder/'IDENTITY.json');item=r['handover']
    checks['handover_SHA']=item['path']==str(folder/'HANDOVER.json') and item['sha256']==sha(folder/'HANDOVER.json');artifact_hashes[str(folder/'HANDOVER.json')]=sha(folder/'HANDOVER.json')
    if not checks['handover_SHA']:raise ValueError('Actual handover artifact hash required')
    sw,physics,identity,static,Peak,ast_ids=physical_stack()
    import mujoco as mj
    # Construct only the admitted physical object. Install traps only after the
    # physical stack and Core have had their indirect import opportunity.
    c=physics.Core(adhesion=False);report['Core_constructions']+=1
    identity.verify_runtime(ident,c)
    reproduction=dict(action_max_abs_error=0.,q_des_act_max_abs_error=0.,desired_motor_target_max_abs_error=0.,IK_error_max_abs_error=0.,contact_count_max_abs_difference=0)
    restore_differences={};failed_diagnostics=[];mask_failures=[];state_hash_failures=[];count=0
    report.update(reproduction=reproduction,recomputed_measurement_vs_saved_max_abs=restore_differences,diagnostic_mismatch_rows=failed_diagnostics,mask_mismatch_physical_ms=mask_failures,state_hash_mismatch_physical_ms=state_hash_failures)
    def hashes():return {key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in restore.snapshot(c).items()}
    with restore.install_step_traps(mj,core=c) as traps:
        report['step_and_neural_traps']=traps
        restore.trap_neural_calls_if_loaded(traps)
        first=restore.restore(c,t,initial,5800,mujoco_module=mj);handover=restore.handover_check(c,hand)
        checks['own_handover_exact']=first['exact_state'] and handover['all_exact']
        report['first_restore']=first;report['handover_comparison']=handover
        factory_before=hashes();teacher=make_teacher(c,condition);report['teacher_constructions']+=1;factory_after=hashes()
        checks['factory_no_actualstate_write']=factory_before==factory_after
        checks['saved_factory_no_actualstate_write']=hand['teacher_factory_before_sha256']==hand['teacher_factory_after_sha256'] and hand['teacher_factory_created'] is True
        checks['factory_saved7hash_reproduction']=set(hand['teacher_factory_before_sha256'])==set(MECHANICAL) and factory_before==hand['teacher_factory_before_sha256'] and factory_after==hand['teacher_factory_after_sha256']
        if not all(checks.values()):raise ValueError('Teacher must initialize once at exact own handover without state writes')
        for j,(k,reference) in enumerate(zip(candidate,rows)):
            ms=int(d['physical_step_ms'][k]);elapsed=(ms-5800)*.001
            if ms!=5800+j*5 or reference['physical_ms']!=ms or reference['candidate_phase'] is not True or abs(float(reference['teacher_elapsed_s'])-elapsed)>1e-12 or abs(float(reference['time_s'])-float(t['time'][ms-1]))>1e-12 or abs(float(d['time'][k])-float(t['time'][ms-1]))>1e-12:raise ValueError('Contiguous saved preaction teacher clock required')
            restored=restore.restore(c,t,initial,ms,mujoco_module=mj)
            if not restored['exact_state'] or restored['saved_trace_index']!=ms-1:raise ValueError('Exact saved preaction restore required')
            for name,value in restored['recomputed_measurement_vs_saved'].items():
                if value.get('max_abs_error') is not None:restore_differences[name]=max(restore_differences.get(name,0.),float(value['max_abs_error']))
            before=hashes();time_before=float(c.d.time)
            report['teacher_action_attempts']+=1
            action,diag=teacher.action(c,elapsed);report['teacher_action_calls']+=1
            after=hashes()
            if before!=after or float(c.d.time)!=time_before:raise ValueError('Teacher altered restored actual fields/time')
            if not np.isfinite(action).all() or action.shape!=(42,):raise ValueError('Finite42 teacher action required')
            if before!=reference['pre_query_hashes'] or after!=reference['post_query_hashes'] or set(before)!=set(MECHANICAL):state_hash_failures.append(ms)
            expected_action=d['teacher_action'][k]
            action_error=float(np.max(np.abs(action-expected_action)));reproduction['action_max_abs_error']=max(reproduction['action_max_abs_error'],action_error)
            if float(np.max(np.abs(np.asarray(reference['teacher_action'])-expected_action)))>1e-12:raise ValueError('JSONL and NPZ teacher action disagree')
            for target,key in (('q_des_act_max_abs_error','q_des_act'),('desired_motor_target_max_abs_error','desired_motor_target'),('IK_error_max_abs_error','ik_error_norm_per_leg')):
                error=float(np.max(np.abs(np.asarray(diag[key])-np.asarray(reference['diagnostic'][key]))));reproduction[target]=max(reproduction[target],error)
            contacts=int(np.max(np.abs(np.asarray(diag['actual_contact_counts'])-np.asarray(reference['diagnostic']['actual_contact_counts']))));reproduction['contact_count_max_abs_difference']=max(reproduction['contact_count_max_abs_difference'],contacts)
            differences=compare_diagnostic(diag,reference['diagnostic'])
            if differences:failed_diagnostics.append(dict(physical_ms=ms,fields=differences))
            # Source COM/NF come from the actual saved preaction endpoint, never
            # the newly re-solved contact forces of this offline reconstruction.
            mask,unbacked=independent_mask(action,diag,t['com'][ms-1],t['nonfoot'][ms-1],restored['exact_state'])
            qualified=bool(all(mask.values()));mask_array=np.asarray(list(mask.values()),dtype=bool)
            if mask!=reference['label_mask'] or qualified is not reference['label_qualified'] or abs(unbacked-float(reference['unbacked_planned_load_over_W']))>1e-12 or not np.array_equal(mask_array,d['label_mask'][k]) or qualified!=bool(d['label_qualified'][k]):mask_failures.append(ms)
            if not bool(d['teacher_available'][k]) or not np.array_equal(d['supervision_target'][k],d['teacher_action'][k]):raise ValueError('Unmixed available teacher supervision required')
            count+=1;report['replayed_rows']=count
            if restored['traps']['integration_calls']!=0 or restored['traps']['NN_forward_calls']!=0 or restored['traps']['torch_load_calls']!=0:raise ValueError('Nested restore trap observed forbidden operation')
        restore.trap_neural_calls_if_loaded(traps)
        checks['zero_integration_or_tick_attempts']=traps['integration_calls']==0 and traps['integration_attempts']==0 and traps['core_tick_or_reset_attempts']==0
        checks['zero_neural_forward_or_checkpoint_load']=traps['neural_tracking_complete'] and traps['NN_forward_calls']==0 and traps['torch_load_calls']==0 and traps['observed_NN_forward_calls']==0 and traps['observed_torch_load_calls']==0
    checks['trap_patches_restored']=traps['patches_restored']
    checks['action_qdes_target_IK_le1e6']=all(reproduction[key]<=1e-6 for key in ('action_max_abs_error','q_des_act_max_abs_error','desired_motor_target_max_abs_error','IK_error_max_abs_error'))
    checks['actual_contacts_exact']=reproduction['contact_count_max_abs_difference']==0
    checks['full_diagnostic_replay_le1e6']=not failed_diagnostics
    checks['independent11_masks_exact']=not mask_failures
    checks['source_prepost7hash_exact']=not state_hash_failures
    checks['one_persistent_teacher']=report['teacher_constructions']==1 and report['teacher_action_calls']==count==len(candidate)
    checks['source_time_no_integration']=count==len(rows)
    identity.verify_runtime(ident,c);verify(sources);verify(r['source_hashes']);verify(artifact_hashes)
    if sha(folder/'RESULT.json')!=report['result_sha256']:raise ValueError('Result metadata drift during replay')
    report.update(reproduction=reproduction,recomputed_measurement_vs_saved_max_abs=restore_differences,diagnostic_mismatch_rows=failed_diagnostics,mask_mismatch_physical_ms=mask_failures,state_hash_mismatch_physical_ms=state_hash_failures,artifact_hashes=artifact_hashes,selected_legacy_ast=ast_ids,physical_integration_steps=traps['integration_calls'],NN_forward_calls=traps['NN_forward_calls'],torch_load_calls=traps['torch_load_calls'],behavior_pass=None,neural_acceptance=False,teacher_labels_are_conditional=True,solver_caution='Saved preaction mechanics restored; mj_forward re-solves unsaved contacts. Reproduction must independently satisfy declared tolerances; no claim of native contact-force replay or autonomous recovery.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--case','--condition',dest='case',choices=tuple(case for case in CASES if case!='old_walk_control'),required=True);p.add_argument('--execute-parent-teacher-audit',action='store_true');args=p.parse_args()
    if not args.execute_parent_teacher_audit:p.error('Parent saved teacher replay flag required')
    out=ROOT/'trials'/args.case/'TEACHER_VERIFY.json'
    if out.exists():raise FileExistsError(out)
    report=dict(kind=FORMAT+'_SAVED_TEACHER_REPLAY_AUDIT',case=args.case,checks={},Core_constructions=0,teacher_constructions=0,teacher_action_attempts=0,teacher_action_calls=0,replayed_rows=0,physical_integration_steps=None,NN_forward_calls=None,torch_load_calls=None,integrity_all_pass=False,training_updates=0,scope='TEACHER_ASSISTED_DATA_NOT_NEURAL_ACCEPTANCE; BodyANNcomponent_NOTnewbrainneckdeployment',no_automatic_retry=True)
    error=None;started=time.monotonic()
    try:
        _run(args.case,report);report['integrity_all_pass']=bool(report['checks']) and bool(all(report['checks'].values()))
        if not report['integrity_all_pass']:raise ValueError('Saved teacher replay failed; retain actual evidence without rerun')
    except BaseException as exc:
        error=exc;report.update(error_type=type(exc).__name__,error=repr(exc),integrity_all_pass=False)
        traps=report.get('step_and_neural_traps')
        if traps is not None:report.update(physical_integration_steps=traps['integration_calls'],NN_forward_calls=traps['NN_forward_calls'],torch_load_calls=traps['torch_load_calls'])
    report['wall_s']=time.monotonic()-started;save(out,report)
    if error is not None:raise error
    print(json.dumps(dict(case=args.case,integrity_all_pass=True,replayed_rows=report['replayed_rows'],physical_integration_steps=report['physical_integration_steps'],NN_forward_calls=report['NN_forward_calls'])),flush=True)


if __name__=='__main__':main()
