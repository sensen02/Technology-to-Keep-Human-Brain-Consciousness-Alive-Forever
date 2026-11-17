"""Saved-array audit only. NumPy is imported ONLY inside parent-called audit_all.
No file writes, Core/MuJoCo/teacher/NN imports, model execution, or training.
The caller owns the explicit execution flag, resource guard, and output files.
"""
from pathlib import Path
import hashlib
import json
import math

MASK_NAMES=('actual_COM','actual_NF_1ms','IK','unbacked_plannedload','teacher_supported','qdot_clip','rate_clip','target_bounds','tau_ff_clip','finite42','actual_restore')
SNAPSHOT_KEYS=('qpos','qvel','ctrl','xfrc_applied','qfrc_applied','target','buffer')


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def _tick_guard(guard):
    if guard is not None:
        guard()


def _close(a,b,tol=1e-10):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a,bool) or isinstance(b,bool):
        return type(a) is type(b) and a==b
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b))<=tol


def _intervals(valid,physical_ms,elapsed):
    """Pure interval bookkeeping; last time is last queried decision, not rollout."""
    spans=[];start=None
    for i,accepted in enumerate(valid):
        accepted=bool(accepted)
        contiguous=i==0 or int(physical_ms[i])-int(physical_ms[i-1])==5
        if start is not None and (not accepted or not contiguous):
            end=i-1
            spans.append(dict(first_row=int(start),last_row_inclusive=int(end),rows=int(end-start+1),first_physical_ms=int(physical_ms[start]),last_physical_ms=int(physical_ms[end]),first_teacher_elapsed_s=float(elapsed[start]),last_teacher_elapsed_s=float(elapsed[end]),nominal_decision_coverage_s=float((end-start+1)*.005)))
            start=None
        if accepted and start is None:
            start=i
    if start is not None:
        end=len(valid)-1
        spans.append(dict(first_row=int(start),last_row_inclusive=int(end),rows=int(end-start+1),first_physical_ms=int(physical_ms[start]),last_physical_ms=int(physical_ms[end]),first_teacher_elapsed_s=float(elapsed[start]),last_teacher_elapsed_s=float(elapsed[end]),nominal_decision_coverage_s=float((end-start+1)*.005)))
    prefix=0
    for i,accepted in enumerate(valid):
        if not bool(accepted) or (i and int(physical_ms[i])-int(physical_ms[i-1])!=5):
            break
        prefix+=1
    longest=max((span['rows'] for span in spans),default=0)
    return spans,int(prefix),int(longest)


def audit_all(stat,meta,guard=None):
    """Return (JSONable audit result, bool checks) from the nine saved query files.

    The parent must call this only after its execution flag. This function writes
    nothing and does not import the original query or restore modules. The eleven
    recorded mask columns are audited with an explicit limit: current arrays do
    not store per-row restore_exact/qpos/qvel/buffer hashes. The last mask column
    is accepted as recorded assertion evidence, NOT independently re-run here.
    """
    import numpy as np

    checks={};results=[];walks=[]
    entries={entry['id']:entry for entry in meta['teachers']+meta['students']}
    recorded=stat.get('qualification',[])+stat.get('students',[])
    checks['nine_query_case_ids']=len(recorded)==9 and len(entries)==9 and {r['id'] for r in recorded}==set(entries)
    checks['recorded_status_labelquery_only']=stat.get('status')=='LABEL_QUERY_ONLY_NOT_PHYSICAL_SUCCESS'
    checks['recorded_original_integrity_checks_true']=bool(stat.get('checks')) and all(value is True for value in stat['checks'].values())

    def arrays(path):
        with np.load(path,allow_pickle=False) as archive:
            return {name:archive[name].copy() for name in archive.files}

    def describe(value):
        value=np.asarray(value,dtype=float)
        finite=np.isfinite(value)
        out=dict(elements=int(value.size),finite_elements=int(finite.sum()),nonfinite_elements=int(value.size-finite.sum()))
        if finite.any():
            v=value[finite]
            out.update(minimum=float(v.min()),maximum=float(v.max()),RMS=float(np.sqrt(np.mean(v*v))),absmax=float(np.abs(v).max()))
        else:
            out.update(minimum=None,maximum=None,RMS=None,absmax=None)
        return out

    def pearson(a,b):
        if a.size<2 or not np.isfinite(a).all() or not np.isfinite(b).all() or float(a.std())==0. or float(b.std())==0.:
            return None
        value=float(np.corrcoef(a.ravel(),b.ravel())[0,1])
        return value if math.isfinite(value) else None

    def maxdiff(a,b):
        a=np.asarray(a);b=np.asarray(b)
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            return None
        return float(np.max(np.abs(a-b))) if a.size else 0.

    def handover_audit(entry,source_trace,initial,record):
        boundary_ms=5800;i=boundary_ms-1;expected={}
        for out_name,source_name in (('qpos','qpos'),('qvel','qvel'),('ctrl','actualctrl'),('target','target')):
            expected[out_name]=np.ascontiguousarray(source_trace[source_name][i],dtype=np.float64)
        expected['buffer']=np.ascontiguousarray(source_trace['target'][boundary_ms-40:boundary_ms],dtype=np.float64)
        force_proof={}
        for name,maximum in (('xfrc_applied','external_body_force_max'),('qfrc_applied','external_generalized_force_max')):
            if name in source_trace:
                expected[name]=np.ascontiguousarray(source_trace[name][i],dtype=np.float64)
                force_proof[name]='saved_full_trace_array'
            elif maximum in source_trace and float(source_trace[maximum][i])==0. and name in initial:
                expected[name]=np.zeros_like(np.asarray(initial[name],dtype=np.float64))
                force_proof[name]='all_elements_zero_proved_by_saved_max0; shape_from_explicit_initial_array'
            else:
                force_proof[name]='UNKNOWN_NO_FULL_ARRAY_OR_ZERO_MAX_PROOF'
        hashes={name:hashlib.sha256(value.tobytes(order='C')).hexdigest() for name,value in expected.items()}
        own=_read(Path(entry['folder'])/'HANDOVER.json');containers={}
        for container in ('physical_before_sha256','physical_after_sha256','before','before_sha256','after_sha256'):
            candidate=own.get(container)
            if isinstance(candidate,dict):
                keys=[name for name in SNAPSHOT_KEYS if name in candidate]
                if keys:
                    containers[container]=dict(covered_keys=keys,equal_per_key={name:name in hashes and hashes[name]==candidate[name] for name in keys})
        coverage=any(set(value['covered_keys'])==set(SNAPSHOT_KEYS) for value in containers.values())
        source_exact=coverage and all(all(value['equal_per_key'].values()) for value in containers.values())
        original_query=record.get('handover',{});query_hashes=original_query.get('current_hashes',{})
        query_exact=set(hashes)==set(SNAPSHOT_KEYS) and all(query_hashes.get(name)==hashes[name] for name in SNAPSHOT_KEYS)
        first_hashes=record.get('first_restore',{}).get('snapshot_hashes',{})
        first_exact=set(hashes)==set(SNAPSHOT_KEYS) and all(first_hashes.get(name)==hashes[name] for name in SNAPSHOT_KEYS)
        return dict(independently_rebuilt_source_hashes=hashes,force_proof=force_proof,buffer_source_rows=[5760,5800],source_containers=containers,all_seven_source_hashes_match=bool(source_exact),all_seven_query_handover_hashes_match=bool(query_exact),all_seven_first_restore_hashes_match=bool(first_exact),source_handover_time_s=float(own['time']) if 'time' in own else None,source_trace_time_s=float(source_trace['time'][i]),scope='Independently rebuilt boundary5800 mechanical hashes only, not per-row Core restoration replay')

    for record in recorded:
        _tick_guard(guard)
        case_id=record['id'];entry=entries[case_id];folder=Path(entry['folder']);artifact=Path(record['artifact']['path'])
        checks[case_id+'_query_artifact_SHA']=_sha(artifact)==record['artifact']['sha256']
        q=arrays(artifact);t=arrays(entry['trace']);s=arrays(entry['samples']);initial=_read(folder/'INITIAL_STATE.json')
        _tick_guard(guard)
        n=len(q['physical_ms']);start=int(entry['query_start']);selection=slice(start,start+n)
        checks[case_id+'_query_rows']=n==int(entry['query_rows'])==int(record['rows'])
        checks[case_id+'_core_query_shapes']=q['query_action'].shape==(n,42) and q['label_mask'].shape==(n,11) and q['actual_COM'].shape==(n,3) and q['IK_error'].shape==(n,6)
        if not checks[case_id+'_core_query_shapes']:
            raise ValueError('Query array schema is not auditable: '+case_id)
        physical_ms=q['physical_ms'].astype(np.int64)
        source_ms=s['physical_ms'][selection] if entry['teacher_qualification'] else s['physical_step_ms'][selection]
        checks[case_id+'_source_preaction_indices']=bool(np.array_equal(physical_ms,source_ms) and np.array_equal(physical_ms,5800+np.arange(n)*5) and np.all(physical_ms>=1) and np.all(physical_ms<=len(t['time'])))
        if not checks[case_id+'_source_preaction_indices']:
            raise ValueError('Invalid source pre-action indices: '+case_id)
        index=physical_ms-1;error=q['query_error'].astype(bool);ok=~error
        checks[case_id+'_preaction_time_exact']=bool(np.array_equal(q['time'],t['time'][index]))
        expected_elapsed=s['elapsed_teacher_s'][selection] if entry['teacher_qualification'] else np.arange(n)*.005
        checks[case_id+'_teacher_clock_exact']=bool(np.array_equal(q['teacher_elapsed_s'],expected_elapsed))
        checks[case_id+'_actual_source_values_exact']=bool(np.array_equal(q['actual_COM'],t['com'][index]) and np.array_equal(q['actual_NF_1ms'],t['nonfoot'][index]) and np.array_equal(q['actual_target'],t['target'][index]) and np.array_equal(q['actual_ctrl'],t['actualctrl'][index]))
        expected_action=s['Y42'][selection] if entry['teacher_qualification'] else s['actually_applied_action'][selection]
        checks[case_id+'_actual_source_action_exact']=bool(np.array_equal(q['actual_action'],expected_action))
        checks[case_id+'_action_difference_exact']=bool(np.array_equal(q['action_difference'],q['query_action']-expected_action,equal_nan=True))
        checks[case_id+'_source_normalized_target_fields_exact']=bool(np.array_equal(q['source_target_error_normalized'],s['X238'][selection,42:84]) and np.array_equal(q['source_target_offset_normalized'],s['X238'][selection,126:168]))
        quaternion=np.asarray(t['qpos'][index,3:7],dtype=float);quaternion=quaternion/np.linalg.norm(quaternion,axis=1,keepdims=True);w,x,y,z=quaternion.T
        source_yaw=np.arctan2(2*(w*z+x*y),1.-2*(y*y+z*z))
        yaw_error=float(np.max(np.abs(np.angle(np.exp(1j*(q['actual_yaw'][ok]-source_yaw[ok])))))) if ok.any() else None
        checks[case_id+'_actual_yaw_from_source_quaternion']=yaw_error is None or yaw_error<=1e-8
        finite_action=np.isfinite(q['query_action']).all(1)
        finite_diff=np.isfinite(q['action_difference']).all(1)
        unbacked=np.sum(np.where(q['actual_contacts']==0,q['planned_load_over_W'],0.),axis=1)
        checks[case_id+'_unbacked_load_independent']=bool(np.allclose(unbacked[ok],q['unbacked_planned_load_over_W'][ok],atol=1e-12,rtol=0))
        # The first ten columns are recomputed, with the original missinglabel-row
        # convention: query_error means the entire mask is false, even if source
        # COM/NF was safe. No missing proposal is replaced by zero action.
        first10=np.column_stack((t['com'][index,2]>=1.2,t['nonfoot'][index]==0.,np.max(q['IK_error'],axis=1)<=.01,unbacked<=.01,~q['unsupported'].astype(bool),q['qdot_clip_count']==0,q['rate_clip_count']==0,~q['target_bound_mask'].any(1),~q['tau_clip_mask'].any(1),finite_action))
        first10[error]=False
        recorded_restore=q['label_mask'][:,10].astype(bool)
        expected_mask=np.column_stack((first10,recorded_restore));expected_mask[error]=False
        good=expected_mask.all(1)
        checks[case_id+'_first10_mask_conditions_independent']=bool(np.array_equal(first10,q['label_mask'][:,:10]))
        checks[case_id+'_restore_mask_recorded_assertion_consistency']=bool(np.array_equal(recorded_restore,ok) and record.get('first_restore',{}).get('exact_state') is True and record.get('handover',{}).get('all_exact') is True)
        checks[case_id+'_qualified_mask_conjunction']=bool(np.array_equal(good,q['label_qualified']))
        checks[case_id+'_recorded_mask_names']=list(record['label_mask_names'])==list(MASK_NAMES)
        pass_counts=expected_mask.sum(0).astype(int).tolist();reject_counts={name:int(n-count) for name,count in zip(MASK_NAMES,pass_counts)}
        checks[case_id+'_recorded_mask_counts']=pass_counts==record['label_mask_pass_counts']
        count=int(good.sum());fraction=float(good.mean()) if n else None
        all_rmse=float(np.sqrt(np.mean(q['action_difference'][finite_diff]**2))) if finite_diff.any() else None
        qualified_rmse=float(np.sqrt(np.mean(q['action_difference'][good]**2))) if good.any() else None
        absmax=float(np.max(np.abs(q['action_difference'][finite_diff]))) if finite_diff.any() else None
        checks[case_id+'_recorded_counts_and_action_metrics']=count==int(record['label_qualified_count']) and _close(fraction,record['label_qualified_fraction']) and _close(all_rmse,record['all_finite_action_difference_RMSE']) and _close(qualified_rmse,record['qualified_action_difference_RMSE']) and _close(absmax,record['all_action_difference_absmax'])
        spans,prefix,longest=_intervals(good,physical_ms,q['teacher_elapsed_s'])
        largest=max(reject_counts.values(),default=0);most=[name for name,value in reject_counts.items() if value==largest and value>0]
        target_delta=q['desired_motor_target']-q['actual_target']
        relation=dict(teacher_desiredtarget_minus_actual_integrator_RMS=float(np.sqrt(np.mean(target_delta[finite_diff]**2))) if finite_diff.any() else None,querycorrection_vs_teacher_targeterror_Pearson_all=pearson(q['action_difference'][finite_diff],target_delta[finite_diff]),querycorrection_vs_teacher_targeterror_Pearson_qualified=pearson(q['action_difference'][good],target_delta[good]))
        original_relation=record['teacher_actual_register_relation']
        checks[case_id+'_register_relationship_independent']=all(_close(value,original_relation.get(name)) for name,value in relation.items())
        handover=handover_audit(entry,t,initial,record)
        checks[case_id+'_boundary_seven_hashes_independent']=all(handover[k] for k in ('all_seven_source_hashes_match','all_seven_query_handover_hashes_match','all_seven_first_restore_hashes_match'))
        result=dict(id=case_id,cohort=entry.get('cohort'),condition=entry['condition'],teacher_qualification=bool(entry['teacher_qualification']),rows=n,qualified_count=count,qualified_fraction=fraction,longest_valid_prefix_rows=prefix,longest_valid_run_rows=longest,most_common_rejection=most[0] if most else None,most_common_rejection_ties=most,rejection_counts=reject_counts,rejection_counts_overlap=True,rejection_sum_is_not_unique_invalid_count=True,unique_invalid_rows=int((~good).sum()),query_error_rows=int(error.sum()),valid_intervals=spans,valid_interval_time_semantics='first/last queried5ms decision; nominalcoverage is not physical recovery',label_mask_pass_counts=pass_counts,label_mask_names=list(MASK_NAMES),all_finite_action_difference_RMSE=all_rmse,qualified_action_difference_RMSE=qualified_rmse,all_action_difference_absmax=absmax,actual_yaw_source_quaternion_max_wrapped_error=yaw_error,handover_independent_boundary_audit=handover,normalized_source_target_error=describe(q['source_target_error_normalized']),normalized_source_target_offset=describe(q['source_target_offset_normalized']),teacher_actual_register_relation=dict(**relation,scope='Descriptive correlation; teacher rate law explicitly depends on target error, not causal identification or proof of actor unobservability'),restore_column_evidence=dict(independently_recomputed_per_row=False,accepted_source='Recorded label_mask column10 plus original query invariant assertions, STAT.first_restore and sevenhash handover',verified_per_row_source_fields=['physical_ms','time','actual_COM','actual_NF_1ms','actual_target','actual_ctrl','actual_action','source_normalized_target_fields','actual_yaw_for_nonerror_rows'],missing_per_row_artifact_fields=['restore_exact','qpos','qvel','buffer','seven_snapshot_hashes'],limitation='No Core/restoredsolver execution here; cannot independently prove all per-row mechanical restoration from these artifacts'),native_and_NN_execution_evidence='Original STAT/steptraps only, not re-executed by savedarray recovery')

        if entry['teacher_qualification']:
            maxima=dict(action_max_abs_error=maxdiff(q['query_action'],s['Y42'][selection]),private_qdes_max_abs_error=0.,desired_target_max_abs_error=0.,IK_error_max_abs_error=0.,contact_count_max_abs_difference=0.)
            observed=0;clock_exact=True;missing_numeric=False
            with (folder/'teacher_diagnostics.jsonl').open() as stream:
                for line in stream:
                    row=json.loads(line)
                    if not row['candidate_phase']:
                        continue
                    j=observed;observed+=1
                    if j>=n:
                        clock_exact=False;continue
                    if j%256==0:_tick_guard(guard)
                    clock_exact=clock_exact and int(row['physical_ms'])==int(physical_ms[j])
                    diag=row['diagnostic']
                    for destination,saved,source_name in (('private_qdes_max_abs_error','q_des_act','q_des_act'),('desired_target_max_abs_error','desired_motor_target','desired_motor_target'),('IK_error_max_abs_error','IK_error','ik_error_norm_per_leg'),('contact_count_max_abs_difference','actual_contacts','actual_contact_counts')):
                        value=maxdiff(q[saved][j],diag[source_name])
                        if value is None:
                            missing_numeric=True
                        else:
                            maxima[destination]=max(maxima[destination],value)
            reproduced=not missing_numeric and not error.any() and observed==n and clock_exact and all(maxima[k] is not None and maxima[k]<=1e-6 for k in ('action_max_abs_error','private_qdes_max_abs_error','desired_target_max_abs_error'))
            checks[case_id+'_teacher_JSONL_clock_and_rows']=observed==n and bool(clock_exact)
            checks[case_id+'_teacher_reproduction_vectors_independent']=bool(reproduced)
            checks[case_id+'_teacher_reproduction_maxima_match_STAT']=all(_close(value,record['teacher_reproduction'].get(key)) for key,value in maxima.items()) and bool(reproduced)==record['teacher_reproduction_pass']
            result['teacher_reproduction_independent']=dict(maxima=maxima,source_candidate_JSONL_rows=observed,source_clock_exact=bool(clock_exact),passes_original_1e6_tolerance=bool(reproduced),missing_numeric=missing_numeric,scope='Every saved sequential proposal compared with archived actual teacher diagnostics, not a new teacher execution; contact/IK differences reported without assuming bitexact')
        results.append(result)
        del q,t,s
        _tick_guard(guard)

    original_walk={row['id']:row for row in stat.get('walk_saved_oldmean_contrasts',[])}
    for entry in meta.get('walks',[]):
        _tick_guard(guard)
        s=arrays(entry['samples']);start=int(entry['query_start']);delta=np.tanh(s['oldmean'][start:])-s['policy_action'][start:]
        row=dict(id=entry['id'],cohort=entry['cohort'],source_rows=int(len(delta)),action_difference_RMSE=float(np.sqrt(np.mean(delta**2))),action_difference_absmax=float(np.max(np.abs(delta))),teacher_query=False,teacher_labels_available=False,scope='Saved conditional oldmean algebraic contrast only, no new NN or physical counterfactual')
        recorded_walk=original_walk.get(entry['id'],{})
        checks[entry['id']+'_walk_saved_algebraic_contrast']=row['source_rows']==recorded_walk.get('source_rows') and _close(row['action_difference_RMSE'],recorded_walk.get('action_difference_RMSE')) and _close(row['action_difference_absmax'],recorded_walk.get('action_difference_absmax'))
        walks.append(row)
        del s
    _tick_guard(guard)
    result=dict(kind='INDEPENDENT_SAVED_TEACHER_QUERY_ARRAY_AUDIT_V1',status='LABEL_QUERY_ONLY_NOT_PHYSICAL_SUCCESS',cases=results,walk_saved_algebraic_contrasts=walks,all_checks_passed=bool(all(checks.values())),case_count=len(results),independent_scope='Source-time/index/action/target/ctrl/COM/NF/yaw, first10 mask columns, statistics, qualified intervals, teacher archive vector differences, and boundary sevenhashes',nonindependent_scope='Per-row actual_restore column and historical integration/NN/RNG/compiledsolver guarantees accepted as original recorded assertions, not re-executed',invalid_rows_retained=True,physical_success_inferred=False,Core_constructions=0,NN_forwards=0,teacher_executions=0,physical_integration_steps=0,optimizer_updates=0,files_written=0)
    return result,checks
