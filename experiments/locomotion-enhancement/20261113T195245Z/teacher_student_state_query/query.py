"""Parent offline sequentialteacherqueries at archived actualstudentstates.
LABEL_QUERY_ONLY_NOT_PHYSICAL_SUCCESS. No model/NN/optimizer or integration.
"""
from pathlib import Path
from contextlib import ExitStack
import argparse
import hashlib
import random
import time
from metadata import ROOT,read,save,verify,sha,qualify
from common import cpu1,stack,load_arrays,resources,make_teacher,query_one,array_record,summarized,legacy_diagnostics,TeacherLocalQueryError


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute-parent-saved-teacher-query',action='store_true');a=p.parse_args()
    if not a.execute_parent_saved_teacher_query:p.error('ParentstageB offlinequeryrelease required')
    start=time.monotonic();cpu1();meta=qualify();pre=read(ROOT/'PREFLIGHT.json')
    if not pre['all_passed'] or pre['timing_status']!='READY_QUERY_300S' or pre['source_hashes']!=meta['source_hashes']:raise ValueError('Actualsame-source0integration preflight/timing required')
    if (ROOT/'STAT.json').exists():raise FileExistsError('No overwrite/repeatedqueries')
    verify(meta['source_hashes'])
    import numpy as np
    import restore
    physics,identity,mj=stack();rng_python=random.getstate();rng_np=np.random.get_state();core_count=0;qualification=[];students=[];walks=[];all_stage_ok=True
    out=ROOT/'queries';out.mkdir(exist_ok=False)
    sources={**meta['source_hashes'],str(ROOT/'PREFLIGHT.json'):sha(ROOT/'PREFLIGHT.json')}

    def case_query(entry,qualifying,counters):
        nonlocal core_count
        folder=Path(entry['folder']);trace=load_arrays(entry['trace']);samples=load_arrays(entry['samples']);initial=read(folder/'INITIAL_STATE.json');ident=read(folder/'IDENTITY.json');hand=read(folder/'HANDOVER.json');rows=[];errors=[];restored_force_diffs={};reproduction=dict(action_max_abs_error=0.,private_qdes_max_abs_error=0.,desired_target_max_abs_error=0.,IK_error_max_abs_error=0.,contact_count_max_abs_difference=0)
        with ExitStack() as guards:
            c=physics.Core(adhesion=False);core_count+=1;core_guard=guards.enter_context(restore.block_core_ticks(c));restore.trap_neural_calls_if_loaded(counters)
            identity.verify_runtime(ident,c)
            first=restore.restore(c,trace,initial,5800,mujoco_module=mj);handover=restore.handover_check(c,hand)
            if not first['exact_state'] or not handover['all_exact']:raise ValueError('Ownhandover exactmechanicalrestore required')
            teacher=make_teacher(c,entry['condition']);reference_diagnostics=legacy_diagnostics(folder/'teacher_diagnostics.jsonl') if qualifying else None
            for j in range(entry['query_rows']):
                if j%32==0:resources(start)
                index=entry['query_start']+j
                ms=int(samples['physical_ms'][index] if qualifying else samples['physical_step_ms'][index])
                if ms!=5800+j*5:raise ValueError('Sourcequeryeventclock not contiguous5ms')
                elapsed=float(samples['elapsed_teacher_s'][index]) if qualifying else j*.005
                actual_action=samples['Y42'][index] if qualifying else samples['actually_applied_action'][index]
                try:
                    record=query_one(c,teacher,trace,initial,ms,elapsed,restore,mj)
                    output=array_record(record,actual_action);output['query_error']=False
                    for name,comparison in record['restore']['recomputed_measurement_vs_saved'].items():
                        value=comparison.get('max_abs_error')
                        if value is not None:restored_force_diffs[name]=max(restored_force_diffs.get(name,0.),float(value))
                    if qualifying:
                        reference=next(reference_diagnostics)
                        if reference['physical_ms']!=ms:raise ValueError('Archivedteacherdiagnostic clockmismatch')
                        diag=reference['diagnostic'];got=record['diag']
                        for target,key in (('action_max_abs_error',None),('private_qdes_max_abs_error','q_des_act'),('desired_target_max_abs_error','desired_motor_target'),('IK_error_max_abs_error','ik_error_norm_per_leg')):
                            diff=float(np.max(np.abs(record['action']-actual_action))) if key is None else float(np.max(np.abs(np.asarray(got[key])-np.asarray(diag[key]))))
                            reproduction[target]=max(reproduction[target],diff)
                        reproduction['contact_count_max_abs_difference']=max(reproduction['contact_count_max_abs_difference'],int(np.max(np.abs(np.asarray(got['actual_contact_counts'])-np.asarray(diag['actual_contact_counts'])))))
                except TeacherLocalQueryError as exc:
                    # Only explicit numerical/load-solver infeasibility is local.
                    # Restore, source, forbidden-call and state-write errors are fatal.
                    # Algorithm queries can be locally infeasible. Keep an explicit
                    # missinglabel row; no zeros pretending to be teacher output.
                    errors.append(dict(row=j,physical_ms=ms,error=repr(exc),teacher_history_not_reset=True))
                    nan42=np.full(42,np.nan);nan6=np.full(6,np.nan)
                    output=dict(time=float(trace['time'][ms-1]),physical_ms=ms,teacher_elapsed_s=elapsed,actual_target=trace['target'][ms-1].copy(),actual_ctrl=trace['actualctrl'][ms-1].copy(),query_action=nan42.copy(),actual_action=actual_action.copy(),action_difference=nan42.copy(),desired_motor_target=nan42.copy(),q_des_act=nan42.copy(),IK_error=nan6,qdot_raw=nan42.copy(),rate=nan42.copy(),tau_ff=nan42.copy(),raw_PD_torque_at_desired_target=nan42.copy(),raw_PD_torque_at_next_target=nan42.copy(),raw_PD_desired_exceed_mask=np.ones(42,dtype=bool),raw_PD_next_exceed_mask=np.ones(42,dtype=bool),qdot_clip_count=-1,rate_clip_count=-1,target_bound_mask=np.ones(42,dtype=bool),tau_clip_mask=np.ones(42,dtype=bool),planned_load_over_W=nan6.copy(),actual_contacts=np.full(6,-1),unsupported=True,unbacked_planned_load_over_W=np.nan,actual_COM=trace['com'][ms-1].copy(),actual_NF_1ms=float(trace['nonfoot'][ms-1]),actual_yaw=np.nan,label_qualified=False,label_mask=np.zeros(11,dtype=bool),query_error=True)
                    if qualifying:reproduction['action_max_abs_error']=float('inf')
                output['source_target_error_normalized']=samples['X238'][index,42:84].copy()
                output['source_target_offset_normalized']=samples['X238'][index,126:168].copy()
                rows.append(output)
            if core_guard['core_tick_or_reset_attempts']:raise ValueError('ForbiddenCoretick/reset attempted')
            identity.verify_runtime(ident,c)
        artifact=out/(entry['id']+'.npz');np.savez_compressed(artifact,**{name:np.asarray([r[name] for r in rows]) for name in rows[0]})
        report=summarized(rows);report.update(id=entry['id'],condition=entry['condition'],cohort=entry.get('cohort'),teacher_qualification=qualifying,handover=handover,first_restore=first,recomputed_measurement_vs_saved_max_abs=restored_force_diffs,errors=errors,source_result_sha256=sha(folder/'RESULT.json'),artifact=dict(path=str(artifact),sha256=sha(artifact)),status='LABEL_QUERY_ONLY_NOT_PHYSICAL_SUCCESS',qualifiedmask_notphysicalrecovery=True)
        if qualifying:
            # Serialize explicit failure as None, never NaN/Infinity fabricated data.
            repro_finite=all(np.isfinite(v) for v in reproduction.values())
            passed=repro_finite and not errors and reproduction['action_max_abs_error']<=1e-6 and reproduction['private_qdes_max_abs_error']<=1e-6 and reproduction['desired_target_max_abs_error']<=1e-6
            report['teacher_reproduction']={k:(float(v) if np.isfinite(v) else None) for k,v in reproduction.items()};report['teacher_reproduction_pass']=bool(passed)
        save(out/(entry['id']+'.json'),report)
        resources(start);return report

    with restore.install_step_traps(mj) as counters:
        for entry in meta['teachers']:qualification.append(case_query(entry,True,counters))
        all_stage_ok=all(q['teacher_reproduction_pass'] for q in qualification)
        if all_stage_ok:
            for entry in meta['students']:students.append(case_query(entry,False,counters))
        # Walking is a saved oldmean algebraic NN contrast, never teacher query.
        for entry in meta['walks']:
            d=load_arrays(entry['samples']);oldaction=np.tanh(d['oldmean'][1160:]);actual=d['policy_action'][1160:];delta=oldaction-actual
            walks.append(dict(id=entry['id'],cohort=entry['cohort'],teacher_query=False,source_rows=len(actual),oldmean_definition='saved mean-residual conditionaloldcomponent',action_difference_RMSE=float(np.sqrt(np.mean(delta**2))),action_difference_absmax=float(np.max(np.abs(delta))),teacher_labels_available=False,new_physical_counterfactual=False))
        noNN=counters['NN_forward_calls']==0 and counters['torch_load_calls']==0
        zero=counters['integration_calls']==0
    verify(sources);rss=resources(start);np_after=np.random.get_state();rng_exact=rng_python==random.getstate() and rng_np[0]==np_after[0] and np.array_equal(rng_np[1],np_after[1]) and rng_np[2:]==np_after[2:]
    checks=dict(three_teacher_reproduction_qualified=all_stage_ok,sixstudentcases_queried=len(students)==6,Core_budget=core_count<=9,zero_physical_integration=zero,zero_NNforward_or_weightload=noNN,RNG_unchanged=rng_exact,all_inputs_unchanged=True)
    stat=dict(kind='OFFLINE_STUDENT_STATE_TEACHER_QUERY_STAGE_B_V1',status='LABEL_QUERY_ONLY_NOT_PHYSICAL_SUCCESS',checks=checks,qualification=qualification,students=students,walk_saved_oldmean_contrasts=walks,Core_constructions=core_count,physical_integration_steps=counters['integration_calls'],NN_forwards=counters['NN_forward_calls'],optimizer_updates=0,step_and_neural_traps=counters,source_hashes=sources,wall_s=time.monotonic()-start,peak_RSS_MiB=rss,interpretation='Allteacherlabels are localofflinequeries on restoredsavedstates, includinginvalidrows. NoDaggertraining, no newbodyrollout/recovery, no brainneckdeployment.')
    save(ROOT/'STAT.json',stat)
    with (ROOT/'REPORT.zh-CN.md').open('x') as stream:stream.write(report_text(stat))
    resources(start);verify(sources)
    save(ROOT/'VERIFY.json',dict(all_passed=bool(all(checks.values())),checks=checks,STAT_sha256=sha(ROOT/'STAT.json'),report_sha256=sha(ROOT/'REPORT.zh-CN.md'),source_hashes=sources,Core_constructions=core_count,physical_integration_steps=counters['integration_calls'],NN_forwards=counters['NN_forward_calls'],optimizer_updates=0,status='LABEL_QUERY_ONLY_NOT_PHYSICAL_SUCCESS'))
    resources(start)
    if not all(checks.values()):raise ValueError('Teacherqueryqualification incomplete/failed; keepoutputs,noautomaticretry')
    print('OFFLINE_STUDENT_STATE_LABEL_QUERY_COMPLETED_0INTEGRATION',flush=True)


def report_text(stat):
    lines=['# 阶段B：真实学生保存状态上的教师离线查询','','**LABEL_QUERY_ONLY_NOT_PHYSICAL_SUCCESS**。这是保存状态重建和教师局部标签可行性分析，不是物理复跑、恢复证明、神经控制成功或新训练。','','每次查询前状态来自trace[physical_ms−1]，40ms目标缓冲来自target[ms−40:ms]；5.8s用第5799行，不借下一物理步。离线复制qpos/qvel/ctrl/target/buffer不属于实际运动控制；mj_forward只重新求解当前约束，不是原生接触力逐步重放。','',f"Core={stat['Core_constructions']}；原生积分={stat['physical_integration_steps']}；NNforward={stat['NN_forwards']}；优化更新=0。",'','## 教师自身轨迹资格','']
    for row in stat['qualification']:lines.extend([row['id']+'：reproduction_pass='+str(row['teacher_reproduction_pass']),'```json',json.dumps(row['teacher_reproduction'],ensure_ascii=False,indent=2),'```'])
    lines.extend(['','## 学生状态查询','|cohort／条件|实际查询行|合格标签行|合格比例|全部有限动作差RMSE|合格动作差RMSE|','|---|---:|---:|---:|---:|---:|'])
    for row in stat['students']:lines.append(f"|{row['id']}|{row['rows']}|{row['label_qualified_count']}|{row['label_qualified_fraction']}|{row['all_finite_action_difference_RMSE']}|{row['qualified_action_difference_RMSE']}|")
    lines.extend(['','## 实际body寄存器与actor访问差异','已读BASE/sources/proprioception.py:38-43：42:84为(target−q)/halfspan目标误差，126:168为(target−q0)/halfspan目标位置偏移，均感知裁剪[-10,10]后float32；不是effort。physics.obs238将这些当前body寄存器原样编入观测。','新残差／fast encoder和old256 recurrent编码器的physicalmask排除42:84、126:168及195:237(previousaction)。但整个旧motor并非完全不读target：BASE/models.py:140-142在base输出均值显式减去gain()*obs126:168，故最终oldmean/newmean已有冻结的身体本地target-position prior；只是可训练新分支和recurrent memory没有该当前寄存器的直接通道。不能据此断不可观测／不可能，原walking曾通过。直接encoder的autograd梯度为0仅说明该被测分支的连接／冻结detach，不能证明整个mean对相应body寄存器数值不敏感，尤其不能忽略已存在的冻结target-position prior。','每行保存实际target、query desiredtarget及源X238的两个归一化target字段；每case统计它们与教师纠正动作的描述关系。教师本身的rate律明确依赖当前target，所以相关性不构成唯一根因。后续若考虑body-local efference／target寄存器显式观测，那是另立未授权候选，真实body寄存器不等于教师privateq_des，不得将理想教师目标塞actor。','','标签mask在跑前固定：实际COM≥1.2／源1ms NF0、queryIK≤.01、无unsupported、unbacked计划载荷≤.01W、qdot/rate/target/tau不裁剪、有限42、恢复exact。全部invalid及算法错误行保留，不用mask掩盖标签失败；1ms源NF不冒称该时刻native峰为0。','每个case的私有q_des、clock、世界anchors从同一handover初始化后连续推进，不逐tick冷reset。重解的足载/NF/速度/actuator_force与归档差异在每case JSON中明确保留。老师自身轨迹复现不合格时不会继续给学生标签。','walk两cohort只比较保存oldmean的tanh与原NN动作，没有生成walkingteacher纠正标签。','所有标签仍是算法教师的条件性局部建议；合格mask不证明真实身体施加后能恢复。没有Dagger训练、新权重、真实root轨迹、脑neck部署。今后actor只准真实physical112/old256/task2/new96，不把teacherclock/anchor/rootstate当隐藏输入。',f"wall={stat['wall_s']:.3f}s，峰RSS={stat['peak_RSS_MiB']:.3f}MiB；源/RNG/编译参数与禁止step/NN调用证据见VERIFY。"])
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        if not (ROOT/'FAILURE.json').exists():
            save(ROOT/'FAILURE.json',dict(status='OFFLINE_LABEL_QUERY_FAILED_NOT_PHYSICAL_SUCCESS',error_type=type(exc).__name__,error=repr(exc),partial_outputs_preserved=True,no_automatic_retry=True,counter_status='Inspect preflight/partial artifacts; no fabricated integration or NN totals'))
        raise
