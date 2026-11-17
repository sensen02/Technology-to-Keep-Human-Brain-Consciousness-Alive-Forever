"""StageA parent-only savedarrays/NNfixedobservations; no Core/teacherquery/train."""
import argparse
from pathlib import Path
import importlib.util
import json
import os
import random
import resource
import sys
import time
from metadata import ROOT,BODY,CAL,SHA,CASES,read,save,sha,verify,qualify


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def guard(start):
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
    if rss>1800:raise RuntimeError('DiagnosticRSS>1800MiB')
    if time.monotonic()-start>300:raise RuntimeError('Diagnosticwall>300s, noautoextension')
    return rss

def windows(times):
    import numpy as np
    return [(str(lo)+'..'+str(hi),(times>=lo-1e-10)&(times<(hi-1e-10) if hi<9.6 else times<=hi+1e-8)) for lo,hi in ((0.,.05),(.05,.5),(.5,2.),(2.,9.6))]

def numeric_summary(v):
    import numpy as np
    v=np.asarray(v)
    return dict(RMS=float(np.sqrt(np.mean(v*v))),absmax=float(np.max(np.abs(v))),mean=float(np.mean(v))) if v.size else dict(RMS=None,absmax=None,mean=None)

def first_event(mask,times):
    import numpy as np
    index=np.flatnonzero(mask)
    return None if not len(index) else float(times[index[0]])

def intervention_summary(action,residual,hidden,baseline_action,baseline_hidden,times):
    import numpy as np
    out=dict(action=numeric_summary(action),residual=numeric_summary(residual),action_difference_from_baseline=numeric_summary(action-baseline_action),old256_difference_absmax=float(np.max(np.abs(hidden[:,:256]-baseline_hidden[:,:256]))),new96_difference=numeric_summary(hidden[:,256:]-baseline_hidden[:,256:]),windows={})
    for label,mask in windows(times):out['windows'][label]=dict(action_difference=numeric_summary(action[mask]-baseline_action[mask]),residual=numeric_summary(residual[mask]),new96=numeric_summary(hidden[mask,256:]))
    return out

def parent_run():
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(key)!='1':raise ValueError(key+'=1 required')
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    if any(k in sys.modules for k in ('mujoco','physics','flygym','student_wave')):raise ValueError('Fresh NNonly processrequired')
    if any((ROOT/name).exists() for name in ('STAT.json','VERIFY.json','REPORT.zh-CN.md')):raise FileExistsError('No overwrite/repeateddiagnostic')
    started=time.monotonic();meta=qualify();verify(meta['source_hashes']);checks={};partial=[]
    import torch
    import numpy as np
    import stats
    torch.set_num_threads(1)
    rng=dict(torch=torch.get_rng_state().clone(),numpy=np.random.get_state(),python=random.getstate())
    ci=module('_failure_strict_calibration_io',CAL/'common.py');m,payload=ci.load_checkpoint(meta['checkpoint'],SHA,final=True);m.requires_grad_(False)
    modelmod=sys.modules[m.__class__.__module__]
    weight_before=modelmod.state_digest(m.state_dict())
    # caldata's fromsources must bind frozen calibration paths, not diagnosticROOT.
    sentinel=object();prior=sys.modules.get('sources',sentinel)
    try:
        cs=module('_failure_actual_calibration_sources',CAL/'sources.py');sys.modules['sources']=cs
        cd=module('_failure_actual_calibration_data',CAL/'data.py')
    finally:
        if prior is sentinel:sys.modules.pop('sources',None)
        else:sys.modules['sources']=prior
    training=cd.prepare(m);scan=modelmod.make_scan(m);train_rows=[]
    guard(started)
    # Baseline replay of all4 must qualify before any dependent statistics/probes.
    baselines=[]
    def arrays(path):
        with np.load(path,allow_pickle=False) as archive:return {key:archive[key].copy() for key in archive.files}
    with torch.no_grad():
        for entry in meta['rows']:
            folder=entry['folder'];d=arrays(BODY/'trials'/entry['condition']/'neural_decisions.npz');state=m.initial_state(1);replay={key:[] for key in ('mean','rawmean','oldmean','residual','hidden_before','hidden_after','policy_action')};features=[];primed=None
            for k,x in enumerate(d['X238']):
                if k%128==0:guard(started)
                hb=torch.cat(state,-1)[0].numpy().copy();xx=torch.tensor(x[None],dtype=torch.float64);command=torch.tensor(d['command'][k:k+1],dtype=torch.float64)
                if k<200:mean=torch.zeros((1,42),dtype=torch.float64);res=torch.zeros_like(mean)
                elif k<1160:mean,state=m.prime_old(xx,command,state);res=torch.zeros_like(mean)
                else:
                    if primed is None:primed=tuple(v.clone() for v in state)
                    mean,state,res=m.step_details(xx,command,state)
                    features.append(torch.cat((xx[:,m.old.base.physical_mask.bool()],*state[:3],command),-1)[0])
                values=dict(mean=mean[0].numpy().copy(),rawmean=mean[0].numpy().copy(),oldmean=(mean-res)[0].numpy().copy(),residual=res[0].numpy().copy(),hidden_before=hb,hidden_after=torch.cat(state,-1)[0].numpy().copy(),policy_action=mean[0].tanh().numpy().copy())
                for key,value in values.items():replay[key].append(value)
            exact={key:bool(np.array_equal(np.asarray(values),d[key])) for key,values in replay.items()};checks[entry['condition']+'_baseline_full352_replay_exact']=all(exact.values())
            partial.append(dict(condition=entry['condition'],baseline_replay=exact))
            if not all(exact.values()):
                save(ROOT/'VERIFY.json',dict(all_passed=False,failed_stage='baseline_replay_stop_before_analysis',cases=partial,physical_steps=0,source_hashes=meta['source_hashes']));raise ValueError('Actualbaseline mismatch; stop dependentanalysis')
            baselines.append(dict(entry=entry,d=d,primed=primed,features=torch.stack(features)))
    # No training-fit/statistical analysis is performed before all4baseline gates.
    with torch.no_grad():
        train_res,train_h=scan(training['features']);train_action=(training['oldmean']+train_res).tanh()
        for i,raw in enumerate(training['raw']):
            train_rows.append(dict(id=raw['entry']['id'],action_error=numeric_summary((train_action[:,i]-training['y'][:,i]).numpy()),oldmean=numeric_summary(training['oldmean'][:,i].numpy()),residual=numeric_summary(train_res[:,i].numpy()),windows={label:numeric_summary((train_action[mask,i]-training['y'][mask,i]).numpy()) for label,mask in windows(np.arange(1920)*.005)}))
        zero_readout=(6.*torch.tanh(m.head(m.head.weight.new_zeros((1,96)))))[0].numpy().copy()
    guard(started)
    outputs=[]
    for baseline in baselines:
        entry=baseline['entry'];case=entry['condition'];folder=BODY/'trials'/case;d=baseline['d'];t=arrays(folder/'trace_1ms.npz');g=arrays(folder/'geometry_1ms.npz');r=read(folder/'RESULT.json');ident=read(folder/'IDENTITY.json');hand=read(folder/'HANDOVER.json');reference=Path(entry['reference'])
        rt=arrays(reference/'trace_1ms.npz');rd=arrays(entry['reference_sample']);rh=read(reference/'HANDOVER.json')
        raw=stats.raw_stats(t,d,g,r,ident);prefix=stats.compare_prefix(t,d,rt,rd,entry['reference_is_teacher'],hand,rh)
        domain,per_tick=stats.domain_stats(baseline['features'].numpy(),training['features'].numpy(),d['hidden_after'][1160:])
        elapsed=np.arange(1920)*.005;actual_action=d['policy_action'][1160:];actual_hidden=d['hidden_after'][1160:];oldmean=d['oldmean'][1160:]
        variants=dict(baseline=intervention_summary(actual_action,d['residual'][1160:],actual_hidden,actual_action,actual_hidden,elapsed),residual_zero=intervention_summary(np.tanh(oldmean),np.zeros_like(oldmean),actual_hidden,actual_action,actual_hidden,elapsed))
        with torch.no_grad():
            for mode in ('new96_frozen_zero','speed_flip'):
                state=tuple(v.clone() for v in baseline['primed']);actions=[];residuals=[];states=[]
                for k,(x,u) in enumerate(zip(d['X238'][1160:],d['command'][1160:])):
                    if k%128==0:guard(started)
                    if mode=='new96_frozen_zero':state=(*state[:3],torch.zeros_like(state[3]))
                    command=u.copy()
                    if mode=='speed_flip':command[0]=1.-command[0]
                    mean,state,res=m.step_details(torch.tensor(x[None],dtype=torch.float64),torch.tensor(command[None],dtype=torch.float64),state)
                    actions.append(mean[0].tanh().numpy().copy());residuals.append(res[0].numpy().copy());states.append(torch.cat(state,-1)[0].numpy().copy())
                variants[mode]=intervention_summary(np.asarray(actions),np.asarray(residuals),np.asarray(states),actual_action,actual_hidden,elapsed)
        # Matched-clock source action is descriptive, not teacher query at studentstate.
        index=next(i for i,rawsource in enumerate(training['raw']) if rawsource['entry']['id']==('walk' if case=='walk' else case))
        train_x=training['x'][:,index].numpy();train_features=training['features'][:,index].numpy();train_u=training['u'][:,index].numpy();target=training['y'][:,index].numpy()
        firstmatch=dict(X238_first_exact=bool(np.array_equal(d['X238'][1160],train_x[0])),command_first_exact=bool(np.array_equal(d['command'][1160],train_u[0])),old_after256_first_exact=bool(np.array_equal(d['hidden_after'][1160,:256],train_features[0,112:368])),new96_input_first_zero=bool(np.all(d['hidden_before'][1160,256:]==0)),actual_first_action_vs_source_target=numeric_summary(actual_action[0]-target[0]),actual_first_residual_vs_traincache=numeric_summary(d['residual'][1160]-train_res[0,index].numpy()),trained_zero_h96_readout_residual=numeric_summary(zero_readout),warm_last_to_first_autonomous_action=numeric_summary(d['actually_applied_action'][1160]-d['actually_applied_action'][1159]))
        source_gap={label:numeric_summary(actual_action[mask]-target[mask]) for label,mask in windows(elapsed)}
        action_delta=np.max(np.abs(np.diff(np.vstack([d['actually_applied_action'][1159],actual_action]),axis=0)),axis=1)
        onset=dict(actiondelta_above_p1=first_event(action_delta>.1,elapsed))
        for group in ('physical','oldhidden','task'):
            fraction=np.asarray(per_tick[group+'_exceed_fraction']);z=np.asarray(per_tick[group+'_max_abs_z']);onset[group+'_domain_threshold']=first_event((fraction>.05)|(z>5.),elapsed)
        onset['new96_saturation_above_p01']=first_event(np.asarray(per_tick['new96_saturation_fraction'])>.01,elapsed)
        outputs.append(dict(condition=case,actual_behavior=r['candidate_metrics'],actual_native=r['native'],baseline_replay_exact=True,raw_saved_statistics=raw,matched_warm_prefix=prefix,training_domain=domain,first_cohort_NN_domain_onsets_s=onset,initial_source_match=firstmatch,sameclock_source_action_gap_notcorrective_target=source_gap,fixed_observation_variants=variants,variant_scope='No newbodytrajectory/yawprediction; same savedX underconditionalNNintervention only'))
        del t,g,rt,rd
        guard(started)
    checks['allweights_before_after_exact']=modelmod.state_digest(m.state_dict())==weight_before
    checks['oldweights_frozen_verified']=m.verify_old()
    checks['no_optimizer_or_physicalbackend']=not any(k in sys.modules for k in ('mujoco','physics','flygym','student_wave'))
    checks['RNG_before_after_unchanged']=torch.equal(rng['torch'],torch.get_rng_state()) and rng['python']==random.getstate() and rng['numpy'][0]==np.random.get_state()[0] and np.array_equal(rng['numpy'][1],np.random.get_state()[1]) and rng['numpy'][2:]==np.random.get_state()[2:]
    verify(meta['source_hashes']);rss=guard(started)
    data=dict(kind='INDEPENDENT_MOTOR_FIXED_OBSERVATION_DIAGNOSTIC_A_V1',checkpoint_sha256=SHA,baseline_replay_all4_exact=all(checks[c+'_baseline_full352_replay_exact'] for c in CASES),training_source_metrics=train_rows,cases=outputs,checks=checks,source_hashes=meta['source_hashes'],wall_s=time.monotonic()-started,peak_RSS_MiB=rss,physical_steps=0,optimizer_updates=0,teacher_state_query_executed=False,scope='Savedarrays and fixedobsNN only. Domain/exposure/saturation/action gaps are descriptions, not uniquecauses or newphysicalcounterfactuals. Componentnotbrainneck.')
    save(ROOT/'STAT.json',data)
    with (ROOT/'REPORT.zh-CN.md').open('x') as stream:stream.write(report(data))
    save(ROOT/'VERIFY.json',dict(all_passed=bool(all(checks.values())),checks=checks,STAT_sha256=sha(ROOT/'STAT.json'),report_sha256=sha(ROOT/'REPORT.zh-CN.md'),source_hashes=meta['source_hashes'],physical_steps=0,optimizer_updates=0,teacher_state_query_executed=False))
    if not all(checks.values()):raise ValueError('Diagnosticintegritycheckfailed')
    print(json.dumps(dict(STAT=str(ROOT/'STAT.json'),report=str(ROOT/'REPORT.zh-CN.md'),all_passed=True,physical_steps=0,wall_s=data['wall_s'])),flush=True)


def report(data):
    lines=['# 独立motor首版失败：阶段A保存数据／固定观察神经诊断','',f"实际checkpoint SHA：{SHA}。四条baseline352状态／mean／residual／action完整保存重放均精确一致；0物理积分、0优化器更新、0教师状态查询。","","所有干预固定原观察X，仅解释神经输出/状态敏感度，不预测新身体yaw或成功。所有8条来源都是训练数据；同一时刻teacher动作不是学生实际状态的纠正标签。完整性通过不改变四项身体任务实际FAIL。","","|条件|实际yaw°|漂移/体长|支撑均值|COM最低|native NF峰|行为通过|","|---|---:|---:|---:|---:|---:|---|"]
    for row in data['cases']:
        m=row['actual_behavior'];lines.append(f"|{row['condition']}|{m['yaw_change_deg']:.6f}|{m['max_COM_xy_drift_bodylength']:.6f}|{m['support_mean_over_weight']:.6f}|{m['COM_min_native']:.6f}|{row['actual_native']['post_nonfoot_peak_native']:.6f}|{m['task_pass_candidate']}|")
    lines += ['','## 预热／交接与训练来源','机械qpos/ctrl/targets与神经old256 priming分别核查；prefix不预设完全相同，逐字段最大差与首次差时间保留。教师没有归档old256 hidden，不伪造其重放资格。训练缓存来自冻结789f实际8条源，new96从零起，不依赖训练目标进入actor。','']
    for row in data['cases']:
        lines += ['### '+row['condition'],'','初始匹配与接管输出：','```json',json.dumps(row['initial_source_match'],ensure_ascii=False,indent=2),'```','','最早神经/域阈值（ANN秒，null表示本窗口未达）：','```json',json.dumps(row['first_cohort_NN_domain_onsets_s'],ensure_ascii=False,indent=2),'```','','物理onset、四固定窗口原始统计、机械/神经prefix、分组训练域与干预量见STAT.json该case；全部是实际保存数组统计。','']
    lines += ['## 解释边界与下一步','先比较四固定窗口0–.05/.05–.5/.5–2/2–9.6s的动作跳、基底/残差抵消、新96状态和机械失稳先后；域外比例或高z只是相关描述。任务v翻转保持turn与old256连续，不能据其输出推测身体会停转。残差置零和每tick新96输入归零均未改变任何真实身体。','原生NF峰摘要无法定位首次原生子步，报告onset只来自1ms记录。控制器动作/target/ctrl触边不等于已证实唯一根因，完整时窗也不等于安全门通过。','本阶段不增加训练或候选，不改任何科学来源。教师在学生状态上的查询需要完整姿态/速度/执行器历史与FK/支撑重建，是尚未执行的阶段B；必须单独授权并验证教师目标可行性后再讨论有限纠正学习。当前仍是身体组件，不是经真实neck的新模型部署。',f"本次wall={data['wall_s']:.3f}s，峰RSS={data['peak_RSS_MiB']:.3f}MiB；源/权重前后SHA、RNG和全部重放证据见VERIFY.json。"]
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--execute-parent-fixed-observation-diagnostic',action='store_true');a=p.parse_args()
    if not a.execute_parent_fixed_observation_diagnostic:p.error('ParentstageA fixedobservations release required')
    parent_run()
