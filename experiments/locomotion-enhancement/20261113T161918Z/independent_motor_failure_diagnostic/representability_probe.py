"""Parent-only bound6 algebraic action-range addendum; no train/Core/teacherquery."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import resource
import sys
import time
import xml.etree.ElementTree as ET
from metadata import ROOT,BODY,CAL,SHA,CASES,read,save,sha,verify,qualify

XML=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-triad-pad/20261110T010000Z/MODEL_TRIAD_x200.xml')
LEGS=('LF','LH','LM','RF','RH','RM')


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def actuator_groups():
    root=ET.parse(XML).getroot();parent=root.find('actuator')
    if parent is None:raise ValueError('Fullsource actuator declarations required')
    entries=list(parent)
    if len(entries)!=42 or any(e.tag!='motor' for e in entries):raise ValueError('Exact42source motors required')
    groups={leg:[] for leg in LEGS};rows=[]
    for index,element in enumerate(entries):
        joint=element.attrib.get('joint','');matches=[leg for leg in LEGS if joint.startswith('joint_'+leg)]
        if len(matches)!=1:raise ValueError('Unambiguouslegprefix source mapping')
        values=[float(x) for x in element.attrib['ctrlrange'].split()]
        if len(values)!=2 or not values[0]<0<values[1]:raise ValueError('NativeXMLcontrolrange')
        leg=matches[0];groups[leg].append(index);rows.append(dict(index=index,actuator=element.attrib['name'],joint=joint,leg=leg,native_limit=min(-values[0],values[1])))
    if any(len(v)!=7 for v in groups.values()):raise ValueError('Source7jointmotors perleg')
    return groups,rows


def range_floor(oldmean,target,bound=6.):
    """Coordinate/time-independent CLOSED interval gives conservative infimum."""
    import numpy as np
    old=np.asarray(oldmean,dtype=float);y=np.asarray(target,dtype=float)
    if old.shape!=y.shape or not np.isfinite(old).all() or not np.isfinite(y).all():raise ValueError('Finitealigned oldmean/actualtarget')
    lo=np.tanh(old-bound);hi=np.tanh(old+bound);best=np.minimum(np.maximum(y,lo),hi);error=np.abs(y-best)
    return lo,hi,best,error


def statistics(lo,hi,y,error,prediction=None):
    import numpy as np
    outside=(y<lo)|(y>hi);outside_tol=(y<lo-1e-12)|(y>hi+1e-12);violations=error[outside]
    out=dict(coordinates=int(y.size),strict_out_of_range_fraction=float(outside.mean()),out_of_range_fraction_tolerance1e12=float(outside_tol.mean()),action_MSE_lower_bound=float(np.mean(error**2)),action_RMSE_lower_bound=float(np.sqrt(np.mean(error**2))),absolute_error_floor_p95=float(np.quantile(error,.95)),absolute_error_floor_max=float(error.max()),unreachable_coordinates_abs_error_p95=None if not violations.size else float(np.quantile(violations,.95)),ticks_with_any_out_of_range_fraction=float(outside.any(-1).mean()),interval_width_min=float((hi-lo).min()),interval_width_p05=float(np.quantile(hi-lo,.05)),interval_width_p50=float(np.quantile(hi-lo,.5)),interval_width_p95=float(np.quantile(hi-lo,.95)))
    if prediction is not None:
        learned=float(np.mean((prediction-y)**2));out.update(actual_learned_action_MSE=learned,actual_learned_action_RMSE=math.sqrt(learned),learned_MSE_minus_algebraic_lower_bound=learned-out['action_MSE_lower_bound'])
    return out


def guard(start):
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
    if rss>1800 or time.monotonic()-start>180:raise RuntimeError('Boundprobe resourcecap exceeded, noextension')
    return rss


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute-parent-representability-probe',action='store_true');a=p.parse_args()
    if not a.execute_parent_representability_probe:p.error('Parent separateboundprobe release required')
    start=time.monotonic()
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(key)!='1':raise ValueError(key+'=1 required')
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    if any((ROOT/name).exists() for name in ('REPRESENTABILITY.json','REPRESENTABILITY_VERIFY.json','representability_cache.npz')):raise FileExistsError('No overwrite repeatedboundprobe')
    add=read(ROOT/'ADDENDUM_PREREG.json');stage=read(ROOT/'STAT.json');prior=read(ROOT/'VERIFY.json')
    if add['checkpoint_sha256']!=SHA or stage['checkpoint_sha256']!=SHA or not prior['all_passed'] or not all(prior['checks'].values()) or prior['STAT_sha256']!=sha(ROOT/'STAT.json') or prior['report_sha256']!=sha(ROOT/'REPORT.zh-CN.md') or not stage['baseline_replay_all4_exact']:raise ValueError('Actualcompleted stageA all4baselineproof required beforeaddendum')
    meta=qualify();hashes=dict(meta['source_hashes'])
    for path in (ROOT/'ADDENDUM_PREREG.json',Path(__file__),ROOT/'STAT.json',ROOT/'VERIFY.json',ROOT/'REPORT.zh-CN.md',XML):hashes[str(path)]=sha(path)
    groups,mapping=actuator_groups()
    for case in CASES:
        identity=read(BODY/'trials'/case/'IDENTITY.json');limits=identity['actuator']['torque_limit']
        if identity['files'].get(str(XML),{}).get('sha256')!=sha(XML):raise ValueError('MappingXML not bound to actualcompiled identity')
        if len(limits)!=42 or any(not math.isclose(float(v),row['native_limit'],rel_tol=0.,abs_tol=1e-10) for v,row in zip(limits,mapping)):raise ValueError('Sourceactuatororder/nativecaps do notmatch actualcompiled identity')
    verify(hashes);guard(start)
    import torch
    import numpy as np
    torch.set_num_threads(1);rng=dict(torch=torch.get_rng_state().clone(),numpy=np.random.get_state(),python=random.getstate())
    if any(k in sys.modules for k in ('physics','mujoco','flygym','student_wave')):raise ValueError('No physicalbackend allowed')
    ci=module('_boundprobe_strict_motor_io',CAL/'common.py');m,payload=ci.load_checkpoint(meta['checkpoint'],SHA,final=True);m.requires_grad_(False);modelmod=sys.modules[m.__class__.__module__];before=modelmod.state_digest(m.state_dict())
    sentinel=object();previous=sys.modules.get('sources',sentinel)
    try:
        cs=module('_boundprobe_frozen_cal_sources',CAL/'sources.py');sys.modules['sources']=cs;cd=module('_boundprobe_frozen_cal_data',CAL/'data.py')
    finally:
        if previous is sentinel:sys.modules.pop('sources',None)
        else:sys.modules['sources']=previous
    data=cd.prepare(m)
    with torch.no_grad():old=data['oldmean'].numpy().copy();target=data['y'].numpy().copy()
    if old.shape!=(1920,8,42) or target.shape!=old.shape:raise ValueError('Actual8fulltraining1920x42')
    evaluation=read(CAL/'EVALUATION.json');evpath=Path(evaluation['arrays']['path'])
    if sha(evpath)!=evaluation['arrays']['sha256']:raise ValueError('Actualfinal600evaluationartifact SHA')
    hashes[str(evpath)]=sha(evpath)
    with np.load(evpath,allow_pickle=False) as archive:
        if not np.array_equal(archive['actual_target'],target) or not np.array_equal(archive['old_mean'],old):raise ValueError('Frozenoldcache/target differ from actualfinalcalibration')
        learned=archive['new_action'].copy()
    lo,hi,best,error=range_floor(old,target,6.);rows=[]
    for i,raw in enumerate(data['raw']):
        per_leg={leg:statistics(lo[:,i,indices],hi[:,i,indices],target[:,i,indices],error[:,i,indices],learned[:,i,indices]) for leg,indices in groups.items()}
        rows.append(dict(id=raw['entry']['id'],teacher=raw['entry']['teacher'],all42=statistics(lo[:,i],hi[:,i],target[:,i],error[:,i],learned[:,i]),per_leg=per_leg))
    student=[]
    for case in CASES:
        # CurrentstudentX has no qualified corrective teacher label. NoMSEfloor here.
        with np.load(BODY/'trials'/case/'neural_decisions.npz',allow_pickle=False) as archive:current=archive['oldmean'][1160:].copy()
        lower=np.tanh(current-6.);upper=np.tanh(current+6.);width=upper-lower
        student.append(dict(condition=case,saved_oldmean_definition='savedproduction mean-residual decomposition',target_label_available=False,MSEfloor_computed=False,lower_min=float(lower.min()),lower_max=float(lower.max()),upper_min=float(upper.min()),upper_max=float(upper.max()),width_min=float(width.min()),width_p05=float(np.quantile(width,.05)),width_p50=float(np.quantile(width,.5)),width_p95=float(np.quantile(width,.95)),interval_straddles_zero_fraction=float(np.mean((lower<=0.)&(upper>=0.))),scope='Intervaldescription only, zero signstraddling isnot a teacherlabel or actualmotion assertion'))
    checks=dict(actual8oldcache_target_exact_final_evaluation=True,all42finite=bool(np.isfinite(old).all() and np.isfinite(target).all() and np.isfinite(error).all()),all_targets_bounded=bool(np.max(np.abs(target))<=1.+1e-12),best_actions_in_closedrange=bool(np.all(best>=lo) and np.all(best<=hi)),learned_error_not_below_bound=all(row['all42']['learned_MSE_minus_algebraic_lower_bound']>=-1e-12 for row in rows),source_legmapping42nativecaps_exact=True,weights_unchanged=modelmod.state_digest(m.state_dict())==before,RNG_unchanged=torch.equal(rng['torch'],torch.get_rng_state()) and rng['python']==random.getstate() and rng['numpy'][0]==np.random.get_state()[0] and np.array_equal(rng['numpy'][1],np.random.get_state()[1]) and rng['numpy'][2:]==np.random.get_state()[2:],no_physics_or_teacherquery=not any(k in sys.modules for k in ('physics','mujoco','flygym','student_wave')))
    verify(hashes);rss=guard(start)
    cache=ROOT/'representability_cache.npz';np.savez_compressed(cache,old_mean=old,actual_target=target,interval_lower=lo,interval_upper=hi,nearest_interval_action=best,absolute_error_floor=error)
    rss=guard(start)
    result=dict(kind=add['kind'],checkpoint_sha256=SHA,bound_mean_residual=6.,mathematical_status='Closedinterval relaxed pertime/percoordinate infimum; conservative lowerbound, notfiniteANNattainment proof',global_equal8_action_floor=statistics(lo,hi,target,error,learned),training_sequences=rows,actuator_source_mapping=dict(xml=str(XML),xml_sha256=sha(XML),indices_by_leg=groups,rows=mapping,qualification='XMLactuatordeclarationorder+exactactual42nativecapvector'),actual4_student_interval_descriptions=student,cache=dict(path=str(cache),sha256=sha(cache)),checks=checks,source_hashes=hashes,wall_s=time.monotonic()-start,peak_RSS_MiB=rss,physical_steps=0,optimizer_updates=0,teacher_state_query=False,newcontroller_outputs_applied=False)
    save(ROOT/'REPRESENTABILITY.json',result)
    with (ROOT/'REPRESENTABILITY.zh-CN.md').open('x') as stream:stream.write(report(result))
    rss=guard(start);verify(hashes)
    save(ROOT/'REPRESENTABILITY_VERIFY.json',dict(all_passed=bool(all(checks.values())),checks=checks,report_sha256=sha(ROOT/'REPRESENTABILITY.zh-CN.md'),result_sha256=sha(ROOT/'REPRESENTABILITY.json'),cache_sha256=sha(cache),source_hashes=hashes,whole_call_wall_s=time.monotonic()-start,peak_RSS_MiB=rss,physical_steps=0,optimizer_updates=0))
    guard(start)
    if not all(checks.values()):raise ValueError('Boundprobe checkfailure, retainedevidence')
    print(json.dumps(dict(result=str(ROOT/'REPRESENTABILITY.json'),all_passed=True,physical_steps=0,wall_s=result['wall_s'])),flush=True)


def report(r):
    lines=['# bound6在真实训练轨迹上的动作表示下界','','只对冻结789f缓存oldmean及8条真实训练目标做代数区间计算，0物理、0训练、0教师状态查询。原阶段A科学代码与结果未改。','','理论闭区间为[tanh(oldmean−6), tanh(oldmean+6)]，将实际目标夹到区间得到逐时刻／逐动作最小可能误差。由于6*tanh(head)有限权值严格位于(−6,6)，该值是闭包放松后的保守下界／infimum，不代表共享ANN／GRU一定可实现，幅度6也不是物理力矩上限。','','|训练序列|目标区间外比例|MSE下界|RMSE下界|95%绝对误差下界|实际训练RMSE|','|---|---:|---:|---:|---:|---:|']
    for row in r['training_sequences']:
        s=row['all42'];lines.append(f"|{row['id']}|{s['strict_out_of_range_fraction']:.8g}|{s['action_MSE_lower_bound']:.8g}|{s['action_RMSE_lower_bound']:.8g}|{s['absolute_error_floor_p95']:.8g}|{s['actual_learned_action_RMSE']:.8g}|")
    lines += ['','逐腿7通道组依据原完整XML actuator声明与joint前缀，并与四实际IDENTITY的42个原生力矩上限逐项一致校验；分腿统计和mapping完整列于REPRESENTABILITY.json，不由数组shape臆测。','训练下界大意味着此固定残差域的逐点放松问题本身无法精确拟合这些目标；若近零，仅说明未见大的逐点幅度下限，不能排除网络容量、历史表示、观测别名、优化不足或闭环暴露差异，也不证明某一个机制唯一成立。','四条学生实际轨迹只报告保存oldmean对应动作区间的宽度与符号覆盖，不存在其真实teacher纠正目标，不计算学生状态MSE下界。旧teacher同时间动作不能冒充新学生状态标签。','本补充不改bound、conditioning、网络输出或权重，不产生候选，任何后续变化须独立预登记与验证。所有原组件行为FAIL保留，当前尚无brain-realneck新控制部署。',f"源／权重／RNG前后核验见REPRESENTABILITY_VERIFY.json；wall={r['wall_s']:.3f}s，峰RSS={r['peak_RSS_MiB']:.3f}MiB。"]
    return '\n'.join(lines)+'\n'


if __name__=='__main__':main()
