"""Stdlib saved-JSON/streaming JSONL diagnostic only; never imports project backend."""
from pathlib import Path
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'pivot_teacher_design'
BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')


def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(p):return json.loads(Path(p).read_text())


def save(p,v):
    with Path(p).open('x') as stream:json.dump(v,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')


def summary_row(row,line,previous):
    d=row['diagnostic'];ref=d.get('teacher_reference',{})
    result=dict(decision=row['decision'],JSONL_line=line,physical_ms=row['physical_ms'],candidate=row['candidate_phase'],phase=d['phase'],desired_z=[p[2] for p in d['desired_foot_xy_bottom_z']],achieved_z=[p[2] for p in d['achieved_foot_xy_bottom_z']],IK_error_max=max(d['ik_error_norm_per_leg']),qdot_unclipped_absmax=max(map(abs,d['qdot_des_unclipped'])),qdot_clip_count=d['qdot_des_clip_count'],rate_clip_count=d['rate_clip_count'],target_bound_count=len(d['motor_target_bound_indices']),tau_clip_count=len(d['tau_ff_native_clip_indices']),actual_raw_torque_exceed_count=len(d['actual_state_raw_torque_native_exceed_indices']),contact_activation=d['contact_activation'],planned_stance=d['planned_stance'],actual_contact_counts=d['actual_contact_counts'],planned_load_fractions=d['per_leg_fz_over_W'],unbacked_support=d['unbacked_support_indices'],unsupported=d['unsupported'],root_balance_residual_norm=d['root_balance_residual_norm_over_W'],reference_heights=ref.get('planned_clearance'),reference_swing_leg=ref.get('swing_leg'),reference_yaw_rad=ref.get('desired_yaw_relative'),q_des_act=d['q_des_act'],desired_motor_target=d['desired_motor_target'],qdot_des=d['qdot_des'],rate_rad_s=d['rate_rad_s'],tau_ff=d['tau_ff'])
    if previous is not None:
        old=previous['diagnostic'];result.update(previous_actual_decision=previous['decision'],q_des_delta_max=max(abs(x-y) for x,y in zip(d['q_des_act'],old['q_des_act'])),desired_motor_target_delta_max=max(abs(x-y) for x,y in zip(d['desired_motor_target'],old['desired_motor_target'])),qdot_des_delta_max=max(abs(x-y) for x,y in zip(d['qdot_des'],old['qdot_des'])),rate_delta_max=max(abs(x-y) for x,y in zip(d['rate_rad_s'],old['rate_rad_s'])))
    return result


def main():
    prereg=read(ROOT/'PREREG.json');chosen=set(prereg['selected_decisions'])
    paths=[ROOT/'PREREG.json',Path(__file__),SOURCE/'teacher.py',SOURCE/'runner.py',SOURCE/'common.py',SOURCE/'PREREG.json',BASE/'static_teacher.py',BASE/'physics.py',BASE/'sources/repair_core.py']
    for case in prereg['conditions']:
        folder=SOURCE/'trials'/case
        paths.extend(folder/n for n in ('RESULT.json','VERIFY.json','HANDOVER.json','BODY_LENGTH.json','IDENTITY.json','teacher_diagnostics.jsonl'))
    before={str(p):sha(p) for p in paths}
    cases=[]
    for case in prereg['conditions']:
        folder=SOURCE/'trials'/case;r=read(folder/'RESULT.json');audit=read(folder/'VERIFY.json');hand=read(folder/'HANDOVER.json');length=read(folder/'BODY_LENGTH.json');selected=[];last=None;previous=None;last_summary=None;onsets={};warm_digest=None
        with (folder/'teacher_diagnostics.jsonl').open() as stream:
            for lineno,line in enumerate(stream,1):
                row=json.loads(line);d=row['diagnostic']
                if row['decision']==959:warm_digest=hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                if row['candidate_phase']:
                    flags=dict(qdot_clipping=d['qdot_des_clip_count']>0,rate_clipping=d['rate_clip_count']>0,target_bounds=bool(d['motor_target_bound_indices']),tau_clipping=bool(d['tau_ff_native_clip_indices']),actual_raw_torque_exceed=bool(d['actual_state_raw_torque_native_exceed_indices']),lost_any_leg_contact=any(n==0 for n in d['actual_contact_counts']),unbacked_planned_support=bool(d['unbacked_support_indices']),unsupported=d['unsupported'],IK_error_above_p01=max(d['ik_error_norm_per_leg'])>.01,IK_error_above_p1=max(d['ik_error_norm_per_leg'])>.1)
                    for key,hit in flags.items():
                        if hit and key not in onsets:onsets[key]=dict(decision=row['decision'],physical_ms=row['physical_ms'],candidate_elapsed_s=(row['physical_ms']-5800)/1000,JSONL_line=lineno)
                if row['decision'] in chosen:selected.append(summary_row(row,lineno,previous))
                last=row;last_summary=summary_row(row,lineno,previous);previous=row
        if last is not None and all(x['decision']!=last['decision'] for x in selected):selected.append(last_summary)
        cases.append(dict(condition=case,actual_ms=r['actual_ms'],candidate_metrics=r['candidate_metrics'],native=r['native'],early_stop=r['early_stop'],original_five=r['original_five'],audit_integrity_all_pass=audit['integrity_all_pass'],HANDOVER=hand,body_length=length,warm_decision959_diagnostic_sha256=warm_digest,first_onsets=onsets,selected=selected,actual_JSONL_rows=last['decision']+1))
    unchanged=all(sha(p)==h for p,h in before.items())
    if not unchanged:raise ValueError('Saved scientific input changed during diagnostic')
    data=dict(kind=prereg['kind'],scientific_inputs_unchanged=True,cases=cases,matched_HANDOVER_all3=all(c['HANDOVER']==cases[0]['HANDOVER'] for c in cases),matched_warm959_diagnostic_all3=len({c['warm_decision959_diagnostic_sha256'] for c in cases})==1,execution=dict(numeric_imports=0,NPZ_array_reads=0,model_loads=0,physical_steps=0,optimizer_updates=0),limitations='No actual actuator target or actual qvel at5.8 read from NPZ; desired_motor_target is a teacher proposal, not actual persistent target; IK achieved feet refer to privateFK, not actual feet')
    save(ROOT/'SELECTED_SUMMARY.json',data)
    report=report_text(cases,data)
    with (ROOT/'REPORT.zh-CN.md').open('x') as stream:stream.write(report)
    before[str(ROOT/'SELECTED_SUMMARY.json')]=sha(ROOT/'SELECTED_SUMMARY.json');before[str(ROOT/'REPORT.zh-CN.md')]=sha(ROOT/'REPORT.zh-CN.md')
    save(ROOT/'sourcehashes.JSON',dict(kind='READONLY_PIVOT_FAILURE_SOURCE_AND_SAVED_JSON_SHA256_V1',all_scientific_inputs_before_after_exact=True,source_hashes=before,arrays_read=False,models_loaded=False,physics_run=False))
    print(json.dumps(dict(report=str(ROOT/'REPORT.zh-CN.md'),summary=str(ROOT/'SELECTED_SUMMARY.json'),hashes=str(ROOT/'sourcehashes.JSON'),source_files=len(before),scientific_inputs_unchanged=True,numeric_imports=0,physical_steps=0)))


def report_text(cases,data):
    rows=[]
    for c in cases:
        m=c['candidate_metrics'];rows.append(f"|{c['condition']}|{c['actual_ms']}|{m['duration_s']:.3f}|{m['yaw_change_deg']:.6f}|{m['COM_min_native']:.6f}|{m['nonfoot_1ms_peak']:.6f}|{c['native']['post_nonfoot_peak_native']:.6f}|{m['max_COM_xy_drift_bodylength']:.6f}|{str(m['task_pass_candidate'])}|")
    onsets=[]
    for c in cases:
        s=c['first_onsets'];onsets.append(c['condition']+'：'+ '；'.join(k+'='+str(s[k]['candidate_elapsed_s'])+'s' for k in ('qdot_clipping','rate_clipping','lost_any_leg_contact','unbacked_planned_support','IK_error_above_p1') if k in s))
    return '''# 首批停步／pivot教师失败：只读交接诊断

本报告只读取已保存JSON及流式选定JSONL行，0 NumPy/Torch/MuJoCo导入、0 NPZ数组、0模型／物理／训练。所有源和实际输入SHA前后完全一致。三回合RESULT与VERIFY均已存在，审核完整性均通过，但教师行为未通过。本诊断不是新的物理反事实，不改旧成败；所有数值是实际短窗口，不能补成9.6s。

## 实际结果

|条件|实际总ms|候选秒|实际yaw°|COM最低原生|1ms NF峰原生|native NF峰原生|max漂移/体长|任务候选通过|
|---|---:|---:|---:|---:|---:|---:|---:|---|
'''+ '\n'.join(rows)+'''

stop门保持null，因为没有冻结站停速度阈值；同时其COM和非足接触安全门已明确FAIL。左右pivot均FAIL、全六腿周期0、完整窗口false、额外力0。三个early_stop均是连续100ms COM<.8或非足>.5W的预登记安全终止；不是完整窗口验收通过。支撑均值仍≥.9不代表身体稳定。

## 1. 悬空脚高度被hold固化：本批具体假说被实际交接记录否定

三个HANDOVER文本及959行warm诊断完全一致。交接时COMz=1.275868111950224，实际状态qpos/qvel/ctrl/target/buffer和外力在构造候选前后hash相等。decision960（JSONL第961行，physical5800ms）六足desired_z分别为[-.00757711796,-.02512267165,-.01628010624,-.00909556299,-.02666549191,-.01687186453]，全为负，实际接触计数[3,3,3,3,3,3]。因此这里没有“交接时某足正高度悬空、stop永久保持不落地”的实证。

源码确实有一般性风险：`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/pivot_teacher_design/teacher.py:95-97`捕获rest/bottom0；170行hold构造继承的rest_pad_bottom来自实际测量，sample110-141在stop给target_z=0；原 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/static_teacher.py:83-84,264`将原底高继续加参考高度。如果在别的walking相位构造，正高度会被保留，但该一般缺陷不能冒充本批已发生原因。这里的负底高对应实际接触几何/接触穿透，不能仅凭负号宣布异常。

## 2. 构造不reset，不等于教师目标／导数连续；本批初始跳变较小

旧warm959→新960最大q_des变化.001973347rad、教师desired_motor_target变化.001967310rad；warm末一步958→959对应.000435280/.000421979rad。新teacher从实际关节姿态初始化，未复制旧teacher的私有q_des；新第一步qdot接近9.33e-13，而旧warm末最大未裁剪qdot=.0870559rad/s。960/961两步无qdot、rate、target bound或tau饱和，IK误差约1e-15，实际接触全在。因此可以说控制器切换改变私有目标历史／导数，但没有观察到第一步巨幅目标跳变或直接饱和证据，不能把首帧不连续定为唯一原因。

重要字段区分：diagnostic.desired_motor_target是教师提议目标，实际持续积分的c.actuator.target_rad并不在JSONL中逐值保存；HANDOVER仅保存其hash。实际qvel也仅在NPZ轨迹中，而本轮禁止读取数组。所以本报告不能据JSONL恢复5.8s真实target误差、真实执行器速度或PD机械功；没有虚构这些值。`qdot_des_unclipped`是私有IK目标有限差分，`achieved_foot`也是私有FK，不是实际足轨迹。

## 3. stop也失稳，需先隔离基础支撑／姿态任务

stop在候选.2s仍IK误差约2.02e-13，接触全[3,...,3]、无饱和，但未裁剪qdot需求已升1.8262rad/s；.4s qdot=5.4243rad/s、7个qdot被裁，LF实际接触0而支撑求解仍给LF .04082W计划载荷，unbacked=[0]、unsupported=true。末.785s私有IK最大误差1.52295原生长度、qdot需求299.297rad/s、目标边界7项、tau裁剪2项，属于后续明显失配，不是已实现稳定停步。

首个阈值时间（从5.8s交接算，按5ms决策诊断）：

'''+ '\n\n'.join(onsets)+'''

左右pivot都在首个.8s摆腿结束前停止，第一条摆动腿是RM（index5；实际顺序LF/LH/LM/RF/RH/RM）。.1s两方向qdot需求已2.3265/2.4319rad/s并裁剪；.2s LF接触丢失且仍被当计划支撑；.4s IK误差约2.09/1.78并多关节饱和。目标期望yaw在.1s仅±3.315e-5rad，而RM摆动净空已.02296875。两个实际yaw都偏右，不能仅凭短窗符号反推yaw映射一定写反：共同支撑失稳、单腿抬起与目标参考作用尚未隔离。

## 4. 影子根随动后的姿态任务缺口：源码可定位，物理因果仍待验证

`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/pivot_teacher_design/teacher.py:153-159`每决策把私有q_des非执行器分量同步实际root、更新actualThorax旋转／原点；真实root没有写入。原 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/static_teacher.py:75-84,125-147`则始终以构造时固定私有root做FK。新sample136-141只给足XY和world底高；继承IK149-163只解足目标，支撑165-250只分配竖直准静态载荷，**没有显式的期望Thorax高度／roll／pitch恢复任务**。

因此，新随动影子与旧固定影子的机械含义不同：身体实际高度变动时，IK可以通过改变关节目标继续满足固定地面足高，而不是保持旧固定根对应的站姿关节构型。最初IK误差很小但目标速度逐步增长，和这个姿态约束缺口相容；它不证明该机制是唯一充分原因。旧静态分配的数学根残差很小，也不能替代支撑足真实接触：stop.4s已有unbacked但残差仍1.41e-4。后期极大IK速度需求属于错误增长/饱和现象，不能当原生执行器实际速度。

此外水平目标用desired旋转表达worldanchors、私有FK用actual旋转，而Z继续是world底高；这在pitch/roll变化时不能自动等同完整3D期望躯干姿态任务。完整坐标/高度约束需要独立源码设计审查或有限物理对照，不能通过直接写根轨迹“修好”。

## 最小下一步

先把“连续旧walkingwarm→可持续稳停／支撑”作为独立基础问题，和同初态的原运动ANN/旧教师继续行走对照区分。先说明期望身体高度／倾角与足接触的物理约束、控制器切换目标历史，证明不写真实root/额外力的完整身体能稳住；再谈原地教师转向的幅度和摆腿参数。现有stop也失败，不能把所有失败归结为pivot速度太大，也不能据三次有限失败判身体不可行。

本次不创建新candidate、不改参数、不重物理、不训练；下一若要验证假说须父另立新版本、明确唯一主要变量与有限预算，不盲多档sweep或额外ANN epochs。原789f的身体命令诊断与新neck2DN独立二维任务仍是不同身份，尚无新神经停走／原地控制通过。

证据目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/pivot_teacher_failure_diagnostic/` 含SELECTED_SUMMARY.json的实际选行/首阈值及sourcehashes.JSON。每条件JSONL只流式保留958/959/960/961/962/980/1000/1040及末行；没有全日志载入内存或NPZ读取。输入SHA在读取前后再次核验，源保持冻结。
'''


if __name__=='__main__':main()
