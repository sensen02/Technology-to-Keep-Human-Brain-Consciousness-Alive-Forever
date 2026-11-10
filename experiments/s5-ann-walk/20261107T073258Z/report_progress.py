"""Aggregate this iteration without promoting standing/skidding to stable walking."""
import json
import subprocess
import sys
from pathlib import Path
from neural_walk import OUT, dump, sha
HERE=Path(__file__).resolve().parent
PY=Path('/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python')


def main():
    rows=[]
    for tag in ['ann_11s','hold_4s','motor_cut_4s','teacher_11s','ann_disturb_11s']:
        folder=OUT/tag
        assert (folder/'RESULT.json').exists(),tag
        if tag!='ann_11s':
            proc=subprocess.run([str(PY),'-B',str(HERE/'verify_walk.py'),str(folder)],capture_output=True,text=True)
            assert proc.returncode in [0,2],proc.stdout+proc.stderr
            (HERE/(tag+'_audit.log')).write_text(proc.stdout+proc.stderr)
        v=json.loads((folder/'VERIFICATION.json').read_text())
        r=json.loads((folder/'RESULT.json').read_text())
        rows.append(dict(tag=tag,release_s=v['release_seconds'],
            support=v['support']['release'],D_failed=[k for k,val in v['support']['checks'].items() if not val],
            stepping=v['gait']['all_legs_stepping'],motion=v['gait']['locomotion'],
            policy=v['policy']['passed'],physical_replay=v['physics_replay']['performed'],
            verified=v['verified_neural_locomotion'],folder=str(folder)))
    ann=json.loads((OUT/'ann_11s/VERIFICATION.json').read_text())
    old=[]
    for name in ['th_primed','loop_primed']:
        path=OUT/'history_fix/learning'/name/'PRIMED_RESULT.json'
        r=json.loads(path.read_text());old.append(dict(tag=name,walk=r['walk'],Dpassed=r['validation']['passed']))
    fastroot=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T074255Z')
    fast=json.loads((fastroot/'ann_screen/RESULT.json').read_text())
    status=dict(importance='核心必需',goal='S5 stable physical neural walking',complete=False,
        finding='Replacement ANN sustains stepping and foot support through10s without a fall; D5 and contact slip fail, with substantial turning.',
        controller='replacement neural motor generator, not original connectome; no clock/lookup/teacher after1s',
        results=rows,old_weight_history_fix=old,
        material_slip=ann['physics_replay']['material_contact_velocity'],
        faster_candidate=dict(folder=str(fastroot),Dchecks=fast['validation']['checks'],release=fast['validation']['release']),
        video=str(OUT/'ann_11s/WALK_4x_slow.mp4'),
        next='Build ground-consistent stance reference with constant y/z, backward stance velocity and near-fixed sole orientation; then neural imitation and closed-loop physical verification. Full D gates unchanged.',
        units='native; absolute SI force NOT_ESTABLISHED')
    dump(OUT/'STATUS.json',status)
    a=rows[0];s=a['support'];g=a['motion'];sl=ann['physics_replay']['material_contact_velocity']
    text=f'''# S5 稳定行走实施进展：持续迈步已延长，正式验收仍未通过

替代 ANN 在 1s 算法教师预热后连续接管 10s，没有倒地；六足出现持续卸载/着地与身体推进。**仍未完成稳定行走**：质心高度 D5 不通过，实际接触点滑移很大，并持续转弯。不能把这段结果称为稳定直行或原连接体脑的自主行为。

控制来自学习的人工网络：40帧自身目标历史中抽取7个时点，MLP 294→128→128→42，离线多步自预测训练；预热全程填充真实历史，接管后不输入时间、相位或步态表，也不运行算法偏航控制。当前替代臂没有运行原12000细胞脑，脑必要性不适用；原连接体+ANN路线保留并单独复测。

## 同次物理证据

- 预热 0–1s，替代 ANN 1–11s，完整49/48/42/71自由身体、正常原重力与接触、原力矩上限、有限增量PD；无根部驱动和外力注入。
- 足载均值 {s['foot_support_mean']:.6f} W，10分位 {s['foot_support_p10']:.6f} W；非足支撑为0，自由落体持续时间0。质心最低 {s['com_z_min']:.6f}，D5失败（其余原生支撑门通过），没有删去失败帧。
- 以步态周期平均身体运动，前向累计 {g['cycle_mean_body_forward_travel']:.6f} native，平均速度 {g['cycle_mean_forward_speed']:.6f} native/s；这不是世界x的直线位移。航向累计变化 {g['heading_change_rad']:.6f} rad，轨迹接近转圈，横向累计 {g['cycle_mean_body_lateral_travel']:.6f} native。
- 独立重放从初态重新积分，保存的关节/根位置、速度、COM、足位置、力矩和抽样接触全部最大误差0.0；ANN每个输出、目标积分、交接历史和物理快照重算一致。
- 接触物质点切向速度用刚体速度与接触力臂计算，并与接触点Jacobian×qvel交叉核对。六足法向力加权滑移速度为 {', '.join(f'{v:.3f}' for v in sl['force_weighted_mean_tangential_speed'])} native/s，约为身体前进速度的 {min(sl['slip_to_forward_speed_ratio']):.2f}–{max(sl['slip_to_forward_speed_ratio']):.2f} 倍，**滑移门失败**。采样间隔5ms，不能称连续12.5µs接触速度测量。

## 必要对照

| 条件 | 撤教师后时长 | 六足持续迈步 | 足载均值/W | 非足占比 | D失败项 |
|---|---:|---|---:|---:|---|
'''
    for r in rows:
        s=r['support']
        text+=f"| {r['tag']} | {r['release_s']:.1f}s | {r['stepping']} | {s['foot_support_mean']:.4f} | {s['nonfoot_share_mean']:.4f} | {', '.join(k.split('_')[0] for k in r['D_failed']) or '无'} |\n"
    text+='''
零增量hold保留执行器自身PD保持，不能等同断力矩；motor_cut真正关闭力矩。扰动是1.30–1.35s声明的关节目标速率扰动，最大0.3rad/s，不能扩大称外力平衡恢复。算法教师仅作对照。

## 原路线的接口修复与失败

旧TH/loop wrapper到1s才首次调用，用同一状态重复填满历史；现已在原模型不重训、不改身体下补齐预热期间的真实历史。TH修复后仍倒地；loop修复后通过短时支撑门，但身体前移仅约0.026，多个足几乎不抬起，不能计为行走。这证明修复必要，也证明该修复本身没有完成目标。

之前“时间教师无位置纠错”解释错误：原公式包含 `(ref-target)/.005`；“frozen与ann逐位相同”也不准确，二者非常接近但原数组不同。近自交不等于数学不可观测。此前用足body-origin世界x端点位移称精确滑移、用原始COM振动路径称行走距离，均不用于本轮成功判据。

5.5Hz的单独神经候选未改变身体或执行器上限，4s筛查仍失败D5，并新增短时自由落体D2失败，不继续以更快频率冒充稳定。

## 视频与下一动作

视频直接读取本次BODY轨迹，1s处标明算法教师→替代ANN，4倍慢放，总44s视频对应11s模拟。1320帧解码通过、71mesh不透明、渲染COM与原数组误差0.0。已打开8时点PREVIEW与SUMMARY图，未逐帧目视1320帧。

下一步修正支撑期足轨迹：在离线完整身体FK里约束支撑足的共同后移速度、固定横向与高度、足底方向，摆动高度形状保留真实步行数据来源；该参考只作教师，仍须重新训练ANN并撤除教师验证。不能靠改验收窗、放宽D5、加入根驱动或声称原脑能力收口。

绝对SI力与生物有效性：NOT_ESTABLISHED。
'''
    text+=f'\n核心证据：{OUT}/ann_11s/VERIFICATION.json\n\n视频：{OUT}/ann_11s/WALK_4x_slow.mp4\n\n预览：{OUT}/ann_11s/PREVIEW.png\n\n完整状态：{OUT}/STATUS.json\n'
    path=OUT/'REPORT.zh-CN.md'
    if path.exists():path.read_text()
    path.write_text(text)
    inputs=list(OUT.rglob('*.json'))+[OUT/'REPORT.zh-CN.md']
    inputs=[p for p in inputs if p.name!='MANIFEST_SHA256.json']
    manifest=OUT/'MANIFEST_SHA256.json'
    if manifest.exists():manifest.read_text()
    dump(manifest,{str(p):sha(p) for p in inputs})
    print(json.dumps(dict(report=str(path),verified=False,rows=len(rows)),ensure_ascii=False))


if __name__=='__main__':main()
