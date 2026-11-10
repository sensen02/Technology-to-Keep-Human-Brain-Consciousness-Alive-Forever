"""Report actual search/contact diagnostics; distinguish learned-policy evidence."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk')
OUT=BASE/'20261107T115537Z'


def read(p):return json.loads(p.read_text())


def main():
    search=read(OUT/'SEARCH_RESULTS.json');audit=read(OUT/'AUDIT_ADAPTIVE.json');best=read(OUT/'BEST_PARAMETERS.json')
    assert len(search['results'])==25 and all(audit['checks'].values())
    specs=[('Nominal selected','20261107T112003Z/selected/teacher_selected_4s'),
           ('Static preload','20261107T120826Z/support_feedforward_screen'),
           ('Contact damping','20261107T122002Z/contact_damping_screen'),
           ('Filtered damping','20261107T122432Z/filtered_damping_screen'),
           ('Support dwell','20261107T122951Z/neural/teacher_dwell_4s')]
    rows=[]
    for label,rel in specs:
        folder=BASE/rel;v=read(folder/'VERIFICATION.json');g=read(folder/'GOAL_ASSESSMENT.json')
        rows.append(dict(label=label,folder=str(folder),controller='algorithmteacher',
            D=v['support']['passed'],failed_D=[k for k,x in v['support']['checks'].items() if not x],
            real_steps=v['gait']['all_legs_stepping'],heading=all(g['heading_checks'].values()),
            body_velocity=v['gait']['locomotion']['cycle_mean_forward_speed'],
            heading_change=v['gait']['locomotion']['heading_change_rad'],
            COM_min=v['support']['release']['com_z_min'],
            slip_ratio=v['physics_replay']['material_contact_velocity']['slip_to_forward_speed_ratio'],
            material_pass=v['material_slip_passed'],replay=v['physics_replay']['passed']))
    scores=np.array([r['score']['objective'] for r in search['results']])
    fig,axes=plt.subplots(2,1,figsize=(12,10),constrained_layout=True)
    axes[0].plot(np.arange(25),scores,'o-',label='Allactualtrialobjectives')
    axes[0].plot(np.arange(25),np.minimum.accumulate(scores),lw=2,label='Bestactuallymeasuredsofar')
    axes[0].set(xlabel='Trial index',ylabel='Worstleg +0.1mean +penalties',title='25 adaptive physical comparisons: stable center was not improved')
    axes[0].legend()
    x=np.arange(6);width=.8/len(rows)
    for i,r in enumerate(rows):axes[1].bar(x-.4+width/2+i*width,r['slip_ratio'],width=width,label=r['label'])
    axes[1].axhline(.5,color='r',ls='--',label='unchanged0.5bound');axes[1].set(xticks=x,xticklabels=['LF','LM','LH','RF','RM','RH'],ylabel='Materialslip / bodyforward speed')
    axes[1].legend(ncol=2,fontsize=8);fig.savefig(OUT/'ROUND5_COMPARISON.png',dpi=150);plt.close(fig)
    status=dict(stage='S5 goalround5 actualphysicsdiagnostics',complete=False,
        adaptive_search=dict(trials=25,actual_steps=search['actual_physics_steps'],audit_pass=True,best=best['selected'],
            prior_search_trials=26,all_search_trials=51,all_search_steps=12240000,improvement=False),
        contact_regime=str(OUT/'CONTACT_REGIME.json'),contact_solver='Unchangedoriginal noslip0, mu1, solref/solimp, Newton. NoSlipchangesnotusedorclaimedasoriginalmodel.',
        teacher_results=rows,neural_baseline=str(BASE/'20261107T103607Z/CURRENT_STATUS.json'),
        main_direction='DirectPPOlearnedsensoryresidual onfrozenlearnedvelocitymotorANN; sameoriginalfullbodyandactuator, no teacherpostwarm. PureteachercontrolsabovefailedandarenotdeployedasANN.',
        residual_training=str(BASE/'20261107T123634Z/STATUS.json'),
        neural_policy_criteria='ActualtrainednetworkmuststillpassfullnativeD/sixlegsteps/heading/perlegslip<=.5/independentreplay>=10s andcuts; trainingrewardnotacceptance',
        units='native; originalbrainnecessity/absoluteSIforce/biologicalvalidity NOT_ESTABLISHED')
    p=OUT/'ROUND5_STATUS.json'
    if p.exists():p.read_text()
    p.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    text='''# S5 第五轮：原接触下的实际神经改进准备与物理排查

全目标未完成。25次自适应常量偏移物理搜索已全部完成，最优仍是进入本轮前已实际验证的中心参数；没有新偏移候选改善。所有失败/负结果保留，不用继续增加摩擦或修改接触求解器来把本结果改为通过。

## 搜索与独立验证

新25条每条3s，共6,000,000个原生12.5µs物理步；加前轮26条，总51搜索条/12,240,000步。每批两个单位随机方向及反向、固定seed115537/σ.025、全角偏界±.08，下一中心只能是已实际测得的有效候选；没有把拟合或平均姿态当成新结果。独立审计重建所有RNG状态、方向、裁剪、25条评分、参数、预算、排序和所有源码hash，全部通过。搜索分数不是最终逐足验收。

## 接触机制：不能简单归因摩擦系数不足

保存实际状态重建的LH/RH支撑材料速度约.2386/.2286native/s，法向绝对速度约.1724/.1445native/s。摩擦是原PYRAMIDAL/condim3/μ1；RH金字塔L1牵引利用率最大仅.3536，靠近边界的载荷份额0；LH接近.9边界份额约.0026。大部分滑动发生在摩擦锥内部。J×qvel与世界刚体twist交叉误差约3.6e-15，重建Fn与保存原生记录误差约9e-15，不能用错误的足原点漂移替代材料速度。

负载交接窗口会放大切向及法向运动，但只包含约25%LH、32%RH滑移分子；其余滑移发生于窗口之间。因此没有把“增加交接重叠”冒称唯一解。

MuJoCo官方文档说明软摩擦正则化可能产生漂移，NoSlip后处理会改变该优化问题的处理。主线保持noslip_iterations=0、原solref/solimp和μ，没有把该后处理当成无影响的数值修复。详见同目录SOLVER_INTERPRETATION.zh-CN.md及[MuJoCo建模文档](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-settings)。这不能反推所有滑移都不可由控制消除，仍须实测。

## 有限教师对照：全部单列失败项

| 条件 | 原D全部 | 实际六足迈步 | 直行门 | 前进native/s | 最差滑移比 | 全足滑移门 |
|---|---|---|---|---:|---:|---|
'''
    for r in rows:text+=f"| {r['label']} | {r['D']} | {r['real_steps']} | {r['heading']} | {r['body_velocity']:.5f} | {max(r['slip_ratio']):.5f} | {r['material_pass']} |\n"
    text+='''
静态足载前馈只将规划力矩转换为≤.0015625rad目标偏置，原物理力矩上限不变；600tick静态分配方程/记录复算通过，实际反力与计划反力RMS仍约1.34native，不声称计划即实现。实测未解决后足滑移。

接触速度关节反馈只用内部有界电机实现，没有外力或物理摩擦增加。未滤波的200Hz反馈使实际材料滑移大幅恶化，D5高度和偏航门失败；加入唯一20ms低通后恢复承重/直行，但材料滑移仍未达标。原始与滤波公式、因果递推、真实感觉量和执行器目标均独立审计。反馈不保证关节总功率为负，因为自由根未驱动；不能把“阻尼”命名称为被动摩擦。

六足共同支撑时间对照保留逐足原姿态路径，只将duty5/6变.9，并将频率设.8Hz保证原30rad/s界，精确峰速26.82608rad/s、平台足底间隙.2486native；实际物理全D/步序/直行通过，滑移仍失败，不能只看名义6足支撑比例约.4。

## 主线转向直接实际物理奖励的神经残差

固定现有111149速度损失运动ANN，真实40帧目标历史保持；新残差ANN只看当前183本体感觉和294自身因果目标历史，输出6个关节小目标修正。部署只允许两组学习权重+原有限增量执行器，不允许接触算法、教师、时间相位表或根部轨迹在撤教师后继续控制。

当前PPO实现与导出/独立验证器放在123634Z目录，训练计划8192神经决策，不从训练奖励推断成功。必须零修正与原ANN动作一致、导出NumPy/Torch一致，再做真实物理训练和完整原生撤教师验收。原11s替代ANN保持承重/直行但滑移失败的基线没有被新模型继承为通过证据。

主要产物：

- `/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T115537Z/AUDIT_ADAPTIVE.json`
- `/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T115537Z/CONTACT_REGIME.json`
- `/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T115537Z/ROUND5_COMPARISON.png`
- `/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T120826Z/AUDIT_FEEDFORWARD.json`
- `/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T122002Z/contact_damping_screen/AUDIT_DAMPING.json`
- `/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T122432Z/filtered_damping_screen/AUDIT_DAMPING.json`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T123634Z/run_residual.py`

这些教师、重放或诊断不能算原连接体神经成功。目标保持active；绝对SI力、原脑必要性和生物有效性NOT_ESTABLISHED。
'''
    p=OUT/'ROUND5_REPORT.zh-CN.md'
    if p.exists():p.read_text()
    p.write_text(text)
    print(json.dumps(dict(report=str(p),complete=False,adaptive_trials=25,teacher_cases=len(rows)),ensure_ascii=False))

if __name__=='__main__':main()
