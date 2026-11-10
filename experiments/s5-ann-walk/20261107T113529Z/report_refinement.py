"""Goalround4 search audit and strict physics outcome, preserving all failed gates."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk')
OUT=BASE/'20261107T113529Z'


def read(p):return json.loads(p.read_text())


def main():
    search=read(OUT/'SEARCH_RESULTS.json');best=read(OUT/'BEST_PARAMETERS.json');audit=read(OUT/'AUDIT_REFINEMENT.json')
    assert all(audit['checks'].values()) and len(search['results'])==13
    specs=[('Original broadlift gain128','20261107T101753Z/teacher_gain128_4s'),
           ('Search1 selected','20261107T112003Z/selected/teacher_selected_4s'),
           ('Search2 selected','20261107T113529Z/selected/teacher_refined_4s'),
           ('Halfcadence control','20261107T114023Z/teacher_half_cadence_6s')]
    if (BASE/'20261107T114444Z/neural/teacher_lower_lift_4s/VERIFICATION.json').exists():
        specs.append(('Lower lift .18','20261107T114444Z/neural/teacher_lower_lift_4s'))
    rows=[]
    for name,rel in specs:
        folder=BASE/rel;v=read(folder/'VERIFICATION.json');g=read(folder/'GOAL_ASSESSMENT.json');r=read(folder/'RESULT.json')
        slip=v['physics_replay']['material_contact_velocity']
        rows.append(dict(name=name,folder=str(folder),release_s=r['release_seconds'],D=v['support']['passed'],
            support=v['support']['release'],steps=v['gait']['all_legs_stepping'],heading=all(g['heading_checks'].values()),
            motion=v['gait']['locomotion'],material_pass=v['material_slip_passed'],slip_ratios=slip['slip_to_forward_speed_ratio'],
            physics_replay=v['physics_replay']['passed'],neural_success=False))
    status=dict(stage='S5 bounded actualphysics parameter search and strict checks',complete=False,
        search1=dict(trials=13,steps=3120000,selection=str(BASE/'20261107T112003Z/BEST_PARAMETERS.json')),
        search2=dict(trials=13,steps=search['actual_physics_steps'],selection=best['selected'],audit_pass=all(audit['checks'].values())),
        aggregate_search=dict(trials=26,steps=3120000+search['actual_physics_steps'],fixed_budget_met=True),
        extra_diagnostics='Strictselected4sforeachsearch plusone6shalf-cadencecontrolandone4slowerliftcontrol, allseparatelypreregistered andnotcountedassparse26searchtrials.',
        results=rows,originalbrain='notrun, thesecontinuousalgorithmteachersarenotANNsuccess',
        thresholds='originalnativeD/steps/heading/perlegmaterialslip<=.5 andoriginalforce/ratecapsunchanged',
        current_neural_baseline=str(BASE/'20261107T103607Z/CURRENT_STATUS.json'),
        failures='Noall-feetslippassfromsearchrank; allindividualtrialstestskept, numericalangleclipandJSONadapterlimitsdisclosed',
        units='native; absoluteSIforce/originalbrainnecessity/biologicalvalidity NOT_ESTABLISHED')
    p=OUT/'ROUND4_STATUS.json'
    if p.exists():p.read_text()
    p.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    fig,axes=plt.subplots(2,1,figsize=(12,9),constrained_layout=True)
    scores=[r['score']['objective'] for r in search['results']];tags=[r['tag'].replace('joint_','').replace('Trochanter','Troch') for r in search['results']]
    axes[0].barh(np.arange(13),scores,color=['tab:orange' if r['tag']==best['selected']['tag'] else 'tab:blue' for r in search['results']])
    axes[0].set(yticks=np.arange(13),yticklabels=tags,xlabel='Worstfoot +0.1mean +declared penalties');axes[0].invert_yaxis()
    axes[0].set_title('13 local actual-physics comparisons; selection is not acceptance')
    x=np.arange(6);width=.8/len(rows)
    for i,r in enumerate(rows):axes[1].bar(x-.4+width/2+i*width,r['slip_ratios'],width=width,label=r['name'])
    axes[1].axhline(.5,color='red',ls='--',label='unchanged perleg threshold0.5');axes[1].set(xticks=x,xticklabels=['LF','LM','LH','RF','RM','RH'],ylabel='Material slip / body forward speed')
    axes[1].legend(ncol=2,fontsize=8);fig.savefig(OUT/'ROUND4_COMPARISON.png',dpi=150);plt.close(fig)
    text='''# S5 第四轮：实际物理搜索与严格逐足滑移验收

当前目标仍未完成。保持原完整71mesh、自由根、重力/接触/摩擦、原力矩上限、30rad/s增量速率与有限PD128，通过真实物理而非FK/静态推断筛选训练参考。所有本轮新评分/教师结果均不算神经自主成果；旧11s替代ANN基线保留，滑移失败结论不变。

## 有限搜索预算与独立记账

第一批13次：原基线+6个实际饱和相关关节各±.02rad，3s/条，实际3,120,000物理步。其最优RH_Trochanter_yaw +.02rad另做4s完整原生严格验收，四足通过材料滑移门，两个后足未通过。

第二批13次：以第一批已实测中心为基准，每个坐标各±.01rad，总角偏界仍±.03。因为最终要求每条腿达标，搜索目标改为最差足滑移比+.1均值+原安全/进度惩罚；最终物理门不放宽。全部候选实际运行，不拼接未经测试的坐标收益。总26次搜索=6,240,000原生物理步，全部有限/无警告/完整3s，所有评分和失败项保留。

独立审计重新计算每条所有评分字段、预算、参数、顺序、同原力矩界与BEST等于实际最优的条件，全部通过。搜索5ms接触/100ms进度平滑只作比较，候选必须另用完整12.5µs支撑记录和0.5ms材料速度重放，不靠平均目标宣布逐足通过。

## 严格物理结果

| 条件 | 全D门 | 六足持续迈步 | 直行门 | 前进native/s | 最大材料滑移比 | 全足滑移门 |
|---|---|---|---|---:|---:|---|
'''
    for r in rows:
        text+=f"| {r['name']} | {r['D']} | {r['steps']} | {r['heading']} | {r['motion']['cycle_mean_forward_speed']:.5f} | {max(r['slip_ratios']):.5f} | {r['material_pass']} |\n"
    text+='''
第二批最优是已实测RH_yaw+.02、RH_roll+.01，不是独立效果的未测相加。所有严格重放从原初态重新积分，仅施关节受限力矩；实际位姿/速度/COM/足位置/力矩/抽样接触与保存轨迹一致。评分接近门槛不算通过，非足接触不能算足支撑。

半步频控制保留首批同几何/偏移/gain128，只改1→.5Hz；仍未消除后足滑移，不扩大频率扫描。下一单独参考只降低宽抬脚的空中幅值.25→.18，保留先抬后摆的XY/步序/支撑段；真实全部跗节平台清距仍要求≥.1native，离线独立几何/原速率门通过才送物理。它不改变摩擦、质量、重力、力矩上限或身体实际root。

## 证据与限制

- 第一批原始记录与审计：`/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T112003Z/`
- 第二批原始记录与审计：`/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T113529Z/`
- 结构化本轮结果：`/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T113529Z/ROUND4_STATUS.json`
- 对照图：`/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T113529Z/ROUND4_COMPARISON.png`
- 仍有效的11s替代ANN视频：`/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T103607Z/ann_12s/WALK_4x_slow.mp4`

首次第二批严格运行bash-106因命令遗漏python解释器而shell exit126，零实际物理；错误日志保留，补上原解释器bash-107运行exit0。首批审计JSON numpy布尔编码失败exit1，外部strict numpy编码适配器保留原审计代码和评分/check，独立出处已记录。没有通过chmod、模型改动或改历史哈希绕过。

本轮尚未训练选中参数的新ANN。原11s网络仍只是通过支撑/步序/直行而未通过材料滑移，不能把教师或搜索改进归功于旧网络。目标持续active，未complete/blocked；绝对SI力、原连接体必要性和生物有效性NOT_ESTABLISHED。
'''
    p=OUT/'ROUND4_REPORT.zh-CN.md'
    if p.exists():p.read_text()
    p.write_text(text)
    print(json.dumps(dict(report=str(p),complete=False,search_trials=26,strict_conditions=len(rows)),ensure_ascii=False))

if __name__=='__main__':main()
