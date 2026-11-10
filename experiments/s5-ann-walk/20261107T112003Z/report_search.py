"""Visualize physically measured candidate effects and separate strict acceptance."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T112003Z')


def main():
    data=json.loads((OUT/'SEARCH_RESULTS.json').read_text());best=json.loads((OUT/'BEST_PARAMETERS.json').read_text())
    audit=json.loads((OUT/'AUDIT_SEARCH_RESULTS.json').read_text())
    assert len(data['results'])==13 and all(audit['checks'].values())
    verify=json.loads((OUT/'selected/teacher_selected_4s/VERIFICATION.json').read_text())
    scores=np.array([r['score']['objective'] for r in data['results']])
    ratio=np.array([r['score']['slip_to_progress_ratios'] for r in data['results']])
    tags=[r['tag'].replace('joint_','').replace('Trochanter','Troch') for r in data['results']]
    fig,axes=plt.subplots(2,1,figsize=(12,10),constrained_layout=True)
    colors=['tab:orange' if r['tag']==best['selected']['tag'] else 'tab:blue' for r in data['results']]
    axes[0].barh(np.arange(13),scores,color=colors);axes[0].set(yticks=np.arange(13),yticklabels=tags,xlabel='Declared search objective (lower better)')
    axes[0].invert_yaxis();axes[0].set_title('13 actual physics diagnostics; average objective does not establish all-feet acceptance')
    im=axes[1].imshow(ratio.T,aspect='auto',vmin=0,vmax=max(1.5,float(ratio.max())),cmap='viridis')
    axes[1].set(xticks=np.arange(13),xticklabels=tags,yticks=np.arange(6),yticklabels=['LF','LM','LH','RF','RM','RH'])
    axes[1].tick_params(axis='x',labelrotation=60);fig.colorbar(im,ax=axes[1],label='Sampled slip/progress ratio (diagnostic5ms)')
    fig.savefig(OUT/'SEARCH_EFFECTS.png',dpi=150);plt.close(fig)
    g=verify['gait']['locomotion'];slip=verify['physics_replay']['material_contact_velocity']['slip_to_forward_speed_ratio']
    text=f'''# 有限实际物理评分：13次单项参考对照

全部13次预登记对照完成，实际3,120,000个12.5µs物理步，无额外根部驱动、外力或摩擦/重力/力矩上限改变。每条3s，前1s预热，后2s以5ms实际接触材料速度和100ms端点平均COM评分；搜索记录不是正式验收。

独立审计重算全部评分字段、参数、顺序、数量、命令/身体记录与源码hash通过。最优单项是RH Trochanter yaw目标+0.02rad，评分{best['selected']['score']['objective']:.6f}，原基线{best['baseline']['score']['objective']:.6f}。没有把独立坐标收益直接拼成未测参数。

## 严格重跑

选中项另用4s完整原生记录重跑，评分窗1–4s，不直接用稀疏搜索分数收口。全部原生D门、六足持续卸载/着地、直行通过；周期平均前进{g['cycle_mean_forward_speed']:.6f}native/s，3s航向变化{g['heading_change_rad']:.6f}rad，横/前向比约{abs(g['cycle_mean_body_lateral_travel'])/g['cycle_mean_body_forward_travel']:.6f}。独立物理重放/命令重建最大误差0.0。

每0.5ms实际材料滑移速度/身体前进速度为{', '.join(f'{x:.4f}' for x in slip)}。四足通过0.5门，LH/RH仍失败。故**搜索有改善，稳定行走目标仍未完成，且这是算法教师不是ANN接管**。

## 失败和限定

- 搜索平均目标可能改善部分足、掩盖最差足，最终验收坚持逐足原门。下一次仅有明确13次局部精调预算，以max逐足滑移比为主要搜索目标，中心是本次实际测得单项；每个新参数都必须实际跑，不假设线性组合收益。
- 搜索原生PD/forward逐物理步与严格runtime相同，显式目标nativeclip造成原基线与旧独立轨迹出现极小浮点差异（qpos约2.7e-10、qvel约2.7e-7），没有伪称逐位一致。严格候选用自己的完整原生记录独立重放。
- 稀疏5ms接触可漏掉短事件，搜索分数不能替代全12.5µs支撑或0.5ms真实材料速度。两个后足的实际失败不因平均分数下降而消失。
- 审计器NumPy布尔JSON序列化exit1未写产物，外部严格numpy-scalar编码adapter保存并执行，原审计源码/评分逻辑未改；完整provenance保留。

主要证据：

- `/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T112003Z/PREREGISTRATION.json`
- `/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T112003Z/SEARCH_RESULTS.json`
- `/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T112003Z/BEST_PARAMETERS.json`
- `/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T112003Z/AUDIT_SEARCH_RESULTS.json`
- `/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T112003Z/selected/teacher_selected_4s/VERIFICATION.json`

源码 `/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T112003Z/`；bash-99搜索exit0，bash-100严格物理exit0，bash-101独立重放exit2表示逐足滑移失败。目标保持active；绝对SI力/生物有效性/原脑必要性NOT_ESTABLISHED。
'''
    p=OUT/'REPORT.zh-CN.md'
    if p.exists():p.read_text()
    p.write_text(text)
    print(json.dumps(dict(report=str(p),trials=13,physical_goal_complete=False)))

if __name__=='__main__':main()
