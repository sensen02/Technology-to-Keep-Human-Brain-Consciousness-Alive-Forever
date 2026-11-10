"""Summarize real physical sensory-ANN training and strict teacher-free evaluation."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T123634Z')


def read(p):return json.loads(p.read_text())


def main():
    final=read(OUT/'FINAL_STATUS.json');audit=read(OUT/'AUDIT_TRAINING.json');logs=read(OUT/'TRAIN_LOG.json')
    assert final['counters']['neural_policy_decisions']==8192 and all(audit['checks'].values())
    rows=[]
    for tag in ['ann_residual_u08','ann_residual_u16','ann_residual_u24','ann_residual_u32','ann_residual_u32_12s','neural_cut_u32']:
        folder=OUT/'evaluation'/tag;v=read(folder/'VERIFICATION.json');g=read(folder/'GOAL_ASSESSMENT.json');r=read(folder/'RESULT.json')
        slip=v['physics_replay'].get('material_contact_velocity',{}).get('slip_to_forward_speed_ratio')
        rows.append(dict(tag=tag,folder=str(folder),controller=r['controller'],release_s=r['release_seconds'],
            D=v['support']['passed'],support=v['support']['release'],steps=v['gait']['all_legs_stepping'],
            heading=all(g['heading_checks'].values()),motion=v['gait']['locomotion'],material_pass=v['material_slip_passed'],
            slip_ratio=slip,replay_performed=v['physics_replay']['performed'],replay=v['physics_replay'].get('passed'),
            both_networks=v['policy']['passed']))
    long=rows[4];cut=rows[5]
    status=dict(stage='S5 learnedproprioceptiveresidualANN actualphysicsrewardtraining',complete=False,
        training=final,audit=str(OUT/'AUDIT_TRAINING.json'),logging_defect=audit['raw_metric_integrity'],
        base_model='Frozen111149learnedvelocity-lossmotorANN',residual='477->128tanh->128tanh->6tanh*.003, current183proprio+294owncausalhistory',
        deployed_model=str(OUT/'MODEL_RESIDUAL_u32.npz'),results=rows,
        outcome='Learnedphysicalfeedback controllerholdsfullbody/realsteps/nearstraighttravel for11safterteacherwithdrawal, actualmaterialslipstillfails. Notfinalstablewalkingpass.',
        originalbrain='Notrun; replacementlearnednetworks andlocalboundedPDactuator, notoriginalconnectomecapability',
        deployment='First1swarmteacherthenweights+actualproprio+ownhistory only, no time/phase/lookup/teacher/contact-controlformula',
        next='Declaredepisode-boundarycontinuation fromsameactor+Adam+Torch/NumPyRNG, fixed64moreupdates, unchangedphysicalcriteria; copy-onlyloggingfix andper-rolloutGAEarrays beforefuturetraining',
        units='native; absoluteSIforce/originalbrainnecessity/biologicalvalidity NOT_ESTABLISHED')
    p=OUT/'NEURAL_STATUS.json'
    if p.exists():p.read_text()
    p.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    fig,axes=plt.subplots(3,1,figsize=(12,11),constrained_layout=True)
    axes[0].plot([x['update'] for x in logs],[x['reward_mean'] for x in logs],'o-',ms=3)
    axes[0].set(xlabel='PPO update',ylabel='Measured training reward',title='8192 actual neural decisions; reward is not walking acceptance')
    x=np.arange(6)
    for i,r in enumerate(rows[:4]):axes[1].bar(x-.3+i*.2,r['slip_ratio'],width=.2,label=r['tag'])
    axes[1].axhline(.5,color='r',ls='--',label='unchanged perleg bound0.5');axes[1].set(xticks=x,xticklabels=['LF','LM','LH','RF','RM','RH'],ylabel='Material speed / forward speed');axes[1].legend(ncol=3,fontsize=8)
    for r in [long,cut]:
        z=np.load(Path(r['folder'])/'BODY.npz');m=z['time']>=1
        axes[2].plot(z['com'][m,0],z['com'][m,1],label=r['tag'])
    axes[2].set(xlabel='World x (native)',ylabel='World y (native)',aspect='equal');axes[2].legend()
    fig.savefig(OUT/'NEURAL_COMPARISON.png',dpi=150);plt.close(fig)
    s=long['support'];m=long['motion']
    text=f'''# S5 感觉反馈残差ANN：11秒撤教师验证，材料滑移仍失败

**完整目标未完成。** 本轮已从单纯离线角度模仿，推进为在原完整身体真实物理中训练感觉反馈网络。最终冻结权重在1s物理教师预热后，连续11s仅由两个学习网络及原有限增量执行器驱动，承重、六足持续迈步、前进和直行通过；实际足地材料滑移仍未通过。

## 真实神经训练及接口

- 冻结111149速度误差训练的基底运动ANN（40帧自身关节目标历史、7个lags），追加477→128tanh→128tanh→6tanh残差ANN。输入是183当前真实本体感觉+294因果自身目标历史；排除时间、步态相位、世界位置、root高度、脚锚和未来教师状态。
- 输出只修正六个关节的下一目标，每关节≤.003rad；所有42关节仍经过原30rad/s增量、1ms积分、有限PD128及原native力矩上限。真实物理12.5µs，原71mesh/49/48/42/neq0、自由root、原重力/摩擦/接触/质量保留。
- 实际完成32×256=8192次神经决策，6次真实1s预热、6个episode，3,756,800个原生步，0触发终止失败。四个模型导出，Torch/NumPy差≤3.6e-7；零残差64段因果历史验证与原基底动作逐位一致，没有零残差仍偷偷加控制。
- 训练奖励依据实际接触材料速度/足载/前进/倾斜等；它只是学习目标，未被当作最终D/逐足滑移/直行验收。

## 最终模型11秒物理结果

- 全11s原生D门通过；足载平均{s['foot_support_mean']:.6f}W，最低COM{s['com_z_min']:.6f}native，无非足承重或自由落体。
- 周期平均身体前进{m['cycle_mean_body_forward_travel']:.6f}native，速度{m['cycle_mean_forward_speed']:.6f}native/s；真实净世界x位移约{m['raw_net_xy'][0]:.6f}native。未将周期身体摆动路径当额外进度。
- 11s真实偏航变化{m['heading_change_rad']:.6f}rad，直线性{m['straightness']:.6f}，直行门通过。各足反力记录确认持续卸载/着地，并非悬空摆腿或仅站立。
- 独立重放从原初态重新积分实际关节力矩；全身体/命令/两个ANN输出、183感觉在重放物理状态上的重建、目标历史交接最大误差0.0。
- 统一0.5ms实际材料接触点速度（Jacobian×qvel与世界刚体twist交叉核对），六足滑移比 **{', '.join(f'{x:.4f}' for x in long['slip_ratio'])}**，门槛逐足≤.5；未全部通过，目标不能标完成。
- 匹配切ANN输出为零增量，保留原目标记忆/有限PD：D门{cut['D']}，持续六足迈步{cut['steps']}，周期前进{cut['motion']['cycle_mean_body_forward_travel']:.8f}native。切断臂为数组/输出复核，没有额外完整物理重放。原已验证断力矩塌陷证据仍保留，但没有伪称本轮重跑断力矩。

## 四个冻结检查点没有挑隐失败窗

| 检查点 | 原D | 实际六足迈步 | 直行 | 最差材料滑移比 | 全足滑移 |
|---|---|---|---|---:|---|
'''
    for r in rows[:4]:text+=f"| {r['tag']} | {r['D']} | {r['steps']} | {r['heading']} | {max(r['slip_ratio']):.5f} | {r['material_pass']} |\n"
    text+='''
所有四个4s物理和完整独立重放均保留。最终网络不是根据短窗“最优分”冒充达标，只测持续性；微小改善与仍失败的LH分别单列。

## 训练审计缺陷与恢复范围

独立审计确认8192决策/原生步/命令/历史/183感觉/logprob/已知检查点前向和四导出hash。发现原训练辅助字段body_omega保存了qvel的view，写盘时变为该episode最终角速度。原奖励在每步当时即时计算正确，保存的BODY_qvel逐步独立copy也正确，因此审计从BODY_qvel[:,3:6]重新计算奖励，误差在数值容差内；原日志、源码和模型没有覆盖或追改。

未保存全部中间critic权重或逐步nextvalue/GAE，所以不能独立重放每次PPO更新的全部GAE；当前只验证公式/重置机制和可得摘要。下一次新目录继续训练会复制辅助速度字段，并新增每rollout nextvalue/advantage/returns记录，不能伪造历史缺失。全部训练记账状态明确WITH_LOGGING_DEFECT，不写成无缺陷验收。

## 同次视频和后续

- 视频（48s，4倍慢放，对应1s预热+11s自主）：`/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T123634Z/evaluation/ann_residual_u32_12s/WALK_4x_slow.mp4`
- 不同步态相位：`/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T123634Z/evaluation/ann_residual_u32_12s/PHASE_PREVIEW.png`
- 独立感觉/网络/物理复核：`/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T123634Z/evaluation/ann_residual_u32_12s/VERIFICATION.json`
- 训练审计：`/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T123634Z/AUDIT_TRAINING.json`
- 本轮神经结果图：`/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T123634Z/NEURAL_COMPARISON.png`

71mesh不透明，视频从同次BODY轨迹渲染，没有生成动作、脚本位移或改写物理。实际打开八个不同接管步态相位；标题标明滑移未通过。不是逐帧目视全部1440帧。

下一步保留当前基底和残差，明确在训练episode边界恢复网络、Adam与随机状态，物理重新做原初始化及1s预热，按固定64额外更新继续实际奖励训练；不改PPO奖励/超参/输出幅度/模型接触，不在部署恢复教师。目标保持active。此臂为人工替代运动网络，不是原连接体能力；绝对SI力、原脑必要性、生物有效性NOT_ESTABLISHED。
'''
    p=OUT/'NEURAL_REPORT.zh-CN.md'
    if p.exists():p.read_text()
    p.write_text(text)
    print(json.dumps(dict(report=str(p),complete=False,neural_release_seconds=11.,material_pass=long['material_pass']),ensure_ascii=False))

if __name__=='__main__':main()
