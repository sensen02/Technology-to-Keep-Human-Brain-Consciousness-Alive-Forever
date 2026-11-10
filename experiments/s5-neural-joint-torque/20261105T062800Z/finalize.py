"""Produce the report and manifest from validated pilot artifacts."""
from pathlib import Path
import json,hashlib
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T062800Z')
C=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-neural-joint-torque/20261105T062800Z')
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
s=js(O/'SUMMARY.json');cov=js(O/'COVERAGE.json');ver=js(O/'VERIFICATION.json')
rows=[]
for label,r in [('P0未训练',s['P0']['baseline']),('P0训练后',s['P0']['best']),('P1读出切断',s['P0']['cut']),('P2随机组最优',s['P2']['best'])]:
    p=r['parts'];rows.append(f"| {label} | {r['J']:.6f} | {p['foot_support_mean']:.4f} | {p['com_z_min']:.4f} | {1000*p['first_nonfoot_s']:.3f} | {p['bearing_legs_median']:.0f} |")
allres=[js(p) for p in O.glob('*/RESULT.json')]
uncovered=[r['joint'] for r in cov['joints'] if not(r['positive_pools'] or r['negative_pools'])]
lines=['# S5 神经→关节力矩首轮试验（手动研究原型）','',
'**结论：闭环运行和切断效应已验证；动态支撑失败，未自主行走。**',
'采用新目录、固定模型和既有66池/204运动神经元，神经网络12000细胞。P0/P2各17次训练评估（基线+16候选），各一次逐位重复和一次切断，共38次。只有两次预热实际积分；读出不参与预热，因此其完整物理、神经和滤波状态可按臂缓存，释放阶段全部重新闭环积分。实际积分时长2×1.3+38×0.2=10.2模拟秒。',
'', '| 条件 | J（越低越好） | 平均足载/W | 最低质心 | 首次非足触地/ms | 承载腿数中位 |','|---|---:|---:|---:|---:|---:|']+rows+['',
'同次辅助窗足载均值1.0000123W、质心均值1.2374997；D3门槛约0.9000W，D5约1.1375，D8约0.9658。足载不能把非足地面力计入。所有候选均未通过完整动态支撑判据。',
'P0的J下降0.304440，切断后J为4.106615，说明神经读出影响了物理轨迹；两者都不能支撑，不能宣称C1承重因果门通过。P2最优3.097709，略优于P0的3.111251，因此预登记C2未通过。本轮单种子、小预算不构成统计显著性证明，更不能据此断言真实连接组无用。历史肌肉路线时间步敏感性带宽不能当作新plant的误差界。',
'', '## 关键结构发现',
'按已登记的主导关节映射，42关节中10个没有读出、18个仅单向、14个双向。非负池权重和关节增益无法填补零行或产生缺失方向。本轮没有补造连接。缺失关节：'+', '.join(uncovered)+'。',
'符号来源重新核查：原构建器实际取的是同姿态激活从0到1引起的关节力矩增量，而不是裸actuator_moment；所以没有遗漏肌肉拉力的符号。旧说明“moment”措辞不精确，本报告作追加澄清，不改历史文件。共享池对同一关节同一方向合并一次；有冲突立即中止，本轮无冲突。',
'下一最小动作：在新登记中核查原模型完整42×90力矩增量矩阵，评估主导关节截断丢掉了哪些已有作用；只有模型支持的映射才能引入。先验证双向覆盖，再考虑离线初始化与闭环训练。零行存在不证明其它姿态或控制策略必然失败，也不能把本轮失败唯一归因于覆盖。',
'', '## 工程核验与限制',
'- 38条轨迹均验证有限值、指令来自已完成神经窗、读出公式可重算、物理与神经交接连续；原始接触记录为12.5微秒。P0/P2最优重复的完整身体数组和J逐位一致。',
'- 撤辅窗口qfrc_applied/xfrc_applied逐步断言为零；MuJoCo警告为零。D1用正非足力最大值判断，避免旧辅助函数均值口径掩盖接触。D6沿用原生时间步低通差分。',
'- 预热用明确标记的算法力矩，前0.2秒无神经预热，0.2–1.3秒SNN接收真实身体感知但读出暂不施力；1.3秒起只有神经力矩，无PD、偏置、目标角度或外部周期发生器。',
'- 神经/感知固定为现有默认参数，直接载入归档A/C循环矩阵；没有继承旧肌肉训练中的阈值、池增益或感知权重。因此不能与旧肌肉训练分数作单变量比较。',
f"- 仿真进程已记录峰值RSS最高{max(r['peak_rss_MiB'] for r in allres):.1f} MiB；串行执行。训练与验证/渲染命令均退出0。旧自动goal仍exhausted，未启用自动续跑。",
'- 输入文件运行前后以及交付前哈希一致；每个候选独立时间戳目录，参数变更关联上一参数文件和新文件哈希。保留全部失败结果，没有挑选性删除。',
'- 工作plant上限来自实际辅助力矩包络，部分关节超过肌肉（最高3.62倍）。省略肌肉被动力、激活动力学和动态力臂映射，不是果蝇肌肉能力证明。绝对力值与SI：NOT_ESTABLISHED。',
'', '## 桌面样本与视觉核验状态',
f"- `{O/'P0_P1_P2_8x_slow.mp4'}`：全程1.3秒算法预热+0.2秒释放，301帧，25fps，8倍慢放。三列P0/P1/P2，相机仅随根部平移取景，不修改身体轨迹。",
f"- `{O/'COMPARISON.png'}`：撤辅后质心、足载、正非足力。",
f"- `{O/'VIDEO_SAMPLES.png'}`：视频抽帧。",
'- 模型71个mesh全部设不透明，物理数组哈希未变；每帧从保存qpos直接forward，质心重放误差0；视频已解码检查首/中/末帧及总帧数。',
'- **人工视觉核验尚未完成**：已调用read_image打开两张图，但当前gpt-6-astra运行时不声明图像输入，两次工具均拒绝。不能声称已肉眼确认姿态、遮挡或足部可见性。',
'', '## 文件重要程度与实际清理',
f"- 核心必需：`{O/'PREREGISTRATION.json'}`、`{O/'CHANGE_LEDGER.json'}`、`{O/'SUMMARY.json'}`、`{O/'COVERAGE.json'}`、`{O/'VERIFICATION.json'}`、`{O/'MANIFEST_SHA256.json'}`；所有候选目录的参数、预登记、BODY/NEURAL/NATIVE/BOUNDARY和RESULT，及WARM_P0/WARM_P2，均用于重算和复现。",
f"- 核心必需：`{C/'run_trial.py'}`、`{C/'verify_and_render.py'}`。重要辅助：`{C/'finalize.py'}`、本报告、对照视频/图片、逐帧来源表和可视化变更表。",
'- 本轮未产生需要删除的废弃临时文件；未清理文件。所有失败候选仍有审计用途，保留。没有修改旧封存目录。','']
(O/'REPORT.zh-CN.md').write_text('\n'.join(lines))
dump(O/'CLEANUP_LEDGER.json',dict(deleted=[],reason='No disposable intermediate files generated. All failed candidates retained for audit; no old sealed files modified.',importance='重要辅助'))
manifest={str(p):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='MANIFEST_SHA256.json'}
manifest.update({str(p):sha(p) for p in sorted(C.glob('*.py'))})
dump(O/'MANIFEST_SHA256.json',manifest)
assert all(sha(Path(p))==h for p,h in manifest.items())
print('manifest',len(manifest),'files, mismatches 0')
print('\n'.join(rows))
