"""Report targeted RH neural interventions, including preserved audit console failure."""
from pathlib import Path
import json,hashlib
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T181843Z')
C=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-neural-joint-torque/20261105T181843Z')
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
s=js(O/'SUMMARY.json');a=js(O/'CHAIN_AUDIT.json');rh=next(r for r in a['per_leg'] if r['leg']=='RH');best=s['best'];base=s['baseline'];verification=js(O/'VERIFICATION.json');recovery=js(O/'AUDIT_RECOVERY_VERIFICATION.json')
assert verification['total']==15 and recovery['persisted_spike_load_and_foot_trajectory_recomputed']
rows=[]
for r in s['results']:
    contact='无' if r['RH']['first_contact_s'] is None else f"{1000*r['RH']['first_contact_s']:.3f}"
    nf='无' if r['parts']['first_nonfoot_s'] is None else f"{1000*r['parts']['first_nonfoot_s']:.3f}"
    rows.append(f"| {r['label']} | {r['J']:.6f} | {r['parts']['foot_support_mean']:.4f} | {r['RH']['spikes']} | {r['RH']['foot_load_mean']:.4f} | {contact} | {nf} |")
identity=js(Path(best['folder'])/'NEURAL_INTERVENTION.json');intervention=js(identity['file'])
text=['# S5 右后腿神经活动与失载因果诊断','',
'**结论：右后腿并非整体无放电；降低其既有兴奋性输入改善了接触和全身支撑，但仍未自主支撑/行走。** 本轮只做冻结读出下受限因果干预，没有改变模型上限、感知编码、预热或添加外置运动控制器。',
'', '## 审计事实',
f"基线右后腿32个运动细胞中{rh['active_cells']}个放电，释放0.2s内{rh['spikes_full']}次尖峰（前60ms {rh['spikes_first60ms']}次）；11个池中仅1个滤波率全窗低于1e-9。平均池率{rh['mean_pool_rate_hz']:.3f}Hz，右后腿指令限幅占比{rh['saturation_fraction']:.6f}，未限幅峰值相对执行器上限{rh['max_raw_normalized']:.3f}。这不是整腿没有命令。",
f"真实原生接触记录显示右后腿平均足载0、触地占比0。足端Tarsus5刚体原点高度从{rh['foot_origin_z_at_release']:.6f}升至60ms时{rh['foot_origin_z_at_60ms']:.6f}，变化{rh['foot_origin_displacement_60ms'][2]:.6f}模型单位。刚体原点不是脚底间隙，是否承重依据接触力而非高度。",
'静默细胞为1317、7502、11415，仅描述这条轨迹，不等于这些细胞先天无功能。逐池及右后腿逐关节的正/负读出贡献见CHAIN_AUDIT.json和PER_POOL.csv。',
'', '## 因果干预（相同完整释放初态）',
'所有神经干预仅在1.3s释放边界修改参数，不重置膜电位、电导、延迟队列、滤波率或身体。第一窗仍使用已完成神经窗的读出，新的参数影响按正常神经/指令时钟进入身体。每条运行独立新时间戳候选配置，跑前登记。',
'| 干预 | J↓ | 全身足载/W | RH尖峰 | RH足载/W | RH首次接触/ms | 首次非足触地/ms |','|---|---:|---:|---:|---:|---:|---:|']+rows+['',
f"最优：{s['best_intervention']}。仅对32个右后腿运动细胞对应的We入连接行乘0.5，实际改变{intervention['synapses']['We']['changed_weights']}个现有权重。We/Wi的索引、稀疏结构和符号未变，非目标细胞入连接未变；不等于隔离网络，因为目标细胞输出仍能通过原回路影响其它细胞。",
f"J从{base['J']:.6f}降到{best['J']:.6f}；全身足载{base['parts']['foot_support_mean']:.6f}→{best['parts']['foot_support_mean']:.6f}W；最低质心{base['parts']['com_z_min']:.6f}→{best['parts']['com_z_min']:.6f}；首次非足触地{1000*base['parts']['first_nonfoot_s']:.4f}→{1000*best['parts']['first_nonfoot_s']:.4f}ms。右后腿尖峰497→{best['RH']['spikes']}，首次接触{1000*best['RH']['first_contact_s']:.4f}ms，平均足载{best['RH']['foot_load_mean']:.6f}W。",
'阈值下降唤醒更多活动没有解决问题；提高阈值也不能复现减半兴奋性输入的改善。单纯右后腿读出缩放0.25/0.5均更差，乘2只小幅改善，完全切断则恶化。由此不能把所有现象简化成输出总幅度，神经动态及实际身体反馈值得进一步检验；本轮没有分离所有中介路径，也不能判定唯一根因。',
f"最优重复完整BODY数组与J逐位相同；全读出切断J={s['cut']['J']:.6f}，感觉切断J={s['sensory_cut']['J']:.6f}。因果差异是固定初态确定性结果，没有独立种子置信区间。当前改动只针对P0，没有以改变后的网络与此前P2分数比较来声称连接组优势。",
'', '## 仍未达标',
'15条运行全部未通过完整动态门；新最优D1/D2/D3/D4/D4b/D5/D7/D8未通过，仅D6力平衡自洽通过。足载均值约0.711仍低于约0.900要求，最低质心约0.685低于D5约1.1375及D8约0.9658，承载腿数中位2。右后腿重新触地只是部分恢复，不是协调行走。',
'本轮结果支持继续沿已有兴奋/抑制输入平衡做局部、预登记的调参，但不能在未验证下直接把六腿都减半。下一步先在右后腿入兴奋增益0.5附近做剂量响应，并检验是否依赖释放相位；若改善只在单一初态成立，不能升级为稳定行为。',
'', '## 工程验证、错误与资源',
'- 15次评估，单次1.3s实际预热缓存+15次0.2s释放，共4.3模拟秒；单重任务串行。物理12.5us、神经0.5ms、指令1ms。原生接触、真实神经脉冲、滤波率、力矩、完整身体状态全部保存。',
'- 原始基线及最优重复均逐位一致。全部15条指令由保存的神经率与候选矩阵独立重算通过；感觉切断电流恰0；物理交接连续，释放辅助/外力逐步断言0，零MuJoCo警告。输入哈希前后相同。',
'- **自身错误保留**：audit_hindleg.py在末尾控制台json.dumps遇到ndarray，退出1。之前CHAIN_AUDIT/PER_POOL/前后输入哈希已成功保存。未覆盖原源码或旧哈希；AUDIT_EXECUTION.json记错误，独立验证器重新计算全部腿/池尖峰、足载与60ms足端位置，验证通过，见AUDIT_RECOVERY_VERIFICATION.json。不能把首次进程退出码说成0。后续预登记、15次试验、独立验证和报告命令均退出0。',
'- 视频BASELINE_BEST_CUT_8x_slow.mp4三列为原冻结基线/最优干预/全读出切断，均来自本次实际轨迹，含1.3s算法预热+0.2s释放，301帧、25fps、8倍慢放。71mesh不透明，物理数组不变，质心重放误差0；视频首中末帧解码通过。',
'- 图像工具仍因当前gpt-6-astra无图像输入而拒绝打开抽帧，人工视觉核验未完成；不声称肉眼确认姿态或遮挡。',
'- 绝对力值/SI仍NOT_ESTABLISHED。模型使用此前辅助包络匹配的理想力矩上限，本轮未提高上限；神经参数仅工程假设，不是果蝇生理标定。',
'', '## 文件重要程度与清理',
f'- 核心必需：`{O}/CHAIN_AUDIT.json`、`{O}/AUDIT_EXECUTION.json`、`{O}/AUDIT_RECOVERY_VERIFICATION.json`、`{O}/PREREGISTRATION.json`、`{O}/SUMMARY.json`、`{O}/VERIFICATION.json`、`{O}/MANIFEST_SHA256.json`；所有候选干预身份、参数、神经/身体/原生接触轨迹及输入哈希。',
f'- 核心必需代码：`{C}/probe_neurons.py`、`{C}/verify_and_render.py`；`{C}/audit_hindleg.py`保留作首次退出1的可追溯源码。重要辅助：`{C}/report.py`。',
f'- 重要辅助：`{O}/REPORT.zh-CN.md`、`{O}/BASELINE_BEST_CUT_8x_slow.mp4`、`{O}/RH_CHAIN.png`、`{O}/COMPARISON.png`、`{O}/VIDEO_SAMPLES.png`。',
'- 实际未删除文件：失败审计与所有候选仍用于审计/复现；没有散落临时帧或确认无用重复产物。','']
(O/'REPORT.zh-CN.md').write_text('\n'.join(text))
dump(O/'CLEANUP_LEDGER.json',dict(deleted=[],reason='All candidates and original console-failed audit remain required for verification and failure provenance; no temporary frame files.',importance='重要辅助'))
manifest={str(p):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='MANIFEST_SHA256.json'};manifest.update({str(p):sha(p) for p in C.glob('*.py')});dump(O/'MANIFEST_SHA256.json',manifest);assert all(sha(Path(p))==h for p,h in manifest.items())
print('manifest',len(manifest),'mismatches 0');print('best',best['label'],best['J'],best['RH'])
