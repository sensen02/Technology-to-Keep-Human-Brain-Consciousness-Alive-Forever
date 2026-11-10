"""Report dose selection and frozen held-out phase comparisons."""
from pathlib import Path
import json,hashlib
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T182927Z')
C=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-neural-joint-torque/20261105T182927Z')
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
s=js(O/'SUMMARY.json');v=js(O/'VERIFICATION.json');best=s['best'];assert v['total']==19
rows=[]
for r in sorted(s['dose_results'],key=lambda r:r['dose']):
    rows.append(f"| {r['dose']} | {r['J']:.6f} | {r['parts']['foot_support_mean']:.4f} | {r['parts']['com_z_min']:.4f} | {1000*r['parts']['first_nonfoot_s']:.3f} | {r['RH']['foot_load_mean']:.4f} |")
phase=[]
for pair in s['phase_pairs']:
    b=pair['baseline'];t=pair['treated'];phase.append(f"| {pair['release_time_s']:.3f} | {b['J']:.6f} | {t['J']:.6f} | {pair['delta_J']:+.6f} | {b['parts']['foot_support_mean']:.4f} → {t['parts']['foot_support_mean']:.4f} | {t['validation']['passed']} |")
deltas=[r['delta_J'] for r in s['phase_pairs']];improved=sum(d<0 for d in deltas)
text=['# S5 右后腿兴奋输入剂量与释放相位复验','',
'**本轮19次评估完成；行为结论仍为未自主支撑/行走。** 只改变既有右后腿运动细胞的入兴奋连接强度，执行器上限、身体、读出矩阵、感知编码及验收门均不变。',
'', '## 剂量选择：固定1.300s交接',
'| RH入兴奋倍数 | J↓ | 平均足载/W | 最低质心 | 首次非足触地/ms | RH足载/W |','|---|---:|---:|---:|---:|---:|']+rows+['',
f"七个预登记剂量中选择{s['selected_dose']}，J={best['J']:.6f}。两次重复完整BODY数组和J逐位一致；原1.0与0.5倍分别逐位复现此前结果。选择只使用1.300s交接数据。较低J并非所有行为指标都改善，完整D1-D8见RESULT。",
f"最优全读出切断J={s['cut']['J']:.6f}、感觉切断J={s['sensory_cut']['J']:.6f}。神经与反馈依赖仍在，但不足以证明足部稳定承载。",
'', '## 留出相位：固定选中剂量，不重新调参',
'交接时间1.225/1.250/1.350/1.375s，对应同一算法预热步态的不同相位。每个时间点使用自己连续形成的物理、神经和滤波状态，做未干预与固定选中剂量成对比较。初态不同是有意测试；同一相位内两臂起始状态相同。每条释放仍为0.2s，辅助参考取该次预热末0.2s。',
'| 交接时间/s | 未干预J | 固定剂量J | ΔJ（负为改善） | 足载/W：基线→干预 | 干预动态门 |','|---|---:|---:|---:|---|---|']+phase+['',
f"四个留出相位中{improved}个J改善；平均ΔJ={np.mean(deltas):+.6f}，最差ΔJ={max(deltas):+.6f}。这些是四个确定性交接状态的配对差异，不是独立动物样本或统计显著性证据；不能据平均分掩盖最差相位。",
'所有19次完整动态门均未通过。即使固定剂量跨相位改善，也只能称该设置的支撑损失改善，不能称稳定站立/行走。足载、非足接触、自由落体、最低高度和承载腿数仍分别验收。',
'关键限制：留出相位干预后J仍3.40–3.64、足载仅0.204–0.335W，明显差于选参相位。改变释放时刻同时改变身体姿态/速度与SNN状态，因此本轮证明的是交接状态依赖，不能把全部差异归因于步态相位或单独神经记忆。下一步应先用单列诊断对照区分这两类状态的贡献，而不是依据选参相位继续宣称稳健提升。',
'', '## 复验与实现细节',
'- 原运行器只在内存派生的预热方法中改变实际停止时间与最后0.2s辅助参考窗起点；原源码不改写，WARMUP_ADAPTER.json登记来源与哈希。SNN仍从0.2s开始接收实际感知，身体在交接时不重置。',
'- 原损失函数写死交接时间1.3s；本轮仅在评分用的数组副本中把时间轴平移到该原点，保存原生记录保留实际时间。独立验证器从各相位真实参考窗重算J与D1-D8，验证一致，没有把不同相位的首次接触时间错算成改善。',
'- 19条完整轨迹的滤波率→力矩重算、神经干预身份、交接qpos/qvel连续、感觉切断零电流、0.2s释放时长及12.5us原生时间步全部通过；释放每步辅助力和外力为0，零MuJoCo警告，输入哈希不变。',
'- 单进程串行：五次实际预热时长之和6.5s，19×0.2s释放，总10.3模拟秒。全部命令退出0；没有并行重任务。',
'- 交付1.300s相位的完整预热+释放三列视频（未干预/选中剂量/读出切断），301帧25fps，8倍慢放。71mesh不透明且物理数组不变，重放质心误差0；视频首/中/末帧解码通过。图像工具不支持当前模型图像输入，人工视觉核验未完成；不将数值复验称为肉眼确认。',
'- 绝对力值与SI仍NOT_ESTABLISHED；工作plant是此前辅助包络匹配的理想力矩执行器，部分能力高于肌肉，本轮未提高限幅。',
'', '## 下一步',
'根据留出相位最差表现决定下一轮：若方向不一致，优先解释交接姿态/神经状态依赖；若一致但仍不达标，则固定这一局部改善，审计其它腿的真实负载与驱动，避免把RH剂量照搬到六腿。当前结果不授权省略失败门或用外置控制器接管。',
'', '## 完整路径与重要程度',
f'- 核心必需：`{O}/PREREGISTRATION.json`、`{O}/FROZEN_SELECTION.json`、`{O}/SUMMARY.json`、`{O}/VERIFICATION.json`、`{O}/MANIFEST_SHA256.json`；所有PHASE目录中的预热、原生身体/神经/接触、候选干预参数和实际交接时间身份。',
f'- 核心必需代码：`{C}/dose_phase.py`、`{C}/verify_results.py`。重要辅助：`{C}/report.py`。',
f'- 重要辅助：`{O}/REPORT.zh-CN.md`、`{O}/DOSE_PHASE.png`、`{O}/PHASE_1.300/BASELINE_DOSE_CUT_8x_slow.mp4`、`{O}/PHASE_1.300/VIDEO_SAMPLES.png`。',
'- 实际清理：没有删除文件。全部剂量、留出相位、切断和重复结果仍为证据依赖；没有散落临时帧。','']
(O/'REPORT.zh-CN.md').write_text('\n'.join(text))
dump(O/'CLEANUP_LEDGER.json',dict(deleted=[],reason='All dose/phase/cut/repeat artifacts remain necessary evidence; no temporary frame files.',importance='重要辅助'))
manifest={str(p):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='MANIFEST_SHA256.json'};manifest.update({str(p):sha(p) for p in C.glob('*.py')});dump(O/'MANIFEST_SHA256.json',manifest);assert all(sha(Path(p))==h for p,h in manifest.items())
print('manifest',len(manifest),'mismatches0');print('\n'.join(rows));print('\n'.join(phase))
