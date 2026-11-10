"""Final causal diagnosis and escalation report, with all unsuccessful branches retained."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque')
CODE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-neural-joint-torque')
D=ROOT/'20261105T152908Z';E=ROOT/'20261105T153224Z';O=ROOT/'20261105T153657Z'
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
diag=js(D/'DIAGNOSIS.json');edge=js(E/'SUMMARY.json');final=js(O/'SUMMARY.json')
for p in [D,E,O]:assert (p/'VERIFICATION.json').exists()
assert all(not r['validation']['passed'] for p in [D,E,O] for r in [js(f) for f in p.glob('*/RESULT.json')])
rows=[]
for label,r in [('原完整映射最优',diag['results'][0]),('离线初始化+闭环训练P0',edge['P0']['best']),('保留最优后独立权重训练P0',final['P0']['best']),('最终P0读出切断',final['P0']['cut']),('最终P0感觉切断',final['P0']['sensory_cut']),('同预算P2随机组最优',final['P2']['best'])]:
    v=r['parts'];tt='无' if v['first_nonfoot_s'] is None else f"{1000*v['first_nonfoot_s']:.3f}"
    rows.append(f"| {label} | {r['J']:.6f} | {v['foot_support_mean']:.4f} | {v['com_z_min']:.4f} | {tt} | {r['saturation_fraction']:.4f} |")
drows=[]
for r in diag['results']:drows.append(f"| {r['label']} | {r['J']:.6f} | {r['J']-diag['results'][0]['J']:+.6f} | {r['parts']['foot_support_mean']:.4f} |")
allr=[js(f) for p in [D,E,O] for f in p.glob('*/RESULT.json')]
assert len(allr)==118
text=['# 因果诊断与更激进的神经读出训练','',
'**已完成因果干预、离线读出初始化和保留最优的独立权重训练；仍未通过动态支撑验收，未自主行走。** 所有失败分支保留，执行器限幅未提高，SNN仍是释放后唯一指令来源。',
'', '## 因果诊断回答了什么',
'固定原最优控制器，12个干预从同一完整物理、神经、滤波初态释放。各腿切断仅关闭对应既有池的读出列；感觉切断仅将释放期传入SNN的感觉电流置零，不重置网络。',
'| 干预 | J↓ | 相对基线ΔJ | 平均足载/W |','|---|---:|---:|---:|']+drows+['',
'所有干预均未改善原基线J，且都未通过支撑门。输出缩小/放大均更差：高饱和本身不能证明“幅度过大是失撑原因”。感觉切断使J显著数值恶化，证明当前反馈对该轨迹有作用；这是确定性因果差异，不是统计显著性结论，也不证明反馈编码正确。逐腿切断中左中腿影响最大，右后腿较小；影响小不等于可以删除该腿，也不能直接推断单一腿是失撑根因。',
'', '## 已实施的升级',
'升级一：固定174个有模型依据的读出作用和方向，将原66池×42关节因子化增益改为每项独立非负强度。预热真实神经输出作离线输入，实际辅助力矩仅作离线训练目标；训练窗0.3–1.1s，留出窗1.1–1.3s。严格用前一已完成神经窗的滤波率对应下一指令窗，无姿态/时间/相位/目标角度输入、无偏置。每臂5个尺度+16个闭环候选，共21次训练评估；另各4个基线/复验/切断控制，共50次。',
f"P0离线训练MSE={edge['P0']['fit']['train_mse']:.6f}，留出MSE={edge['P0']['fit']['holdout_mse']:.6f}（零预测{edge['P0']['fit']['zero_predictor_holdout_mse']:.6f}），非零系数{edge['P0']['fit']['nonzero_coefficients']}/174。但其闭环最优J={edge['P0']['best']['J']:.6f}，比原2.513150差；拟合误差下降不能当自主控制成功。该分支乘性搜索不能唤醒精确零系数，限制如实保留。",
'升级二：从每臂实际闭环最优（旧控制器或离线分支，取更低J者）继续独立权重训练。采用允许零系数恢复的有界加性扰动，sigma0.15/0.08/0.04，每代8候选，并始终保留最优。每臂25次训练评估+3次复验/切断，共56次。两臂累积训练预算相同。没有强行用较差教师初值覆盖此前结果。',
'', '| 条件 | J↓ | 平均足载/W | 最低质心 | 首次非足触地/ms | 指令限幅占比 |','|---|---:|---:|---:|---:|---:|']+rows+['',
f"最终真实组与随机组J差（随机减真实）={final['P2']['best']['J']-final['P0']['best']['J']:+.6f}。单种子和有限搜索不足以一般性判定连接组优劣；两者都未过行为门。",
'验收足载门槛约0.9000W、D5最低质心约1.1375、D8约0.9658；不能将躯干部触地承载或跌倒位移算行走。完整D1-D8结果见各RESULT.json。绝对力值与SI仍NOT_ESTABLISHED。工作plant部分上限高于原肌肉，最高3.62倍；本轮保持不变。',
f"最终真实组平均足载升至{final['P0']['best']['parts']['foot_support_mean']:.6f}W、首次非足触地推迟到{1000*final['P0']['best']['parts']['first_nonfoot_s']:.4f}ms，但最低质心{final['P0']['best']['parts']['com_z_min']:.6f}反而低于原控制器0.598086；综合J改善并非每项都改善。六腿平均负载（LF/LM/LH/RF/RM/RH）={final['P0']['best']['validation']['release']['per_leg_mean_load']}，右后腿全窗足载0。这个具体失衡值得下一轮跟踪，不能仅凭消融效应小就删掉右后腿。", 
'', '## 验证、资源和边界',
'- 本轮总118次释放评估，分三次预登记：12诊断+50离线初始化/训练+56保留最优细化；共5次实际预热，每个1.3s，释放每次0.2s，总实际积分30.1模拟秒。重任务始终串行。',
f"- 118条保存轨迹独立核验，池率→指令重算通过；感觉切断电流恰为零，读出切断指令恰为零。基线和最优可复现，释放交接状态连续，物理/神经时钟固定；所有释放步辅助力/外力断言0、零MuJoCo警告。进程记录峰值RSS最高{max(r['peak_rss_MiB'] for r in allr):.1f} MiB。",
'- 因果基线与原最优逐位一致。独立系数表示旧控制器时乘法重排允许1e-8容差，实测误差见升级一SUMMARY；后续固定新表示的最优重复均逐位一致。运行前后输入哈希一致。',
'- 自主控制没有在线教师、PD、目标角度、偏置、预设步态或额外神经连接。辅助前1.3s明确标注算法预热；之后仅由真实感知驱动SNN再经线性读出生成力矩。离线拟合和参数搜索只改变读出权重，不在释放时产生动作序列。',
'- 两个升级包各交付同轨迹三臂视频：完整1.3s预热+0.2s释放，301帧、25fps、8倍慢放；71mesh不透明，物理数组不变，质心重放误差0，视频首/中/末帧已解码。',
'- **人工视觉核查未完成**：本会话read_image被当前gpt-6-astra未声明图像输入阻断；已尝试读取新样本并保留工具错误。不把数值和解码检查称为肉眼验证。',
'', '## 下一步',
'本轮只定位了当前闭环对反馈和各腿读出的依赖，没有找到一个能单独修复失撑的开关。更激进的独立读出训练已执行，但不将有限预算失败解释为数学不可能。下一层应把诊断推进到神经活动可达性和固定符号近似：识别释放前后静默/缺方向池、关节支撑力矩与限幅的对应，再登记调整现有神经元阈值、时间常数或已有突触增益的受限实验。继续禁止用外置PD/CPG替代神经元。',
'', '## 文件与重要程度',
f'- 核心必需：`{D}/DIAGNOSIS.json`、`{D}/VERIFICATION.json`；`{E}/SUMMARY.json`和FIT_P0/FIT_P2；`{O}/SUMMARY.json`、`{O}/VERIFICATION.json`；三包所有预登记、变更台账、候选权重、原生接触/身体/神经轨迹、输入哈希及清单。',
f'- 核心必需代码：`{CODE}/20261105T152908Z/causal_diagnosis.py`、`{CODE}/20261105T153224Z/train_edge_readout.py`、`{CODE}/20261105T153224Z/verify_and_report.py`、`{CODE}/20261105T153657Z/train_from_incumbent.py`、`{CODE}/20261105T153657Z/verify_all.py`。',
f'- 重要辅助：`{O}/REPORT.zh-CN.md`、`{O}/P0_P1_P2_8x_slow.mp4`、`{O}/COMPARISON.png`、`{O}/VIDEO_SAMPLES.png`；`{D}/CAUSAL_DIAGNOSIS.png`；`{CODE}/20261105T153657Z/final_report.py`。',
'- 实际清理：没有删除文件。没有生成散落临时帧；所有失败候选、各候选配置目录及旧输入仍是验证和回溯所需，保留并纳入清单。','']
(O/'REPORT.zh-CN.md').write_text('\n'.join(text))
# Smaller stage summaries link to the final full report rather than duplicating it.
for p,label in [(D,'冻结最优因果诊断'),(E,'离线初始化与独立读出升级')]:
    (p/'REPORT.zh-CN.md').write_text(f'# {label}\n\n本阶段全部失败/对照保留。完整解读见 `{O}/REPORT.zh-CN.md`。\n\n本目录PREREGISTRATION/结果/原始轨迹/VERIFICATION核心必需；图像视频及本说明重要辅助。实际未删除文件。绝对力值及SI：NOT_ESTABLISHED。\n')
for p in [D,E,O]:
    dump(p/'CLEANUP_LEDGER.json',dict(deleted=[],reason='No disposable temporary frames; all candidates including failed branches remain audit dependencies.',importance='重要辅助'))
    files={str(f):sha(f) for f in sorted(p.rglob('*')) if f.is_file() and f.name!='MANIFEST_SHA256.json'}
    files.update({str(f):sha(f) for f in sorted((CODE/p.name).glob('*.py'))})
    dump(p/'MANIFEST_SHA256.json',files);assert all(sha(Path(f))==h for f,h in files.items());print(p.name,len(files),'manifest mismatches 0')
print('\n'.join(rows))
