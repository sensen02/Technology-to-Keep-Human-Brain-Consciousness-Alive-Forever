"""Outcome report and file manifest for corrective-data translator continuation."""
from pathlib import Path
import json,hashlib
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-translator/20261105T231405Z');C=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-neural-translator/20261105T231405Z')
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
s=js(O/'SUMMARY.json');audit=js(O/'DISTRIBUTION_AUDIT.json');prefix=js(O/'PREFIX_AUDIT.json');verify=js(O/'VERIFICATION.json');labelverify=js(O/'LABEL_VERIFICATION.json');cleanup=js(O/'CLEANUP_LEDGER.json');video=js(O/'VIDEO_VERIFICATION.json');allruns=[js(p) for p in (O/'RUNS').glob('*/RESULT.json')];assert len(allruns)==verify['total']==56
warmcontrol=js(O/'WARM_ONLY_CONTROL.json')
rows=[]
for c in s['candidates']:
    rs=c['results'];rows.append(f"| {c['name']} | {max(r['J'] for r in rs):.6f} | {np.mean([r['J'] for r in rs]):.6f} | {sum(r['validation']['passed'] for r in rs)}/4 |")
final=[];control_map={r['release_time_s']:r for r in warmcontrol['final']}
for p in s['final_pairs']:
    a=p['old'];b=p['trained'];w=control_map[p['phase']];final.append(f"| {p['phase']:.3f} | {a['J']:.6f} | {w['J']:.6f} | {b['J']:.6f} | {a['parts']['foot_support_mean']:.4f} → {b['parts']['foot_support_mean']:.4f} | {b['validation']['passed']} |")
ref=s['zero_reproduction'];new=s['representative'];lines=['# S5 闭环分布诊断与补充训练结果','',
'**补充实际闭环状态的数据后，四个新留出交接状态的支撑损失均改善，但原来表现最好的1.300s交接明显退步。全部56条新运行仍未通过完整支撑门；未自主站立/行走，未证明保留走路意图。**','',
'## 先回答本轮问题','',
f"正常预热验证输入到训练样本的标准化最近邻距离中位数为{audit['warm_dev_to_train']['median']:.6f}；严格首次非足触地前的闭环训练记录为{prefix['closed_train']['strict_distance_median']:.6f}，闭环开发验证为{prefix['closed_dev']['strict_distance_median']:.6f}。输入偏移在首次触地之前已经存在，不只是跌倒后的现象。距离依赖特征尺度与采样覆盖，不能当信息理论上界。",'',
'对先前“失撑前3052个样本”的口头说法作精确更正：代码与预登记用的是局部下一1ms内非足力为0且最低COM≥0.965802的筛选；3052个合格训练样本里2592个严格处于首次触地前，460个曾触地后重新达到局部筛选条件。215个开发验证样本里相应为210/5。实际训练遵循预登记筛选，没有事后换数据；PREFIX_AUDIT单列严格前缀统计。','',
f"相近神经输入的参考修正标签仍存在分歧：开发集到扩充训练集的最近邻距离中位{audit['reference_label_conflicts']['nearest_distance_median']:.6f}，归一化力矩RMS差中位{audit['reference_label_conflicts']['normalized_target_RMS_gap_median']:.6f}。这些输入并非完全相同，标签也只是旧参考控制器的建议，不能据此证明神经输入缺少所有必要信息，或宣称相反动作都是唯一正确答案。",'',
'## 如何补训练，以及哪些边界没变','',
'- 只在已保存的物理状态上调用mj_forward，离线计算原预热参考器在该状态的建议力矩；没有把这个参考器放进撤辅阶段施力。暖态和闭环标签统一为指令窗起点，避免平均未来窗与瞬时标签口径混用。',
'- 沿用原612/64/64/42无偏置网络、204细胞三尺度神经历史、原SNN/感觉编码/RH0.375/174项旧读出/执行器限幅。运行时仍只有神经活动输入，没有身体状态补偿、时刻或步态相位输入。',
'- 用上一轮训练相位的160条轨迹构建16000个抽样记录，局部筛选后3052个参与训练；开发标签来自12条完整开发轨迹共1200点，215点合格；前轮最终留出数据不参加本轮训练或选参。',
'- 比较闭环数据权重0.25/0.5/0.75三个模型，再按开发物理表现选非零候选，在5个训练相位实际运行、离线补标签、重训两轮。两次新增1000点各有90/127点合格。第二轮重训没有优于第一轮，保留退步结果。',
'- 全程确定性神经推理，没有本轮探索噪声。局部教师目标不是已证明能恢复平衡的最优动作，结果仍由真实闭环和D1–D8裁定。','',
'## 开发验证：按预登记最差状态优先排序','',
'| 候选 | 最差J↓ | 平均J↓ | 完整支撑门 |','|---|---:|---:|---:|']+rows+['',
f"预登记规则选中{s['incumbent']['name']}，模型为`{s['incumbent']['model']}`。它在这四个开发状态的排序优于旧读出；这不意味着每个状态都改善，更不意味着可移除旧版。本轮保留新旧两者与完整候选。",'',
'## 四个新留出状态：冻结之后才测试','',
'| 交接/s | 旧读出J | 同标签口径、只用预热数据J | 加闭环数据J | 足载/W：旧→新 | 新模型支撑门 |','|---|---:|---:|---:|---|---|']+final+['',
'为排除标签时间口径变化的混淆，在主试验之后追加登记了只用预热数据的对照：同样瞬时标签、架构、初始化、优化器和每阶段epoch上限，连续训练两阶段按开发数据选模型，再用同四个最终状态比较；没有重调主模型。该对照也比旧版部分改善，但在四个状态都落后于补闭环数据模型，支持数据覆盖的贡献。它是事后登记的诊断补充，候选搜索数和实际早停轮数不完全相同，不冒充严格同预算的预先确认性实验。新增12条物理评估；主模型及选择结论保持冻结。','',
f"新时刻1.218/1.268/1.318/1.368s均未参与拟合与选择。旧读出平均J={np.mean([p['old']['J'] for p in s['final_pairs']]):.6f}，新模型平均J={np.mean([p['trained']['J'] for p in s['final_pairs']]):.6f}；四对J均下降，但仍全部失败。",'',
'## 必须同时报告的退步与切断结果','',
f"固定1.300s时刻，旧J={ref['J']:.6f}、足载={ref['parts']['foot_support_mean']:.4f}W；新J={new['J']:.6f}、足载={new['parts']['foot_support_mean']:.4f}W。此处明显退步，不能用四个新状态的平均改善掩盖。",
f"在这个退步状态，切断新分支恢复旧J={s['translator_cut']['J']:.6f}，感觉切断J={s['sensory_cut']['J']:.6f}反而优于正常反馈的新J={new['J']:.6f}，全运动输出切断J={s['motor_cut']['J']:.6f}。新翻译分支与在线反馈的响应在该状态下有问题；这些结果不支持“反馈一定有益”或可靠神经接管，不能推广为所有相位的结论。",'',
'没有已验证的走/停神经条件，意图保持仍NOT_ESTABLISHED。支撑门也没有通过，稳定站立、多周期迈步、前进或语义意图都不能宣称实现。','',
'## 验证、可见性与资源','',
f"- 全{verify['total']}条释放独立重算实际时间、J/D1-D8、204细胞滤波和旧66池率、网络输出、力矩裁剪、身体交接，均通过；原控制器逐位复现，训练模型重复也逐位相同。全部释放每步外力/辅助力0、零MuJoCo警告。",
f"- 全部局部筛选掩码重算；分轨迹抽查{sum(r['labels_recomputed'] for r in labelverify['checks'])}个离线力矩标签，归一化最大误差0；24点float64直接最近邻重算与优化版本最大差{labelverify['nearest_distance_float64_max_error']:.9g}。不把抽查标签写成逐点全量复验。",
f"- 新1.4s物理预热+56×.2s释放，共12.6模拟秒；旧轨迹离线标注仅forward，不计成新物理rollout。串行单重任务，峰值RSS约{max(r['peak_rss_MiB'] for r in allruns):.1f}MiB，复用上一轮隔离CPU环境，没有再安装或改动原环境。",
f"- 视频301帧/25fps/8倍慢放，四列旧读出/上一轮训练MLP/本轮补数据MLP/全切断，完整1.3s预热+.2s释放。全部71mesh不透明、物理数组不变，质心重放误差{video['replay_com_error']}，首中末解码通过；当前模型无图像输入，实际read_image尝试被拒绝，肉眼核验未完成。",
'- 绝对力值与SI仍NOT_ESTABLISHED；本轮没有提高此前理想关节力矩plant限幅。','',
'## 下一最小动作','',
'围绕1.300s这个已知退步状态，定位新分支何时开始给出与参考修正冲突的力矩，比较神经滤波滞后与即时参考动作之间的关系；同时对改善状态做同样检查。优先区分“输入历史不足/响应延迟”和“目标冲突”，再决定是否改时间表示或训练目标，继续保持推理不直接读身体状态的边界。不能把停住或降低活动本身作为走路意图的替代。','',
'## 完整路径与重要程度','',
f'- 核心必需：`{O}/PREREGISTRATION.json`、`{O}/DISTRIBUTION_AUDIT.json`、`{O}/PREFIX_AUDIT.json`、`{O}/SUMMARY.json`、`{O}/FROZEN_SELECTION.json`、`{O}/LABEL_VERIFICATION.json`、`{O}/VERIFICATION.json`、`{O}/MANIFEST_SHA256.json`；DATA组件、CONFIGS模型、全部RUNS轨迹保留供重算。',
f'- 核心必需代码：`{C}/continuation.py`、`{C}/verify.py`、`{C}/prefix_audit.py`、`{C}/warm_only_control.py`；追加对照证据`{O}/WARM_ONLY_CONTROL.json`。重要辅助代码：`{C}/report.py`、`{C}/cleanup.py`。',
f'- 重要辅助：`{O}/REPORT.zh-CN.md`、`{O}/INPUT_SHIFT.png`、`{O}/COMPARISON.png`、`{O}/OLD_PREVIOUS_NEW_CUT_8x_slow.mp4`、`{O}/VIDEO_SAMPLES.png`、`{O}/CLEANUP_LEDGER.json`。',
f"- 实际清理{len(cleanup['deleted'])}项、累计{sum(d['bytes'] for d in cleanup['deleted'])}字节：无引用旧覆盖表、明确不用于训练的新增重复教师数组、两份训练集拼接副本。拼接副本逐字段验证可从保留组件精确重建后才删除，台账保存原哈希与重建来源；没有删除模型、失败轨迹或历史证据。",'']
(O/'REPORT.zh-CN.md').write_text('\n'.join(lines))
manifest={str(p):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='MANIFEST_SHA256.json'};manifest.update({str(p):sha(p) for p in C.glob('*.py')});dump(O/'MANIFEST_SHA256.json',manifest);assert all(sha(Path(p))==h for p,h in manifest.items())
print('manifest',len(manifest),'mismatches0');print('\n'.join(rows));print('\n'.join(final));print('REPRESENTATIVE',ref['J'],new['J'],ref['parts']['foot_support_mean'],new['parts']['foot_support_mean']);print('cleanup bytes',sum(d['bytes'] for d in cleanup['deleted']))
