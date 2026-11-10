"""Outcome, regression guard and evidence manifest for frozen timing/gain probes."""
from pathlib import Path
import json,hashlib
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-translator/20261105T234531Z');C=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-neural-translator/20261105T234531Z')
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
s=js(O/'SUMMARY.json');v=js(O/'VERIFICATION.json');lag=js(O/'TIMING_COMMON_SUPPORT.json');diag=js(O/'TIMING_DIAGNOSIS.json');video=js(O/'VIDEO_VERIFICATION.json');cleanup=js(O/'CLEANUP_LEDGER.json');results=js(O/'PROGRESS.json');assert len(results)==s['total']==v['total']
base=s['candidates'][0];reference=next(r for r in base['results'] if r['release_time_s']==1.3);safe=s['guarded'];best=s['best_nonzero'];safe_rep=next(r for r in safe['results'] if r['release_time_s']==1.3);nonzero_pass=sum(c['guard_pass'] for c in s['candidates'][1:]);failed=sum(not r['validation']['passed'] for r in results)
rows=[]
for c in s['candidates']:
    representative=next(r for r in c['results'] if r['release_time_s']==1.3);rows.append(f"| {c['name']} | {max(r['J'] for r in c['results']):.6f} | {np.mean([r['J'] for r in c['results']]):.6f} | {representative['J']:.6f} | {representative['parts']['foot_support_mean']:.4f} | {c['guard_pass']} |")
final=[]
for p in s['final_pairs']:
    a=p['zero'];g=p['guarded'];n=p['best_nonzero'];final.append(f"| {p['phase']:.3f} | {a['J']:.6f} | {g['J']:.6f} | {n['J']:.6f} | {a['parts']['foot_support_mean']:.4f} → {n['parts']['foot_support_mean']:.4f} |")
lagrows=[]
for item in lag['cases']:
    common=item['common_support'];lagrows.append(f"| {item['phase']:.3f} | {0 if common is None else common['best']['n']} | {'不足' if common is None else common['best']['lag_ms']} |")
text=['# S5 神经翻译器时序与幅度诊断','',
f"**本轮{s['total']}条实际释放评估完成，其中{failed}条完整支撑门失败。24个非零候选中{nonzero_pass}个满足原1.300s状态的回归保护，实际受保护选择为{safe['name']}。** 未自主站立/行走，意图保持仍NOT_ESTABLISHED。",'',
'## 这轮回答什么','',
'上一轮补闭环数据改善了多个状态，却使1.300s足载从0.7321W跌到0.2640W。本轮冻结同一个MLP全部权重，只比较神经滤波时间和新分支修正幅度，检查是否能通过统一快慢/强弱调整消除该取舍。',
'- 滤波原版5/20/50ms，另测半倍2.5/10/25ms、五分之一1/4/10ms、双倍10/40/100ms。只改新分支204细胞历史，旧66池读出始终5ms。',
'- 修正幅度0.05/0.1/0.25/0.5/0.75/1，另有0幅度旧控制基线，共25个设置，各测5个开发交接状态。矩阵权重、SNN、感觉编码、RH0.375、身体和执行器限幅不变。',
'- 所有滤波初态由真实预热尖峰从0.2s起因果重算，取交接边界以前的样本。没有在交接时清零记忆、拼接不同神经状态、偷看未来或直接输入身体状态。',
'- 1.300s回归保护同时要求J不升、平均足载/最低COM/首次非足触地时间不降（仅1e-9数值容差）。先过滤，再按5状态完整门通过数、最差J、平均J排序；单列不受保护的最佳非零版本，不能冒充部署选择。','',
'## 全部开发对照','',
'| 设置 | 五状态最差J↓ | 平均J↓ | 1.300s J↓ | 1.300s足载/W | 通过回归保护 |','|---|---:|---:|---:|---:|---|']+rows+['',
f"旧基线1.300s J={reference['J']:.6f}、足载={reference['parts']['foot_support_mean']:.6f}W；受保护选择J={safe_rep['J']:.6f}、足载={safe_rep['parts']['foot_support_mean']:.6f}W。无保护最佳非零为{best['name']}，模型`{best['model']}`。",
'若受保护选择为zero，含义是本轮没有获得替换旧控制器的效果依据；新分支只是单列实验候选。若某个非零候选在更差状态更好，也不能掩盖已知好状态的退步。','',
'## 四个新交接时刻：冻结之后才看','',
'| 交接/s | 旧基线J | 受保护选择J | 无保护最佳非零J | 足载/W：旧→非零 |','|---|---:|---:|---:|---|']+final+['',
'新时刻1.206/1.256/1.306/1.356s不参与选择。同一个候选同时担任多臂时复用相同运行，没有重复计数制造样本量。仅4个确定性状态，不是统计显著性或生物重复。','',
'## 时序诊断的证据与限制','',
'在已保存轨迹上，用原预热参考器给出假想修正，比较当前网络实际修正与不同时间的参考修正；从不把未来参考或身体状态送入在线控制。reference[t+lag]与prediction[t]比较，负lag表示与过去参考更相似，不能直接当成实测神经反应延迟。',
'初始诊断各lag的有效窗口不同，可能产生边界偏差；独立复核另外固定每个状态的同一组预测时间点，只保留共同支持至少10点的对称lag范围。以下为共同窗口结果：','',
'| 状态/s | 相同预测点数量 | 最低误差lag/ms |','|---|---:|---:|']+lagrows+['',
f"1.300s严格首次非足触地前可用窗为{next(x['strict_prefix_samples'] for x in diag['cases'] if x['phase']==1.3)}个1ms样本；相对于同状态的旧读出，参考误差首次增大的时间约{1000*next(x['first_reference_worsening_s'] for x in diag['cases'] if x['phase']==1.3):.3f}ms。参考器只是比较对象，不证明唯一必要动作。",
'时间常数改变还会改变输入幅度、平滑和训练分布；幅度改变会影响饱和及闭环后续状态。因此本轮能证明这些参数干预的实际结果，不能把其效果唯一归因于延迟。若全部非零候选不能过保护，只说明本轮测试范围没有找到兼顾解，不证明所有神经翻译器都不可行。','',
'## 重复与切断','',
f"无保护最佳非零在1.300s J={s['representative']['J']:.6f}，重复逐位相同；切全部运动J={s['motor_cut']['J']:.6f}，切感觉J={s['sensory_cut']['J']:.6f}，切新分支J={s['translator_cut']['J']:.6f}。只切新分支精确恢复旧J。切断只用于因果诊断，不拿更好切断分数宣称感觉不需要或意图得到保持。",'',
'## 验证、资源与交付边界','',
f"- 全{s['total']}条轨迹的历史从原始预热尖峰重建，再逐步验证神经特征/旧池率/幅度变换/实际命令；MLP权重逐数组保持不变，旧读出5ms不变。原参数完整复现上一轮退步轨迹，候选重复精确。所有交接qpos/qvel连续，原生J/D1-D8独立重算一致，输入哈希未变。",
f"- 幅度变换为复用原运行器采用atanh(gain*tanh(raw))，增益0与1分别有精确零/原值分支；独立验证最终实际幅度的浮点误差小于1e-6，实际施加力矩由保存latent精确重算。",
f"- 每步释放外力/辅助力0、零MuJoCo警告；1.4s新预热+{s['total']}×.2s释放，共{1.4+s['total']*.2:.1f}模拟秒；串行单重任务，峰值RSS约{max(r['peak_rss_MiB'] for r in results):.1f}MiB，复用已有CPU环境。",
f"- 视频301帧/25fps/8倍慢放，旧读出/原全幅翻译器/最佳非零/受保护选择四列，完整1.3s算法预热+.2s释放。71mesh可见、物理数组不变，质心重放误差{video['replay_com_error']}，首中末解码通过。当前模型无图像输入，实际读取抽帧失败，肉眼核验未完成。",
'- 身体状态补偿未加入；绝对力值/SI仍NOT_ESTABLISHED；仍没有经验证的走/停神经条件，站稳也不能算意图保持。本轮仍是研究试验。','',
'## 下一最小动作','',
'若全局幅度/快慢设置无法兼顾，应在后续学习中显式加入已知较好轨迹的回归约束，并检查是否能仅根据神经历史学会何时不作破坏性修正；在线不得用交接时间/相位/身体状态作人工切换。先证明这种输入依赖的区别能够在留出状态成立，再扩展时长或宣称协调运动。','',
'## 完整路径与重要程度','',
f'- 核心必需：`{O}/PREREGISTRATION.json`、`{O}/TIMING_DIAGNOSIS.json`、`{O}/TIMING_COMMON_SUPPORT.json`、`{O}/FROZEN_SELECTION.json`、`{O}/SUMMARY.json`、`{O}/VERIFICATION.json`、`{O}/MANIFEST_SHA256.json`；CONFIGS参数、DATA缓存/时序参考、全部RUNS候选保留。',
f'- 核心必需代码：`{C}/timing.py`、`{C}/verify.py`；重要辅助代码`{C}/report.py`。',
f'- 重要辅助：`{O}/REPORT.zh-CN.md`、`{O}/TIMING_GAIN_GRID.png`、`{O}/COMPARISON.png`、`{O}/TIMING_GAIN_GUARD_8x_slow.mp4`、`{O}/VIDEO_SAMPLES.png`、`{O}/CLEANUP_LEDGER.json`。',
f"- 已实际清理{len(cleanup['deleted'])}项、{sum(d['bytes'] for d in cleanup['deleted'])}字节：本轮无读者的旧覆盖表，以及未使用的新重复教师数组。真实历史尖峰源、候选模型与失败轨迹均保留。",'']
(O/'REPORT.zh-CN.md').write_text('\n'.join(text))
manifest={str(p):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='MANIFEST_SHA256.json'};manifest.update({str(p):sha(p) for p in C.glob('*.py')});dump(O/'MANIFEST_SHA256.json',manifest);assert all(sha(Path(p))==h for p,h in manifest.items())
print('FINAL',s['total'],'failed',failed,'safe',safe['name'],'nonzero',best['name'],'nonzero_guard_pass',nonzero_pass,'manifest',len(manifest),'mismatches0');print('\n'.join(final));print('COMMON LAGS',lagrows)
