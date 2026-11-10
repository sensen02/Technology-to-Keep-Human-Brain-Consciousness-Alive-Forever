"""Final outcome report, file importance and immutable evidence manifest."""
from pathlib import Path
import json,hashlib
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-translator/20261105T200021Z')
C=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-neural-translator/20261105T200021Z')
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
s=js(O/'SUMMARY.json');controls=js(O/'FINAL_CONTROLS.json');v=js(O/'VERIFICATION.json');fit=js(O/'OFFLINE_FIT.json');initial=js(O/'INITIAL_SELECTION.json');engineering=js(O/'ENGINEERING.json');video=js(O/'VIDEO_VERIFICATION.json')
all_results=[js(p) for p in (O/'RUNS').glob('*/RESULT.json')];assert len(all_results)==v['total'];failed=sum(not r['validation']['passed'] for r in all_results)
rows=[]
for cand in initial['candidates']:
    rs=cand['results'];rows.append(f"| {cand['name']} | {max(r['J'] for r in rs):.6f} | {np.mean([r['J'] for r in rs]):.6f} | {sum(r['validation']['passed'] for r in rs)}/4 |")
history=js(O/'VALIDATION_HISTORY.json');groups={}
for r in history:
    if r['label'].startswith('checkpoint_'):groups.setdefault(r['model_file'],[]).append(r)
for path,rs in groups.items():rows.append(f"| {Path(path).stem} | {max(r['J'] for r in rs):.6f} | {np.mean([r['J'] for r in rs]):.6f} | {sum(r['validation']['passed'] for r in rs)}/4 |")
pairs=[]
for p in controls['paired_final']:
    a=p['old'];b=p['trained_MLP'];pairs.append(f"| {p['release_time_s']:.3f} | {a['J']:.6f} | {b['J']:.6f} | {b['J']-a['J']:+.6f} | {b['parts']['foot_support_mean']:.4f} |")
old=s['representative'];trained=controls['representative'];zero_selected=s['incumbent']['name']=='zero'
text=['# S5 神经翻译网络首轮实测报告','',
'**人工神经翻译网络已实现并完成实际训练，但没有学会自主支撑或行走，也没有建立走路意图保持证据。**',
f"实际选择保留版本：`{s['incumbent']['name']}`。"+('它是原控制器加零修正，新网络没有获得替换原控制器的效果依据。' if zero_selected else '它只代表本轮选择规则下的最优，不代表通过行为门。'),
'', '## 做了什么',
'- 原12000细胞SNN、感觉编码、RH入兴奋倍率0.375和已有174项读出冻结；原身体、71mesh和执行器限幅不变。新接口为204个运动细胞的5/20/50ms因果滤波→612/64/64/42无偏置tanh网络，45952权重，作为旧读出的有界力矩修正。新接口放开固定符号/稀疏限制，属于人工训练接口，不是生物连接发现。',
'- 新接口不读取身体姿态、速度、接触、目标角度、时刻或步态相位。原SNN继续接收原来的物理感觉通道。没有新增身体状态补偿、在线PD、CPG、姿态锁定或辅助外力。',
'- 零修正完整BODY轨迹及J精确复现旧最优；204细胞三尺度滤波由保存尖峰独立重算逐项相同；保存/恢复连续；Torch与NumPy推理一致。',
'', '## 离线拟合确实提高，但不是行为成功',
f"完整轨迹分组留出的归一化力矩MSE：旧读出{fit['mlp']['old_readout_dev_mse']:.6f}；放开符号的线性修正{fit['linear']['best_dev_mse']:.6f}；非线性MLP修正{fit['mlp']['best_dev_mse']:.6f}。训练轨迹预热相位偏移0/0.4/0.8/1.2rad，验证0.2/0.6rad；没有随机拆相邻帧。教师标签为真实辅助预热下一完整1ms窗的平均力矩，覆盖整段步行动作而非平均站姿。",
'这些是本仿真参考动作分布上的误差，不是可解码信息上界。离线候选进入闭环后未优于旧读出的最差交接表现；因此PPO从零修正MLP开始，而没有强行采用离线误差最低版本。',
'', '## 闭环选择：失败也完整保留',
'开发验证相位1.225/1.275/1.325/1.375s，同一权重覆盖所有状态；先比较完整D门通过数，再比较最差J，最后比较平均J。',
'| 模型 | 最差J↓ | 平均J↓ | 支撑门通过 |','|---|---:|---:|---:|']+rows+['',
f"实际完成{s['training_evaluations']}次PPO训练rollout。探索动作明确记录为随机训练动作，不能计为确定性神经自主行为。其余确定性评估使用冻结神经输入→网络推理；合计{len(all_results)}条释放轨迹，其中{failed}条完整支撑门失败。",
'没有候选达到进入长时阶段的支撑条件，所以本轮没有训练0.5/1.0s阶段、没有声称多周期运动，也未触发同预算随机连接组比较。这里是本轮实验的结果与阶段决策，不是恢复逐次调用审批或将今后工作限制在某个额度。',
'', '## 最终配对留出与直观样本',
'另冻结开发集上最好的非零MLP，仅供展示失败与配对诊断；此选择不看最终留出结果，不改变部署选择。最终时刻在开跑前由计划的半毫秒点向下对齐0.5ms到整1ms指令边界，避免快照含未来半窗神经信息。',
'| 最终交接/s | 原读出J | 冻结训练MLP J | ΔJ（负为改善） | MLP足载/W |','|---|---:|---:|---:|---:|']+pairs+['',
f"固定代表时刻1.300s：旧/保留版本J={old['J']:.6f}、足载={old['parts']['foot_support_mean']:.4f}W；训练MLP J={trained['J']:.6f}、足载={trained['parts']['foot_support_mean']:.4f}W。不拿单一时刻覆盖其它状态结果。",
f"保留版本全运动读出切断J={s['motor_cut']['J']:.6f}，感觉切断J={s['sensory_cut']['J']:.6f}，只切断新分支J={s['translator_cut']['J']:.6f}。"+('因为保留的新分支本来为零，最后一项与原版相同是预期结果，不能声称证明了训练MLP的因果贡献。' if zero_selected else ''),
'', '## 关于“走路意图不能被翻译成稳定”',
'- 支撑只是门槛，稳定站着不算运动目标。此次连支撑门都没通过，不能说学会走路，也不能把失败解释成已经忠实保持了意图。',
'- 查到的两个DNp09源身份不在当前12000细胞网络中，当前网络没有经验证的走/停驱动条件。这不证明任何运动信号都不存在，但意味着本轮无法完成来源明确的意图条件对照。未添加细胞、未把放电高低任意命名为走/停，也未将标签直接喂给翻译器。',
'- 保存尖峰历史的重排已单列离线输入敏感性诊断，不能代替真实身体闭环或走停因果证据。神经切断依赖性也不等于语义意图被保留。',
'', '## 历史更正与本轮验证',
'- 原先“残差0.577意味着只能做到四成/必然不能站”的说法不成立；旧方向锥共享池系数测试不是后来174独立项读出的严格上界。保留历史文件，追加更正。',
'- 对照时发现上一轮20261105T182927Z的PHASE_1.225预热继承了前一sensory-cut开关，保存感觉电流和尖峰全为0；其余相位没有该问题。这一对照不能作为正常感觉预热下的纯交接相位证据，撤回此前“四相位正常预热都改善”的解释。旧数据不改写；HISTORICAL_PHASE_CORRECTION.json含具体来源哈希。新运行器每次预热显式恢复感觉，全部13个缓存神经状态独立验证非零。',
f"- {v['total']}条释放的实际时间轴、原生J/D1-D8、204细胞滤波、旧读出+新修正→最终力矩、物理交接和输入哈希均独立验证。随机训练动作按保存的采样动作复算，不冒称能由均值唯一复现。全部释放每步辅助/外力0、零MuJoCo警告；峰值进程RSS约{max(r['peak_rss_MiB'] for r in all_results):.1f}MiB。",
f"- 单进程串行，6×1.4s预热+{sum(r['duration_s'] for r in all_results):.1f}s释放，共{8.4+sum(r['duration_s'] for r in all_results):.1f}模拟秒；CPU PyTorch2.6.0隔离安装，原venv_body未改动。", 
f"- 视频{video['frames']}帧/25fps/8倍慢放，四列原版/训练MLP/实际保留版本/全运动切断，完整1.3s算法预热与释放，全部71mesh可见、物理数组不变，重放质心误差{video['replay_com_error']}；首中末解码通过。图像读取工具当前模型不支持图像输入，肉眼核验未完成，不冒称视觉确认。",
'- 绝对力值及SI仍NOT_ESTABLISHED；执行器仍是此前匹配辅助包络的理想关节力矩plant，部分限幅高于原肌肉能力，本轮未再提高。',
'', '## 下一最小动作',
'不因这次失败否定所有神经翻译器。优先检查离线教师分布与撤辅后实际神经输入分布的差异，以及相似神经输入是否对应相反的必要动作；若覆盖不足，再按真实闭环偏离状态补训练数据。继续保持“推理只读神经活动”的边界。走停意图验收需要独立核查上游指令通路，不能拿更低拟合误差代替。',
'', '## 文件重要程度与完整路径',
f'- 核心必需：`{O}/PREREGISTRATION.json`、`{O}/SUMMARY.json`、`{O}/FINAL_CONTROLS.json`、`{O}/VERIFICATION.json`、`{O}/INTENT_AUDIT.json`、`{O}/HISTORICAL_PHASE_CORRECTION.json`、`{O}/MANIFEST_SHA256.json`；MODELS权重、DATA预热/交接缓存及所有RUNS候选原生记录均用于重算。',
f'- 核心必需代码：`{C}/runtime.py`、`{C}/translator.py`、`{C}/experiment.py`、`{C}/final_controls.py`、`{C}/verify_delivery.py`、`{C}/audit_previous_phase.py`。',
f'- 重要辅助：`{O}/REPORT.zh-CN.md`、`{O}/OLD_NEW_CUT_8x_slow.mp4`、`{O}/VIDEO_SAMPLES.png`、`{O}/COMPARISON.png`、`{O}/ENVIRONMENT.txt`、`{O}/CLEANUP_LEDGER.json`、`{C}/report.py`、`{C}/.venv/`（隔离环境可按锁定依赖重建，目前仍用于验证）。',
'- 已实际清理继承旧运行器产生且不被本轮读取的COVERAGE.json；构造最终对照时同路径再生后也即时清理。两次删除同一类40247字节可再生临时输出，不删除训练失败、候选权重、原始身份或历史哈希。','']
(O/'REPORT.zh-CN.md').write_text('\n'.join(text))
manifest={str(p):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='MANIFEST_SHA256.json'}
manifest.update({str(p):sha(p) for p in C.glob('*.py')});manifest[str(C/'.gitignore')]=sha(C/'.gitignore');dump(O/'MANIFEST_SHA256.json',manifest);assert all(sha(Path(p))==h for p,h in manifest.items())
print('RESULT',s['incumbent']['name'],'runs',len(all_results),'failed',failed,'manifest',len(manifest),'mismatches0');print('\n'.join(rows));print('\n'.join(pairs))
