"""Summarize full-mapping audit and trial without claiming walking."""
from pathlib import Path
import json,hashlib
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T064731Z')
OLD=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T062800Z')
C=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-neural-joint-torque/20261105T064731Z')
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
s=js(O/'SUMMARY.json');old=js(OLD/'SUMMARY.json');audit=js(O/'MAPPING_AUDIT.json');verify=js(O/'VERIFICATION.json')
rows=[]
for label,r in [('旧主导映射P0基线',s['P0']['dominant_replay']),('完整映射P0基线',s['P0']['baseline']),('旧映射P0训练最优',old['P0']['best']),('完整映射P0训练最优',s['P0']['best']),('完整映射P1切断',s['P0']['cut']),('完整映射P2随机组最优',s['P2']['best'])]:
    p=r['parts'];ms=p['first_nonfoot_s']*1000 if p['first_nonfoot_s'] is not None else None
    rows.append(f"| {label} | {r['J']:.6f} | {p['foot_support_mean']:.4f} | {p['com_z_min']:.4f} | {ms} | {r['validation']['passed']} |")
flip=[r['joint'] for r in audit['joints'] if r['opposite_to_fixed_sign_entries']]
one=[r['joint'] for r in audit['joints'] if not(r['full_positive'] and r['full_negative'])]
median=lambda k:float(np.median([r[k] for r in audit['samples']]))
allres=[js(p) for p in O.glob('*/RESULT.json')]
text=['# S5 完整关节映射审计与闭环试验','',
'**结果：完整映射同预算训练改善真实组，但仍未自主支撑/行走。** 所有40次评估均未通过完整动态支撑门。这是手动研究原型，位移、视频和较低损失不等于自主行走。',
'', '## 从原模型找回了什么',
'原主导映射覆盖32/42关节，14个双向，18个单向；完整映射覆盖42/42，35个双向，7个单向。读出非零项67→174，增加107项，原有符号翻转0项。符号矩阵秩32→33，覆盖齐全并不等于42自由度可独立控制。',
'计算使用原肌肉模型同一姿态下激活0→1的力矩增量，按原66个运动池合并后取符号。不添加神经元、突触或跨腿任意映射。零阈值相对1e-10/1e-8/1e-6三档结果一致，非浮点残差制造覆盖。',
'关键帧及25个真实预热姿态的激活仿射误差均小于1e-9。35个双向仍是当前固定参考姿态的结论：9项在预热样本中出现相反方向，涉及 '+', '.join(flip)+'。固定符号仍是工程近似，未加入运行时几何补偿。',
'仍单向的7关节：'+', '.join(one)+'。所有90肌肉（包括无运动池者）合计也只有38个双向，因此既有池缺口和身体作用几何都需要区分。',
f"25样本上，放宽为每时刻任意非负池系数的力矩方向锥拟合，旧映射相对残差中位{median('dominant_cone_relative_residual'):.6f}，完整映射{median('full_cone_relative_residual'):.6f}。这只是必要能力检查，允许的自由度比实际共享w/g和神经活动多，不能当作闭环可行证明。",
'', '## 同预算实测',
'每组17次训练评估，完全相同种子、两代各8候选、sigma0.35/0.20、108维非负指数增益、固定限幅；另各一次旧映射基线重现、最优重复、读出切断，共40次。真实组和随机组各自从零向量开始，不挑选图身份。',
'| 条件 | J↓ | 平均足载/W | 最低质心 | 首次非足接触/ms | 全动态门 |','|---|---:|---:|---:|---:|---|']+rows+['',
f"完整映射真实组相对旧映射同预算最优J变化为 {s['P0']['best']['J']-old['P0']['best']['J']:+.6f}；随机组最优减真实组最优为 {s['P2']['best']['J']-s['P0']['best']['J']:+.6f}。这些是单种子小预算结果，无统计显著性结论。",
'所有动态门结果完整保存在各RESULT.json。足载只计足部；辅助参考平均1.0000123W，释放D3约0.9000W。辅助PD仅在前1.3秒运行，释放0.2秒只有神经力矩。模型上限未提高，未改变神经参数、感知或预热。',
f"真实组最优的指令限幅占比为{s['P0']['best']['saturation_fraction']:.6f}（时间×关节样本），固定符号读出的幅度大量被限幅；这是一项需要解释的限制，不能靠继续抬力矩上限掩盖。新映射未经训练J反而较旧基线高，因此不把覆盖改善本身宣称为行为改善。当前因果证据是同预算训练后的实际差异。", 
'', '## 核验与交付限制',
'- 两个旧映射基线的J及全部BODY数组逐位复现，证明同输入下仅换读出映射；两组最优也各自逐位重复。所有40条指令重算通过、真实感知随状态改变；交接qpos/qvel连续。释放辅助力和外力逐步断言0，零MuJoCo警告。',
f"- 40条轨迹独立核验，仿真最高记录RSS {max(r['peak_rss_MiB'] for r in allres):.1f} MiB；串行执行。旧自动续跑goal仍exhausted，未启动新并行服务。",
'- 视频301帧、25fps、8倍慢放，覆盖完整1.3秒预热及0.2秒释放，71mesh不透明、物理数组哈希未改，重放质心误差0，首/中/末帧解码通过。',
'- 图像读取能力限制：当前模型不声明图像输入，read_image被工具拒绝。不能声称已肉眼核验；数值/解码检查不能替代视觉检查。',
'- 原工作plant部分关节上限超过原肌肉（最高3.62倍）。绝对力值与SI仍NOT_ESTABLISHED。不能把理想关节执行器结果说成果蝇肌肉能力。',
'', '## 下一步依据',
'若完整映射仍未过门，停止仅扩大同一种随机增益搜索。优先分析最优运行首次失撑时刻的神经活动、指令饱和和缺失方向，区分固定映射能力与实际神经活动可达性。之后可在新登记中利用辅助阶段的真实感知/神经输出离线初始化读出；教师只作离线目标，撤辅推理仍必须只读神经活动。',
'', '## 完整路径与重要程度',
f'- 核心必需：`{O}/AUDIT_PREREGISTRATION.json`、`{O}/MAPPING_AUDIT.json`、`{O}/FULL_MAPPING.npz`、`{O}/PREREGISTRATION.json`、`{O}/CHANGE_LEDGER.json`、`{O}/SUMMARY.json`、`{O}/VERIFICATION.json`、`{O}/MANIFEST_SHA256.json`，所有候选和预热原始状态/轨迹/参数。',
f'- 核心必需：`{C}/audit_mapping.py`、`{C}/run_full_mapping.py`、`{C}/verify_results.py`；前轮运行器和验证器仍为只读依赖，路径及哈希已登记。重要辅助：`{C}/report.py`。',
f'- 重要辅助：`{O}/REPORT.zh-CN.md`、`{O}/P0_P1_P2_8x_slow.mp4`、`{O}/COMPARISON.png`、`{O}/VIDEO_SAMPLES.png`、`{O}/VIDEO_FRAME_MAP.json`。',
'- 清理：本轮未生成需删除的无用临时文件，实际未删除文件；保留全部失败和旧输入用于回溯。DOMINANT_COVERAGE.json是旧映射复验依赖，不是废弃副本。','']
(O/'REPORT.zh-CN.md').write_text('\n'.join(text))
dump(O/'CLEANUP_LEDGER.json',dict(deleted=[],reason='All retained runs and mappings remain necessary for verification, failures and reproduction.',importance='重要辅助'))
manifest={str(p):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='MANIFEST_SHA256.json'}
manifest.update({str(p):sha(p) for p in sorted(C.glob('*.py'))})
dump(O/'MANIFEST_SHA256.json',manifest);assert all(sha(Path(p))==h for p,h in manifest.items())
print('manifest',len(manifest),'mismatches 0');print('\n'.join(rows))
