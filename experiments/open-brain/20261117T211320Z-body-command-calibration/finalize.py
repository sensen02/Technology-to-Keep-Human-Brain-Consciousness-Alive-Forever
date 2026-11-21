"""Publish scoped teacher-relative command calibration without rewriting history."""
from scope import *
import shutil,platform,importlib.metadata

choice=json.loads((HERE/'SELECTION.json').read_text());checkpoint=Path(choice['checkpoint'])
audit=json.loads((HERE/'INDEPENDENT.json').read_text());formal=[r for r in audit['records'] if r['group'] in ('HELD','HELD_A01')]
assert len(formal)==8
relative=all(r['relative_pass'] for r in formal)
parking=json.loads((HERE/'PARKING_REGRESSION_RESULT.json').read_text());parking_pass=all(r['relative_pass'] for r in parking)
verify_sources(audit['source_sha256']);command_audit=json.loads((HERE/'COMMAND_SOURCE_AUDIT.json').read_text());assert len(command_audit['rows'])==24
for row in command_audit['rows']:
 path=HERE/'COMMANDS'/f"{row['family']}_{row['variant']}"/'TRACE.npz';assert sha(path)==row['sha256']
physical=[]
for p in sorted(HERE.glob('*/*/TRACE.npz')):
 if p.parent.parent.name=='COMMANDS':continue
 with np.load(p,allow_pickle=False) as z:
  if 'qpos' not in z.files:continue
  count=len(z['qpos'])
 result_path=p.parent/'RESULT.json'
 meta=json.loads(result_path.read_text()) if result_path.exists() else dict(status='completed physical trace; SMOKE metric JSON serialization failed, no scientific pass claimed')
 physical.append(dict(path=str(p),steps=count,result=meta))
steps=sum(r['steps'] for r in physical)
assert steps==1468000 and len(physical)==74
sources=freeze([checkpoint,HERE/'OWNER_ACCEPTANCE.json',HERE/'SELECTION.json',HERE/'A01_EXTENSION.zh-CN.md'])
env=dict(executable=sys.executable,python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in ('torch','mujoco','flygym','numpy')},threads=1,render='EGL')
result=dict(project='P1_open_source_brain',status='TEACHER_RELATIVE_FOUR_COMMAND_CLASSES_VERIFIED' if relative and parking_pass else 'COMMAND_CALIBRATION_HAS_REMAINING_DEVIATIONS',
 owner_accepted_prior_stage=True,teacher_relative_pass=relative,parking_regression_pass=parking_pass,
 checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),policy_loader=str(STOP/'policy_repair.py'),
 physical_model_entry=str(BASE/'BODY_MODEL_CURRENT.json'),formal_pairs=formal,
 teacher_absolute_pass=sum(r['teacher_absolute'] for r in formal),student_absolute_pass=sum(r['student_absolute'] for r in formal),
 first_calibration=dict(initial_relative_pass=9,total=16,after_adaptation_relative_pass=16),
 training=dict(frames=52400,new_teacher_trajectories=16,new_teacher_frames=32000,retained_terrain_frames=20400,epochs=40,actual_updates=4120),
 new_physical_steps=steps,new_body_episodes=len(physical),new_neural_ticks=500000,
 neural_tick_ledger='24 complete postneck traces x20000=480000, failed stride generation completed20000 unsaved neural ticks before archive API error; no ticks for body replay.',
 independent_replays=len(audit['replays']),raw_arrays_each=20,
 source_sha256=sources,environment=env,physics=physical,evidence=str(HERE/'INDEPENDENT.json'),
 limitations=['Teacher-relative engineering tolerance is not biological calibration or independent reliability certification.',
 'Only one held phase seed6211 with new stimulus timing/period; conditions are not independent animal samples.',
 'Neck command traces generated using fixed true-source BANC pathways and specified DN stimuli; no Eon autonomous command selection or sensory closed loop.',
 'DNa01/DNa02/DNg13 identities differ; current body interface shares two side-specific amplitude drives, not independently implemented steering mechanisms.',
 'Old 69 cases as a whole, all-terrain capabilities, flight, full D1-D8 and full adhesion/slip ledger not regressed here.',
 'Previous absolute failures remain original records; owner accepted prior stage, no historical metrics changed.'])
write(HERE/'FINAL_ACCEPTANCE.json',result)
labels={'a01_left':'DNa01 左走弧','a01_right':'DNa01 右走弧','arc_left':'DNa02 左走弧','arc_right':'DNa02 右走弧','stride_left':'DNg13 左步幅差动','stride_right':'DNg13 右步幅差动','back':'MDN 后退','speed':'DNg100/DNg97 低→高→低'}
lines=[]
for row in formal:
 f=row['family'];group='HELD_A01' if f.startswith('a01') else 'HELD'
 t=json.loads((HERE/group/f'{f}_2_teacher/RESULT.json').read_text());s=json.loads((HERE/group/f'{f}_2_student/RESULT.json').read_text())
 detail=f"速度 {t['forward_speed']:.3f}/{s['forward_speed']:.3f} mm/s；偏航 {t['yaw_degrees']:.2f}/{s['yaw_degrees']:.2f}°"
 if f.startswith('stride'):detail+=f"；右/左步幅比 {t['stride_right_left_ratio']:.3f}/{s['stride_right_left_ratio']:.3f}"
 if f=='speed':detail='稳态速度（教师/学生）：'+', '.join(f'{a:.3f}/{b:.3f}' for a,b in zip(t['speed_windows'],s['speed_windows']))+' mm/s'
 lines.append(f"| {labels[f]} | {'相符' if row['relative_pass'] else '偏离'} | {detail} |")
report=f'''# 身体四类指令的教师相对校对（P1）

按用户最新裁定，前一身体课程登记为已接受的限定阶段；保留旧绝对失败，不要求学生超过同体教师。此轮校对左右走中转向、步幅差动、后退和速度变换，结论：**新相位正式教师相对 {sum(r['relative_pass'] for r in formal)}/8，已接受基本走停回归 {sum(r['relative_pass'] for r in parking)}/2**。这是手动研究实施，非成品或生物有效性认证。

## 实际做了什么

保持70body、66有限铰链、42目标＋6足垫、原有限PD/附着/站姿、固定颈部结构/电导膜动力学/静态读出、局部CPG与相位运动读出全部不变。DNa01/DNa02左右走中转向，DNg13左右步幅差动、MDN后退、DNg100推进＋DNg97加速脉冲均实际经过固定颈部，保存原始发放/到达/两维读出，再让教师和学生各自独立完整物理执行同一份命令。不用另一只自由跑教师按时间给学生标签，不以影子标签替代完整教师对照。

两个开发相位及发放时刻/周期共16组完整教师示范32,000帧；原学生相对9/16（六基础族7/12，DNa01左右2/4），主要是右转偏小、一例DNa01右转方向反了、速度切换伴随偏航偏差。比如DNa02右教师−97.3°/学生−46.6°，DNg13右−56.4°/−25.9°。这轮教师本身能完成对应动作，这些是学生适配问题。

用16条当前教师示范＋保留原地形20,400帧，共52,400帧，40epoch、4,120次真实AdamW更新；仅感觉ANN参数学习，没有改脑/颈部、物理、幅度模板、足垫分类器、相位约束或观测维度。网络仍无GRU/上一动作/场地标签，局部CPG保留。实际复合监督损失中心差分最大误差1.28e−11、Adam/RNG起点精确续接，每update完整可加载权重/SHA及定期全优化器保存，终局独立核验。固定epoch40后校准16/16，再冻结权重后首次使用6211第三套相位/时刻/周期8组正式配对；校准条件已用于训练，不冒称留出。

## 新相位正式结果

表中数值按“教师/学生”，速度为同一冻结窗胸部前向投影，偏航带符号。步幅为足端在胸部坐标中完整往返周期的中位幅度，非命令值。

| 通道/动作 | 教师相对校对 | 实际量测（教师/学生） |
|---|---|---|
{chr(10).join(lines)}

速度三个阶段及加减速方向逐项验收，不仅比整段平均。所有相对容差在PROTOCOL预登记：速度差≤max(1mm/s,教师25%)、偏航差≤max(10°,教师25%)、左右步幅比差≤.20且方向相同；停车/高度/实际越限等共同边界也分别校对。绝对门单列：正式教师{result['teacher_absolute_pass']}/8、学生{result['student_absolute_pass']}/8；若双方绝对失败，不会冒称任务达成。当前转向末端都使用两维侧别驱动/CPG幅度，通路不同不等于已实现彼此独立的运动机制。

原来异常高抬左中腿的.15mm/1723及平地走停1151，用新权重与独立教师再次完整配对，均在相对门内，学生原绝对门均通过；未以新动作覆盖原站稳成果。完整地形/69旧条件未在此轮全部重跑。

## 可复验性与边界

权重：`{checkpoint}`，SHA `{sha(checkpoint)}`。核验同体编译模型逐字节一致、同份颈后命令SHA、无教师出动作的学生调用计数、保存输出逐拍重算、原生高度/实际越限/前向速度/缓存几何时序；五个代表动作在新进程物理重放，原生20数组全相同。所有正式运行无物理警告。原始动作通路/发放到达详见COMMAND_SOURCE_AUDIT.json。

本轮实际74个身体回合、1,468,000原生步（包含失败JSON烟测20,000步、两个基本回归共68,000步）；固定颈部500,000 tick，其中24份保存命令480,000、首版stride封装API错误前已实际完成但未保存20,000。没有Eon全脑新积分或新环境感觉闭环。第一版Ground没有neck.roots，在archive构造时报错exit1，两个已存颈后轨迹保留；commands_v2改用circuit.port_roots，失败输入留存。首教师烟测NumPy bool无法JSON序列化exit1，20000步TRACE已保存，修复版metrics_v2/v3另存、未篡改旧源。其他实际生成、批次、fit、选择、正式、重放、独立核验、渲染/解码与交付命令均exit0，详细清单见RUN_CHECKLIST。

环境Python {env['python']}、Torch {env['packages']['torch']}、MuJoCo {env['packages']['mujoco']}、FlyGym {env['packages']['flygym']}，单线程CPU/EGL。来源包含实际导入模块、身体资产、模型/初态、固定颈部参数/映射和权重，前后SHA核对；P2资产及旧BODY_CURRENT通过模型不动。新BODY_COMMAND_CURRENT/BODY_STAGE_CURRENT仅登记本轮限定教师相对能力，不能据此把原69测试条件叫69种指令，亦不扩成飞行、任意地形/爬墙、全D1–D8或离体意识实验已完成。

同次视频TEACHER_STUDENT_COMMANDS.mp4四行教师/学生并排，960×1280、400帧、25fps、16s、.125倍慢放；从同次保存原生qpos渲染，只去地面反射，不补动作、不写根轨迹。实际全解码并打开完整身体关键帧和原生指标图。当前仅一个新相位种子配一套新时序，范围有限，不作为广泛鲁棒性统计证明。

下一最小动作：以这版作为已校对的身体指令基线；若扩范围，可按用户次序加入行走中的指令组合/更长多轮变速起停，或再做复杂地形课程，先验同体教师并按相同标准蒸馏，不自动进入脑导航或飞行。
'''
(HERE/'REPORT.zh-CN.md').write_text(report)
checklist=f'''# 本轮运行检查表（P1，S2/S3/S5）

跑前协议PROTOCOL.zh-CN.md、同类DNa01补齐A01_EXTENSION.zh-CN.md；用户裁定OWNER_ACCEPTANCE.json。当前正式teacher-relative {sum(r['relative_pass'] for r in formal)}/8、原接受走停2/2；原绝对结果分栏，旧历史不改。

- 身份：同规范48路/70body/66有限铰链，原PD/附着/初态/力限；ANN＋来源化工程读出/局部CPG；没有新脑/颈部可塑性或物理调参。
- 来源：各INPUTS.json实际导入依赖/教师资产/初态/模型/固定颈部参数和读出/输入权重SHA；同命令、同编译模型，终局核对。
- 命令环境：本目录drive.sh，{json.dumps(env,ensure_ascii=False)}。
- 实际命令：commands.py dev首版exit1（2完成源保留，额外20,000tick未存）；commands_v2.py dev/held、commands_a01.py dev/held退出0；run_body.py烟测exit1（20,000积分TRACE保存，NumPy bool JSON）；batch_pair_v2.py dev DEV、batch_a01.py dev DEV_A01、fit_commands_v3.py、evaluate_candidate.py CALIBRATED --checkpoint FIT/epoch040.pt、freeze_candidate.py CALIBRATED FIT/epoch040.pt均0；batch_pair_v2.py held HELD、batch_a01.py held HELD_A01均同checkpoint且0；parking_regression.py、replay.py、audit_command_sources_v2.py、verify.py、media.py均0。
- PPO/GAE不适用：真实监督复合损失中心差分、Adam/RNG续接、每个4,120update可加载权重/参数SHA、固定40轮选择、独立重放核验完成；模板/足垫/相位/正则化缓冲不变。
- 数据：16当前教师32,000帧＋20,400保留地形，共52,400；校准16/16，正式6211新相位8组（首次使用）；早期模型右转/速度偏离保留。没有不报告失败的换种子，未用正式标签微调。
- 实际积分：身体74回合1,468,000步，颈部500,000tick含失败未存20k；新Eon全脑0。额外只读FK/核验/渲染不计积分。
- 核验：五代表新进程20数组逐值一致，学生teacher_calls0，逐决策动作重算、身体与命令精确绑定，所有警告0。未执行：旧69完整回归/全D1–D8/逐足完整附着滑移账本/飞行/新脑上行闭环。
- 媒体：同次原生状态四动作并排960×1280、400帧、25fps、16s/.125x；ffmpeg全帧解码与ffprobe0，实际打开COMMAND_STATES/COMMAND_METRICS；只去反射不改动作。
- 收尾：手动研究原型；新限定教师相对指令入口，旧通过BODY_CURRENT及源资产不覆盖。桌面报告/视频及源证据关联，tracker/路线图更新，无后台常驻任务。下一动作由用户次序在此身体基线扩组合或地形课程。
'''
(HERE/'RUN_CHECKLIST.zh-CN.md').write_text(checklist)
entry=dict(project='P1_open_source_brain',status=result['status'],implementation_root=str(HERE),checkpoint=str(checkpoint),sha256=sha(checkpoint),
 policy_loader=str(STOP/'policy_repair.py'),body_model_entry=str(BASE/'BODY_MODEL_CURRENT.json'),accepted_for_stage_use=relative and parking_pass,
 scope='teacher-relative walking steer DNa01/DNa02 L/R, stride DNg13 L/R, backward MDN, speed low-high-low-stop; physical48 body',
 evidence=str(HERE/'FINAL_ACCEPTANCE.json'),report=str(HERE/'REPORT.zh-CN.md'),old_full_ground_entry=str(BASE/'BODY_CURRENT.json'))
if relative and parking_pass:
 write(BASE/'BODY_COMMAND_CURRENT.json',entry);write(BASE/'BODY_STAGE_CURRENT.json',dict(**entry,prior_owner_acceptance=str(HERE/'OWNER_ACCEPTANCE.json')))
terrain=json.loads((BASE/'TERRAIN_BODY_CURRENT.json').read_text());terrain.update(command_calibration_entry=str(BASE/'BODY_COMMAND_CURRENT.json'),command_calibration_status=result['status'])
write(BASE/'TERRAIN_BODY_CURRENT.json',terrain)
delivery=Path('/home/sensen/Desktop/engine修复验收/身体四类指令校对-20261117T211320Z');delivery.mkdir(exist_ok=True)
files=('REPORT.zh-CN.md','FINAL_ACCEPTANCE.json','RUN_CHECKLIST.zh-CN.md','OWNER_ACCEPTANCE.json','COMMAND_STATES.png','COMMAND_METRICS.png','TEACHER_STUDENT_COMMANDS.mp4')
delivered={}
for name in files:
 shutil.copy2(HERE/name,delivery/name);assert sha(HERE/name)==sha(delivery/name);delivered[name]=sha(delivery/name)
write(HERE/'DELIVERY.json',dict(directory=str(delivery),files=delivered))
print('Final teacher-relative calibration:',relative,'parking:',parking_pass,'desktop',delivery)
