from scope import *
import shutil,platform,importlib.metadata

audit=json.loads((HERE/'INDEPENDENT.json').read_text());verify_sources(audit['source_sha256'])
rows=[];details=[]
from metrics_v2 import compare
for f in FAMILIES:
 t=json.loads((HERE/'HELD'/f'{f}_2_teacher/RESULT.json').read_text());s=json.loads((HERE/'HELD'/f'{f}_2_student/RESULT.json').read_text());rows.append(compare(t,s));details.append(dict(family=f,teacher=t,student=s))
assert all(r['relative_pass'] for r in rows)
physics=[]
for p in sorted(HERE.glob('*/*/TRACE.npz')):
 if p.parent.parent.name=='COMMANDS':continue
 r=json.loads((p.parent/'RESULT.json').read_text());assert sha(p)==r['trace_sha256'];physics.append(dict(path=str(p),steps=r['steps'],mode=r['mode']))
command_audit=json.loads((HERE/'COMMAND_AUDIT.json').read_text());assert len(command_audit['rows'])==21
assert len(physics)==47 and sum(r['steps'] for r in physics)==940000
sources=freeze([HERE/n for n in ('PIVOT_SELECTION.zh-CN.md','SMALL_PIVOT_PREREG.zh-CN.md','PIVOT_RATE_CALIBRATION.zh-CN.md','WEIGHT_SELECTION.json')])
env=dict(python=platform.python_version(),executable=sys.executable,packages={n:importlib.metadata.version(n) for n in ('torch','mujoco','flygym','numpy')},threads=1,render='EGL')
result=dict(project='P1_open_source_brain',status='LIMITED_PIVOT_AND_THREE_ACTIVE_BRAKES_TEACHER_RELATIVE_VERIFIED',
 relative_pass=True,formal_pairs=rows,details=details,teacher_absolute_pass=sum(r['teacher_absolute'] for r in rows),student_absolute_pass=sum(r['student_absolute'] for r in rows),
 checkpoint=str(WEIGHT),checkpoint_sha256=sha(WEIGHT),training_updates=0,body_and_neck_parameters_changed=False,
 pivot_scope='DNa02 single-side period120ticks / 250ms pulse train, completed finite-angle turn and stand; continuous large-angle low-translation not passed',
 pivot_yaw_contract='Formal completed yaw at final2s compared to initial, old pulse-off active yaw separately retained; selection and version change before held.',
 brake_scope='DNg60 / DNg74_a / DNg74_b individual drive, forward DNg100 emissions and arrivals continue during hold',
 new_physical_steps=940000,new_physical_episodes=47,new_neural_ticks=420000,
 replay_cases=5,replay_arrays_each=18,source_sha256=sources,physics=physics,environment=env,
 evidence=str(HERE/'INDEPENDENT.json'),
 limitations=['Formal pivot student min thorax height .685035/.684471 below original .7initial gate; teachers .696136/.694260, difference about .01mm; relative pass not absolute pass.',
 'Original long-turn high stimulation and period120/180 long-train failures retained; period180 left also student deviation.',
 'Small-pivot teacher development left transient height failure retained; selection explicitly revised before held to motion-goal + student full-gate criterion.',
 'Only one held phase8861 with new timing; not broad robustness or arbitrary-angle guarantee.',
 'Commands actually generated through fixed true-source neck then identically replayed to teacher and student; not new live Eon sensory-decision closed loop.',
 'Three braking groups share existing body readout; not complete biological mechanism distinction.',
 'No full69 regression, full D1-D8/adhesion-slip ledger, flight, or P2 model changes.'])
write(HERE/'FINAL_ACCEPTANCE.json',result)
lines=[]
for x in details:
 f=x['family'];t=x['teacher'];s=x['student']
 if f.startswith('pivot'):lines.append(f"| {'左' if f.endswith('left') else '右'}有限角度转向 | {t['yaw_degrees']:.2f}° / {s['yaw_degrees']:.2f}° | 最大平移 {t['peak_excursion']:.3f} / {s['peak_excursion']:.3f} mm |")
 else:lines.append(f"| {dict(stop60='DNg60',stop74a='DNg74_a',stop74b='DNg74_b')[f]} 主动制动 | 首合格200ms窗起点 {t['first_still_delay']*1000:.1f} / {s['first_still_delay']*1000:.1f} ms | 持稳RMS {t['hold']['rms']:.4f} / {s['hold']['rms']:.4f} mm/s；推进到达 {s['forward_arrived_during_hold']} |")
report=f'''# 有限角度原地转向与主动停住站稳（P1）

**同体教师相对正式5/5通过；三组主动制动原绝对门3/3通过。** 小角度转向的转角/平移/站稳达到目标，但学生两例有瞬时胸高略低于原绝对线，不能写成全部绝对门5/5。此次是手动研究校对，非成品。

## 本轮实际实施

沿用户接受的教师相对标准，使用当前5e513...ff57e身体网络、同一规范70body/66有限铰链/42位置＋6足垫、原有限PD和附着、同一站姿、局部CPG/工程相位读出。脑侧指定DNa02左/右及DNg60/DNg74_a/DNg74_b的发放，实际经过固定真实来源颈部的轴突传导、受体/膜动力学及静态读出，保存原始发放/到达与颈后两维驱动。教师和学生各自完整独立身体回合消费同一记录。学生teacher_calls=0，没有教师在线给动作，没有额外根/姿态锁定。

初始两相位10组都满足教师相对门，三制动开发绝对6/6。现有网络已具响应能力，**本轮没有重训、没有改脑/颈部/身体物理/读出权重**；只为有限角度原地转向校准脑侧脉冲频率和时长。原四类指令参数与权重逐字节不变，不重复训练覆盖已接受模型。

## 正式新相位8861结果

下表均为“教师 / 学生”。原地转向用周期120tick（12ms）、250ms单侧脉冲串；整个物理回合保持2秒，最大位移按完整2秒计算，没有截掉停止尾段。

| 条件 | 主要响应（教师 / 学生） | 运动质量或持续推进证据 |
|---|---|---|
{chr(10).join(lines)}

三组主动停是在DNg100推进脉冲持续到达时完成，持稳窗支撑比例全部1.0、颈后驱动≤.035，不是撤掉推进假装主动停。首合格时延是首个连续200ms速度/漂移/承重合格窗口的起点，不等同单神经元反应延迟。正式学生DNg60约192.5ms、DNg74_a/DNg74_b约157.5ms进入该窗口；最终200ms也站稳，无物理警告。

原地转向学生最终完成左25.02°/右−23.06°、最大XY偏离初始1.605/1.712mm，均≤原2.25mm；末窗RMS .016/.014mm/s左右。两例学生最低胸高.685035/.684471mm，教师.696136/.694260mm，低约.01mm；学生原高度门失败，但教师相对门通过，实际关节越限0、非足力0。保留为明确残差，不改高度线，也不否定用户已接受的阶段。

## 大角度限制与口径修正保留

初始较强刺激持续约1.3秒时教师/学生可转约±200度，最大平移约3.8–4.8mm，不能叫低平移原地旋转。降低发放率到周期120、180ticks的长脉冲串仍未通过位移门，180档左侧学生还额外偏离教师；所有原始RESULT/TRACE保留。

补做有限250ms脉冲串，开发7013学生左右原绝对门通过；教师左侧有短暂胸高失守。正式前PIVOT_SELECTION明确采用“双方转角/位移达标、学生完整门通过”的开发选择，未冒称教师身体全门通过，也未根据正式结果重选。

短脉冲的脑侧off并非颈后驱动消失或身体停稳时刻。新正式指标将完成转角定义为全2秒末站稳后的偏航相对起始；旧脉冲off瞬时角度active_yaw_degrees同时保存。正式学生脉冲off瞬间约12.98°/−11.18°，最终约25.02°/−23.06°。终点修订在正式有限角度试验前记录，原18度角度线、全2秒2.25mm位移线、全部姿态/关节/站稳线未改；旧RESULT不追写。结论仅是有限角度低平移转向，不外推任意角度、持续多圈或精确原地零位移。

## 复验、环境、交付

权重 `{WEIGHT}`，SHA `{sha(WEIGHT)}`，与上轮相同。五正式学生新进程重放，每例18数组逐值相同；全部26套学生保存动作新加载逐决策复算，与执行的42目标/6垫状态精确一致，同指令SHA、编译身体字节一致。原生高度/越限/位移/速度及持稳推进到达重新核算；旧active和新completed角度并列，未对旧失败静默改判。

实际47身体回合940,000原生步dt=.0001；21份实际颈后命令420,000神经tick；新增Eon全脑积分0、训练/优化器更新0。颈部结构、参数/模型/初态/身体资产/权重及实际导入依赖前后SHA核对。Python {env['python']}、Torch {env['packages']['torch']}、MuJoCo {env['packages']['mujoco']}、FlyGym {env['packages']['flygym']}、CPU单线程/EGL。commands/batch_v2/commands_rate/pivot_rate/commands_small/stopping_commands/stopping_formal/pivot_formal/replay/audit_commands/verify/media实际入口及ffmpeg/ffprobe均exit0；科学未通过另列，不把过程exit0冒充行为通过。旧P2、旧BODY_CURRENT及所有历史通过/失败源不改。

PIVOT_STOP.mp4来自同次原生qpos，800×1200、400帧/25fps/16s/.125倍，五行教师/学生并排，逐列标真实绝对PASS/FAIL。只去地面反射，无后期改动作。全帧解码与实际打开PIVOT_STOP_STATES、PIVOT_STOP_METRICS和旧长转限制轨迹图完成。基本研究样本范围有限，完整D1–D8/69新体整套/完整逐足附着滑移账本/飞行/新全脑感觉闭环不在本轮。

新PIVOT_STOP_CURRENT和BODY_COMMAND_CURRENT追加本轮证据；地面9组19根基础身体响应已覆盖，剩四飞行组8根仍暂缓，持续大角度低平移转向作为行为质量缺口单列，不能从指令组计数中抹去。
'''
(HERE/'REPORT.zh-CN.md').write_text(report)
(HERE/'RUN_CHECKLIST.zh-CN.md').write_text(f'''# P1 / S2、S3、S5 本轮检查表

- 跑前PROTOCOL、补充PIVOT_RATE_CALIBRATION/SMALL_PIVOT_PREREG、正式前PIVOT_SELECTION完整保留。用户教师相对标准，不追改绝对旧结果。
- 固定权重5e513...ff57e、70body/66限位/48执行、固定真实来源颈部/工程动力学/静态读出/局部CPG；变更仅脑侧DNa02脉冲周期/持续时长。训练0，PPO/GAE/Adam变更不适用。
- 实际入口本目录drive.sh：commands.py dev；batch_v2.py dev DEV；commands_rate.py120/3、180/4与pivot_rate.py3/4；commands_small.py120/5、pivot_rate.py5；freeze_weight.py；stopping_commands.py、stopping_formal.py；commands_small.py120/2、pivot_formal.py；replay.py、audit_commands.py、verify.py、media.py均exit0。ffmpeg全解码和ffprobe0。未运行原run_body.py/批次旧副本；打印字段兼容在执行前另存v2。无未收集后台任务。
- 环境{json.dumps(env,ensure_ascii=False)}，各INPUTS实际导入及身体资产/初态/权重/颈部参数/读出/命令SHA，结束复核。47回合940000原生步、420000颈部tick、新Eon0/新训练0。
- 原始基线2相位10组相对通过，三制动开发绝对6/6；长转/两低率长转失败保留；短串开发教师左高度失败保留、学生完整通过；正式相对5/5、教师绝对5/5、学生绝对3/5（两原地瞬时高度FAIL），不混淆。正式转角终点修订在运行前声明，旧active值保留。
- 五新进程正式学生18数组精确；26学生逐决策输出精确、身体/命令同一、无警告；保存原生qpos/qvel/接触载荷/实际动作/足端轨迹。完整D1–D8/69整套/全附着滑移账本/飞行/脑感觉闭环未执行。
- 媒体同次原生姿态800×1200、400帧/25fps/16s/.125x，仅去反射；实际打开3图含长转共同限制，源视频全解码0。报告桌面交付；手动研究原型，不作成品/生理真实性保证。
- 下一动作可按用户次序使用有限角度转向/主动制动基线，或研究持续大角度转向的全身承重与摆腿协同；飞行暂缓，不自动扩任务。
''')
entry=dict(project='P1_open_source_brain',status=result['status'],checkpoint=str(WEIGHT),sha256=sha(WEIGHT),policy_loader=str(STOP/'policy_repair.py'),
 implementation_root=str(HERE),body_model_entry=str(BASE/'BODY_MODEL_CURRENT.json'),evidence=str(HERE/'FINAL_ACCEPTANCE.json'),report=str(HERE/'REPORT.zh-CN.md'),
 teacher_relative_pass=True,absolute_student_pass='3/5; pivot transient height fails',pivot_scope=result['pivot_scope'],active_brake_groups=['DNg60','DNg74_a','DNg74_b'])
write(BASE/'PIVOT_STOP_CURRENT.json',entry)
for name in ('BODY_COMMAND_CURRENT.json','BODY_STAGE_CURRENT.json'):
 data=json.loads((BASE/name).read_text());data.update(pivot_stop_entry=str(BASE/'PIVOT_STOP_CURRENT.json'),ground_groups_with_basic_response=9,ground_axons=19,flight_deferred_groups=4,pivot_large_angle_low_translation='NOT_PASSED')
 write(BASE/name,data)
terrain=json.loads((BASE/'TERRAIN_BODY_CURRENT.json').read_text());terrain.update(pivot_stop_entry=str(BASE/'PIVOT_STOP_CURRENT.json'));write(BASE/'TERRAIN_BODY_CURRENT.json',terrain)
delivery=Path('/home/sensen/Desktop/engine修复验收/原地转向与主动停稳-20261117T224845Z');delivery.mkdir(exist_ok=True)
files=('REPORT.zh-CN.md','FINAL_ACCEPTANCE.json','RUN_CHECKLIST.zh-CN.md','PIVOT_SELECTION.zh-CN.md','PIVOT_STOP_STATES.png','PIVOT_STOP_METRICS.png','LONG_PIVOT_LIMITATION.png','PIVOT_STOP.mp4');delivered={}
for n in files:
 shutil.copy2(HERE/n,delivery/n);assert sha(HERE/n)==sha(delivery/n);delivered[n]=sha(delivery/n)
write(HERE/'DELIVERY.json',dict(directory=str(delivery),files=delivered))
print('Final: teacher relative5/5, student absolute3/5, same body weights; delivered',delivery)
