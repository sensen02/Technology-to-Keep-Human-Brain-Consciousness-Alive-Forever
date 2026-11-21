"""Record the verified local fix and explicitly reject whole-body promotion."""
from common import *
import shutil,platform,importlib.metadata

selection=json.loads((HERE/'SELECTION.json').read_text());audit=json.loads((HERE/'INDEPENDENT.json').read_text())
reference=json.loads((HERE/'FOOT_HEIGHT_COMPARISON.json').read_text());body_entry=BASE/'BODY_CURRENT.json'
old_body_sha=sha(body_entry);old_terrain=json.loads((BASE/'TERRAIN_BODY_CURRENT.json').read_text())
write(HERE/'PREVIOUS_ENTRIES.json',{str(body_entry):json.loads(body_entry.read_text()),str(BASE/'TERRAIN_BODY_CURRENT.json'):old_terrain})
physics=[]
for path in sorted(HERE.glob('*/*/TRACE.npz')):
 r=json.loads((path.parent/'RESULT.json').read_text());assert sha(path)==r['trace_sha256']
 physics.append(dict(folder=str(path.parent),**r))
assert len(physics)==50 and sum(r['steps'] for r in physics)==850000
verify_sources(audit['source_sha256'])
original=next(r for r in physics if r['folder']==str(HERE/'DEV_ROUTING/block015_1723'))
assert original['pass_'] and original['joint_excess']==0
new_held=[r for r in audit['records'] if r['group']=='HELD']
development=[r for r in audit['records'] if r['group']=='DEV_ROUTING']
assert sum(r['pass_'] for r in new_held)==4 and sum(r['pass_'] for r in development)==10
sources={str(p):sha(p) for p in HERE.glob('*.py')}
for p in ('PROTOCOL.zh-CN.md','SELECTION_REVISION.zh-CN.md','BODY_QUALIFICATION.zh-CN.md','drive.sh'):sources[str(HERE/p)]=sha(HERE/p)
environment=dict(executable=sys.executable,python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in ('numpy','torch','mujoco','flygym')},threads=1,render='EGL')
result=dict(project='P1_open_source_brain',status='SPECIFIC_STOP_LIFT_FIXED_WHOLE_BODY_NOT_QUALIFIED',
 production_promoted=False,deployable=False,checkpoint=selection['checkpoint'],checkpoint_sha256=selection['sha256'],
 control='unchanged sensory neural weights, fitted motor/pad readout + explicit local CPG; signed phase-feasible motor readout correction',
 original_stop_case=dict(pass_=True,left_middle_foot_peak_before_mm=reference[0]['left_middle_peak'],left_middle_foot_peak_after_mm=reference[1]['left_middle_peak'],
   comparison_window_s=[1.3,1.7],joint_excess=original['joint_excess'],stand_rms=original['stand_rms'],dx=original['dx']),
 development=dict(pass_count=10,total=12),first_frozen_held=dict(pass_count=4,total=9,stand_count=sum(r['stand_rms']<=.30 for r in new_held)),
 new_matched_regression='0.30mm/181: parent original pass, selected routing actual joint excess .027009> .02; retained FAIL, do not promote.',
 replays=dict(episodes=6,arrays_each=17,all_exact=True),actual_experimental_optimizer_updates=2920,
 training='37,112 frames, 40 epochs; actual update weights/gradient/Adam/RNG verified, not selected because worse physical result8/12',
 neural_only_probe='fine-tuned weights with legacy readout: 1/4; rejected, no extra training or new held4313/5903 run',
 new_physical_steps=850000,new_physics_episode_count=50,new_neural_ticks=0,
 source_sha256=sources,environment=environment,evidence=str(HERE/'INDEPENDENT.json'),physics=physics,
 full_ground_69_regression='NOT_RUN',whole_D1_D8='NOT_RUN',new_brain_neck_closed_loop='NOT_RUN',
 original_production_entry_sha256=old_body_sha,source_body_unchanged=True,
 limitations=['Old end-point stance does not prove whole transient safe; original local lift fixed but whole body candidate fails.',
 'Sign/phase intervention supports a causal contribution; teacher shadow is not proof of only biological mechanism.',
 'Development raw all-gates and complete-foot-cross both audited; held4/9 not reliability estimate over all terrains.',
 'Selection revised explicitly before first new held; later neural-only probes use already-seen cases only; no hidden reuse as new held.',
 'Only declared engineering readout; no biological neural CPG claim. Neck fixed, replayed archived postneck command, no current neural simulation.'])
write(HERE/'FINAL_ACCEPTANCE.json',result)
report=f'''# 停车突然高抬腿：已定位并消除原例，整体验收未通过

本轮P1开源脑项目身体侧手动研究修复，非成品；P2人工脑、规范肌肉和旧已验收权重未改。

## 发现和改动

用户观察正确：高抬的是左中腿。原.15mm块/1723回合已跨过块，1.3秒撤令后左中跗节5高度达到2.11167mm。其前右中腿在支撑相位仍收到很大的正纠错驱动，随后左中腿的网络纠错输出也升到约30–36。影子教师只读取同一只学生的实际当前状态、不施加动作，qpos/qvel/control/pad/x/phase与旧回合逐值相同；约1.35秒左中腿教师标签0而学生约30.73，1.36秒0对32.77。异常不是正确脑指令要求高抬腿。

旧学生把教师带符号的净相位纠错裁成非负、没有强制纠错方向与相位一致。教师的冻结运动读出摆动相位为正、支撑相位可为负。修复把六神经驱动投影到逐腿当前相位允许的方向/幅度区间[0,80g]或[80g,0]，g来自同一冻结教师相位曲线（数值核验最大差3.26e−8）。保留原感觉网络权重、CPG、运动模板、足垫分类器和全部物理参数。没有按腿ID/具体时刻/障碍ID写抬腿或停车脚本。此投影是显式工程运动读出，不是新训练生物VNC；当前只能把它作为有来源的工程约束。该配对干预支持相位错配对本次异常有因果贡献，不能说它是所有失败唯一根因。

同一原失败回合，撤令后左中足峰高 **2.11167→.076982mm**，实际关节越限 **.055078→0rad**，最低胸高从约初高.694提高到.862，前进6.46568mm、六足已跨过凸块，末200ms速度RMS .221933≤.30。跨越/停车/限位原门全部通过，六个代表回合独立新进程17数组逐值相同，45,900保存决策逐行重算与执行命令完全一致，全部新编译BODY.mjb字节与旧同场景身体相同。

## 为什么还不能替换身体版本

修复不是整体验收成功。12个已用相位/场景开发 **10/12**，仍有.30mm/181（实际软限位越界.027009）和263（.034198）失败；181是旧学生原来通过的匹配回合，属于明确新回归，必须保留。新留出2039/2711/3253×三场景 **4/9**，2711三例末窗RMS .323063/.342149/.368839均>.30，3253低块胸高下降、高块卡住/越限；不是把最后站住或前进了就算通过。留出后未用这些结果重选当前ROUTING_FIXED权重，不再称这些种子未见。

还做了真实补训尝试：原成功教师/两轮纠偏安全前缀＋同状态影子安全前缀共37,112帧，40epoch、2,920次AdamW更新，实际复合损失中心差分通过、原Adam/RNG精确续接，每更新完整可评估权重和SHA、每10epoch全优化器保存。监督开发RMSE .005205→.003304但物理开发退到8/12，故拒绝采用；把同一补训权重配回旧读出另4例仅1/4、拒绝。没有额外重训或跑新4313/5903留出。

原预登记对未全通过候选只按MSE排序不够合理，留出前明确追加SELECTION_REVISION.zh-CN.md，先完整物理通过数、再MSE；不是静默改指标。所有原胸高、限位、站稳、跨越等门不变。相位修复候选仍不合格，故 **deployable=false，不覆盖BODY_CURRENT或旧地形学生入口**。研究状态单列BODY_STOP_REPAIR_CURRENT.json；这种失败不能用“修复了一个视频现象”掩盖。

## 核验与交付

本轮50物理回合、850,000原生步（dt=.0001s），新脑/颈部tick0，可塑性关闭；下行是已归档真实颈部到达驱动，不是本轮脑感觉新闭环。补训为实验候选，最终展示修复版的感觉参数未更新。

运行环境Python {environment['python']}、Torch {environment['packages']['torch']}、MuJoCo {environment['packages']['mujoco']}、FlyGym {environment['packages']['flygym']}，CPU单线程/EGL。drive.sh调用diagnose_saved/inspect_phase/shadow_batch/policy_repair/probe_repair/batch_repair/fit_repair/probe_fit/select/probe_neural/verify_repair/media_repair均exit0；行为失败逐例留档。初次调用旧drive.sh传绝对脚本导致拼接路径错误exit2，在导入/积分前；新本目录drive.sh正常。来源清单含实际已导入项目依赖、教师资产、初态、权重，前后SHA核对，INDEPENDENT和FINAL_ACCEPTANCE含证据。原神经脑/颈部与生物硬限位结论不扩张。

STOP_REPAIR.mp4三栏为旧失败/原例修复/仍失败的高块，1440×320，340帧、25fps、13.6秒、.125倍慢放，同次保存原生姿态，无后期改动作。左中腿仅渲染标红、去地面反射，几何不变。ffmpeg全帧解码及ffprobe通过；已实际打开原例高抬/修复/失败保留全身图和原生指标图。不能以视频替代完整物理合格。

## 身体阶段还需什么

详见BODY_QUALIFICATION.zh-CN.md。当前原例修复只属于局部课程成果；“地面实验平台身体合格”还要：新48路身体原69条地面指令回归，覆盖速度/转向/后退/停车并列原地转向漂移缺口；多个步态相位/速度/块前块上块后停车再起步；限定未见块/台阶/浅沟/温和坡度/摩擦变化；轻推/短暂足滑后的局部恢复和持续多轮起停；原生支撑/滑移/附着/力矩/质心账本；脑命令实际过固定颈部、切断恢复与上行到达验证。标称范围和可恢复扰动范围分开，不要求网络在超负荷条件下永不跌倒。

爬墙/爬树需独立验方向性附着、曲面脱附和视觉—触觉协调；飞行继续暂缓。下一个最小动作应是先核同体教师在不同停车相位/承重组合下的有效纠正，构建相位分组的停车换支撑课程，复核新回归与软限位响应，而不是继续用现有固定撤令时刻堆迭代。
'''
(HERE/'REPORT.zh-CN.md').write_text(report)
checklist=f'''# 本轮执行检查表（P1 / S2、S3、S5）

跑前协议：{HERE/'PROTOCOL.zh-CN.md'}。选模的留出前显式修订：SELECTION_REVISION.zh-CN.md。后续未采用的训练-only读出试探：NEURAL_ONLY_PREREG.json，不复用旧留出冒充新数据。

- 问题/身份/目标/预算/种子/原门/控制来源：见PROTOCOL；冻结70body、66有限铰链、42＋6、原有限PD、工程附着；ANN＋局部CPG与来源化读出。未改物理、无根写入/额外力。
- 输入身份：实际导入本地模块和教师PREPARE资产、初态、权重在各INPUTS.json；源码未在活动运行中修改。新文件新入口独立保存，全部终局SHA检查。
- 实际命令：本目录`bash drive.sh diagnose_saved.py`、inspect_phase.py、shadow_batch.py、policy_repair.py、probe_repair.py、`batch_repair.py dev DEV_ROUTING --checkpoint ROUTING_FIXED.pt`、fit_repair.py、probe_fit.py、`batch_repair.py dev DEV_FIT --checkpoint FIT_REPAIR/epoch030.pt`、select.py、`batch_repair.py held HELD --checkpoint ROUTING_FIXED.pt`、`batch_repair.py replay REPLAY --checkpoint ROUTING_FIXED.pt`、probe_neural.py、verify_repair.py、media_repair.py；均退出0。原错误旧drive绝对路径exit2，0积分。
- 实际环境：{json.dumps(environment,ensure_ascii=False)}。
- 正确性：非PPO/GAE不适用；真实监督复合损失中心差分、Adam/RNG续接、全2,920更新可加载权重SHA、每更新参数digest逐项核验。选用候选实际0SGD，补训2,920updates为未采用实验；不混淆二者。
- 物理：50回合850,000原生步；新脑/颈tick0。开发10/12、新留出4/9，不全合格。六新进程复跑各17数组精确；27开发/留出/重放45,900决策输出精确；compiled BODY字节相同。
- 失败/未执行：高块181新回归、263旧失败、新2711三停车失败/3253两地形失败保留；补训开发8/12/旧读出探针1/4拒绝。原69回归、全D1–D8、逐足完整滑移/附着账本及新脑感觉闭环未执行。
- 媒体：原生qpos同次渲染，只标红左中腿/去反射；340帧/.125x、1440×320、25fps/13.6s；ffmpeg/ffprobe0，实际打开STOP_REPAIR_STATES与STOP_REPAIR_METRICS。
- 当前结论：原指定停车高抬腿在对照中修复，但整体验收NOT_QUALIFIED、deployable=false；原生产及旧地形学生指针不覆盖，P2不动；无后台任务/真实动物或硬件副作用。下一动作见报告及BODY_QUALIFICATION。
'''
(HERE/'RUN_CHECKLIST.zh-CN.md').write_text(checklist)
entry=dict(project='P1_open_source_brain',status=result['status'],deployable=False,implementation_root=str(HERE),
 checkpoint=selection['checkpoint'],sha256=selection['sha256'],policy_loader=str(HERE/'policy_repair.py'),
 evidence=str(HERE/'FINAL_ACCEPTANCE.json'),report=str(HERE/'REPORT.zh-CN.md'),qualification_plan=str(HERE/'BODY_QUALIFICATION.zh-CN.md'),
 development='10/12',held='4/9',specific_stop_lift='fixed paired original case, six independent representative replays exact',
 old_production_preserved=True,original_terrain_policy_unchanged=True)
write(BASE/'BODY_STOP_REPAIR_CURRENT.json',entry)
old_terrain.update(latest_stop_repair_entry=str(BASE/'BODY_STOP_REPAIR_CURRENT.json'),latest_stop_repair_status=result['status'],
 next_actions=['qualify_teacher_support_transitions_across_stop_phases','preserve_old_passes_while_fixing_known_regressions','new_69_case_regression_and_terrain_course'])
write(BASE/'TERRAIN_BODY_CURRENT.json',old_terrain)
assert sha(body_entry)==old_body_sha
review=dict(full_video_decode_exit_code=0,ffprobe_exit_code=0,frames=340,width=1440,height=320,fps=25,duration_s=13.6,playback=.125,
 actually_opened=['STOP_REPAIR_STATES.png','STOP_REPAIR_METRICS.png'],observations='Red left middle leg visibly shoots above body in old1.36s and stays near ground in paired routing repair; full body/legs/wings retained; repaired high block FAIL remains labelled. Metrics show original stopping dip/overshoot resolved but separate physical regressions kept.',new_physical_steps=0)
write(HERE/'MEDIA_REVIEW.json',review)
delivery=Path('/home/sensen/Desktop/engine修复验收/停车异常抬腿修复-20261117T202200Z');delivery.mkdir(exist_ok=True)
files=('REPORT.zh-CN.md','BODY_QUALIFICATION.zh-CN.md','FINAL_ACCEPTANCE.json','RUN_CHECKLIST.zh-CN.md','STOP_REPAIR_STATES.png','STOP_REPAIR_METRICS.png','STOP_REPAIR.mp4','MEDIA_REVIEW.json')
delivered={}
for name in files:
 shutil.copy2(HERE/name,delivery/name);assert sha(HERE/name)==sha(delivery/name);delivered[name]=sha(delivery/name)
write(HERE/'DELIVERY.json',dict(directory=str(delivery),files=delivered))
print('Recorded local fix, whole-body NOT_QUALIFIED, old entries preserved; desktop',delivery)
