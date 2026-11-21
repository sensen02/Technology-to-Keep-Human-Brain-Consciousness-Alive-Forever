"""Truthful course result and delivery; never promotes the unqualified student."""
from shared import *
import platform
import shutil
import importlib.metadata

audit=json.loads((HERE/'INDEPENDENT.json').read_text())
assets=json.loads((TEACHER/'PREPARE_INPUTS.json').read_text())['source_sha256']
assert all(sha(p)==v for p,v in assets.items())
sources={str(p):sha(p) for p in sorted(HERE.glob('*.py'))}
sources.update({str(HERE/'drive.sh'):sha(HERE/'drive.sh'),**assets})
physics=[]
for p in sorted(HERE.glob('*/*/TRACE.npz')):
    result=p.parent/'RESULT.json'
    if not result.exists():continue
    r=json.loads(result.read_text());assert sha(p)==r['trace_sha256']
    physics.append(dict(path=str(p),steps=r['steps'],pass_=r['pass_'],mode=r['mode']))
assert len(physics)==60 and sum(p['steps'] for p in physics)==1020000
old_body=BASE/'BODY_CURRENT.json';old=json.loads(old_body.read_text())
assert old['sha256']=='44e0ca425de238a95e0aafe8fa865ace39b5c3bbe5503039578910ea5f79af23'
checkpoint=HERE/'FIT2/epoch030.pt'
environment=dict(python=platform.python_version(),executable=sys.executable,
 packages={n:importlib.metadata.version(n) for n in ('torch','mujoco','flygym','numpy')},
 threads='OMP/OPENBLAS/MKL=1',render='EGL')
result=dict(project='P1_open_source_brain',scope='body terrain imitation course; manual research prototype',
 status='TRAINED_PARTIAL_NOT_QUALIFIED',full_course_pass=False,
 checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),selected_before_held=True,
 development=dict(pass_count=audit['dev_pass'],total=6),held=dict(pass_count=audit['held_pass'],total=6),
 held_stand_count=6,held_complete_cross_count=4,held_block_total=4,
 actual_optimizer_updates=7320,teacher_success_training_episodes=12,original_frames=20400,
 correction_frames=14754,final_dataset_frames=35154,
 new_physical_steps=1020000,new_physics_episode_count=60,new_neural_ticks=0,
 replay_episodes=3,replay_arrays_each=17,replay_exact=True,
 no_runtime_teacher=True,local_CPG_retained=True,pad_head='teacher-label identified phase classifier, not Adam-trained',
 original_body_entry_preserved_sha256=sha(old_body),neck='unchanged fixed nonplastic; archived postneck commands replayed',
 environment=environment,failures=[r for r in audit['records'] if not r['complete_cross_and_original']],
 evidence=str(HERE/'INDEPENDENT.json'),physics=physics,source_sha256=sources,
 limitations=['Original 69 ground-command cases not regressed on this new body.',
 'No new Eon/upstream sensory closed-loop simulation.',
 'No D1-D8 or full slip/adhesion force ledger for this course; prior pad qualification is separate.',
 'Every batch has loss and parameter digest; complete optimizer checkpoints every 10 epochs, not every batch.',
 'FIT1 initial parent hash absent from pre-run INPUTS; full state/Adam/RNG identity verified after run.',
 'New runtime import manifest not fully pinned pre-run; inherited qualified teacher assets and listed stage sources rehashed unchanged.',
 'Changed proprioception and additional data jointly; no isolated causal claim for either.',
 'Biological joint hard stops and pad microstructure unknown; not biological validation.'])
assert audit['dev_pass']==5 and audit['held_pass']==5
write(HERE/'FINAL_ACCEPTANCE.json',result)
report=f'''# 身体局部地形学习实测（P1，S2/S3/S5）

已实际完成教师模仿和两轮同状态纠偏补训，当前是手动研究原型。**开发5/6、独立留出5/6，课程尚未全通过，不替换原已验收身体。**

## 学了什么

使用已合格的同一70body、66有限铰链、42关节位置＋6足垫身体及末端触觉教师，物理、初始化、足垫、关节范围、执行力均未改。12条成功教师回合共20,400个1ms帧；两轮DAgger在实际学生身体当前状态查询隔离教师，教师占比先.5后.2，失败保留且只取物理安全前缀，最终35,154帧，其中14,754个纠偏帧。没有把另一只自由运行教师按时间对齐作标签。

局部CPG保留；逐腿傅里叶运动读出、纠错协同方向和六足垫分类器由教师标签辨识，目标重构RMSE约1.04e−7rad、最大3.20e−6rad，足垫25,122次标签查询，训练标签分类完全一致。感觉网络为当前观察→192→192→6纠错驱动，再经学得读出生成42目标；非GRU，不输入上一动作、教师隐藏计数器、场地ID或绝对位置。第二轮补24项当前足端本体观察，原权重迁移、新连接零初始化，Adam因维度改变明确重置。外部局部CPG、有限PD和学得相位模板均显式声明，不冒称原连接组直接控制。

监督实际执行60＋40＋40轮，共7,320次AdamW更新；每batch记实际损失/参数SHA，每10epoch存模型/Adam/RNG。实际复合损失两次中心差分通过，最大误差2.90e−10；不是PPO，GAE不适用。FIT1全状态含Adam/RNG与父检查点逐值一致；FIT2迁移及优化器重置独立核验。预处理首版重复解压导致慢，训练前中断130，0更新/0积分；新不可变实现一次读入并逐帧核对当前几何时刻，误差0，未改观测时序。

## 走路和站稳同时验收

冻结所选权重后才用1151/1723两新步态种子，每场景2例、每例1.7s，原生dt=.0001s。下面均纯学生执行，教师调用0；命令来自真实固定颈部已归档的到达输出，本轮没有重新积分脑或颈部。

| 场景 | 留出通过 | 前进距离mm | 末200ms速度RMS mm/s |
|---|---:|---:|---:|
| 平地 | 2/2 | 7.495–7.673 | .02149–.02219 |
| .15mm低凸块 | 1/2 | 7.222–7.471 | .02251–.02744 |
| .30mm低凸块 | 2/2 | 6.497–6.965 | .02095–.03476 |

六例均达到末窗站稳门≤.30，非足接触力均0，所有关节目标限幅合规；四例凸块六足均过远缘。但是.15mm/1723在已经跨过凸块后的停车过渡约1.3362s，右中腿coxa→trochanterfemur pitch实际超软限位.055078rad>.02，胸高最低约初高.694<.7；即使最终站稳仍判失败。开发.30mm/263右中腿coxa yaw约1.0452s超限.045791rad，后足未完全跨过，也保留失败。目标没有越界不意味着实际关节不越界。

原BC开发4/6，首轮纠偏仍4/6，新增本体观察＋第二轮数据后5/6；不能据此单独归因某项修改。已保存的其他第二轮检查点只作开发筛查，没有完全解决困难例；留出后不追加调参，也不将它重新叫未见集。新开发教师高块两例也失败，说明上一轮6/6资格仅适用于原先种子，不保证任意相位；失败教师不充当成功示范。

## 核验、限制与下一动作

所选权重：`{checkpoint}`，SHA256 `{sha(checkpoint)}`。12条开发/留出共20,400次保存决策逐行新加载重算，与实际42关节及6垫命令逐值相同；独立核对原生高度/姿态/越限/速度和足端跨越。首留出种子三场景在新进程重放，每例17数组全部逐值相同。共60条实际物理回合、1,020,000原生步；渲染/FK/只读核验不计积分。完整身份见FINAL_ACCEPTANCE.json/INDEPENDENT.json。

视频STUDENT_HELD.mp4来自同次原生状态，四栏含留出失败，不改轨迹。1920×320、340帧、25fps、13.6s，物理采样5ms，.125倍慢放；ffmpeg全帧解码、ffprobe及实际查看全身状态图/指标图/失败瞬间图已完成。只去地面反射。没有拼教师动作。

归档环境：Python {environment['python']}、Torch {environment['packages']['torch']}、MuJoCo {environment['packages']['mujoco']}、FlyGym {environment['packages']['flygym']}，CPU单线程/EGL。入口drive.sh；所有完成训练、采集、开发、留出、重放、verify、media、ffmpeg/ffprobe退出0，行为失败独立记录。首次慢预处理130及先前语法预检失败保留。每batch只有摘要而非完整可重载权重，FIT1父hash跑前未写（跑后全状态一致），新运行完整导入清单跑前未补齐；不得倒填为全面符合执行规范。继承教师资产与各阶段已声明来源SHA终局未变。

原BODY_CURRENT.json及其44e0...af23权重保留。TERRAIN_BODY_POLICY_CURRENT.json仅登记未合格研究候选、deployable=false。原69通道完整回归、新体完整D1–D8/逐足滑移与附着逐点力矩账本、任意扰动/复杂地面/陡坡爬墙、新脑上行闭环均未执行；P2人工脑/肌肉模型未动，颈部无可塑性。

下一最小动作：分开检查跨块受阻例与停车承重切换例，先验证教师在这些状态上的有效纠正及实际软限位响应，再决定新增支撑协同/停车数据或独立物理修复；不能只继续堆迭代或放宽越限/胸高门。今后追加训练须另立新的留出相位，当前1151/1723已被使用。
'''
(HERE/'REPORT.zh-CN.md').write_text(report)
checklist=f'''# 本轮运行检查表（P1 / S2、S3、S5）

权威跑前协议：{HERE/'PROTOCOL.zh-CN.md'}；观察扩充：{HERE/'OBSERVATION_EXTENSION.zh-CN.md'}。报告及FINAL_ACCEPTANCE列全部结果、限制与下一动作。本表为收尾如实汇总，不冒称额外跑前审批或预登记。

- 教师：20261117T184709Z冻结terminal，同体42+6、有限PD、局部CPG；同快照，不改脑/颈；已归档真实颈后两维驱动。
- BC/纠偏：`bash {HERE/'drive.sh'} batch.py train TEACHER_TRAIN`；dev_teacher TEACHER_DEV；`fit.py FIT0 --epochs 60`；`batch.py dev DEV0 --checkpoint FIT0/epoch030.pt`；dagger DAGGER1 beta .5；fit_dagger.py FIT1 --epochs 40 --initial FIT0/epoch030.pt --correction DAGGER1；dev DEV1；dagger DAGGER2 beta .2。输入均绝对路径由脚本HERE解析。
- 扩观察：首fit_plus.py预处理被停止130，0更新0积分；fit_plus_fast.py FIT2 --epochs 40 --initial FIT1/epoch050.pt --correction DAGGER1 --correction DAGGER2退出0。初fit_plus语法多余括号预检失败，在启动前修正；无该版本训练结果。
- 开发/留出/独立：batch_plus.py dev DEV2；screen_checkpoints.py仅开发筛查已有10/20/40轮；batch_plus.py held HELD、replay REPLAY均`--checkpoint FIT2/epoch030.pt`，全exit0；两套最终5/6不冒称通过。原物理门和完整六足跨越要求未放宽。
- verify.py实际运行exit0，后续一次重复只读核验同0；原生物理总1,020,000，新神经tick0，训练7,320updates。PPO/GAE不适用，监督梯度、Adam/RNG和保存重载实核。
- media_student.py、failure_states.py退出0，无新增积分；ffmpeg全解码/ffprobe退出0。340帧/.125倍，同次状态含失败；实际打开STUDENT_STATES、STUDENT_METRICS、FAILURE_STATES。
- 身份：每阶段INPUTS、最终source_sha256；继承教师PREPARE_INPUTS逐项终局重算。缺项：每batch完整checkpoint未存、新运行完整导入跑前清单不足、FIT1父hash仅跑后绑定。见报告，不伪装全规范符合。
- 结果：TRAINED_PARTIAL_NOT_QUALIFIED；原BODY_CURRENT不覆盖、旧产物不删；无后台训练或真实服务/动物副作用。
'''
(HERE/'RUN_CHECKLIST.zh-CN.md').write_text(checklist)
previous=HERE/'PREVIOUS_ENTRIES.json'
if not previous.exists():write(previous,{str(p):json.loads(p.read_text()) for p in (BASE/'TERRAIN_BODY_CURRENT.json',old_body)})
course=dict(project='P1_open_source_brain',status='TRAINED_PARTIAL_NOT_QUALIFIED',deployable=False,
 implementation_root=str(HERE),checkpoint=str(checkpoint),sha256=sha(checkpoint),policy_loader=str(HERE/'policy_plus.py'),
 body_model_entry=str(BASE/'BODY_MODEL_CURRENT.json'),development='5/6',held='5/6',old_body_entry=str(old_body),
 evidence=str(HERE/'FINAL_ACCEPTANCE.json'),report=str(HERE/'REPORT.zh-CN.md'))
write(BASE/'TERRAIN_BODY_POLICY_CURRENT.json',course)
terrain=json.loads((BASE/'TERRAIN_BODY_CURRENT.json').read_text())
terrain.update(status='MODEL_TEACHER_QUALIFIED_STUDENT_TRAINED_PARTIAL',student_policy_status='5/6_DEVELOPMENT_5/6_HELD_NOT_QUALIFIED',
 student_policy_entry=str(BASE/'TERRAIN_BODY_POLICY_CURRENT.json'),student_report=str(HERE/'REPORT.zh-CN.md'),
 next_actions=['verify_teacher_corrections_on_failed_support_transitions','separate_joint_stop_dynamics_from_policy_error','new_holdout_required_if_retraining'])
write(BASE/'TERRAIN_BODY_CURRENT.json',terrain)
assert sha(old_body)==result['original_body_entry_preserved_sha256']
delivery=Path('/home/sensen/Desktop/engine修复验收/身体地形学生训练-20261117T192322Z');delivery.mkdir(exist_ok=True)
names=('REPORT.zh-CN.md','FINAL_ACCEPTANCE.json','INDEPENDENT.json','RUN_CHECKLIST.zh-CN.md','STUDENT_STATES.png','STUDENT_METRICS.png','FAILURE_STATES.png','STUDENT_HELD.mp4')
delivered={}
for name in names:
    shutil.copy2(HERE/name,delivery/name);assert sha(HERE/name)==sha(delivery/name);delivered[name]=sha(delivery/name)
write(HERE/'DELIVERY.json',dict(directory=str(delivery),files=delivered))
print('FINAL: trained partial, dev5/6 held5/6; old model preserved; delivery',delivery)
