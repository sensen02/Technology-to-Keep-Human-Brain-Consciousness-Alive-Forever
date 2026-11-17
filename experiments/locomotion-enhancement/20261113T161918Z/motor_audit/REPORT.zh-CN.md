# 运动增强 ONLYCODE 源码审计（P1/P2/S2/S5）

本报告为只读设计审计，未导入项目 Python 模块，未加载 Torch／MuJoCo／NPZ 数组，未运行训练、网络 forward 或物理。仅新增本目录报告和源码 SHA 清单；不修改旧科学文件与全局 tracker（由父代理统一登记）。工作树既有 AGENTS／ROADMAP／TRACKER 修改保留。当前权威为 `/run/media/sensen/Data2/cell_wound_prototype/NECK_CURRENT_STATE.json:6` 指向的 20261113T064313Z，不能用早期状态覆盖。本报告所有实验数量、阈值均为建议，尚未执行或通过。

## 1. 决定实施路线的结论

1. **motor789f 的 command_dim=2 实际只有一维自由度**。源码显式要求 `command[0]=relu(command[1])/3`；传独立速度会抛 ValueError。不能以“数组有两列”当作已有独立 speed/turn 能力。
2. **第一列为 0 不等于前进速度为 0**。右转及中性命令都继续运行不接 command 的 walking base128；第一列是残差网络的规范化任务量而不是实测 COM 速度。当前颈部 decoder 也只有一个输出，再固定投影成两列。因此现版没有神经 zero-forward pivot 的独立控制通道；代码不排除残差偶然抵消前行，但那只能经物理证明，不能从 `[0,turn]` 推定。
3. **原样冻结789f无法做独立2维身体命令实验**。第一批应含严格ABI拒绝检查及合法耦合身体基线。独立2维对照可借旧 FastMotor（仅小幅减速域），或另建新 ABI 的 ANN 身体适配器；后者即使冻结789f所有张量，也须标新结构/新接口，不能继续称原789f同模型。
4. **来源 neck 有2个DN维度，当前 decoder 主动压成1维**。可以另立 `2→16→2` ANN decoder，配合脑侧独立2维任务量与新身体控制器训练，仍沿真实ID队列。不增加虚构纤维，不赋予DN已知生理 speed/yaw 含义。仅改 decoder 无法消除冻结 motor 的耦合与持续前行 base。
5. **现身体对地面的识别固定为 PLANE**。直接加 hfield 或盒子会漏统计足载/非足接触；足底高度也是相对世界 z=0。粗地形必须连同 ground mask、局部净空、接触切向速度与记录器一起派生，避免“粗地形上指标看起来好”来自漏计接触。

## 2. 当前 runtime → BASE 的准确链路

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T064313Z/sensory_body_repair/runtime.py:4-12`：加载045302 runtime，复制函数 code object 并重绑 globals；新 fusion400/odor250 绑定，14-32严格装载四网络。第28行检查 motor 状态 `[128,64,64]`；总递归状态仍336。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T064313Z/sensory_body_repair/adapter.py:4-12`：runner/audit 等直接编译045302原脚本 AST，并注入新 runtime 命名空间，不是新独立数值循环。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T045302Z/sensory_body/runtime.py:4-13`：继承033622 visual runtime 的 BASE/MOTOR/project；66-96 的 `BrainNeckLoop` 是实际多感官闭环。第80行只从身体抽111允许通道给 AN 编码器，85行脑只接延迟AN+视觉/嗅觉神经量，86行 `decoder(dnheld)→turn→project(turn)`。第87-92行通过真实来源队列传输。新任务不能直接将 cue/speed/turn 从脑侧送 motor。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T033622Z/visual_body/runtime.py:6-9`：`BASE=/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z`；motor=`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning/COMBINED.pt`，SHA `789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd`。第52-58行 `pin()`严格检查 BASE 导入位置，`project(turn)`实现 `[relu(turn)/3,turn]`。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/student_wave.py:21-35,103-105`：部署观测238、决策5ms；`TARGET_SPEED=.075` 为旧观测特征常量，不是当前速度命令。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/physics.py:24-35,43-54`：动态加载同目录 `sources/repair_core.py`、增量执行器与本体感觉快照；12.5µs原生步。`tick_rate()`139-164每1ms做80物理步，`tick_policy()`166-172可保持5ms动作。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/sources/repair_core.py:31-36,84-100`：实际全身 XML `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-triad-pad/20261110T010000Z/MODEL_TRIAD_x200.xml` 与姿态 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-triad-stand-pose/20261110T030000Z/STAND_POSE_TRIAD_1.45.json`。本次仅读引用，未读数组或执行装载。

## 3. ABI 耦合与 pivot 的具体位置

`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning/model.py`：

- `CombinedTurning.initial_state(batch_size)`32-33：base128、RIGHT64、LEFT64，总256。
- `step(obs,command,state)`34-45：要求 `(B,238)/(B,2)`；39行严格拒绝 `|turn|>1` 或 `|f-relu(turn)/3|>1e-12`。41行 `base.step(obs,state[0])` 与 command 无关。42行内部重新构造 RIGHT=`[0,-relu(-turn)]`、LEFT=`[relu(turn)/3,relu(turn)]`；43行两分支每次都前进，44行按 turn 正负加 RIGHT bound3 / LEFT bound6 残差。53行 identity 完整声明域。
- 因而 `[0,0]` 是原 walking base 的中性行走；`[0,-.5]` 是合法右转行走；`[0,+.5]` 被拒绝；`[-1,0]` 所谓“零速度”也被拒绝。移除39行检查并不会产生独立speed能力，因为42行仍丢弃外部第一列。

`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/neural_calibration/model.py:11-18`：反馈111→32→1、脑GRU16→DN2、decoder **2→16→1**。当前fusion继承它：`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T045302Z/sensory_fusion_calibration/model.py:5-14`只把脑输入改成3，不扩展 decoder；`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T064313Z/fusion_aggregation_repair/model.py:4-9`继续继承。旧公共身体接口 `/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/body_interface/interface.py:72-89`也明确是单turn decoder+固定投影。不能拿其 `motor_command()`直接当新版独立任务接口。

## 4. 已存在可借的 speed2 训练与教师

### 独立命令ANN存在，但不支持停走／pivot

`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/fast_motor.py` 提供 `FastMotor(base)`、`initial_state(B)`、`step(obs,command,state)`、`sequence(...)`、`load(path,expected_sha)`：36-63行是冻结base128+即时ANN残差，两个命令分别进入特征；60行限制 **forward_scale_minus1∈[-.15,0]，turn∈[-1,1]**，75-79行 identity 说明归一化 `[.15,1]`。故只能测试名义速度 `.075*(1+u0)` 的0.85—1倍，不能给u0=-1、不能称stop或pivot已有实现。全零时gate为0且保留walking base。

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/fast_data.py:11-41` 的 `prepare(model)`校验实际 teacher samples与manifest；覆盖 left_full/right_full 和 straight_085/straight_090 四类，**不是完整speed×turn因子网格**。不能把代码域等同物理资格域。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/train_fast.py:17-60` 有固定1000步Adam/每100保存/独立进程保存重放；`fast_data.metrics()`56-57是拟合MSE，不是运动成功。可借训练骨架，但旧超参数与旧成败不自动迁移，须新预登记。

### 教师是 speed/yaw 两量，但原地实现明确缺失

`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/teacher/turning_teacher.py`：`ConditionalTwistReference(base,speed_scale=1.,yaw_rate=0.,prime_s=4.8,ramp_s=1.)`33行，`command(elapsed_s)`46-53返回phase、scale、curvature，`sample(elapsed_s)`55-68输出身体坐标足端目标；`make_teacher(core,*,speed_scale=1.,yaw_rate=0.,prime_s=4.8)`71-78。曲率=yaw_rate/(.075*speed_scale)，**36-37显式拒绝speed_scale=0且yaw_rate≠0**。旧减速API `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/deceleration_moderate/api.py:10,16-22`已有0.85、0.90、1.0课程与 `[scale-1,0]`编码。其24-29减速门保留旧前进门，只适用仍前行的任务。

另一教师 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_finite_twist/turning_teacher.py:22-57,60-90` 有有限SE(2)足端映射 `finite_xy_velocity(rest,target,velocity,k,kdot)`；62-65只准speed_scale=1，不能当pivot教师。新pivot教师应从有限足运动/IK教师独立实现并物理自检，保持 `v=0,ω≠0` 时足端旋转速度仍非零，避免除v或随speed将整个相位冻结。教师仅采集/预热/损失监督；最终部署必须ANN，经neck控制；不允许把上述算法教师作为外置CPG继续运行。

## 5. 实测 yaw 与记录链

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T045302Z/sensory_body/runner.py:36-51`：ANN均值经tanh为42动作；teacher仅预热，随后真实 `tr.tick()`；45行从实际 `c.d.xmat[c.th]`取 `atan2(R10,R00)`，50行unwrap，51行以5.8s交接边界计算yaw。它不是根部轨迹命令。54-65保存trace/neural/geometry与各SHA。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/diagnose.py:34-63`：实际教师动作入物理、每ms记录yaw；39行名义speed/yaw是命令，54行另存规范化command2；57-63用实际yaw/COM算 achieved yaw、身体朝向投影速度，不能把请求yaw_rate当实现值。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/sources/repair_core.py:139-182`：每接触测力，159-167足载加权实际接触点速度；175-179足端几何与胸部坐标足位置。
- 现代码提供实测躯干yaw与足接触运动，**未发现本次已读路径里独立“由实际足速度反推yaw”的验收实现**。新诊断可对承载足在身体平面的相对足速度做加权最小二乘：`p_dot_body_i≈-v_body-ωJp_i`，给ω估计、残差和秩/承载有效性；这是分析指标，不得反馈生成关节或root。需注明body原点与COM平移差及滑移影响，并与躯干实测yaw并列，不能强制二者相等。

## 6. 粗糙地形已存在实现及移植风险

`/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/scene.py`：

- `RoughnessConfig`315-385：kind=none/heightfield/grass/scattered，amplitude、wavelength、seed、rows/cols、extent；非平地要求正幅度波长，base_offset>0、至少4格/波长。
- `_heightfield_data(mj,cfg,scene_cfg=None)`970-998：确定性seed，sin/cos基底+随机高频量并归一化[0,1]；amplitude为实际峰峰值。**seed改变随机项，并不改变主要sin/cos结构**，多seed不能冒充多结构覆盖。
- `_roughness_scatter_positions(cfg)`1030-1037：seed+7919的球体散布；接触拓扑不同，需另立结构等级。
- `apply_scene(world,fly,cfg,ledger=None,*,measure=True)`1272-1399：编译前构造hfield、替换ground pairs及contact sensors，未换pair可能根本碰不到地形；`SceneApplicationReport.measured_after_compile(...)`1083-1190读回真实hfield数据/网格/幅度/接触配置。

该API依赖FlyGym world/fly命名和映射，不是当前Core(PAD_XML)的直接入口。应在新派生完整身体XML/loader中复用算法或小型独立地形生成器，不替换全身平台。原代码字段后缀mm不能证明当前native长度有SI标定；以native单位和腿长比重新声明。

必须修正的测量假设：

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/sources/repair_core.py:100` ground仅PLANE；149-154足载与非足、227-245附着/接触计数均依赖此mask。新增显式ground geom ID集合覆盖hfield/boxes，并核对所有接触被分类，不能仅改画面。
- 同文件169-179 `padz`是球底世界z，非局部地面净空；`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/physics.py:78-84`把它喂给感觉，CycleTracker也用padz。新任务同时记录 `pad_bottom_world_z` 与 `pad_clearance_above_local_surface`，给新版周期/感觉明确版本，保持旧值可复核。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z/runtime.py:116-135` PeakRecorder同样只认c.planes。NativeRecorder及teacher支撑几何也须按ground mask审查；仅把c.planes改名仍须检查语义。
- 粗面滑移需按实际接触面法向投影切向速度；旧139-167只取worldXY速度。旧滑移仍保留，新增terrain-tangent速度和载荷分母/缺失标志。

## 7. 最小有限第一批建议：先身体控制能力，再颈部新decoder

### A. 仅身体命令诊断：计划9回合，条件不足则停止依赖分支

先做不含物理的源码/ABI检查：合法789f `[0,0]、[1/6,.5]、[0,-.5]`，独立 `[0,+.5]、[-.15,0]`预期被拒绝，不绕检查。实际检查由父执行；本报告未执行。

A1：冻结789f三完整回合，以上三个合法命令，采用既有连续1s settle+4.8s教师预热+9.6s ANN，不reset交接，左右同幅并保留体态、六腿、71mesh/42有限PD、全原生力审计。命令由明确实验协议施于身体侧，**不称新脑控制、不是绕颈部署**。

A2：选定一个实际可校验SHA与独立保存资格的冻结FastMotor（不能凭latest选权重），用其合法域做 `u0∈{0,-.15} × turn∈{0,-.5,+.5}` 六回合。可诊断速度列是否影响身体、转向是否仍偏慢；全部是身体侧命令实验，不能称neck已控制speed。不同motor为架构对照，不能将差异唯一归因于speed。若找不到合格旧checkpoint则只完成A1并准备新body ANN适配器，不能冒用未知权重。

每回合15.4s/1,232,000native步，九回合预算11,088,000步；每个回合一次保存并独立审核，不为渲染重物理。先固定配对初态和一个种子做诊断，非稳健性证明。停止条件沿既有数值异常/连续危险阈值，失败保留；没有表现问题不临时扩批。

A3（后续另立有限训练）：现冻结789f的speed独立性不成立，若要用其张量作初始化/教师，在新时间戳实现新的command ABI或新增ANN speed residual；需有ANN输出可抵消/重塑base、可覆盖v=0而保持转向足运动。冻结原张量的事实与新网络训练事实分别记。不得用乘速度直接把全关节动作压0来代替神经停走学习。

### B. neck独立2维任务量

新脑接收脑侧实验二维任务量 `[desired_v/v_ref, desired_omega/omega_ref]`（或声明的等价编码），仍只接delayedAN反馈；两DN保持来源ID、顺序、范围与2ms队列，不插入原始task字段。身体 `decoder(delivered_dn)`改为ANN 2→16→2，输出新body motor的两个任务量。训练覆盖独立speed×turn与v=0左右pivot；current brain3/decoder1/checkpoint不兼容，另立格式、loader、loss、rollout和记录字段。两个来源纤维的工程编码不等于原生功能归因。

至少独立验证speed/turn扰动矩阵或Jacobian的二维有效秩、四角与中性/停止留出、DN通路切断、AN切断、同源队列/保存重放、无脑→身体旁路。工程秩仅说明输出自由度，行为仍须完整身体物理证明。

### C. pivot新任务指标，旧forward门保留

`/run/media/sensen/Data2/cell_wound_prototype/WALKING_RESEARCH_PROTOCOL.zh-CN.md:23-28`要求旧五门不变。`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/physics.py:332-385`实际旧forward是worldX≥.05，旧slip分母是COM净XY位移/时长。对原地任务二者不适用，但仍计算并标旧门FAIL/不适用解释，不能删除或降低旧前行验收。

新版本pivot契约建议在跑前冻结：从静态全身几何定义体长L_body（头到腹尾的身体本轴投影长度，排除腿/翅，保存所用mesh/body边界和姿态/hash；未实测前不写数值），旋转窗口与命令ramp分离；测 `Δyaw=unwrap(yaw_end-yaw_start)`、yaw_rate(rad/s)、请求符号一致率、达到目标角的时间，另报相对匹配zero-command yaw。漂移给 `||COMxy_end-COMxy_start||/L_body`、最大偏离/L_body与累计路径/L_body；速度给瞬时身体前/侧速度/L_body每秒，避免绕圈回原点假装无漂移。可预登记探索目标如9.6s达到30°且最大偏离≤.1L_body，但这是新工程建议，非生物标准或旧成功门。六腿支撑换足须实测报告，原地不强求旧前摆advance条件。

滑移另报逐腿承载切向速度/L_body每秒、滑移路程/L_body、载荷积分分母和接触有效性，必要用实测ω和足半径定义转动参考速度；禁止用趋零COM位移归一化生成虚假的pivot滑移结论。保留原生非足峰、额外力0、COM/姿态安全与完整模型，不能用新yaw门覆盖原5门失败。

### D. 粗地形从低到高，首批最多追加2回合

定义腿长L_leg为选定站姿各腿从coxa到tarsal pad沿关节链段长之和，取中位数并保存六腿值/定义/源码hash；体长与腿长不能混用。建议结构等级：平地0；低起伏A_pp/L_leg=.02、λ/L_leg=2；中等.05、λ/L_leg=1；高阶.10、λ/L_leg=.5，再另加块状/离散障碍结构。不同尺度/结构分别研究，摩擦固定当前值先隔离几何。

先修派生ground mask/clearance并静态检查编译后terrain实际可接触、hfield幅度/方向/分辨率、全71mesh及42PD参数保持。第一实际粗地形批仅低起伏同一结构两个预定seed（例如101/102），相同合法前行命令、同身体权重；预热区可预声明平坦起步区，过渡结构与位置固定，不能临时搬地。两回合2,464,000native步，叠加A上限13,552,000步；独立审核后才决定后续更高尺度和held-out结构/seed。seed101/102不等于广泛鲁棒。指标需包括滑移、有效承载、非足接触、局部COM净空、roll/pitch、倒地/跌倒时间、每腿周期、实际进度、地形覆盖及几何/seed/hash；不要只改摩擦称粗糙地形。

## 8. 验证范围与交接

本次证据是所列源码及JSON文本；不以代码存在声称模型行为通过。全局权威旧结果仍是064313 empty/Onoodor旧五门通过，右联合RH周期与odorcut .050696>.05失败、转弯原slip失败。新运动授权不追改旧成败。

本报告与 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/motor_audit/sourcehashes.JSON` 为本次主输出。清单只哈希审计源码/状态文本，不读取NPZ数组、不反序列化权重；它不是当前body完整运行输入manifest。父实际实现仍须按 `/run/media/sensen/Data2/cell_wound_prototype/WALKING_RESEARCH_PROTOCOL.zh-CN.md:15-19,32-38,62-67`固定所有实际导入依赖、XML/资产、初始权重及编译后变更，保存后独立重放。
