# S4/S5 实施前接口文本审阅

这是父代理对已有源码与证据的静态审阅，不是新模型运行或身体刺激验收。实施前Git提交与远端核验见 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z/GIT_PREIMPLEMENTATION.json`。

## 已确认的接线约束

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning/model.py`：每决策只调用一次共享walking base128；RIGHT64与LEFT64分支每步都更新。外部命令为 `[relu(t)/3,t]`，t在[-1,1]；不应仍用旧行为输出 `[0,t]`。该源SHA256为 `b7ac89e780ecfc0b70f6cace8dfcdade34b79cf55914709dd4214f46b6e91424`。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/steering_behavior.py`：输入恰为2受体与32记忆，慢GRU32后经tanh输出转向，旧速度通道恒0。该源SHA256为 `ab69488c792335bb1f14b8dc13df4b731521929b3110d07f96c50f07c044ac37`。
- 新实现需同时保存motor256链、behavior32链和拼接288链，不能用旧256审核代替总链审核。A为显式高层命令探针，behavior保持初始；B在撤辅后真实推进行为网络，不能人为清除无气味输出偏差。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/odor_loop.py` 旧实现使用pending_response：t动作收到上个区间已积分的受体，当前C仅更新下一步。保留这种因果关系，不能把t+5ms状态提前给t动作。该源SHA256为 `2a2f2293d0dfb7c6b6c1fcfc26ed72d4b701d2165f955f8a53c4ffc1a873eba0`。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/odor/food_field.py` 定义工程C/(K+C)转导、tau=.05s与真实头部旋转后的双侧XY采样；该源SHA256为 `d8bbee489d76cfc9ebaaf1bab8843f826025a09cd7d211dd6332ecc1c1205fe8`。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/odor/HEAD_GEOMETRY_AUDIT.json` 的local_xyz为前0.5713176560335599、左右±0.4591571757690745、z0。新runner以此作为实际采样与几何显示的同一身份，避免图示位置和输入位置不同。

## 新实验必须检验的空白

旧感觉模型的结果 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T035953Z/EVALUATION/RESULT.json` 仅为静态感觉预训练。无气味RMS约.015146和bias约.011003来自约1.2s片段；9.6s持续运行、撤除和反转尚缺实证，不可直接复制旧结论。

旧物理评估器 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/evaluation/evaluate_motor.py` 保存1ms真实身体状态、5ms决策、实际动作与完整预热；新runner需保留这些记录和原生峰值/外力统计，并在新进程重算原指标与周期。现有同模型neutral完整轨迹可供A匹配；B必须另用真实行为网络的无气味回合作匹配，不能强制turn0。

首批B六条件为无气味、等强、左右脉冲和两向反转，均为CONCENTRATION_CLAMP。没有“先建立响应后切断受体”的真实身体回合，不能将无气味条件冒称该干预。实际切断/交换与空间源趋近留在后续阶段。

## 本次预登记审阅

已在实际物理前审阅 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z/PREREG.json`。确认冻结模型、先A3再B6、连续状态、单线程与限定experimental准入，严格历史FAIL保留。向核心实现提出两项跑前澄清：B持续刺激末段符号统计窗口，以及无气味相对原motor-neutral的额外航向≤2°目标。若源码已冻结，只能新增绑定的跑前补充登记，不能改写运行输入。

## 新实现第一次文本审阅（尚未物理）

已读 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z/runtime.py` 与 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z/runner.py` 当前草稿：神经ReLU投影和motor-first/behavior-last的288拼接符合接口；ClampFilter先复制pending给当前动作，再计算下一步滤波；行为网络仅在B的ANN阶段推进，预热源关闭/行为记忆初始；实际动作在ANN直接取tanh(mean)，每决策保持5个1ms tick，未见实际qpos/qvel写入或reset。警告中止由冻结底层 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/physics.py` 的warning检查承接。

已向实现者提出：legacy30°端点资格应与半幅/无气味的原五门分列，不能让“未达30°”被误解成原物理五门失败；`old_motor_qualification_unchanged=False` 字段语义应更正为“旧资格为false且未改变”为true的独立字段。这些是跑前文本审阅意见，最终冻结源码/哈希与独立audit为准。

实现者已在冻结前确认修正上述意见：`engineering_pass`仅对应原五门；`strict_endpoint_qualified`单列且`endpoint30_applicable_to_stimulus=false`；旧运动资格明确为false且unchanged=true。B末段符号统计窗口与无气味相对原neutral的额外航向目标已加入冻结前PREREG。最终仍以冻结版本和运行后审核为准。

后续独立审核与实际身体结果以新trial的RESULT/VERIFY为准。本静态审阅不构成任何刺激/导航成功证据。
