# 调速与弧后接续的学习及整体验证（S2/S5）

本轮目录为 `/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T075254Z/learn`。只使用冻结平地 71 mesh、原自由根和有限力矩的 42 关节执行器。所有新回放均从冻结初态和 0.5 秒静立开始，决策间隔 5 ms、求解器 12.5 μs。独立保留上轮身体台架、脑和真实颈部。停走原型作为回归，同期记录新增工程意图。

## 定义与运行安排

- 学习候选输入为冻结身体 213 项即时观测和两个当前工程意图（前向原生长度/秒、带符号偏航 rad/s）。不存在上一拍动作、私有路径、时钟或循环状态。物理教师可以使用明示的连续时间、支撑锚点和前视落脚。透过它们生成动作和观测样本前先检查原生力、完整性、滑移、连续周期与终末支撑。
- 各合格教师的 `RESULT.json`、决策、毫秒轨迹、原生记录和存在的资格文件逐件哈希。未通过物理结果拒绝进入拟合。运动库按当前关节目标、关节状态和意图联合查询，保留原型目标误差反馈。拟合有界前馈残差的幅度上限为 0.10 归一化速率。240 个更新均记录到 `UPDATE_LOG.jsonl`，另保存临界更新的权重和散列。预测误差不作为闭环门。
- `STOCK.json` 验证连续接口的零偏航和标称推进。调速型回放按预置低、高、减速后窗口测出身体切向速度，高速对两端对照各至少增加 0.005 原生长度/s。出弯窗至少 4.8 秒，对实际退出朝向作前向投影，保留旧水平净位移滑移分母。整段原生支撑至少 0.90 W、COM 最低 1.20、地面非足力零、逐足归一化最大滑移小于等于 0.50，继承站稳和六足循环。短调速窗单列绝对和相对滑移，不能代替整段原始门。完整 D1–D8 另计。
- 模型通过 `physical.py` 模块在实际冻结物理中供应动作。匹配已通过的左右弧线教师意图用 `replay_turn.py`，公开表明存在教师内部航向校正信号。混合整段是独立声明的 `COMBINED_LEFT.json` 和 `COMBINED_RIGHT_HELD.json`，只输入当前意图。若学习接续失败，可在同一状态地形里接合两个工程教师作为明确标记的对照。
- 证据使用毫秒轨迹、原生力和姿态，以及独立物理重积分。媒体读取保留姿态，禁止调用任何物理步函数。实际核对录像每帧解码、关键帧和设备身份。

## 已执行

- 解释器为 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python`。开关和调用通过本目录 `drive.sh` 保留，`MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。
- `physical.py STOCK_PHYSICS STOCK.json` 退出 0，`verify_stock.py` 对比同一历史标称轨迹，全部共同轨迹字段和 5 ms 动作逐字节相同，固定旧世界轴比值 0.4813864263。综合台架的弧向瞬时朝向投影比值 0.481687，计算范围明确区分。
- 初次编译视频脚本成功。720 宽的视频试跑超出冻结模型的离屏帧缓存尺寸，退出 1，保存原日志；正式复跑选择 620 宽。
- 监督拟合 `fit_program.py` 的第一及第二版均退出 0。第一版 15,998 条动作、第二版 22,243 条动作，使用同一训练环境，且全部接受源匹配各自哈希。完整已通过工程回合作为第二版新增合格教师。
- 原型、固定调速、留出调速、修正索引后的左右弧线及可选回退消融均完成物理试跑。`RAMP_HELD_REPEAT/REPLAY_AUDIT.json` 独立重跑退出 0，三份原始数组全同。`STUDENT_RIGHT` 保留修复前转向失败。`STUDENT_COMBINED_LEFT` 与 `COMBINED_SIGNED_LEFT` 保留自由拼接失败。`INTEGRATED_TEACHER_LEFT` 保留未经相位匹配教师失败。
- `engineering_teacher.py left INTEGRATED_PHASE_LEFT --profile PACE_PHASE_ALIGNED.json --plan ../turn/LEFT_SOFT_TRIM.json` 退出 0，7 项均通过。四类正式录像全部解码且关键帧实际打开。冻结旧直走及能复现的 body-only 学习课均从同一站姿和求解器状态开始。
- `engineering_teacher.py left INTEGRATED_PHASE_LEFT_REPEAT --profile PACE_PHASE_ALIGNED.json --plan ../turn/LEFT_SOFT_TRIM.json` 及 `cmp` 三份原始文件均退出 0，两个物理工程回合逐字节匹配。
- `replay_commands.py INTEGRATED_PHASE_LEFT LEARN_PHASE_LEFT policy_directional.py FIT_WITH_COMPOSITE/step_0240.pt` 退出 0，主审计 7 项过。`audit_composite_legacy.py INTEGRATED_PHASE_LEFT LEARN_PHASE_LEFT LEARN_PHASE_LEGACY_AUDIT.json` 退出 0，旧窗、整段出弯、直走和终末门全部通过，且报告与同次源数据散列关联。
- `replay_turn.py right RIGHT_COMPOSITE_WEIGHT FIT_WITH_COMPOSITE/step_0240.pt --provider policy_directional.py` 退出 0，原始右弧台架 7 项过。`audit_policy.py CURRENT_ONLY_COMPOSITE.json policy_directional.py FIT_WITH_COMPOSITE/step_0240.pt LEARN_PHASE_LEFT RIGHT_COMPOSITE_WEIGHT` 退出 0，逆序观察复验两份决策的动作逐位相等。
- `render_body.py VIDEO_LEARN_PHASE_LEFT LEARN_PHASE_LEFT '学得的加减速与左弧接续' --mode student --audit LEARN_PHASE_LEGACY_AUDIT.json` 退出 0，949 帧全解码并打开关键帧。保存输出可和两类审计匹配。
