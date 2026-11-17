# 实施运行检查（S2/S5）

- 输入按 `PREREG.zh-CN.md` 冻结。几何、原料与编译后物理在 `TURN_IDENTITY.json` 保存完整散列。原地对照见 `CENTER_PREREG.zh-CN.md`，弧线复核见 `ARC_REFINEMENT.zh-CN.md` 和 `TARSUS_CLEARANCE_PREREG.zh-CN.md`，原生与接续口径见 `NATIVE_PREREG.zh-CN.md`。
- 物理环境为归档解释器 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python`，参数 `-B -u`。环境 `MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。
- 固定 12.5 μs 原生物理、1 ms 目标更新、5 ms 决策和 2 rad/s 限速。无 PPO 或优化器。采用观测—动作示范原型，保存入库轨迹及其样本散列。当前观测与即时前进/偏航意图是模型输入。
- `turning.py` 初批四条教师及两条左弧参数变体全部退出 0。原始左弧和两参数变体的非足触地门失败。右弧及原始原地两向工程通过。原地中心修正 `pivot_center.py` 左右退出 0，低于新 0.15 体长目标。
- 原生接触程序 `native_contact.cpp` 使用实际同一 MuJoCo 3.9.0，g++17，`-O3 -ffp-contract=off -shared -fPIC`。55 ms 独立短测退出 0，两次冻结原生记录器交叉核对通过。此前误执行的临时探针导入不存在模块，退出 1，零物理步。
- `tarsus_clearance.py TEACH_Clear_012 .012`、`tarsus_clearance.py TEACH_Clear_028 .028` 均退出 0 并达原门。`phase_one.py --prepare` 后，`turning.py --build CENTER_pivot_left CENTER_pivot_right TEACH_Clear_028 TEACH_arc_right` 退出 0，分期固化的三个原型与正式原型文件逐字节相同。
- 标称原地两向和右弧分别用 `phase_one.py --mode <mode> --start nominal --native --out NATIVE_<mode>`。名义左弧用 `verify_native.py arc_left NATIVE_arc_left --entry 4800 --exit-walk 4800`，均退出 0。留出体态在 `HELD_*`，用 `phase_one.py` 或 `turning.py --kind model` 配相应初态，均退出 0。
- 正式教师与候选合计 18 条回合，21,648,000 原生步。每回合台架准备 80,000 步，另有 4,400 原生步采样短测，合计 23,092,400 步。正式标称原生记录 5,152,000 步，固定间距抽查 1,280 次，与既有读取器一致。追加分析只读归档。
- `audit_turns.py AUDIT_FINAL.json NATIVE_pivot_left NATIVE_pivot_right NATIVE_arc_left NATIVE_arc_right HELD_pivot_left HELD_pivot_right HELD_arc_left HELD_arc_right` 退出 0。基线 2,260 次动作逐位相同，八条新回合倒序动作重放误差 0。只读整体坐标变换试验退出 0，工程门一致，旋转/平移不改变身体坐标下的关键指标。
- `render_turns.py NATIVE_pivot_left NATIVE_pivot_right VIDEO_PIVOT_VERIFIED` 与 `render_turns.py NATIVE_arc_left NATIVE_arc_right VIDEO_ARC` 退出 0，完整解码及关键帧目视完成。首轮原地渲染在生成媒体后写清单时因临时变量名冲突退出 1。随后 `accept_turns.py` 退出 0，产出 `FINAL_ACCEPTANCE.json`，并列出出弯原直走滑移诊断。
- `EXIT_SLIP_PREREG.zh-CN.md` 冻结出弯低滑移附加门。`exit_teacher.py arc_left EXIT_TEACH_left` 与 `exit_teacher.py arc_right EXIT_TEACH_right` 均退出 0，原始末态逐位一致，两条合计 3,344,000 原生积分加 160,000 台架准备，240 次原生力交叉检查。左臂滑移失败，右臂站稳失败，保留轨迹和未入库标签。含此补采合计 26,596,400 原生积分。只读 `EXIT_DIAGNOSTIC.json` 记录直接切换时的原型距离和执行器限幅。
- 同一录像增加出弯滑移诊断字幕，`render_turns.py NATIVE_arc_left NATIVE_arc_right VIDEO_ARC_VERIFIED --exit-slip-diagnostic` 退出 0，完整解码 1,046 帧，已目视检查字幕和转向阶段。首次合并清单因旧输出存在退出 1，原清单改名归档为 `ACCEPTANCE_TURN_ONLY.json` 后退出 0。最终验收 JSON 同时列出功能与附加滑移结果。桌面报告与两个视频的 `cmp` 均退出 0。
