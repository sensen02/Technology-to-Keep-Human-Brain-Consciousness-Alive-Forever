# 身体侧单足踏空与补撑

目标 S2/S5。完整 71 mesh 身体、42 路有限 PD、不启用附着、原 nominal 全态站姿、原相同关节限时与限力。原生积分 12.5 µs。正式回合连续静立 0.5 s，受推进意图 9.6 s，撤令后站稳 1.2 s。台架意图不经过上层或颈部。实验位置 `/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T063132Z/teacher`。

## 地形和取证

把原整块无限平面替换为固定于 world 的四块齐平地形箱和一块下沉箱底。对照填充相同开口的底面到 z=0。原平面关闭所有碰撞，新地形几何对各身体段使用同一全局掩码。每次隐藏几何过滤均未参与控制；缺口边界只给证据程序定位真实落点。原身体碰撞掩码、惯量、质量、关节自由度、驱动和求解时间常数保持原实物台架编译值。形状、证据和各输入的路径和 SHA-256 见每个回合 `RUN_IDENTITY.json`。

记录六足每个原生步的支持和滑移分子，区分右前足的上表面、下表面和侧壁接触力，以及足垫底端的 z。C 记录与 Python 接触参考每 4031 步交叉检查，原回放关节状态和原生采样时间核对。实际动作在 ±1 的限幅内送往原增量执行器，未修改连续根状态和外力。其他脚载荷、失载、抬高及上缘外重新承重实测独立记录。

准入必须同时满足同缺口控制隐藏变量匹配、填充地形中的预期承重、实际空中超深失载、失载后纠正、出现右前足重置落点、终点站稳及原完整 9.6 秒工程步态门。补撑比较同一收腿算法仅关闭右中腿主动输出的配对。既有支撑脚增载与一条原先离地的脚新植足为两个物理事实。总支撑不替代具名替代脚受力证据。跨颈实验维持既定独立安排，本轮没有完整 D1–D8 对照。

## 环境、命令和失败

使用 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python`，MuJoCo 3.9.0，`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl`。探针 `floor_contacts.so` 由 `g++ -O3 -ffp-contract=off -shared -fPIC` 编译，头文件和动态库来自 `/run/media/sensen/Data2/cell_wound_prototype/venv_body/lib/python3.12/site-packages/mujoco`。 `drive.sh` 运行 `physical.py` 并在同名 `.exit` 写进程返回码。闭环失败和程序正常退出是两项独立记录。

- Y 优先初期铺面 `DEV_GAP_A`、`DEV_GAP_B`、`DEV_FILLED_A` 在 2114 ms 停止。接触缝位于右后脚，从原状态的载荷被缝误改变；改变排列至 X 优先后初态观测完全一致。原代码保留为 `geometry_dev_y_first.py`，早期目录未覆盖。
- `./drive.sh control CONTROL_AN --scenario AN.json --native` 及对应 `CONTROL_AN_FILLED` 完整运行。填充的 11,301 帧 qpos 与平面原始差异上限 1.40e-14，原生滑移比 .481386。未干预缺口在 2764 ms 出现右前脚深降；上表面重新承重 7596 ms，原生滑移比 .867119，地面非足接触力峰值 .32656 W，行走门未通过。
- `./drive.sh teacher TEACH_SERVO_AN --scenario AN.json --plan PLAN_SERVO.json --native`、 `RF_ONLY_SERVO_AN` 均实际跑到站稳。早期笛卡尔控制的完整 RM variant 和消融分别在 3283／3829 ms 重获上表面接触。背景姿态轨迹受扰，原生滑移比 1.1638 和 1.0817，均未准入。`PAIRED_AN_DEVELOPMENT.json` 记录右中脚相对冻结缺口增载，但相对收腿单支路的 435 ms 平均增量仅 −.000616 W。AX 激进短程五脚支持样本的前 200 ms 有 +.02697 W 具名右中脚增载；同样未通过完整门。
- `TEACH_EDGE_EARLY` 使用 `PLAN_EDGE.json`，深度 .055 的 AE 和补回表面的 `CONTROL_AE_FILLED` 均成对计算。原生滑移比 1.6321，相同受扰身体以相位参考维持步态仍不稳定，未准入。
- `gap_hold.py TEACH_HOLD_AE AE.json PLAN_HOLD.json --native` 固定受支撑脚的有限执行器目标以恢复，随后尝试接回相对步态；站稳及推进门失败，未准入。
- 晚步几何 `AL` 填充对照原生滑移比 .476751 通过；有效缺口在 4815 ms 失去右前承重，冻结分支 7596 ms 重获上表面，原生滑移 .750231。暂停及再进步态的 `TEACH_HOLD_AL` 更早于 6321 ms 重获上表面，原生滑移 2.2497、末端失稳，未准入。
- 末版 0.024 深浅缺口 `AS` 的冻结对照原生滑移 .503791，填充对照 .479451 通过。 `gap_bias.py TEACH_BIAS_AS AS.json PLAN_BIAS.json --native` 的浅量修正和同场 `TEACH_BIAS_AS_RF_ONLY` 全部正常退出，原生完整联合门未通过。修正比冻结对照未缩短上表面复踏时间，原生滑移 .580300。 `BOUNDED_GAP_EVIDENCE.json` 包含对照身份和对 RM 动作消融的完整原生证据；未产生可入库的继续行走教师。
- `gap_stop.py BRAKE_AS AS.json PLAN_BRAKE.json --native` 和 `gap_stop.py BRAKE_AS_RF_ONLY AS.json PLAN_BRAKE_RF_ONLY.json --native` 使用独立 6.45 s 停稳窗口，各 516,000 原生步及退出 0。旧站立控制支路在真失载时接管其余关节，实际先抬右前足、移出洞口、重新连续承重 200 ms，再退出全部移动修正。两次经末段站稳验证。常规继续走门仍独立且未通过。轻 RM 转移版本相对不主动 RM，仅早期 +.00036 W，不能据此声称主动补撑通过。 `SAFE_STOP_EVIDENCE.json` 记录该局部恢复与其因果消融。
- `gap_weight.py BRAKE_AS_FIXED_WEIGHT` 原生 516,000 步且退出 0。维持与 `BRAKE_AS_RF_ONLY` 完全相同的制动／右前足路由，唯一新增右中脚短暂目标偏置。满辅助试验 2764 ms 失载，2765 ms 介入，恢复窗右前足垫高度跨度 .041835 原生，3127.0 ms 开始连续 200 ms 真实上表面承重，基线同制动但去掉该偏置为 3681.0 ms。最终站稳、非足力及外力均通过；其余脚首 200 ms 平均 .99986 W。偏置未证明较消融增载，右中脚前 200 ms 差值 −.003197 W，因此抬腿制动证明保留，主动受力替代仍未通过。 `SAFE_STOP_WEIGHT_EVIDENCE.json` 与相同原生转移证据可复核。
- 所有上述正式及发展物理任务均结束。当前无符合联合门的继续行走教师，准入文件及训练未执行。若继续，先以已实证站稳的制动样本为种子，在同一连续模拟加入实测重启再接自然步态，并让支持腿的控制目标直接约束当前测得的负荷，分别核对闭环再推进和实际具名增载。随后仅用满足所选任务真实物理门的完整回合生成监督目标。
