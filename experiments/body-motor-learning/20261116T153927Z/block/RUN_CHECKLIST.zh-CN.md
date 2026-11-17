# S2/S5 实心低障碍运行检查

- 父实验为冻结身体行走和原地转向证据 `20261116T000536Z`、`20261116T004235Z`，窄障碍反射来自 `20261116T060654Z`。当前路径为 `/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/block`。
- 本轮主要条件为实心块是否存在、几何是否提供给教师、实际侧碰后是否启用局部抬升。相应依赖／XML／资产、设备、编译后参数、初态和哈希见每个试验的 `RUN_IDENTITY.json`；原生接触探针 C++ 使用 MuJoCo 3.9.0、`g++ -std=c++17 -O3 -ffp-contract=off -fPIC -shared` 编译于 `terrain_contacts.so`。
- 解释器为 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python`。运行单线程 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl`。
- 8 秒短探针使用 `bash drive.sh aware DEV_A SOURCE.json --probe-ms 8000`、`bash drive.sh blind DEV_B SOURCE.json --probe-ms 8000`。两进程退出 0，均出现 RF 在块顶承重并到尾缘后回地，未执行长程停走。
- 完整对照命令 `bash drive.sh aware COURSE_AWARE SOURCE.json`、`bash drive.sh blind COURSE_BLIND SOURCE.json`、`bash drive.sh flat COURSE_FLAT SOURCE.json`、`bash drive.sh aware COURSE_AWARE_REPEAT SOURCE.json --compare COURSE_AWARE`。教师两臂和独立重复均进程退出 0，原生完整 3,208,000 步。平地保存了完整轨迹、原生数据、身份复核和结果，执行工具在退出边界收集到的状态为 143，原始记录在 `COURSE_FLAT.TOOL_EXIT`，同参平地独立复跑另存为 `COURSE_FLAT_REPEAT`。
- 实体侧碰对照命令 `bash drive_tactile.sh PROBE_TACTILE_ON --on --ms 5500`、`bash drive_tactile.sh PROBE_TACTILE_OFF --ms 5500`、`bash drive_tactile.sh PROBE_TACTILE_REPEAT --on --ms 5500`。三进程退出 0，两个训练教师开启轨迹 440,000 原生步逐位一致。
- 验收脚本 `certify.py matched MATCHED_EVIDENCE.json COURSE_AWARE COURSE_BLIND COURSE_FLAT COURSE_AWARE_REPEAT` 和 `tactile_audit.py TACTILE_EVIDENCE_V2.json PROBE_TACTILE_ON PROBE_TACTILE_OFF PROBE_TACTILE_REPEAT` 皆退出 0。闭环失败/通过严格以 JSON 判定。
- 几何必须在原始站姿全部足垫前。顶载 ≥0.02 W 持续 10 ms 来自原生力解，侧碰阈值不混同。六足回地判定需实际足垫外沿越过后界加 0.01，连续 20 个 1 ms 帧负荷 ≥0.02 W；继续证据需同回合全部超过、无块载并持续。滑移比为各腿原生 Fz 加权平面接触速度的最大值，除以同窗 COM 世界水平位移速度；前向按初始偏航单位向量投影，非足力取原生峰值，完整行走窗为 0.5–38.9 s。最后 200 ms 复算支持分布、无附着无额外力、COM/直立、末端稳站。
- 体态/关节速度和命令目标留存于 1 ms 轨迹；接触力、接触速度分子、实际 COM 和姿态及额外力留存于每个原生步的紧凑记录，并周期性与既有 Python 物理记录交叉核对。完整子步联合力分解和 D1–D8 未执行。此轮教师验证未进行 PPO、SAC、刻意扰动泛化或神经权重拟合。
- `render.py` 以相同几何和归档 qpos/qvel 渲染，不推进物理仿真；`VIDEO_TACTILE` 的 280 帧和过块配对视频通过全帧解码。目视查看了稳定、碰撞、收腿、块顶承重、中后足、末端落地与稳站帧。另见下方延展验收。


- 延展至 43.3 秒的命令为 `bash drive.sh aware COURSE_EXTENDED EXTENDED.json`、`bash drive.sh aware COURSE_EXTENDED_REPEAT EXTENDED.json --compare COURSE_EXTENDED`。首个进程写出 3,464,000 步完整产物、身份执行复核及完整结果，工具边界记录 143。独立重复退出 0，所有数据哈希逐位相同。平地重复 `bash drive.sh flat COURSE_FLAT_REPEAT SOURCE.json --compare COURSE_FLAT` 退出 0，三个数据文件逐位相同。
- `certify.py extended EXTENDED_EVIDENCE.json COURSE_AWARE COURSE_EXTENDED COURSE_EXTENDED_REPEAT` 退出 0。额外验收从真实网格顶点证明头胸腹过全球尾缘、全部 71 网格的水平包围盒在末端 200 ms 与实体块足迹分离、仍前进后停住。记录原生力、姿态、推进及每足周期；该延展与 40.1 秒版在第一个不同动作前的物理轨迹逐位一致。
- `render.py VIDEO_EXTENDED COURSE_EXTENDED_REPEAT --trunk-ms 39280` 退出 0。完整头胸腹、足端和实体块均可见，实际目视检查起始、三条足实际块顶承重、侧碰、后足回地、39.280 秒头胸腹过尾与 43.300 秒末态。保留帧到实际仿真时间的映射，逐帧通过视频解码。
