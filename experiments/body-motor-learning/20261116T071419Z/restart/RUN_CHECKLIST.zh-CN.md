# AS 踏空制动后的同回合再起步

本项属于人工搭建大脑项目身体侧 S2/S5。沿用 AS 真实 0.024 原生长度深缺口和上一轮 `BRAKE_AS_FIXED_WEIGHT` 的已验证失载、制动、重新承重和站稳。原机体、原限幅 42 PD、无附着、原 nominal 完整站姿快照及同一 12.5 µs MuJoCo 积分时间。每个新回合从初始快照连续完整运行；全体自由根，禁止推进接管时恢复快照、根状态写入、外力或力学调整。台架意图程序独立于脑及真实颈部。前缀有局部制动教师，续行以原已验收冻结策略的当前观测路径执行。

## 预登记及复验契约

- 绝对实验目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T071419Z/restart`。源安全制动为 `/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T063132Z/teacher/BRAKE_AS_FIXED_WEIGHT`，对应 `SAFE_STOP_WEIGHT_EVIDENCE.json`。原地形 AS、逐步记录和其 SHA 见每次 `RUN_IDENTITY.json`。
- 三个独立进程均执行相同完整安全制动前缀 6.450 s，必须和旧试验 15 路 1 ms 物理字段、6 路 5 ms 决策字段及 516,000 次原生力学记录逐位相同。推进意图是台架在 6.450 s 明确送达的重新续行事件，解除已经触发的制动接管。无续令配对保持纯站立，均持续相同时间。每次推进区间 6.450–16.050 s，撤令后观察 1.2 s，窗口按旧工程门和原生时间重定位。
- 续行验收继续用原完整工程门。至少一个真实首个完整周次，继而所有六足完成真实周次，包含起始站稳和末尾站稳。持续推进不低于 .05 原生长度/s、质心最低不低于 1.2、足载均值至少 .9 W、原生最大滑移比不大于 .50、末端无超量非足触力。该时间段记录左右非足和地面接触。配对独立检查 AS 底面或侧壁再进入及无续行物理漂移；此附加条件与主项目完整 D1–D8 区分。
- 续行策略只有原已审核模型，真实传感、无历史路由修改、每步通过标准驱动期限及扭矩上限。无修正参数和学习权重，不发生训练或模型更新。相同输入的独立正向重积分须逐字节复现三个 NPZ 文件。
- 无续令组预期没有步行周期。其与续行组的共同前缀须完全一致，最终站立且 9.6 s 配对位移绝对值小于 0.01 原生长度，单列基线。

## 执行环境和命令

沿用 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python`、MuJoCo 3.9.0、`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl`， `drive.sh` 记实际函数退出值。全部路径在本实验目录。物理和源文件每回合后重新哈希。

## 实际结果

```sh
./drive.sh immediate RESTART_AS --native
./drive.sh no_restart HOLD_AS --native
./drive.sh immediate RESTART_AS_REPEAT --native
python -B -u audit_restart.py RESTART_AS RESTART_AS_REPEAT HOLD_AS RESTART_ACCEPTANCE.json
python -B -u render_restart.py VIDEO_RESTART_FINAL HOLD_AS RESTART_AS
```

三组物理进程、审计、两个实际渲染版本均按同名 `.exit` 记为 0。三组共 4,140,000 原生求解步。固定视角最终渲染便于比较两侧位移，保留初版跟随相机文件。无续令控制原本就应通过站立门而没有行走周期，不把其行走分母极小产生的滑移比解释为打滑。修改项仅限本实验目录。三组身份和 SHA 路径见 `RUN_IDENTITY.json` 和 `RESTART_ACCEPTANCE.json`；全流程再起步及停稳、失载支路、具名支持腿载荷的物理范围独立登记。

## 保留事项

用户要求的同回合制动后直接重走在单一 AS 场景、单一巡航窗口和既定局部制动教师下通过工程验收。右前足的洞底与边壁原生载荷均未在重走窗重新出现。下一课在真正扰动变化和更多适配初态复用同一接口。当前结果不涵盖完整 D1–D8 或任何脑／颈通信验证。

父任务可直接移交的最终固定视角视频位于 `VIDEO_RESTART_FINAL`。共有 1,014 帧经完整解码，静态关键帧已实际打开。画面真实显示右前足先进入缺口，再着地，停稳后右侧重新前进而左侧维持原位。视频未执行实际积分。
