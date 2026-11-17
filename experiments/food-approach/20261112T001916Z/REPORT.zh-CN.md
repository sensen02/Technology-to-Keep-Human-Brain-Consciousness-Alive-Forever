# 候选3：成对 batch 长历史行为恢复——离线通过，身体实证待父流程

本报告仅依据已保存的预登记、实际结果与独立审核文本整理；未新增模型前向、训练、数组分析或物理。父唯一 heavy `bash-62` 正顺序执行新行为权重的六个身体 B 回合，最终身体结果尚待完成。本报告不覆盖动态状态、既有检查表或任何冻结核心。

## 1. 候选范围与不变项

固定父行为权重：`fc3a0c20f854afe162afdf3c87cbddea2b8879a7e29d3cde97f487640e101641`。模型仍是两受体输入→32维编码→slow GRU32→1维 tanh 转向；架构、alpha、连续记忆及8个参数张量的训练范围未改。运动模型 SHA `789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd` 保持冻结，候选离线阶段仅核原始哈希，没有加载或训练运动模型。

24条训练配置的 receptors、targets、concentrations、time、weights 和 segment_id 数组全部逐位复用上一候选；四阶段 .30/.20/.40/.10 权重与动作 MSE+.01 latent MSE 不变。唯一训练机制变化是每个 batch 必须由两个同配置 pair 构成，维持均匀 episode 期望权重：

- 4对 left_pulse/right_pulse；
- 4对 left_to_right/right_to_left；
- 4对 noodor/equal（只是零目标 pair，不声称输入镜像）。

每步 PCG64 seed20261111 从12pair均匀无放回抽2pair，按固定 pair 内顺序展开4条，不额外 shuffle，不强制 neutral pair。每个 episode 入选边际仍为2/12=4/24，不能说旧均匀采样在期望上有方向偏置，也不能把这次通过归结为已证明的唯一因果机制。

## 2. 固定预算与执行记录

- 精确 fc3a 参数初始化；fresh Adam ALL8，学习率1e-4，不恢复旧优化器矩。
- 固定500更新，batch4，完整1920步／9.6秒 BPTT，不截断可训练历史。
- 保存 initial、250、500；只最终500做一次新 V4/V5共12配置验证，未选择250或扩大500预算。
- 实际训练用时189.398秒，Core构造0、物理积分0、运动模型加载0。
- 全部运行由父代理托管单CPU单线程，实际成功流程 `bash-61` exit0并已收集。本子未另开重任务。

最终行为权重：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/TRAIN/checkpoint_000500.pt`

SHA256：
`66f423df4f00d801b2f0a194f539baa63ce84973765694e3d3111fbeb62aa034`

## 3. 启动失败及最小恢复

首个父托管 `bash-60` exit1发生在预检源哈希阶段：原 preflight 将尚未生成的 `PAIR_PREFLIGHT.json` 加入 source_paths，附加 pair 检查执行得太晚。此前24train/12validation数据已生成，未写 FROZEN，未建立 TRAIN，实际优化器训练更新0，物理0。这是执行顺序缺陷，不是模型行为失败。

原 adapter、CODEONLY hash、JOBS和失败日志全部保留。只新增 `preflight_v2.py` 将唯一 pair 检查块移至 source_hash 计算前，新增 `run_remaining_v2.py` 使用独立 JOURNAL/日志名继续；已经产生的数据复用，不重生成。新恢复入口在真正冻结中纳入输入身份，没有改旧源码或绕过校验。

失败记录：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/ATTEMPT01_PREFLIGHT_FAILURE.json`

成功恢复及所有 stage PID/命令/退出码：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/JOBS_V2.json`

## 4. 真实数值预检与独立重放

实际预检112项全部通过，包括 production/scan 输出、latent、32维状态、全部8参数梯度、临时 Adam 参数及矩更新一致；同一4序列批量目标和梯度等于两个 pair 目标／梯度的均值（tol1e-12），不要求左右梯度互为负。

还验证旧24训练数组全部 exact，12pair覆盖24各一次、同配置映射、8个定向 pair 受体交换／目标反号／权重相同，以及 no odor/equal只属于零目标 pair；所有 episode 的边际抽样概率保持。真实500步 pair IDs、展开 episode IDs、优化器计数和RNG在独立保存审核中重构。

initial/250/500保存后均进行了新进程实际模型重放、Adam/RNG/curve与调度审核。最终验证另由新进程重新执行12条完整序列并逐位核对 turn、latent、hidden32、输入目标及每条功能判据；独立审核 integrity_all_pass=true。

原无重置／无运行时 bias减法／无 deadzone／无 ifodor 或额外输出门保持。新增 signed mean/std/peak仅为诊断，不用于改写或抵消 RMS 判定。

预检：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/PREFLIGHT.json`

pair专项检查：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/PAIR_PREFLIGHT.json`

## 5. 新12配置离线验证结果

全部12配置的预登记分项目标通过，原门没有放宽。新 V4/V5与此前训练／验证的时序和强度组合不同，但确定性无气味历史仍重复，不能将其当作全新的独立重复。

| 条件／指标 | V4 | V5 | 门槛 |
|---|---:|---:|---|
| 无气味全程 turn RMS | 0.00324279 | 0.00324279 | ≤0.02 |
| 等强全程 turn RMS | 0.01099852 | 0.00599143 | ≤0.05 |
| 左脉冲 off后0.5–1秒 RMS | 0.03832692 | 0.04592063 | ≤0.05 |
| 右脉冲 off后0.5–1秒 RMS | 0.03883199 | 0.04330576 | ≤0.05 |
| 左右脉冲合格平台符号比例 | 1.0 | 1.0 | ≥0.90 |
| LR／RL两平台符号比例 | 1.0 | 1.0 | ≥0.90 |

响应时延门也全部通过。右侧恢复窗仍有约+0.0173／+0.0181的平均输出和短暂反冲，左侧平均约-0.0181／-0.0222；此次是 RMS在规定窗口内通过，不等于输出严格为零或保证所有未测刺激史都能恢复。

验证结果：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/VALIDATION/RESULT.json`

独立验证重放：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/VALIDATION/VERIFY.json`

输出数组：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/VALIDATION/OUTPUTS.npz`

## 6. 当前身体实证状态

父已将真实最终SHA绑定到独立 `physical_B` 运行范围，启动前21项检查exit0、0Core；当前由父唯一 heavy `bash-62` 顺序执行6个新 B 实际回合：无气味、等强、左脉冲、右脉冲、左右切换与右左切换，新行为SHA配同一冻结motor789f。原 A3纯motor数据可按既定身份复用，旧行为权重的 B_noodor 不充当本次匹配对照。

本报告写作时身体回合尚未全部收齐，不能把12条静态长序列通过直接说成身体受刺激恢复成功。最终仍须检查实际受体／神经288状态／命令／身体轨迹的新进程重放与支撑、周期、非足力、实际航向及 off恢复，原运动滑移失败继续独列。视频、空间气味源趋近、到达、停驻和摄入均不由本报告认定完成。
