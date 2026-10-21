# 自主实施阶段成果：派生初态、收敛、饱和及机械扰动

**手工研究原型，不是成品。没有站立成功；没有可靠的扰动恢复证据。**

## 已完成

1. 实施独立初态规则：将历史姿态5个越界标量关节按最近距离投影到原范围，再重新落脚。原始XML和历史姿态不修改；42个标量关节范围通过、初始仅足部触地。它不是生物学来源姿态，也不是力学平衡。
2. 运行5种时钟×3个干预臂，共15个100ms真实BANC身体实验。零背景、连接/增益固定，主输入仅claw/hook，零控制下限为明确编译派生分支。
3. 保存可复用逐执行器饱和审计及6项纯测试：7个历史配对臂滤波率重建误差精确为零。没有修改解码增益。
4. 按预先写定的协议，运行基础及四分步各5个机械扰动对照，共10个100ms实验。外力为模型总重0.1倍、世界X方向1—2ms，不是运动注流；验证每个实际身体步的外力窗口与冲量。
5. 20项当前回归通过；源码快照核验通过。实际查看派生身体关键帧、收敛图及扰动图。

## 初态与收敛结果

|时钟|正常运动脉冲|正常高度门槛|感觉切断门槛|运动切断门槛|
|---|---:|---:|---:|---:|
|基础身体0.1ms/神经0.5ms|534|34.4ms|15.9ms|21.5ms|
|身体半步|553|33.75ms|15.9ms|21.3ms|
|神经半步|534|32.6ms|15.9ms|21.3ms|
|联合半步|523|32.0ms|15.9ms|21.1ms|
|联合四分步|517|34.15ms|15.875ms|21.0ms|

联合四分步对联合半步：正常门槛差2.15ms，超过预定1ms容差，**收敛不通过**；正常高度最大差0.01886模型mm，通过0.05mm高度容差。感觉/运动切断的门槛差分别-0.025/-0.1ms，均通过。不能只挑基础对四分步接近而称收敛。

派生初态比历史初态在基础时钟多支撑约11.2ms，但仍很快下沉，且有旋转/漂移。不能称有效生物学站立。

## 饱和审计

历史原规范141/1800、零下限139/1800执行器端点饱和；零下限42个执行器曾饱和。峰值未裁剪率380.569Hz，对应3.806需求，被裁到1。首次内部滤波越阈11.5ms，首次保存饱和命令/实际平台15ms。

202个唯一运动细胞，74个可用执行器池、66个不同非空池；8对共享别名。本例共享别名未饱和，但累计池脉冲会双计：550池脉冲对530唯一运动脉冲。部分池用已存在细胞作分母，属于须明确报告的映射假设。

声明触觉与主体全网络脉冲84917/10242，运动输出、命令和身体却精确相同。全网脉冲不能替代运动传播证据。运动切断仍有816运动脉冲但解码器清零，故无饱和不意味着无传播。

## 同机械扰动反馈与重放

重放使用无扰动正常臂命令，但施加与反馈臂相同外力；仅对照，不算闭环运行。

|配置|反馈运动脉冲|反馈高度门槛|重放高度门槛|反馈减重放100ms胸部高度|
|---|---:|---:|---:|---:|
|基础|545|30.1ms|30.1ms|+0.03089模型mm|
|四分步|527|29.5ms|29.6ms|-0.02106模型mm|

基础：扰动引起实际入口变化5ms；运动总数首次不同的窗口起点25ms，实际命令30ms，反馈/重放身体采样35ms不同。
四分步：入口仍5ms，运动计数窗口10ms，命令15ms，身体采样20ms不同。

这些是5ms采样/计数窗口证据，不是逐脉冲首次差异精确定位。基础反馈胸部高度略优、旋转略小；细化后优势翻转。图中反馈与重放虽接近，但不能称反馈恢复优于重放。稳定基线尚不存在，且数值敏感，故结果仅为扰动响应。

## 下一自主任务

先审计神经阈值/不应期、解码器逐步滤波与离散事件的时间敏感性，建立相同脉冲时刻重放的跨步长单元测试，并定位从输入到运动首次分歧。禁止用降低阈值、改增益或换符号解决数值失败。精确事件和收敛厘清之前，不把长跑时间当站立进展。

## 完整成果路径

- 初态与收敛数据：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/projected_pose_convergence/20261013_072227/convergence_results.json`
- 收敛轨迹：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/projected_pose_convergence/20261013_072227/convergence_traces.npz`
- 身体关键帧：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/projected_pose_convergence/20261013_072227/body_keyframes.png`
- 收敛图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/projected_pose_convergence/20261013_072227/convergence.png`
- 饱和数据：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/decoder_saturation_audit/audit.json`
- 扰动数据：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/projected_perturbation_response/20261013_102010/perturbation_results.json`
- 扰动图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/projected_perturbation_response/20261013_102010/perturbation_response.png`
- 新初态模块：`/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/declared_pose_projection.py`
- 初态测试：`/run/media/sensen/Data2/cell_wound_prototype/test_declared_pose_projection.py`
- 收敛入口：`/run/media/sensen/Data2/cell_wound_prototype/run_projected_pose_convergence.py`
- 饱和入口与测试：`/run/media/sensen/Data2/cell_wound_prototype/tools_decoder_saturation_audit.py`、`/run/media/sensen/Data2/cell_wound_prototype/test_decoder_saturation_audit.py`
- 扰动入口：`/run/media/sensen/Data2/cell_wound_prototype/run_projected_perturbation_response.py`
- 预先协议：`/run/media/sensen/Data2/cell_wound_prototype/AUTONOMOUS_NEXT_PROTOCOL.zh-CN.md`
