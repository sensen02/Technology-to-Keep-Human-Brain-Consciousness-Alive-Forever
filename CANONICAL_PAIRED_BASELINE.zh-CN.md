# 原规范身体真实输入配对基线：首批实施结果

**手工研究原型，不是成品。没有达成秒级站立，也不宣称生理参数已验证。**

## 本次实施

- 主入口 `/run/media/sensen/Data2/cell_wound_prototype/run_canonical_body_paired_baseline.py`：固定100ms、12000神经元、零背景，使用实际 BANC We/Wi×0.2，神经0.5ms、身体0.1ms、命令5ms。
- 原入口 `/run/media/sensen/Data2/cell_wound_prototype/run_neural_muscle_closure.py` 增加默认兼容的外部触觉开关、通道排除，以及干预后实际网络入口电流的逐通道记录。不改变默认增益、阈值或连接。
- 主实验禁用所有 hairplate 平均关节、campaniform 足力代理和 bristle 足接触广播；仅保留 claw/hook 的身体位置/速度输入。其符号、归一化和增益仍是未校准假设，不是逐根调谐证据。
- 原 XML 不修改。native 臂保留0.0001正控制下限；zero-floor 臂只在编译模型内将90个控制下限置零，实验后恢复。它是独立分支，不冒充原规范本身。
- 修复 `/run/media/sensen/Data2/cell_wound_prototype/tools_apply_sourced_sensory_config.py` 成功生成后错误退出1的问题，保留字典返回API与拒绝行为。

## 结果

跌落门槛为胸部高度首次低于初值90%，不是摔倒/站立的完整生物学定义。

|实验臂|运动脉冲总数|全网络脉冲|跌落门槛时间|100ms胸部高度（模型mm）|
|---|---:|---:|---:|---:|
|原规范身体输入|538|10484|23.2ms|1.32764|
|原规范感觉切断|0|0|14.9ms|0.87879|
|零控制下限身体输入|530|10242|23.2ms|1.40923|
|零控制下限感觉切断|0|0|14.9ms|0.87854|
|零控制下限运动切断（15ms起）|816|22150|19.5ms|0.86749|
|零控制下限正常命令重放|530|10242|23.2ms|1.40923|
|零控制下限声明触觉代理|530|84917|23.2ms|1.40923|

### 能说明什么

1. 在当前未校准但固定的编码下，**真实身体状态输入能够使真实BANC网络产生运动脉冲**，不需要外部感觉注流或运动池注流；最早运动脉冲在10ms。
2. 感觉切断使所有网络/运动脉冲与命令归零。零下限感觉切断臂激活也精确为零，排除了原XML正下限在该分支提供主动激活。
3. 运动切断后身体采样最早在20ms出现差异，下一次20ms入口感觉记录出现差异。结合入口、命令、轨迹，支持本模型内“运动输出影响身体，身体变化再影响感觉输入”的因果链。时间定位受5ms采样限制，不能声称细于该采样的首次身体变化。
4. 正常命令重放的身体轨迹及命令精确相等，只验证确定性，**不能把重放当作反馈效果证据**。
5. 声明触觉臂显著增加感觉/全网脉冲，但运动和身体与零下限主体一致；没有证据它驱动了运动改善。此臂还启用了未验证足接触广播，只能作诊断。

### 不能说明什么

- 身体仍很快跌破门槛。比感觉切断晚8.3ms不等于站立成功，更不等于秒级扰动恢复。
- 历史初始姿态存在5个标量关节范围违规，软限制和被动肌肉作用可能影响初期运动；接触门槛通过不等于有效姿态。
- 最大激活很快达到1，表明当前解码存在饱和；不应以加增益追求站立。
- claw/hook方向、电流、连接缩放、膜参数、运动池映射及肌肉解码仍未完成生物学校准；verified-only递质来源不等于逐目标受体符号验证。
- 原规范正下限保留，不能宣称原规范全无非神经主动激活；只有独立零下限对照有相应零激活验证。

## 验证

- 新输入接线及已有时钟/脉冲记录回归：15项通过。
- CLI与连接组事件/方向相关回归：10项通过；CLI子任务另运行9项通过，存在重叠，不累加为独立覆盖。
- 七臂运动栅格与逐细胞计数一致；源码快照逐哈希验证通过；7项实验不变量通过。
- 已实际调用读图并检查：图中正常与重放重合，感觉切断无脉冲，运动切断后激活快速衰减，身体高度仍下降，主体激活饱和。
- 首次仿真归档调用参数顺序错误，修复后全套重跑成功；首次部分目录不是正式成果。

## 正式成果路径

完整实验目录：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/canonical_paired_baseline/20261012_235335/`

- 数据报告：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/canonical_paired_baseline/20261012_235335/paired_results.json`
- 原始轨迹：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/canonical_paired_baseline/20261012_235335/paired_traces.npz`
- 图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/canonical_paired_baseline/20261012_235335/paired_baseline.png`
- 身份与运动池：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/canonical_paired_baseline/20261012_235335/neural_identity.json`
- 源码快照：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/canonical_paired_baseline/20261012_235335/source_snapshot.zip`
- 成果哈希：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/canonical_paired_baseline/20261012_235335/artifact_manifest.json`

## 后续实验矩阵与门槛

|优先级|任务|约束/验收|
|---|---|---|
|下一步|独立有效初始姿态与短时步长收敛|不改冻结姿态；不按站立结果搜姿态/增益；先满足关节范围、接触与埋入检查，再重复主体/感觉切断/运动切断|
|下一步|解码饱和与输入尺度审计|来源或明确假设范围；不进行站立成功导向增益搜索|
|随后|真实身体机械扰动、正常反馈与同扰动命令重放|匹配初态与外力，记录完整入口—脉冲—命令—身体—新入口时间链，恢复指标预先定义|
|并行有界|34个缺verified源、逐根感觉归属、目标受体证据|严格网络保留，证据扩展独立分支，未知不补符号/边|
|门槛通过后|0.1→0.5→1→2秒|身体指标和因果对照通过才称支撑/恢复；延长运行本身不是成功|

本次完成首批基线与100ms因果对照，并非完成全部六阶段计划。
