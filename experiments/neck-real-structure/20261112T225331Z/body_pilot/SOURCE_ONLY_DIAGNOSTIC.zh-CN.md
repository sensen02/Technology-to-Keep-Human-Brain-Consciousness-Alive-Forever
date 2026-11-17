# 六回合身体pilot后的独立只读源码诊断

本报告只读取已保存JSON和冻结源码，不载入NPZ、Torch、NumPy、模型、身体或新物理；不修改原源码／阈值，不启动训练。完整性核验全过与身体功能未达标同时成立，不能将sourcequeue重放PASS当作良好闭环行为。

## 已保存结果实际支持什么

六回合均stablebody／六腿周期／原生非足0，独立实际X238→AN→源队列→脑16→DN→decoder→motor256重放完整性成立；原五工程门因滑移全部FAIL，范围1.0967—2.0927。P_empty turnRMS.53838，尽管最终净yaw仅−.518°，仍存在大幅偏转／反向抵消，不能用最终航向近0称中性成功。L脉冲撤除后固定比较窗RMS.58883，R为.31175；DNcut将decoder输入静默0，真实输出−.036864537保留；ANcut恢复窗RMS降至.09987。cuezero与empty神经／yaw逐位一致，支持没有通过condition名字暗藏动作，但不证明闭环稳定。

## 身体观察选择没有发现索引或绕过错误

冻结Core.obs238依次为proprio183、clearance12、previous42action、末位target_speed，共238。原proprio实际定义见：

`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/sources/proprioception.py`

111允许索引为0:42(qoffset42)、84:126(qvelocity42)、168:195(负载6、重力3、线速度3、角速度3、足净空／净空速度12)。排除targeterror、targetposition、旧action与target_speed237。在线BrainNeckLoop使用checkpoint.allowed_raw_indices取同一111，再调用同一normalize与feedbackMLP；与离线data.py的raw[:,MASK]一致。未发现把privilegedcritic或目标状态送给人工脑的路径，bodydecoder只接heldDN，brain只接heldAN＋cue。固定motor保留原本地238接口／局部目标状态是已声明身体侧贡献，不是脑旁路。

取当前obs并不推进感觉导数状态；_pad_rate只在物理tick后更新，和旧采样接口一致。新增settle观察不会被直接认作物理驱动；实际旧预热prefix比较也已通过。没有源码证据支持“选错bodyobs索引／时序错位”作为目前结果的解释。

## 预热差异要精确定义，不应直接叫双稳态

离线NeuralBridge.Scan／python_rollout起始h16=0、heldAN=0、heldDN=0；在线warmup期间brainh也始终0，DNzero发送，feedback是无状态MLP，没有自身递归隐藏状态。sourcequeue只保2ms在途、held仅最新已发布值。

所以“5.8秒真实AN预热”不会让脑累计5.8秒GRU状态。第一ANN时h仍0、DNheld仍0，但ANheld来自实际5795ms身体／5797ms发布，这是相对离线**第一次脑更新的一个非零AN输入差异**。首动作均先经过decoder(0)，但下一决策DN可因这一差异而不同。

一个初始AN脉冲是否足以把脑送到持久不同状态，是可检验假说；现有数据未证明吸引域或双稳态。既不能偷偷清AN队列让初值匹配，也不能仅凭“历史不同”断定是唯一根因。

## 更强的现有候选解释：外生数据拟合不等于闭环稳定

离线训练的4条body111序列来自旧视觉模型驱动的实际身体，作为固定外生输入。离线MSE只检验“给定该固定身体轨迹，预测旧rawturn”，未对新脑／decoder产生的动作怎样改变下一身体状态反传或评估。

在线首个decoder(0)=−.036864537已经不同于旧视觉empty近零turn。新转向改变关节／足载／姿态，再改变AN1编码，AN又能驱动脑和DN，产生离线未覆盖的闭环反馈。真实空场出现大turn与ANcut使后段恢复改变，符合反馈通路参与偏转，但不单独证明正反馈符号／机制。

reconstruction从AN1还原111维的离线归一化RMSE约.9—1.06，仅说明单标量压缩较有损，不能称已重建本体感觉。AN08B018的native输入本来也并非任意111特征的真实生理映射，这仍是明确工程适配。

normalize使用四条trainbody111的mean/std，每维std下限1e−6。在线轻微超出近常量训练维度可被放大，tanh编码可能饱和；必须实际比较保存数据，不能仅凭源码存在floor断言饱和。decoder偏置和模型闭环分布变化须与这一数值尺度因素分开测。

离线cue源自旧视觉已发布帧，常见起止约.62／3.62；新实验cue按预登记.6／3.6，20ms差异是明示的新任务时序，不是源队列丢帧。它不能解释cue全零empty的大幅输出，因此不是充分解释。

## 下一最小可证伪诊断：固定权重、只重放保存观察的2×2

由父唯一CPU1执行，单次不训练、不物理、不改旧结果：

| 观察序列 | 首次脑heldAN=0 | 首次脑heldAN=真实预热值 |
|---|---|---|
| 原离线V_empty实际111序列 | 应重现原离线输出 | 单独改变首次AN初值 |
| 新P_empty保存的实际111序列 | 在固定新观察下移除首次AN初值差异 | 应重现新在线ANN输出 |

四格brainh0／heldDN0、相同normalization和5ms单决策双延迟。这里只改第一次heldAN；之后每次编码／释放按同序列推进，不能把整个AN流清0或改为新物理反事实。输出turn／AN／DN／brainstate在0—.1、.1—.6、全程、3.6后与固定恢复窗的RMS／峰值，以及逐时间差。

解释边界：若仅原观察＋warmAN就复现偏转，则支持初值敏感；若新观察＋coldAN仍偏，则仅warm初值不是充分解释；若两因素交互才明显，不能只指定一个根因。固定保存观察重放不证明对应真实身体会走成新轨迹，不替代后续物理。

再做描述性trainstats诊断：111维训练std最小值／floor计数，各body阶段归一化绝对值、超train分位／范围比例，ANtanh输出接近±1的比例、从输入尺度到encoder活动变化。对实际轨迹作这些描述，不能先设有利阈值删掉失败样本。

若需进一步区分脑动力学，固定真实AN片段作小幅初值扰动和cue0长时离线响应，必须另预登记并表明不是新身体结果；现阶段不直接延长200训练或清队列。

## 结论

源码与独立重放没有显示bodyobs选错或neck时钟漏帧。最优先查的是**首次AN初值敏感与新动作引起的观察／反馈分布变化**，并测归一化尺度和一维反馈压缩；目前不能唯一归因或称已证明双稳态。修复应保留真实颈部身份／方向、无旁路和所有物理FAIL，再在新目录预登记明确更改，不在在线循环加手写零turn门或减偏置掩盖问题。
