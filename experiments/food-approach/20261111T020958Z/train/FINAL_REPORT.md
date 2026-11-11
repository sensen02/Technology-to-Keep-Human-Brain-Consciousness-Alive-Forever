# 固定5000非线性正向神经读出训练完成，尚未运行物理验收

最终检查点：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T020958Z/train/TRAIN/checkpoint_005000.pt`

SHA-256：`8e341270d82d923dcf19ec5b7286fc5797806688b3f27567f75ea3336f7216bd`

父检查点固定为既有head2000：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T010218Z/train/TRAIN/checkpoint_002000.pt`，SHA `eeced49c5bdd50a7e5bce2cfd63d56d9929b3bf2d2cb00c1985019d91321e6df`。其全部197608参数及显式128+64神经记忆冻结；旧父元数据路径问题的外部更正清单和真实父权重/身份均以哈希绑定，没有改写旧文件。

新MLP使用112个允许物理通道、当前base128、当前residual64和神经命令2，共306维，经64维tanh隐层到42维零初始化输出。额外latent为relu(actualturn)*.10*tanh(output)，forward为0。新参数实际22378个。没有额外神经记忆、没有时间/相位/教师/食品坐标输入，也没有外力或姿态脚本。新增输入覆盖、非线性容量和正向范围一起变化，不能单独归因某一因素。

## 固定训练与实现检查

原合格left_full实际2880行是唯一训练序列；父模型严格按实际完整历史逐步计算cache，960条priming不计损失，1920条post决策按动作关节均匀MSE。fresh Adam仅覆盖新MLP四个tensor，lr=.001、seed20261111，固定5000更新，没有提前选点或延长预算。

- root及train/PREREG均在优化前落盘。
- preflight `bash-2` 退出0并收集，28项通过。两次临时优化单步后丢弃，无持续权重产物；零输出层使第一次encoder梯度为0属于预期，第二次反传全部四组新梯度有限且非零。
- 初始化left/neutral/right全部2880原输出/动作/双状态精确；临时更新后neutral/right仍全部精确，父权重不变且没有梯度。
- 缓存与完整生产递归输出差1.11e-16、全部新参数梯度差4.66e-21，低于预登记1e-12；当前父双状态逐位一致，原目标位置负Jacobian保持。
- 持续训练 `bash-3` 退出0并收集。更新5000耗时62.58秒，含保存和独立核验总体70.50秒。
- 初始左转动作MSE `8.720976209497367e-5`，最终post-update `3.8116626221184675e-5`。训练拟合下降不代表真实滑移或方向通过。

2500、5000保存点各通过19项独立新进程检查：完整curve0..N、全部父197608权重精确且冻结、新Adam四状态/计数/形状/有限非负二阶矩/超参数/加载一致、RNG恢复、中性/右转实际2880保存流mean/action/192hidden逐位一致、左转完整生产与cache核对。每次保存也记录了live model/optimizer/RNG/config/curve与保存payload全状态精确比较，随后恢复live RNG，避免构造校验模型消耗训练随机状态。

源文件、原父模型、原数据、旧审计与外部元数据更正哈希均在训练前和保存时核验。所有源PY/JSON现已冻结，报告仅新增Markdown，不修改捕获文件。

## 对父代理的实际评估接口

加载模块：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T020958Z/train/model_io.py`

`load_checked(checkpoint, sha, require_final=True)` 返回 `NonlinearMotor,payload`；默认只允许最终5000物理。`initial_state(B)` 返回(base128,residual64)，`step(obsB238,commandB2,state)` 返回pre-tanh mean42和连续tuple状态。`.base`返回原基线actor。最终动作只做一次tanh。

新payload为format_version1、identity、state_dict、optimizer、config、curve、rng。config `kind=fixed_nonlinear_positive_readout5000`、`physical_eval_only_step=5000`，source_hashes/data_hashes均严格验证。未来实际评估由父代理负责，仅最终5000中性/左/右三条件；所有原物理门、六腿周期、NF0与方向门保持。右转和中性函数在相同观察流中保持并不等于任意正负切换历史都已验证。

本代理未运行Core、物理积分、食品行为训练或部署评估，原失败轨迹保持FAIL。全部相关任务已收集，无活动训练或临时单步目录。
