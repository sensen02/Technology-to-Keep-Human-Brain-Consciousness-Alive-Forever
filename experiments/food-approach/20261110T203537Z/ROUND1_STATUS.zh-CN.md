# 食物趋近目标续轮1：即时残差与命令范围验证进行中

当前目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/`。目标active/armed，原行走模型及旧实验源码/哈希保持不变。

旧slow residual500保住中性但满幅左右slip失败。独立新`range/`提前冻结四个半幅/切换物理诊断，仍用原五门/六周期/既定yaw门：已完成半左slip.52277987、半右.52279882、左后右.52200055均FAIL；方向和六周期通过。最后镜像切换尚在运行，不能声称找到可用部署范围。

新`fast_motor.py`保留原SlowGRU128的151636个参数和明确连续记忆，额外53034参数即时MLP神经残差，以112个允许的身体通道、当前basehidden128及命令2为输入；不输入时间、相位、目标或教师信息。中性残差严格0，直接抑制性目标位置Jacobian保持原值。18项离线检查PASS，实际单次优化检查中基线全参数不变、梯度有限非零、保存/Adam/RNG/推理一致；批量缓存与逐步生产计算差异仅1.39e-17，梯度4.07e-19。

固定1000步训练纳入实际合格左右教师与两档温和减速教师的全2880行，base缓存严格因果。root209exit0，10个保存点独立验证PASS；只评1000，未按中途误差挑点。最终SHA`3dd80394239504ec8b88f4cfe07c79569cf08329a9320c749c3c7b25035fda28`，末段损失反弹如实保留。

root211真实中性→左右/减速两档共五条件及独立实际checkpoint输出/hiddenchain/动作/周期/全窗口记录审计已全部结束exit0并收集。实际中性PASS且完整神经/身体数组逐位复现原通过基线；left/right forward+slipFAIL23.6960/3.47357，speed085所有五门FAIL（支撑塌陷），speed090其余四门PASS但slip1.4584FAIL。独立保存证据核验全部PASS（中性71checks、左右各47、speed各50），不把完整性通过当行为通过。

基于实际失败与独立Adam诊断，新增`continuation3000/`提前登记精确1000→固定3000，仅lr.001改.0001，原权重/Adam6moments+counter/RNG/完整曲线/4data/网络/门控/bound全部保持。child218训练运行，1500/2000保存点各22项独立验证PASS，实际中性输出保持，梯度稳定；仅最终3000运行五条件物理，后续结果尚未形成。实际恢复快照`TRAIN/RESUME_STATE.pt`与`RESUME_CHECKS.json`已核对，避免仅凭加载源码宣称精确接续。

每个回合完整1s稳定+4.8s教师预热+9.6s纯ANN；无根写入、外力、附着，原有限PD/完整71mesh/12.5us保持，HANDOVER哈希立即保存，异常保存部分轨迹，JSON标量类型明确转换。

容量诊断`FAST_RANGE_FLOOR.json`显示左右MSE理论下限约1.3–1.5e-6，两档减速约.000160/.000168；最终拟合误差仍高得多。独立只读审查正在检验误差反弹和实际承重段残差，未据此断言物理失败原因，更未证明身体不可行。

用户直接“继续”后goal已成功resume为revision2/armed。原3000物理pipeline219在协作异常后实际无进程，只保留中性部分IDENTITY/geometry/HANDOVER，无完整轨迹/神经数组/结果，不能恢复未落盘状态；原目录/日志/失效RUNNING摘要全部保留。新child220显式恢复runner在fresh`RECOVERY_EVAL_*`完整重做原缺失评估，固定checkpoint/源码/门槛不变。

恢复中性已原5门/6腿/strictNF0PASS，slip.49278883265，96项实际全轨迹/神经/身份独立核验PASS；左右均失败，左slip5.9139613/forward.02263085/yaw-.00937618，右slip8.70404693/forward.03546713/yaw-.04317865，周期也未全，方向只右通过，其他身体门和NF0保留。左右独立72项完整性检查均PASS；两档speed仍运行。稳定拟合并未解决阶跃命令闭环失败。

`onset_diagnosis/`只读分析已封存，25输入hash+源码复核、6checksPASS；4组前5800物理/960prime/firstANNX和hidden精确，训练1s命令ramp实际公式正确而实测首步满幅。初步200ms/1s同教师状态拟合ramp优于step，但不能从离线结果证明物理因果。`ramp3000/`代码准备36项PASS，等待既定5项全收集后才单独注册actual4条件1s渐变诊断；保留full9.6窗口和所有原门、保留当前step失败，不计食物部署资格。

`recovery_data/`混合学生状态标注采集器草案28项离线PASS，只准备未执行；若需要，完整记录25ms学生/175ms教师恢复的真实X/U/Y/隐藏状态和控制来源，不能把短学生段或混合控制称自主行走。后续数据是否安全/有效需实际新预登记验证。

最终恢复3000五项已全完成、独立审计全PASS：只有neutral功能PASS，其他4失败。随后单独登记parent221匹配1s命令渐变四回合也全FAIL，独立审计均成功；full9.6窗口和全部原门不变，说明命令起始不匹配不是完整解释。对照图`ramp_comparison/ACTUAL_STEP_RAMP_COMPARISON.png`父已实际打开并交桌面。

`recovery_data/`两条25ms学生/175ms教师恢复混合轨迹224/225完整10600端点/1920决策，actor/hidden/来源/物理前缀/记录独立27项核验通过，但安全准入因slip.54953/.55191失败，原dataset_safety_admitted=false保持。纯教师同first4.8s对照.47901/.44702通过，不能把混合失败归为窗口问题。标签单独私有状态顺序重放228/229显示全部1920teacher_action和rootbalance精确相同、新增物理积分0，残差max7.43e-5/.0001032<既有.02诊断值。说明标签来源可核验，不说明动作最优或恢复因果。

新证据后另立`correction_fit5000/`纠正学习范围：失败观察＋经审计教师响应作为CORRECTIVEEXAMPLES，不冒充passingdemonstrations，旧安全准入失败/原验收门全部保留。新预登记精确parent3000→固定5000、同network/bound/lr.0001/model+Adam/RNG/curve，只增加2纠正轨迹与原4示范作6轨迹postprime各等权；数据来源和labelvalidity全部哈希核验通过后才优化。最终模型必须过原五门/六腿/strictNF0/转角/减速验收，不因使用失败状态学习而降低门槛。子e6准备并获授权执行固定训练和final5000真实五条件，尚未出结果。

`feature_conditioning/`独立只读诊断没有发现广泛饱和或hidden幅度被淹没，却发现输入方差异质；函数保持标准化代数验证通过但未训练，且Adam坐标变换不能照搬moment冒称精确接续。为控制变量，本次数据实验不同时标准化。

同次失败视频已按全身可见要求修正相机，`visual3000/ACTUAL_SPEED085_FAIL_WIDE/VIDEO.json`770帧全解码/全部分割轮廓不裁边，71opaque/source/compiled/原状态不变、物理0；父已实际打开最终六阶段蒙太奇。原项目裁边版本保留，桌面旧重复在哈希核验后替换。

纠正学习最终已结束：233追加2000步228.1s，总231.02s，final5000SHA`ddc8d87fa346400ce23b837f3ecc32aa195293fb61f42a69dd3928061f9f5304`，4000/5000各24项保存核验通过，全部旧安全FALSE不改。234五实际回合/五审计exit0收集518.356s；中性原5门/6周期通过，L/R前进.02871168/.03354658、slip2.03706987/5.14954894均失败，左方向/全部6周期通过、右方向通过但缺周期。speed85前进-.02758321/slip4.77047257、speed90前进-.01560978/slip.77514911，两者倒退/塌陷且全部五门/strictNF0/全周期失败，不能计减速。96/73/73/76/76项独立完整性检查均PASS，完整性通过不是能力通过；20实际源码和checkpoint不变。详见`correction_fit5000/FINAL_REPORT.md`和`FINAL5000_SUMMARY.json`。

本轮所有作业结束并收集，无活动训练/仿真或待清理临时文件。行为网络尚未训练，食物到达/摄入/饱腹均未完成。goal active/armed revision2继续，后续最小动作见`NEXT_ROUND_NOTES.zh-CN.md`，不需要用户再授权，也未提前启动下一试验。所有条件/失败/哈希保留。
