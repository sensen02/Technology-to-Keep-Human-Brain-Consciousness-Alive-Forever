# 真实RGB视觉候选1：方向与普通撤除通过，空场和遮挡失败

父唯一托管bash74已完成并收集exit0。真实138项预检通过，production/scan输出、全CNN/GRU参数梯度和Adam误差最大1.1379786002407855e-15；峰值RSS1026.863MiB≤1800。固定freshAdam1e-4、左右配对batch4、full1920GRU与481图像帧完整梯度、500更新完成，训练峰值RSS782MiB，用时约569s。没有模型外归零、死区、坐标输入或记忆reset，没有加载运动模型/新物理积分。

最终权重：`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/visual_policy_v1/TRAIN/checkpoint_000500.pt`
SHA256 f1a339dd86cc2a7e5c7b7d4ba81ef4b39fe8a99b9a74c482ab19980668f137a2。

初始/250/500分别新进程从实际RGB以相同生产形状重放，输出/latent/hidden逐位一致，采样记录/RNG完整核验；只有final500接受一次冻结12验证+8控制。VALIDATION/VERIFY integrity_all_pass=true，RESULT all_targets_pass=false。未挑250或追加500。

## 分项

- 留出左右脉冲末段符号全正确，两向反转符号通过，首次已发布可见帧起2秒响应门通过。
- 普通目标消失后的恢复RMS：左.024220/.012423、右.038275/.021401，均≤.05。
- 空场V0 RMS.0213561264>.02 FAIL，V1 .0054563272 PASS；中心.024273/.032647≤.05。
- 真实遮挡物单独存在时RMS.147153/.159198>.05；遮挡目标后的恢复RMS.150315/.166582/.164004/.178007>.05，全FAIL。原始图像已证明被遮目标与同遮挡物无目标像素相同，因此这是对遮挡物画面的网络偏转，不是“仍看到了后面的目标”的证据。
- 固定旧左帧时输出持续左向，按预登记只作描述，不强制输出0；左右眼交换诊断随几何不同，未伪称等价场景镜像。离线控制不是实际身体控制干预。

本次24训练序列只含空场、中心、左右出现/撤除和反转，没有遮挡物图像；未见图像域是失败的候选解释，不是唯一因果已证实。不能拿方向通过掩盖空场/遮挡门失败，当前不准入视觉身体试验。

## 后续最小候选

保持原架构、相机、20ms延迟/5msheld、运动和气味基线不变，新增训练侧真实遮挡图像及其无目标对照，保留普通方向和撤除历史；新验证使用新几何/新保存身体姿态，旧V0/V1转开发历史。新候选另立预登记、有限预算、保存/验证独立重放，不修改本轮source/FROZEN/结果/门槛，不以runtime“遮住就归零”替代学习。当前只准备CODE_ONLY，未开始下一项训练。

权威结果：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/visual_policy_v1/VALIDATION/RESULT.json`
`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/visual_policy_v1/VALIDATION/VERIFY.json`

全部真实RGB库与已有气味失败保留，无当前确认无用的临时文件。视觉行为完整离线成功、实际视觉身体闭环、视觉气味融合、导航与停驻尚未建立。
