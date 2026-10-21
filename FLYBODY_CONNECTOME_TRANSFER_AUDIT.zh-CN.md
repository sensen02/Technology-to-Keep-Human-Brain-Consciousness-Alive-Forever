# 真实连接组配对输入传播审计

**手工研究原型，不是成品。不是身体闭环、站立成功或生理调谐验证。**

## 方法

重新构建项目现有 BANC tier，逐项核对12,000根顺序与已验证 manifest/identity 完全一致，感觉根3,788个。使用重建的真实 We/Wi，沿用运行时×0.2权重尺度。没有创建、替换或调整连接，没有直接向运动池注流。

三组网络使用相同默认初态、零背景输入，dt=0.5ms，总时长100ms：零输入对照；左前腿hairplate电流脉冲；左前腿hook电流脉冲。两个诊断电流来自上一轮实际身体观测编码测量，分别0.125和0.03702 nA/细胞，持续20ms后撤除。恒定矩形输入是诊断刺激，不是自然运动重放。模型没有驱动身体或使用运动解码器。

## 测得脉冲数

| 组别 | 全部 | 感觉细胞 | 非感觉细胞 | 运动池 |
|---|---:|---:|---:|---:|
| 零输入 | 0 | 0 | 0 | 0 |
| coxa位置代理→hairplate | 116 | 112 | 4 | 0 |
| tibia速度→hook | 41 | 24 | 17 | 0 |

目标感觉通道脉冲分别112和24；本次全部感觉脉冲均在目标通道中。非感觉脉冲出现证明本模型中输入引起下游响应，不是仅有膜电位变化。但是**运动池两组均0脉冲**，不能宣称完成感觉→运动输出链。

hook输入更小但非感觉脉冲更多，不能据此推断普遍感觉强度或生物学优势：目标群大小、连接和模型状态不同。本次只有左前腿两种输入；不能推广所有腿或所有感觉模态。也不能由零运动脉冲断言不存在结构通路：亚阈值响应、时窗及静息状态均可能影响结果。

## 验证

- 真实tier根序列与存档身份完全一致。
- 逐时间步核对全部=感觉+非感觉，以及总脉冲与逐步计数之和。
- 联合测试 **65 tests / OK**，未要求刺激必须产生运动脉冲；阴性结果也是合法测量。
- 已用 read_image 实际查看 `/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_loop/flybody_connectome_transfer_audit.png`：上图112/24感觉脉冲、中图4/17非感觉脉冲、下图运动池全零，与JSON一致；标题及原型限制说明清晰。

## 文件

程序：`/run/media/sensen/Data2/cell_wound_prototype/run_flybody_connectome_transfer_audit.py`

测试：`/run/media/sensen/Data2/cell_wound_prototype/test_flybody_connectome_transfer_audit.py`

数据：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_loop/flybody_connectome_transfer_audit.json`

图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_loop/flybody_connectome_transfer_audit.png`

桌面目录：`/home/sensen/Desktop/cell_wound_prototype/connectome_transfer_audit/`

下一步优先区分结构可达性与动力学阈值：追溯本次发放的非感觉根、其运动池连接及运动池亚阈值响应。不以放大电流、改符号、补连接或隐藏背景来制造运动脉冲。
