# 神经时钟隔离诊断

手工研究原型，不是成品。只运行合成单/两细胞数值夹具，没有改正式BANC连接、阈值、增益或解码器。合成强电流和一条测试边仅用于单元测试，绝不作为项目生物学证据。

## 已验证

12项测试通过（6项新时钟夹具、6项既有饱和测试）。实际看图：粗步长兴奋脉冲响应偏高，细化后接近细步长分裂参考；抑制误差较小。

- 强驱动夹具：2ms不应期下，步长0.5/0.25/0.125/0.0625ms最小ISI为2.5/2.25/2.125/2.0625ms。原因是起点检查可积分、终点检测脉冲，不是已证明的计数器bug。不可偷偷减去dt。
- 两细胞夹具：上述所有步长，源脉冲终点至目标到达起点精确2ms。额外环槽是正确的，不应删掉。
- 恒定亚阈值电流：frozen_exact与解析常系数解一致；后向Euler误差随步长减小而下降。
- 指数电导脉冲：两种方法仍冻结区间起点电导，故frozen_exact并非连续变化电导的精确解。相对于0.00390625ms frozen_exact参考，兴奋脉冲最大电压差：后向Euler从0.67321mV（0.5ms）降到0.07774mV（0.0625ms），frozen_exact从0.91724降到0.10630mV；抑制误差分别从0.01746/0.03978降到0.00190/0.00467mV。该参考自身仍有分裂误差。
- 相同物理脉冲时刻下，解码器bin平均约定相对终点脉冲解析参考的率差为-4.84%/-2.46%/-1.24%/-0.62%。这是建模约定差别，不证明参考更生理。

## 意义与下一步

发现了足以影响阈值/裁剪的数值机制，但尚未证明哪项主导完整身体实验优势翻转。不把简单换frozen_exact当修复，因为本兴奋夹具在粗步长下它误差反而较大。

下一实验：固定同一真实身体输入历史，跨时钟重放到实际BANC网络，保存全体脉冲事件和运动池逐根首次差异，隔离神经传播差异与身体输入分叉。随后才决定独立数值积分分支，不改生理参数、不以站立成功搜索方法。

## 输出

- 工具：`/run/media/sensen/Data2/cell_wound_prototype/tools_neural_clock_fixtures.py`
- 测试：`/run/media/sensen/Data2/cell_wound_prototype/test_neural_clock_fixtures.py`
- 数据：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/neural_clock_fixtures/fixtures.json`
- 图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/neural_clock_fixtures/pulse_clock_comparison.png`
- 解码器时钟工具：`/run/media/sensen/Data2/cell_wound_prototype/tools_decoder_clock_response.py`

正式神经模型源码未修改；合成夹具不能算完成闭环或站立验收。
