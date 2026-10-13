# 递质缺失与预测证据边界

手工研究原型，不是成品。当前没有改变神经边符号或规范物理XML。

## 本地证据

原始Codex CSV按450个LF触觉细胞root ID匹配：Verified NT type全部空白，与缓存一致，因此不是该已验证字段的读取遗漏。Predicted NT type另有ACH424/GABA19/GLUT2/空白5。这是预测而非已验证，不能覆盖原字段。

## 本地版本和置信度复核已完成

原CSV全部158262条ID都能匹配feather的banc_888_id；不可直接混用其他版本root_id列。对目标450细胞，feather的已验证字段仍全空。445个非空CSV预测与feather完全一致，另外5个CSV空白在feather中有预测值；不是5个相反递质标签。

CSV预测置信度：ACH范围0.49–0.98、中位0.87；GABA范围0.50–0.81、中位0.60；GLUT范围0.51–0.57、中位0.54。它们仅是原表描述值，尚不能解释成统一概率。

四项测试通过：ID精度/拒绝浮点ID、空值语义、列映射/重复行、450个根ID与源哈希一致性。源文件保持未变。

- 实现：`/run/media/sensen/Data2/cell_wound_prototype/run_touch_transmitter_source_audit.py`
- 数据：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/touch_transmitter_source_audit.json`
- 测试：`/run/media/sensen/Data2/cell_wound_prototype/test_touch_transmitter_source_audit.py`

原下载脚本使用Codex dataset=banc，但没有显式固定materialization版本。ID重合和文件哈希可追溯本地快照，不能据此虚构统一下载时间/版本。

## 外部来源核对

Virtual Fly Brain官方文档说明BANC预测来源为Codex neurons.csv的Predicted NT type，只在不与Verified NT type冲突时使用；预测限制为选定递质且假定每神经元单一递质，少于100个突触前位点的神经元被排除。置信度直接来自原研究，不同数据集尺度不可混同，VFB本身不设置统一截断。

来源：[VFB Confidence Values](https://flybrain-ndb.virtualflybrain.org/docs/concepts/confidence-value/)
对应BANC研究：[Bates et al. DOI](https://doi.org/10.1038/s41586-026-10735-w)（由VFB文档关联，尚未在此独立通读全文）。

不能将局部缓存中的pair突触计数直接当成完整神经元突触前位点总数，也不能直接把置信度当后验概率。若未来有预测辅助模式，必须与严格已验证模式隔离，记录原字段、预测值、置信度、来源、阈值及符号假设，不能宣称修复了已验证生物通路。当前仍先完成本地字段和版本审计。
