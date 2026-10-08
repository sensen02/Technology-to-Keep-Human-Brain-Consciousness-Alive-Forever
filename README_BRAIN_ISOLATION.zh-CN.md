# 果蝇局部隔离模拟引擎：研究原型

这是一套自行集成、编写和测试的机制原型，不是高保真果蝇数字孪生，也不验证意识维持。主问题是眼—脑保留时，局部断连、胞外刺激和环境改变如何在明确的假设下影响模型。

## 三类结果必须区分
1. **理想化统一场景**：3个点神经元+1条主动电缆组成眼→脑→颈→身体单向图。主动通道是经典鱿鱼HH，非果蝇参数。用于验证唯一所有权、事件时钟、断连和多速率耦合。上游脑在颈断连后保持活动是图结构结果，不能作为生物学发现。
2. **真实FAFB几何子树**：来源ID与nm单位有记录，保留正半径连通子树，不补造半径；人工封闭边界显式记录。仅被动膜响应，非完整神经元或真实颈切口。
3. **BANC粗层/视觉接口**：真实连接表和注释用于覆盖审计与计算测试。边表递质全部空，严禁未知默认为兴奋。纯递质神经元注释映射只是显式受体假设；混合/未知标签排除。短时响应非稳态生理校准。

## 目录与运行
源项目：/run/media/sensen/Data2/cell_wound_prototype/
桌面增量样本：/home/sensen/Desktop/cell_wound_prototype/brain_isolation/
解释器：/run/media/sensen/Data2/cell_wound_prototype/venv/bin/python
依赖：NumPy、SciPy、Matplotlib；主动后端需NEURON。软件版本及源文件hash写入验证报告。

运行所有局部数值测试：
```bash
/run/media/sensen/Data2/cell_wound_prototype/venv/bin/python /run/media/sensen/Data2/cell_wound_prototype/verify_brain_isolation.py
```
生成统一场景：
```bash
/run/media/sensen/Data2/cell_wound_prototype/venv/bin/python /run/media/sensen/Data2/cell_wound_prototype/run_isolation_scenario.py
```
生成膜电流正向记录：
```bash
/run/media/sensen/Data2/cell_wound_prototype/venv/bin/python /run/media/sensen/Data2/cell_wound_prototype/run_neural_recording_demo.py
```
桌面脚本用同一解释器、将脚本前缀换成桌面绝对路径即可。需要大连接组/完整压缩包的审计和数据生成脚本仍要求源项目数据；交付不会复制13GB归档或凭据。桌面保留小SWC样本和已有输出。

## 模块能力与限制
- 电缆：稀疏被动矩阵、切边、单独断端电导；严格单位与拓扑验证。主动NEURON理想电缆独立接口。
- 粗细：固定所有权，精细端输出与输入站点人为规定；无真实突触坐标、无动态升降精度。
- 电极：有限球体电流源+回流、均匀无限导体。非金属等势电极、非电化学模型。空间电流平衡不等于时间电荷平衡。
- 记录：统一场景目前仅伪迹；另有被动膜电流（含电容电流）正向记录示例，不再拿平均Vm当LFP。尚无自洽ephaptic反馈。
- 化学：有限物质量、扩散、受体结合和明确外部库收支。不是已校准激素系统。作用到神经电导的未知增益默认0。
- 支持：availability仍是非生理代理，绝不可叫ATP/O2。另建离子守恒简化模块，不能由K单池推断细胞生存。其无泵/无稳态反馈的胶质结合池可把有限胞外K耗近0并产生极负Ek；这是模型不足的反例，不是胶质保护或真实静息电位。损伤示例明确采用外部电钳中性盐混合、未跟踪counterion和能量；膜电流不反馈离子池，非自洽电扩散。

## 参数和证据
单位、实测/拟合来源、示例值和假设必须分开。DNp01/DNp03被动参数不自动适用于全脑；经典HH不与果蝇测量混称。受体身份/浓度与神经效应并非一回事。逐项对账见 `outputs/brain_isolation/EVIDENCE_LEDGER.md`；真实数据可用性见 `REAL_DATA_DA_EVIDENCE.md` 与 `da_figure5_digitised.json`。

## 验收解释
数值测试验证代码实现、收支与收敛。示例图必须实际打开检查。输出报告既列成功也列数据缺口。不能从放电、同步、复杂度、断连后残余响应推断意识、长期存活或现实实验可行性。

## 第5轮新增要点（引用前必读）
- 真实突触坐标已取得：`synapse_table`（gzip 2.695 GB，流式扫描 80,215,791 行，峰值 152 MB），目标神经元 54 个真实突触前位点，最近节点分配中位 0.314 µm。复现：`run_synapse_map.py`。
- **首次真实数据定量对照是否定结果**：纯"池缩小/补充变慢"假设在结构上无法重现实测的刺激1排序（年老对照 0.57 > 年轻 0.41 µM，差 5.3×读数误差）；唯一可行路径（池上限截断）也逐点失败。复现：`run_da_real_comparison.py`；实测值出处与目读误差见 `outputs/brain_isolation/da_figure5_digitised.json`。
- 单一共享动力学**无法**同时解释中央复合体与蘑菇体，必须区域特异参数——这与"用一个统一旋钮控制激素"的设想直接冲突。
- 缓存连接表与 `synapse_table` **不是同一构建**（同一细胞 14 行 vs 65 行），基于缓存的连接数量主张需重新标注。
- 断言纪律：只引用定稿产物。本轮我曾据中间产物写出一次错误结论（"空间分布无影响"），定稿后更正为 −37.7%。
