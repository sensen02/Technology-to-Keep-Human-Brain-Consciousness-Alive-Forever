# 开源果蝇脑与真实颈部下行台架

基础下行台架通过。冻结的 Eon/Shiu 全脑输出接入四根具名 BANC 颈部轴突，脉冲在源图的实际后突触靶细胞上产生模拟神经响应。范围到 VNC 第一阶靶点，实验和人工搭建大脑分支各自归档。

## 实测结构与数值

- 原样使用 Eon 固定提交的 FlyWire v783 连接数据与 LIF 动力学。全脑实际积分 138,639 个细胞与 15,091,983 个连接表条目。全部上游文件对发布 Git blob 与 SHA-256。CPU 稀疏计算只累加当前活动突触，和原版模型配对 80 步，五份神经状态逐位相同，含 17 个非零和 3 个混合正负输入步。 [Eon 固定提交](https://github.com/eonsystemspbc/fly-brain/tree/a3db62f9436074e485c0278290c2164ed6150808)
- 两对四根 BANC v888 SWC 重新沿同一祖先分支验证穿越两处颈横截面。原始边表准确匹配 2,791 条定向边，合计 19,196 个标注突触，连至 2,551 个 VNC 第一阶靶点。逐轴突的靶数 DNa02 左 303、右 340，DNg100 左 1,063、右 1,085。DNa02 的 ACh 有确认注释，DNg100 使用 ACh 预测。神经轴突和突触来自解剖数据，同源配对跨标本，时延 2 ms 及膜参数用于工程模拟。 [BANC 发布研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC13518251/)

## 同输入配对结果

每条件 160 ms，先静息 20 ms、刺激 90 ms、撤除 50 ms。以下脉冲为 90 ms 刺激段内具名 FlyWire 神经元的发放次数。PFL3 通过脑内源图侧向驱动 DNa02，同时出现 DNg100 活动。此处发送的是开源模型实际产生的脉冲，同源映射只做身份转换。

| 具名刺激 | DNa02 左／右 | DNg100 左／右 |
| --- | ---: | ---: |
| 空输入 | 0／0 | 0／0 |
| 直接刺激左 DNa02 | 9／0 | 0／0 |
| 直接刺激右 DNa02 | 0／10 | 0／0 |
| 直接刺激双 DNg100 | 0／0 | 16／8 |
| 左 DNa02 上游 PFL3 | 9／0 | 5／5 |
| 右 DNa02 上游 PFL3 | 0／11 | 4／4 |
| 上游公开配置双 DNp09 | 1／1 | 0／0 |

同一脑发放记录下，全切四根轴突丢弃 35 个脉冲。只切左 DNa02 丢弃 8 个，另外三根的到达记录逐位一致。恢复只在接到新脉冲后出现。例如双 DNg100 条件的首个颈后左侧恢复脉冲在 88.7 ms 到达。此条件整个回合的颈后首阶群体累计发放完整时 279 次、全切再恢复时 153 次。三个条件由独立 NumPy 实现逐步核对到达时钟、VNC 脉冲与 85,713,600 个膜电位样本，脉冲逐位一致，最大膜电位差 0 mV。

## 复验

工作目录为项目根。使用 `experiments/banc-walking-recovery/20261110T125712Z/env/bin/python`，Python 3.12.14、Torch 2.5.1 CPU、SciPy 1.18.1，OMP、BLAS、MKL 各设一线程。`run_probe.py NEW_OUTPUT` 退出 0，用时 80.88 s，峰值 1,668,536 KiB。`audit.py OUTPUT` 退出 0，用时 1.57 s。图由同次记录重画并实际打开。完整源码、上游快照及记录保存在 `/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261117T045901Z-foundation/`。 [预登记](/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261117T045901Z-foundation/PROTOCOL.zh-CN.md) · [复算证据](/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261117T045901Z-foundation/BENCH/AUDIT.json)

本轮外部刺激直接施加于具名细胞，尚无复眼转导、上行、身体闭环或行为验收。训练零更新。原版 DNp09 探针在这组刺激及时长下未招募 DNg100，PFL3 两侧探针均有联动。它们可作为下一轮讨论时的冻结基线。
