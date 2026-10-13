# 感觉调谐标注本地审计：可映射到粗亚型，尚不能映射到屈伸极性

这是**手工实现并测试的只读数据审计，不是完成的生理模型**。没有修改核心模型，没有重建网络，没有随机分组，也没有从形态、左右侧、SNpp 编号或坐标猜测极性。

## 结论与下一步决策

当前本地 BANC 标注可确定 claw/position 与 hook/direction 粗亚型、腿侧和根身份；当前实际 tier 中 **149 个 claw、136 个 hook，共 285 个细胞**，没有发现明确的 flexion/extension 标签。不能据此声称生物学上没有相反调谐群。`cell_type` 的 SNpp 编码确实提供额外类型信息，但本审计没有可验证的“该编码→屈曲/伸展生理类型→此版本 root”的交叉表，不能把编码的数字当极性。

**面向后续模型的具体判断：** 现在不能做有数据依据的逐 root 屈伸群拆分。若文献有 hook-flexion / hook-extension 或 claw 对向调谐群，应先获取文献细胞身份/外周组与 BANC v888 root 的可验证 crosswalk；否则只能声明为“群体层面的文献约束假说模型”，不能给现有 root 随机分派极性。现有统一 signed-angle 编码是声明的模型假设，不因粗亚型标注而获得生理验证。该审计不否定父任务的文献证据，也不把形态位置当功能测量。

## 输入、版本与精确筛选

- 运行入口：`/run/media/sensen/Data2/cell_wound_prototype/run_sensory_tuning_annotation_audit.py`
- 测试：`/run/media/sensen/Data2/cell_wound_prototype/test_sensory_tuning_annotation_audit.py`
- 机器产物：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/sensory_tuning_annotation_audit.json`
- 主数据：`/run/media/sensen/Data2/cell_wound_prototype/data/banc888/banc_888_meta.feather`，188,508 行。
- 原始标注补查：`/run/media/sensen/Data2/cell_wound_prototype/data/flywire/banc_neurons.csv.gz`，包括 Community labels、Primary Cell Type、Alternative Cell Type(s)、Hemilineage、Soma side。
- 运行时 lookup：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_rebuild/sensory_channel_lookup.json`，路径来自 `/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/sensory_channels.py`。
- 当前发布 run：`add80e12-4b1b-45fc-90be-2336ddf7757c`，身份文件 `/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/runs/add80e12-4b1b-45fc-90be-2336ddf7757c/neural_identity.json`。

先验证 current_run→manifest→identity 哈希，再以 identity 的 tier-local indices 精确取 roots，并逐个核对 lookup 的 leg/channel；对 feather **只用 `banc_888_id` join**，不回退到其他版本的 root 字段。所有 3,788 个通道细胞在该 join 上恰好匹配一次，也在原始 CSV Root ID 上恰好匹配一次。JSON 保存全部 3,788 个 index/root/leg/channel 以及 285 个 claw/hook 的逐 root 标注、位置行号及分类。

注意 exporter 源码使用过 `root_id` 输出 lookup；本次不重新导出、不假设跨版本通用，而是对运行时实际 lookup、已发布身份和 `banc_888_id` 的实际交集做验证。源版本与内容哈希均记录。

## 数量

当前 tier 12,000 个神经元；感觉通道共 3,788：claw 149、hook 136、hairplate 207、campaniform 60、club 323、bristle 2,879、multidendritic 34。

| 腿 | claw | hook | 合计 |
|---|---:|---:|---:|
| lf | 25 | 24 | 49 |
| lm | 33 | 20 | 53 |
| lh | 33 | 22 | 55 |
| rf | 24 | 25 | 49 |
| rm | 12 | 22 | 34 |
| rh | 22 | 23 | 45 |
| 全部 | 149 | 136 | 285 |

全 feather 的 claw/hook 粗亚型共 304：claw 162、hook 142；这个全表数量**不是**当前 tier 数量。JSON 分开保存全表 304 行与当前 285 行每一列的非空数、缺失数、不同值数、最多 8 个标签例子和极性词分类。

## 细标签、形态名、侧别与谱系

以下为当前 285 个细胞：

| 字段 | 非空 | 有效值/例子 | 能否确定极性 |
|---|---:|---|---|
| cell_sub_class | 285 | 六个腿位×claw/hook 粗亚型 | 否 |
| cell_function_detailed | 285 | position 149；direction 136 | 否 |
| cell_type | 285 | 10 个不同编码；SNpp50 90、SNpp39 88、SNpp41 50、SNpp51 48 等 | 编码尚无经验证的极性 crosswalk |
| manc_cell_type | 285 | SNpp50、SNpp39、SNppxx、auto:IN01B028 等 | 否 |
| malecns_cell_type | 210 | 含 auto: 标签；75 缺失 | 否 |
| fanc_cell_type | 1 | auto:13A-10e-δ | 不可解释为屈伸证据 |
| side / CSV Soma side | 285 | left 157；right 128 | 左右不等于屈伸 |
| hemilineage / CSV Hemilineage | 0 | 285 缺失 | 否 |
| seed_04 | 281 | hind_leg_claw_chordotonal_organ_neuron_left 等 | 形态/亚型+侧名，不是屈伸 |
| CSV Community labels | 33 | SNppxx、SNpp39、claw chordotonal neuron 等 | 否 |
| CSV Alternative Cell Type(s) | 285 | SNpp39,hook；SNpp50,claw 等 | 否 |

`finecelltype`、`morphologylabel`、`sidenames`、`lineage` 这四个字面字段不在 feather schema；已查实际对应的 cell_type、跨数据集 type、side、hemilineage、seed、原始 CSV alternative/community 标签，没有用“字段名不相同”代替检查。扫描所有非 ID 标注列及原始 CSV 标注后，285/285 为 `no_explicit_polarity_tag`。缺失列单独归为 `missing_annotation`，不是“中性细胞”。分类只是显式词检索，不将含糊词解释为生理实测。

发现一行 `cell_class=hair_plate_neuron` 与 claw/hook 的 `cell_sub_class` 不一致；保留运行时实际粗亚型分组，不自动修正该数据冲突。标签也有自动匹配及 glia 等冲突提示，应在未来逐 root 生理映射时复核。

## 是否有目标角、增益、tonic 等数值测量

没有发现 target/preferred angle、tuning、gain、tonic、firing rate 等参数列，不能建立每细胞的目标角/增益/静息放电参数表。schema 的数值列是：neurotransmitter_score、l2_nodes、l2_cable_length_um、volume_nm3、input_connections、output_connections、input_side_index、output_side_index、mitochondria、mitochondria_volume、pd_width、segregation_index。

这些分别属于递质预测评分、几何量、连接数/侧指标或细胞器量，**不是感觉调谐测量**。`position`、`root_position`、`root_position_nm` 是解剖坐标字符串，不是目标关节角。当前编码器的 0.30 rad reference_angle、10 rad/s reference_velocity、0.3 nA/unit current scale 是声明的模型归一化/工作点，不是本数据逐细胞测量。

## 哈希与复跑验证

所有输入与审计/测试源码的完整路径及 SHA-256 在 JSON 的 `source_sha256_before/after`；二者一致。主要数据哈希：

- feather：`86ccf5df0c67419f8c5f43e93a7ed38d23a080e9f7fde26737290252f3780098`
- lookup：`710939c76092d5e60cc261bdd55e3b69c4cbe979db922afae34db230bd7b1b63`
- raw CSV：`40a2201554a8c34d2c4b07c8322543a07c1b3faf5acafbac363fd1a3d0fa617f`
- identity：`efb913ce7dcfc768db528686b0c9ea96d2e770cdcf18143e740f0ac49adc1db9`

使用 pandas 环境 `/run/media/sensen/Data2/cell_wound_prototype/venv/bin/python`，不是 venv_body。共享 `heavy_job` 内存锁，最多等 45 秒；整个审计有 60 秒 alarm，BLAS/OMP/MKL/NUMEXPR 全部 1 线程，512 MB 可用内存门槛、2,048 MB 地址空间上限。实测首次正式执行约 0.86 秒，峰值虚拟内存 1,785 MB，RSS 约 538 MB；不是零内存开销。

四项 unittest 验证：唯一根 join（重复/缺失拒绝、浮点 ID 拒绝）、缺失与极性词有限分类和严格 JSON、分组错误/漏组拒绝、真实产物精确 3,788/285 数量及逐 index-root 关系/全部源哈希。**4/4 通过**。本任务是有界文本/数据审计，不包含图像或物理仿真成果。
