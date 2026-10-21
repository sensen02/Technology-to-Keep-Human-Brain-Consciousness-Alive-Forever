# 实际发放源的递质来源与边符号审计

**手工研究原型，不是成品。只读审计，不改符号、连接或增益。**

## 范围及方法

取上一轮两种配对感觉输入中所有实际发放根，加上每组重点运动根，共84个唯一根；其中实际发放根72个。原始CSV按Root ID、v888 feather按banc_888_id、连接缓存按root_ids严格唯一连接。对照cached ann_nt_verified、CSV Verified NT type及feather neurotransmitter_verified。记录所有输入文件SHA256并核查执行前后不变。

源码 `/run/media/sensen/Data2/cell_wound_prototype/engine/neck_cut_data.py` 的build_edges只在annotation模式使用纯ann_nt_verified突触前标签：acetylcholine→We，gaba/glutamate→Wi。未知、混合、histamine和modulator被排除；不从预测字段或nt_pair补符号。`/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/adapters.py` 实际使用annotation/intact模式。

## 测得结果

- 84根的CSV与缓存逐字一致：0冲突。
- 84根的v888 verified与缓存归一化一致：0冲突。
- 实际已发放源→运动池边记录103条（跨实验可能重复同一边），全部符合突触前缓存标签规则：0符号冲突。
- 72个实际发放根：31 acetylcholine、7 gaba、34 verified缺失。
- 34个缺失根的预测字段：28 acetylcholine、4 glutamate、2 gaba。预测值只记录，没有用来补输出边。
- 在84根中，预测与非空verified标签不一致者7根。这不是CSV/缓存摄取冲突；预测和verified是不同字段、不同证据。

## 重要解释

34个verified缺失的发放根可以被感觉输入或网络激发，但在当前严格规则下，其输出连接被排除。**发放不等于其输出已进入模型。**这构成需要明确披露的模型覆盖缺口；尚不能把它定为运动池零脉冲的唯一原因，也不能推断补入预测边一定能恢复运动。

数据集字段名verified不是本轮独立实验验证。即便标签来源一致，项目的ACh→exc及GABA/glutamate→inh仍是受体映射假设，不是每个靶细胞的受体证据。尤其不能把运动细胞自身预测GABA当作解释其输入膜响应的依据：该标签只可能参与输出符号，而且重点运动根本次未发放。

维持当前严格模式，不按感觉类型、运动功能、站立结果或预测标签填符号。下一步应量化这34根的真实原始输出覆盖损失，并单独核查递质/受体映射证据；不进入增益搜索。

## 验证与交付

新增测试2项通过，检查纯标签规则、未知/混合不回退、唯一根、字段一致性和103条边记录。已实际查看 `/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_loop/active_edge_transmitter_audit.png`，31/34/7计数、轴标签及原型说明清晰。

数据：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_loop/active_edge_transmitter_audit.json`

图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_loop/active_edge_transmitter_audit.png`

源码：`/run/media/sensen/Data2/cell_wound_prototype/tools_active_edge_transmitter_audit.py`

测试：`/run/media/sensen/Data2/cell_wound_prototype/test_active_edge_transmitter_audit.py`

桌面目录：`/home/sensen/Desktop/cell_wound_prototype/active_edge_transmitter_audit/`
