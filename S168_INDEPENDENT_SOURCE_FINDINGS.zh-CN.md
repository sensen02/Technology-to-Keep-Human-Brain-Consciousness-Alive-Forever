# S168 独立离线来源核验（P1/P3）

## 结论

**16个未映射槽都不是“原pool存在，但被17099 tier删掉”。冻结manifest中它们的pool_root_ids、tier_root_ids、local_indices均为空，status均unresolved。** 独立遍历冻结全库153962个root、原CSV158262行和v888原始注释188508行，在“motor身份＋正确腿/侧＋明确peripheral_target_type肌肉注释”规则下，16项均未发现可直接补入的来源。

需要分开两种情况：

* LF/RF pleural_promotor有**8个真实存在、而且已经全部在当前tier里的组合名称候选root**，不是数据集外、不是tier外。然而其原CSV Primary Cell Type为tergopleural_promotor，v888明确peripheral_target_type为tergopleural_promotor_muscle，只有组合cell_type/alternative alias提及pleural_promotor。它们确为motor/leg_motor_neuron，但这不等于已经证实同时支配独立pleural_promotor肌肉。可称“组合名称提出的双支配/共享池假说”，不可称“已验证肌肉来源”或据此直接补映射。
* 其余14槽，在对应腿/侧上连同类明确肌肉来源都未找到。其他腿有同名肌肉源不能移植，不能用普通motor身份或MN类型fallback代替肌肉支配证据。

“没有来源”严格指**所检查快照中未找到满足当前证据规则的来源**，不是证明真实果蝇没有该神经元/肌肉，也不是证明组合别名必然错误。

## 全16槽独立枚举

|槽|执行器|当前结论|
|---:|---|---|
|3|LFC_pleural_promotor|原pool空；4个组合别名候选全部已在tier，肌肉支配未证实|
|18|RFC_pleural_promotor|原pool空；4个组合别名候选全部已在tier，肌肉支配未证实|
|30|LMC_tergopleural_promotor_a|对应腿明确来源未找到|
|31|LMC_tergopleural_promotor_b|对应腿明确来源未找到|
|33|LMC_pleural_promotor|对应腿明确来源未找到|
|36|LMC_sternal_adductor|对应腿明确来源未找到|
|45|RMC_tergopleural_promotor_a|对应腿明确来源未找到|
|46|RMC_tergopleural_promotor_b|对应腿明确来源未找到|
|47|RMC_pleural_remotor_and_abductor|对应腿明确来源未找到|
|48|RMC_pleural_promotor|对应腿明确来源未找到|
|60|LHC_tergopleural_promotor_a|对应腿明确来源未找到|
|61|LHC_tergopleural_promotor_b|对应腿明确来源未找到|
|63|LHC_pleural_promotor|对应腿明确来源未找到|
|75|RHC_tergopleural_promotor_a|对应腿明确来源未找到|
|76|RHC_tergopleural_promotor_b|对应腿明确来源未找到|
|78|RHC_pleural_promotor|对应腿明确来源未找到|

## 8个组合别名候选的精确root/global/local

以下global为冻结153962全库的零基数组索引，local为S163当前17099 tier零基索引。所有root均与v888的root_888相同，未做跨版本猜测或浮点转换。

|腿|root十进制字符串|global|local|
|---|---|---:|---:|
|LF|720575941668683687|147338|16348|
|LF|720575941506261538|54253|6087|
|LF|720575941582911865|111113|12380|
|LF|720575941498960546|48794|5475|
|RF|720575941518204888|63658|7084|
|RF|720575941428654255|7589|857|
|RF|720575941556297811|91733|10243|
|RF|720575941592225831|115572|12923|

## 对历史提案的独立核对

`/run/media/sensen/Data2/cell_wound_prototype/run_leg_motor_coverage_audit.py` 第53—87行的“BOTH/可共享填2槽”来自compound cell_type命名推断，脚本自身明确proposal、未写manifest。原始peripheral_target_type并未提供第二个pleural_promotor target，因此其较强“innervating BOTH”措辞不可升级成已验证解剖事实。

`/run/media/sensen/Data2/cell_wound_prototype/tools_resolve_muscle_anatomy.py` 第35行拒绝REJECT_SHARED_MILLER28_30_POOL，属于保留共享池歧义的规则，不是已发现新root但过滤误丢的证据。

`/run/media/sensen/Data2/cell_wound_prototype/tools_build_design_runtime.py` 第146—156行实际将该池用于tergopleural a/b，并明确pleural留空；不是已存在2个pleural修复。不得将其注释“merges”当作独立来源再次循环引用。

## 方法、输入与验证

* 由`/home/sensen/Desktop/engine修复验收/s162-fast-sensory/20261024T224135142077Z/BOUND_INPUT_FREEZE.json`解析实际冻结manifest/数据路径并核验登记SHA256，不使用活动manifest替代。
* 独立从`/home/sensen/Desktop/engine修复验收/s163-sensory-readout-audit/20261024T224712953805523Z/GRAPH_90_MUSCLE_MAPPING.csv`派生16项；用`/home/sensen/Desktop/engine修复验收/s163-sensory-readout-audit/20261024T224712953805523Z/GRAPH_ALL_NEURONS.csv`逐行核验全部17099项root/global/local。冻结全root唯一性通过。
* 原CSV/feather的root保持十进制字符串，冻结int64只做精确int→str，绝不经float。候选保留motor身份证据与肌肉target证据的分列。
* 全输入SHA256、16项、8个alias候选及跨腿明确target全集均在`/run/media/sensen/Data2/cell_wound_prototype/S168_INDEPENDENT_SOURCE_CHECK.json`。读取前后SHA256相同。
* 独立小脚本`/run/media/sensen/Data2/cell_wound_prototype/s168_independent_source_check.py`使用现有venv/python -B实际exit0；报告内容二次assert验证exit0。一次前置schema读取用venv_body失败exit1（没有pyarrow），改用已经装有pyarrow的venv，无安装、无引擎。
* 新model load/forward/physics step/neural step/optimization全部0；未写活动映射、tracker、历史归档。

**最小安全动作：不自动补任何槽。** 若要推进LF/RF组合别名两槽，需新的、独立且明确的对应肌肉支配/分配依据；新证据前只能独立标注假说，不能把“真实root存在”改说成“目标肌肉支配已证实”。本审计不证明神经可达、承重、站立或行走。
