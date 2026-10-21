# Lacin 2019：取得原文后的实验递质证据

**手工研究原型，不是成品。文献证据审查，未修改网络。**

## 已取得的原文

Lacin H et al. (2019), Neurotransmitter identity is acquired in a lineage-restricted manner in the Drosophila CNS. eLife 8:e43701. DOI 10.7554/eLife.43701。

[PMC全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6504232/)，由[Europe PMC DOI查询](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.7554/eLife.43701&format=json)确认PMCID。

已读正文Glutamatergic lineages、GABAergic lineages、Figure3及Figure4 supplement1说明。不是仅根据摘要或数据库分类。

## 实验证据

正文使用msh/VGlut split-GAL4、VGlut-LexA、VGlut抗体及ChAT/GABA抗体等互补手段。报告14A、21A全群表达VGlut蛋白，未检出对应群的ChAT/GABA免疫标记，结论为两群谷氨酸能。14A证据见Figure3 D–F及Figure4 supplement1 D；21A证据见Figure4 supplement1 G。正文明确指出ChAT reporter可在谷氨酸能/GABA能细胞中出现，不能仅凭reporter判乙酰胆碱能。

Figure3图像已下载并实际read_image查看：D的14A绿色标记与VGlut紫色重叠；E/F的GABA/ChAT标记及分通道图可读。此视觉检查核验图件，不是自行重新判定或量化实验。

PMC Figure3标题显示“14B neurons are glutamatergic”，与正文、图内14A标记和图注内容不一致。本报告以正文/图内标记及图注分项为据，明确记录标题不一致，不静默改写原文。

## 本地根的明确hemilineage字段

本轮从v888以banc_888_id唯一连接并读取hemilineage，不按IN编号猜：

| 根 | 本地cell_type | 本地hemilineage | 预测递质 | 证据关系 |
|---|---|---|---|---|
|720575941466160982|IN14A092|14A|glutamate|与14A实验群证据一致，非本根直接测定|
|720575941542325253|IN14A086|14A|glutamate|同上|
|720575941575683161|IN14A096|14A|glutamate|同上|
|720575941686043254|IN14A038|14A|glutamate|同上|
|720575941533989976|IN21A022|21A|acetylcholine|与21A实验群结论存在冲突，待核对映射/版本|
|720575941503524914|AN05B104|05B|acetylcholine|正文将5B列GABAergic，但本类型的primary/secondary适用范围未解决|

所有六根本地verified仍为空，不覆盖字段。论文主要研究postembryonic hemilineages，早期胚胎神经元可呈不同身份，因此AN05B104的primary身份不能直接套5B群规则。必须核实生命周期/类型映射后再决定证据资格。

[VFB IN21A022](https://www.virtualflybrain.org/blog/2022/01/01/in21a022-fbbt_20010639/)正文仍写预测ACh，且分类含cholinergic/glutamatergic，不能选择性引用分类消除冲突。外部VFB基于不同版本类型体系，仍需检查626/888及MANC对应。

## 谷氨酸不等于抑制

原文明确说谷氨酸在果蝇CNS可作为兴奋或抑制递质，依神经元类型而定；还指出运动神经元以谷氨酸激发肌肉。当前项目统一glutamate→抑制是模型受体映射假设，不能仅凭本论文的谷氨酸能证据验证或补符号。即使14A群递质证据可靠，仍缺具体靶细胞的受体/作用依据。

## 本轮界限与下一步

已经从“仅有EM预测”推进到部分根具有**实验群级支持**及明确冲突，但没有取得这些BANC根逐根直接递质实验。不能宣称34根全部已有verified证据，也不能任意补入预测边。

下一步优先核查21A类型映射与预测冲突、AN05B104 primary群范围，并追查14A→具体运动目标的受体/作用证据。暂不改严格基线、递质符号、增益或连接。

原图：`/run/media/sensen/Data2/cell_wound_prototype/sources/literature/lacin2019/figure3.jpg`

报告：`/run/media/sensen/Data2/cell_wound_prototype/LACIN2019_EXPERIMENTAL_EVIDENCE.zh-CN.md`

桌面报告：`/home/sensen/Desktop/cell_wound_prototype/lacin2019/LACIN2019_EXPERIMENTAL_EVIDENCE.zh-CN.md`

桌面原图：`/home/sensen/Desktop/cell_wound_prototype/lacin2019/figure3.jpg`
