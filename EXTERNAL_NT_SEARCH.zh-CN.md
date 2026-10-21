# 外部递质证据初步检索

**手工研究原型，不是成品。文献/数据库检索，不是网络参数修复。**

## 已取正文核查的结果

1. [VFB IN14A092](https://www.virtualflybrain.org/blog/2022/01/01/in14a092-fbbt_20010044/) 正文明确写predicted neurotransmitter is glutamate，来源Eckstein et al. 2024；属于14A hemilineage。页面列BANC_626根720575941466160982，也列720575941542325253、720575941575683161。但后两根在本地v888分别叫IN14A086、IN14A096，说明类型划分/版本不可不加审计地移植。根字符串一致也不能替代626→888版本交叉表。
2. [IN14A086](https://www.virtualflybrain.org/blog/2022/01/01/in14a086-fbbt_20010038/)、[IN14A096](https://www.virtualflybrain.org/blog/2022/01/01/in14a096-fbbt_20010048/)、[IN14A038](https://www.virtualflybrain.org/blog/2022/01/01/in14a038-fbbt_20009992/) 正文均写预测glutamate。IN14A038页面列本次根720575941686043254，但没有逐根实验递质测定说明。页面的glutamatergic ontology分类不能升级正文的prediction证据。
3. [IN21A022](https://www.virtualflybrain.org/blog/2022/01/01/in21a022-fbbt_20010639/) 正文预测acetylcholine，列BANC_626根720575941533989976，与本次同数字根对应；页面同时列cholinergic和glutamatergic分类，不能只取标题分类判符号。
4. [AN05B104](https://www.virtualflybrain.org/blog/2022/01/01/an05b104-fbbt_20007774/) 正文预测acetylcholine。页面显示的BANC_626实例不是本次根720575941503524914，因此目前只是类型级线索，不是本次根的独立verified。
5. [BANC官方预测代码README](https://raw.githubusercontent.com/htem/synister_banc/main/README.md)与[Zenodo发布](https://zenodo.org/records/20350570)明确该工具是EM突触图像递质分类器，提供8种递质预测，只对size>5突触预测；公开预测parquet路径gs://leelab_fly_cns/files/banc_nt_prediction_w_sizethresh_5_09072025.parquet，另有ground-truth train/test split。ground truth训练标签存在不意味着本次34根全部有独立实验验证。

## 仍需获取原文的实验证据线索

检索发现Lacin et al. 2019《Neurotransmitter identity is acquired in a lineage-restricted manner in the Drosophila CNS》，DOI 10.7554/eLife.43701：[期刊入口](https://elifesciences.org/articles/43701)、[PubMed](https://pubmed.ncbi.nlm.nih.gov/30912745/)。本次期刊返回Client Challenge，PubMed正文也未成功取得。不能声称已经核查其中14A或21A的实验图表，更不能由标题直接给本次根赋递质。

hairplate/CoHP8、SNpp52、SNpp45、SNpp44定向检索本轮没有取得逐根实验递质证据。Pratt论文入口https://www.nature.com/articles/s41467-026-69333-z发生认证重定向；搜索摘要不是可用于升级verified的证据。IN26X001仅发现类型实例线索，尚未完成其递质正文核查。

## 结论与下一步

当前新增的是外部预测来源、部分跨库身份线索，以及一个可能提供lineage实验支持的原始论文入口。**尚未取得足以把34根逐根升级为verified的证据，未改任何连接或参数。**

优先获取Lacin原文/补充表，核对hemilineage实验方法及适用范围；核对626/888及MANC类型映射；审查BANC预测模型ground truth是否覆盖这些类型、训练测试分离和置信度语义。实验类型级支持、EM预测和逐根verified必须分开保存。

本轮无新图或模拟，不重复制作计数图冒充研究进展。报告桌面副本：`/home/sensen/Desktop/cell_wound_prototype/EXTERNAL_NT_SEARCH.zh-CN.md`。
