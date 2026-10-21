# IN21A022逐根映射后续核查

**手工研究原型，不是成品。未修改模型或数据标签。**

## 本地精确连接

用banc_888_id唯一连接720575941533989976，实际读取v888字段：

- root_626/root_850/root_888/root_890均记录720575941533989976。
- cell_type/manc_cell_type/malecns_cell_type均为IN21A022；hemilineage=21A。
- root_region=MANC_vnc_LNp_T1_L。
- manc_match=15914；manc_nblast_match=34059。

这加强了本地626/888根编号连续性的证据，但不证明任意旧外部注释都有效。match与nblast_match不是同一种交叉表；未获取质量指标与定义前不能任选候选当同源确认。没有本地birthtime字段证明该根早生。

## 外部页面的命名陷阱

[VFB MANC:14857页面](https://www.virtualflybrain.org/blog/2022/01/01/in21a022_t3_r-manc14857-vfb_jrcv0bgp/)标题为IN21A022_T3_R，旧MANC描述systematic type也是IN21A022，但当前Classification是adult IN21A023 neuron（FBbt_20010640）。页面的glutamate关系不能直接归给本地IN21A022根；14857也不等于本地15914或34059。

[VFB MaleCNS:802621](https://www.virtualflybrain.org/blog/2022/01/01/in21a022_l-malecns802621-vfb_jrmc2nv1/)当前分类IN21A022，记录acetylcholine secretion关系，位置为T3左侧。该关系来自数据库注释，未取得实验出处，不能把ontology capable-of表述升级为独立测定；它也不是本次T1根已确认的对应实例。

## 出生时段检索状态

搜索IN21A022+birthtime、MANC15914和具体编号未取得可靠元数据行。现有MANC论文文本没有IN21A022字符串，图中类型内容需要另取图/表；上一轮“21A早生群预测ACh”仍只是群级解释线索。

[官方malevnc工具README](https://raw.githubusercontent.com/natverse/malevnc/master/README.md)说明查询NeuPrint需要认证token。没有启动需要用户凭据的查询，也没有绕过认证。当前仅此接口路径受认证要求约束，不代表其他公开表不可获取。

## 结论

当前证据不足以确认本根的早生/次生身份，也不足以解决预测ACh与21A实验群Glut的作用范围差异。已明确排除一种误读：不能拿标题IN21A022、实际分类IN21A023的14857页面给本根赋Glut。

下一步应寻找公开MANC原始metadata/typing补充表中bodyId15914及34059的type、birthtime和版本，或者读取论文类型图表。严格基线保持不变。

报告：`/run/media/sensen/Data2/cell_wound_prototype/IN21A022_MAPPING_FOLLOWUP.zh-CN.md`

桌面副本：`/home/sensen/Desktop/cell_wound_prototype/IN21A022_MAPPING_FOLLOWUP.zh-CN.md`

本轮没有新图/模拟，仅来源与字段核查。
