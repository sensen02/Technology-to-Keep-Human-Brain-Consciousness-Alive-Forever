# P3/S3 LR_TYPE_CONFLICT：类型不一致不等于侧别错误

手动只读研究原型，非成品。无R源码执行/在线SeaTable或CAVE查询、模型/仿真、池/side/root替换。此轮解释官方规则并检查两条本地记录是否足以复现，非标注修复。

## 官方固定代码事实

[状态更新脚本](https://raw.githubusercontent.com/htem/bancpipeline/5c333c12f0b9e03873f88cf4e23cad34c0bb49c1/banc/update/banc-update-status.R)本轮HTTP200读取：

```
.match_ct = cell_type[match(banc_match, root_id)]
cell_type != .match_ct -> append_status(..., "LR_TYPE_CONFLICT")
cell_type == .match_ct -> subtract_status(..., "LR_TYPE_CONFLICT")
```

banc_match为空/NA或任一cell_type为NA时保留旧status，不自动去除；因此已有标记也可能保留，不能仅见标签就声称当前匹配仍冲突。

另一块SIDE_CONFLICT比较side与mirror match侧别：相同侧添加、不同侧删除。两标记语义不能互换。side更新另按root_position，只补空side并保留已有值，不根据LR_TYPE_CONFLICT自动交换左右。

[类型更新脚本](https://raw.githubusercontent.com/htem/bancpipeline/5c333c12f0b9e03873f88cf4e23cad34c0bb49c1/banc/update/banc-update-celltypes.R)HTTP200：新跨数据集match驱动的类型重新推导块在HAS_MANUAL_ANNOTATION或_MATCH_PREFERRED下保留原cell_type。该范围不代表脚本每个较早赋值步骤都全局免变，更不证明本地资产由此commit生成。状态脚本写有ALT_MATCH退役迁移，但本地仍保留ALT标签，不能据此推定运行时间/版本。

## 本地两条记录

输入 `/run/media/sensen/Data2/cell_wound_prototype/data/banc888/banc_888_meta.feather` SHA86ccf5df0c67419f8c5f43e93a7ed38d23a080e9f7fde26737290252f3780098。本轮读取schema后仅选实际存在字段：**无banc_match、无cell_type_source列**。

|工作root|side/目标|cell_type|manc_match/manc_cell_type|
|---|---|---|---|
|720575941644474312|middle/right sternal_adductor|sternal_adductor_MesoLN_dorsal_5|12196 / MNml29|
|720575941553503460|middle/left pleural_remotor_and_abductor|pleural_remotor_and_abductor_MesoLN_dorsal_5|19773 / MNml29|

两条均有HAS_MANUAL_ANNOTATION和LR_TYPE_CONFLICT。相同manc_cell_type只说明本地来源列关联，不证明两条BANC同类型或同肌肉，不能用它把side/target互换。fanc列保留原值，auto:IN21A001不升级为验证类型。

records.json中mirror_in_work_root_lookup=null是**缺banc_match列导致无法查询**，不是原始banc_match=null或mirror不存在的证据；此字段是派生不可用结果，不当观测缺失。官方规则已找到，但本地不能复现两标记比较对象或标注操作历史。

## 决策与精确补证

维持中腿左sternal_adductor、中腿右pleural_remotor两个池unresolved；不恢复输出、不再重复词表/目标扫描。停止本侧别支线，所需新证据具体为：本地feather精确SHA/generation对应的两行banc_match和所指记录cell_type/side/target、比较时刻/root版本；manual annotation来源/时间与对MNml29关联的解剖说明；若有目标更正须版本化行级证据。未向维护者发送请求、未申请凭据，不扩大检索为猜测。

下一有界动作：先建立现有科学缺口的去重任务清单，分别标记“需要外部来源”“可用本地新增验证”“已解释停止”；避免在S4metric/两根lineage/两侧目标之间循环文档。只有能明确新增观测或修复验收条件的项才实施，无新证据不靠调参制造运动。

## 实际核验

web_search两精确查询结果噪声未用作定义，随后官方固定tree/两个update源码HTTP200定向读取；export脚本只作上下文。tree工具spill正文被截断100000字符，初JSON解析退出1（Invalid control character），改为对已返回文本提取完整path字符串退出0，仅用于候选路径发现，不宣称完整树扫描或truncated=false。本地pyarrow11实际字段读取/JSON归档退出0，无新测试/图或全量回归；不是完整源码归档，URL/规则及响应状态保留。

桌面 `/home/sensen/Desktop/engine修复验收/P3-lr-status-20261014-v1/records.json` 与同目录中文报告/verification.json。身体收敛、生物链路、S4标定、稳定站立行走仍未达成。
