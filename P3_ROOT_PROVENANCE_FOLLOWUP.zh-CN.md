# P3 两个缺快照根：声明身份来源后续（2026-10-14 自动续轮2）

**手动只读研究/测试原型，不是成品。P3/S3；没有修改模型、旧数据、连接、历史包、服务或参数。两根 resolution 均为 unresolved，不称动物/生物学缺失。本报告不授权根替换。tracker由父代理更新。**

## 1. 核心结论

精确本地来源已找到：声明生成器使用 `root_id`，但同一feather行的版本列使用另一个ID。两个声明根不是凭空出现，也不是本轮live文件漂移造成：原声明池哈希与P1冻结输入匹配，P1 `motor_manifest.rows[].source_annotation` 与原声明记录逐字段相等。

| 原声明根 | 同一行 banc_888_id / root_626 / root_850 / root_888 / root_890（均相同） | 对应声明执行器 | 本地feather行（0起） |
|---|---|---|---:|
|720575941434056663|720575941458992031|LHF_trochanter_extensor|12743|
|720575941500775137|720575941622252796|LHF_sterno-tergo-trochanter_extensor_b|7247|

这是**同一元数据行关联**，不是两个root字符串相等的断言，也不是自动crosswalk修复。两原声明根在本地原注释CSV、supplemental均0条，在缓存根全集缺失；同行版本ID分别在两CSV恰1条、缓存均存在，**但P1的12000根tier均不含它们**。即使将来证明跨版本等价，仍要独立审计tier选择，不能认为一换ID就会补齐当前输出。

额外精确扫描现有声明池：720575941458992031与720575941622252796均未出现在任何measured.resolved.pool_root_ids，未发现这两同行版本ID已占用另一现有池；这不等于全池crosswalk无冲突，不授权替换。

六缺根全核：另外四根720575941391252886、720575941497037218、720575941511883985、720575941559306355各1行，`root_id` 与上述五个版本列全一致。旧“所有root_id直接是v888”的概括不能从少量相等样本外推到这两根。

## 2. 精确声明链

1. 源资产 `/run/media/sensen/Data2/cell_wound_prototype/data/banc888/banc_888_meta.feather`，188508×81。两个声明根各唯一1行，`flow=efferent`、`super_class=motor`、`side=left`、`body_part_effector=hind_leg`。分别 `peripheral_target_type=trochanter_extensor_muscle` / `tergotrochanter_extensor_muscle`，cell_type分别trochanter_extensor / tergotrochanter_extensor_MetaAN。
2. `/run/media/sensen/Data2/cell_wound_prototype/run_neural_rebuild_11_mn_pools.py`：49指定源；54—76声明名称/腿/侧别alias；119—131按flow及三字段建池；**124读取 `d['root_id']`，153将该列原值写成pool_root_ids**。并未读取banc_888_id用于根输出。这个局部列选择足以解释为何上述精确字符串进入声明，但不足以解释root历史。
3. `/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_rebuild/neural_rebuild_11_mn_pools.json`：1181—1193与1210—1223明确记录两池；文件provenance记录生成器、规范XML身份和当时解释器，**没有根更新操作日志/完整源下载revision**。
4. `/run/media/sensen/Data2/cell_wound_prototype/run_neural_muscle_closure.py`：31/310读取该池；`/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/motor_output.py`：13—50携带原pool并只做tier交集，缺根显式报告，不做替换。
5. `/run/media/sensen/Data2/cell_wound_prototype/outputs/experiment_platform/20261013_233237_7be93a1d/result.json`内嵌motor_manifest；两记录source_annotation与上述原声明精确一致。此目录不存在单独motor_manifest.json，本轮纠正了初始路径假设。
6. `/run/media/sensen/Data2/cell_wound_prototype/data/banc888/b888_to_root.npy`（188508×2）恰各1条 `[同行版本ID, 原声明root_id]`，与feather同行吻合。这是额外本地关联资产，**没有足以独立证明列语义/历史的生成revision**。

## 3. 原CSV、官方本地副本与标注者

原Codex快照 `/run/media/sensen/Data2/cell_wound_prototype/data/flywire/banc_neurons.csv.gz` 共158262条；Root ID=720575941458992031在文本行24564，Root ID=720575941622252796在133841；两者motor、左后腿、对应神经/肌肉名一致，Verified NT均空。原声明ID各0条。下载器 `/run/media/sensen/Data2/cell_wound_prototype/flywire_download.py` 34—49表明由Codex `dataset=banc` 产品接口得到；**这个本地CSV没有可确认的物化版本/下载响应manifest，所以其精确版本和日期保留未知**，不以文件mtime代替数据快照日期。

本地已取得官方副本 `/run/media/sensen/Data2/cell_wound_prototype/data/banc_dataverse/supplemental_data_2.txt` 共155927条×28列；上述同行版本ID分别行12683、7215，各恰1条，肌肉目标、腿、侧别与feather同行一致；原声明ID各0条。历史下载来源记doi:10.7910/DVN/7WTH1N release 3，此轮没有重新下载或网络验证release文件身份。它与本地feather不是逐字节同版本表（188508×81），不以匹配两行推断整体同版本。

标注者资产：`/run/media/sensen/Data2/cell_wound_prototype/data/banc_dataverse/cell_info.parquet` 与 `/run/media/sensen/Data2/cell_wound_prototype/data/banc_dataverse/codex_annotations.parquet`。实际读schema嵌入PANDAS_ATTRS，两表均 **datastack_name=brain_and_nerve_cord、materialization_version=888**；timestamp为null。codex关联代表点表last_updated=2026-04-16T20:24:22.862821，这是表元数据，不能冒充两根变化时刻。历史 `/run/media/sensen/Data2/cell_wound_prototype/AUTONOMOUS_LOOP_ROUND24.zh-CN.md`记v888快照2026-04-16；该历史声明不是本轮实时版本查询。

按四个精确ID过滤标注表，只有同行版本ID返回标注，原声明ID0条。cell_info对720575941458992031有user_id188的trochanter_extensor、user_id17的Sternotrochanter MN；对720575941622252796有user_id1081的tergotrochanter_extensor_MetaAN、user_id103的Tergotr. MN，以及188的左侧/T3/运动标签。完整id、supervoxel、UTC created、valid与字段在机器JSON。codex_annotations的target、腿、侧别和类型与上述同行相符；分类系统user_id列是标签，非根lineage操作者证据。**标签创建时间不等于root split/merge时间；user_id不擅自对应真人姓名；近似/跨数据集类型名不当作肌肉逐根实证。**

## 4. 可以与不可以断言的事

- 已证实：本地声明的root_id列与同表版本列发生身份口径不一致；该不一致已随原声明进入冻结P1，而非本轮修改造成的live-vs-history漂移。
- 已证实：同一行的版本ID在原注释与官方副本有唯一记录；在缓存存在但不在当前tier。本轮没有替换根、扩大tier或运行动态。
- **未证实**：原声明root_id对应哪个时间点，是否过期root、split/merge/重新解析，还是源表生成流程改变列语义。缺操作历史与版本化列字典，不能只凭“root_626/850/888/890均相同”判定两个字符串之间的事件。
- **未证实**：这两精确ID是否存在跨个体混入；MANC/MALECNS/FANC匹配字段只是跨数据集类型/实例关联，不能据此把BANC声明根等同MANC根，亦不能证明原root_id是另一个个体。
- 仍为unresolved：生物身份等价、逐根NT/受体、募集、稳定行为。缓存/原连接出入0只描述该快照，**绝非生物学不存在**。

两条有界公开搜索为 `"BANC" "720575941434056663"`、`"BANC" "720575941500775137"`；仅返回无关金融页面，没有精确ID证据，不引用无关结果，也没有继续一般文献搜索或批量fetch。无在线CAVE查询，无token请求。本地同行版本证据不冒充新的公共版本化crosswalk。

## 5. 独立验证与哈希

工具 `/home/sensen/Desktop/engine修复验收/P3-verify-two-root-provenance.py`只用标准库/NumPy/Arrow，不导入项目仿真模块。成功命令：

```bash
PYTHONPATH=/run/media/sensen/Data2/cell_wound_prototype/vendor/pylibs /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python /home/sensen/Desktop/engine修复验收/P3-verify-two-root-provenance.py
```

最终退出0；六条unique/版本列检查、两原声明与P1逐字段相等、两同行版本ID唯一（banc_888_id各1行、root_id列无该值）、原CSV及supplemental各1条、NPY关联各1条、缓存/tier归属断言通过；P3可比输入哈希与当前字节一致、P1相关可比池与缓存哈希一致。没有全量回归、没有新模拟，没有重扫3990039连接行；连接缺失继承 `/run/media/sensen/Data2/cell_wound_prototype/outputs/p3_path_eligibility/20261014_001205/result.json` 中完整历史扫描，并核原连接哈希未变。

关键SHA-256：

| 完整路径 | SHA-256 |
|---|---|
|/run/media/sensen/Data2/cell_wound_prototype/data/banc888/banc_888_meta.feather|86ccf5df0c67419f8c5f43e93a7ed38d23a080e9f7fde26737290252f3780098|
|/run/media/sensen/Data2/cell_wound_prototype/data/banc888/b888_to_root.npy|9eee49a4974edd3a8feb690c07931a23af38a0771f5c4bd4135c11a0b537a7de|
|/run/media/sensen/Data2/cell_wound_prototype/data/flywire/banc_neurons.csv.gz|40a2201554a8c34d2c4b07c8322543a07c1b3faf5acafbac363fd1a3d0fa617f|
|/run/media/sensen/Data2/cell_wound_prototype/data/flywire/banc_connections_princeton.csv.gz|8772298eb69455759c7b4a23ee1d1843f5b389965e312524e652415f8a09b312|
|/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_rebuild/neural_rebuild_11_mn_pools.json|3244c8c1470ad3009dd3e12fc64b11f97d1672b8bda811f9ceacf0221e1dd94c|
|/run/media/sensen/Data2/cell_wound_prototype/data/banc_dataverse/supplemental_data_2.txt|3f9c78aa5d87ad893e3d4d7f37c58883196f067507c95e29af1a0cc126fed279|

全部14个证据输入完整路径/字节数/SHA-256及原行在 `/home/sensen/Desktop/engine修复验收/P3-two-root-provenance.json`。JSON最终SHA-256：`3b963505f62d2beafb6345970c40096a04ec19a1d8aad2ffbbb5c29e0119531c`。

实际调用read_image打开历史 `/run/media/sensen/Data2/cell_wound_prototype/outputs/p3_path_eligibility/20261014_001205/reachability.png`，限制标题/逐腿计数与既有报告一致；不是新图或新动态成果。

失败项透明记录：一次ls猜测vendor/BANC*不存在退出2；venv_body本身无pyarrow、旧/tmp/pq不存在，各退出1；改用项目已有vendor/pylibs读取成功，未安装依赖。一次哈希循环假设P1目录有独立motor_manifest.json退出1，随后定位内嵌result字段成功。上述失败未写任何源/资产；最终独立审计退出0。

## 6. 最小数据请求与可执行下一步

**先请求两条精确ID关联的版本证据，而不是先要求生物实验：**

1. 源feather不可变URL/revision/checksum，以及 `root_id` 与 `banc_888_id` 的精确列定义、生成时间、脚本revision、datastack/materialization ID/UTC时间；同时索取原Codex两CSV导出版本/date manifest。
2. 对(720575941434056663,720575941458992031)和(720575941500775137,720575941622252796)，提供由supervoxel/nucleus或CAVE版本化root lineage支持的关联，含split/merge操作时间及1:1/1:n/n:1基数。不能靠类型同名或同行关联发明等价。
3. 当前立即可做：只读审计所有声明池的身份列唯一性/冲突、相应tier资格与版本差；另建新报告，不改旧pool或P1。通过版本证据后才讨论另行授权的派生mapping，保留旧哈希且不替换本轮两根。

父代理下一轮协议：`/run/media/sensen/Data2/cell_wound_prototype/P3_VERSION_RECONCILIATION_PROTOCOL.zh-CN.md`。本轮仅交付来源追溯，控制链、生物有效性与稳定站立/行走均未达成。

## 7. 自动续轮3追加纠正：tier类型假阴性（保留上文历史结论）

全208根/74池精确字符串核对发现：冻结P1 `tier_root_ids` 为JSON字符串列表。前轮独立脚本第53行使用 `int(c) not in p1['tier_root_ids']`，第54行同样以int检查字符串列表；这是类型不一致导致的假阴性，不是P1或资产变更。两个同行版本ID720575941458992031、720575941622252796 **均在当前冻结12000根tier**，但不在当前声明208根及运动探针202根。上文第14、35—42等处“对应版本ID不在tier”相关结论以此纠正；不改旧JSON/脚本/历史哈希。

只读版本列投影（非权威修订模型）：banc_888_id/root_850/root_888/root_890列只改变这两根，全池唯一数仍208，固定tier交集204；假设池complete67/partial7，对照当前P1仍65/9、202运动探针。root_626另外18根不同（总20），投影唯一208、缓存190、tier186、假设complete53/partial21。全208声明根各唯一feather行，无缺/多行、各版本ID全表唯一，无不同root_id版本碰撞、无新增ID重叠当前202；74池侧/腿/肌肉目标一致。所有关联仍不证明split/merge，两个已知身份等价仍unresolved，未发布pool、未换根、未改变P1。

新原始证据位于 `/run/media/sensen/Data2/cell_wound_prototype/outputs/p3_motor_version_crosswalk/` 新时间目录；桌面 `/home/sensen/Desktop/engine修复验收/P3-motor-version-crosswalk-` 对应目录。精确最终路径、运行/独立验核及实际看图由本轮父tracker记录。下一动作仍是获取列生成revision、CSV export manifest及版本化supervoxel/nucleus lineage，不需因该假阴性盲目扩tier。
