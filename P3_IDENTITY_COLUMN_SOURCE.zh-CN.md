# P3 身份列公开来源：版本列与工作根的代码区别

**P3/S3；手动只读研究实施，不是成品。没有改P1、旧pool、源代码或源资产，没有在线CAVE查询、凭据请求、安装或大资产下载。tracker由父代理更新。**

## 结论

本轮找到权威公开管线代码，证据比“列名看起来像版本”更强：`htem/bancpipeline` 在提交 `5c333c12f0b9e03873f88cf4e23cad34c0bb49c1`（提交日期2026-06-07T14:34:20Z）中区分 **SeaTable工作列 `root_id`** 与 **由supervoxel按物化版本解析的 `banc_<version>_id` / `root_<version>`**。代码的metrics注释明确称其使用“current root_id”。因此，该公开管线的`root_id`不是无条件等于v888的别名；公开字典中“root_id (= banc_888_id)”主要是发布接口/bancr口径，不能覆盖本地两列不同的实际字节。

**但没有找到本地feather与NPY下载/生成的不可变revision、日志或manifest。不能证明本地root_id是哪个日期的current、当前在线仍有效、由这个提交生成，或两对ID发生过特定split/merge。两根resolution继续unresolved，不授权替换。**

## 1. 精确权威代码与日期

[提交API](https://api.github.com/repos/htem/bancpipeline/commits/main)返回上述SHA和日期。实际生成代码位置为[固定提交 banc/share/banc-data.R](https://raw.githubusercontent.com/htem/bancpipeline/5c333c12f0b9e03873f88cf4e23cad34c0bb49c1/banc/share/banc-data.R)，不是字典所写的旧位置`banc/meta/banc-data.R`。

Section 1的有效代码（不是被注释掉的旧enrichment块）：

```r
version_id_col <- paste0("banc_", banc.version, "_id")
banc.meta.orig <- banctable_query()
banc.meta <- banc.meta.orig
banc.meta[[version_id_col]] <- banc_rootid(banc.meta$supervoxel_id,
                                            version = banc.version)
# mutate中：版本解析NA或"0"时回退root_id，否则保留解析值
# select同时保留version_id_col、root_id、root_626/850/888/890
arrow::write_feather(banc.meta, meta_file)
```

这说明`root_id`从SeaTable携带而来，并未一律重命名成v888；版本列另行计算，且存在fallback。`distinct(version_id_col,.keep_all=TRUE)`还会去重，不能仅靠列定义推定全表跨版本双射。Section 2原注释：`It's keyed on current root_id`，然后通过`root_id`连接metrics，最终删除metrics表中的工作根列。Section 1用`gsutil cp`上传到`gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_<version>/`。

[固定提交 banc/update/banc-ids.R](https://raw.githubusercontent.com/htem/bancpipeline/5c333c12f0b9e03873f88cf4e23cad34c0bb49c1/banc/update/banc-ids.R)先调用`bancr:::banctable_updateids()`，查询工作根及历史根；“Fix version”块明确分别执行`banc_rootid(supervoxel_id, version="626")`、`"850"`、`"888"`。v890独立回填源码[main/banc/update/banc-ids890.R](https://raw.githubusercontent.com/htem/bancpipeline/main/banc/update/banc-ids890.R)明确`target.version <- "890"`。此main文件是本轮读到的浮动源码，不冒充本地资产生成revision；v890准确UTC物化时刻仍未知。

官方项目[htem/BANC-project README](https://raw.githubusercontent.com/htem/BANC-project/main/README.md)将`htem/bancpipeline`列为产生compiled_data的管线。初始教程指向`flyconnectome/bancpipeline`，该旧位置提交API404；纠正到官方htem位置后200。旧`banc/meta/banc-data.R`路径提交查询返回空数组，并不表示整个官方仓库不可访问。

## 2. 官方字典与本地字节不一致

[官方GCS字典](https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/documentation/banc_888_meta.md)HTTP200：v888为2026-04-16 snapshot；`banc_888_id`是v888主键，`root_626`预印本物化、`root_850`中间物化、`root_888`发表物化；supervoxel用于按时间解析。ID列为string。字典**不列root_890，也未给626/850精确UTC时间**。

|项|字典声明的发布feather|本地冻结feather（本轮重新核哈希）|
|---|---|---|
|行×列|188162×79|188508×81|
|bytes|51450978|57503026|
|MD5|8c8babff28b21c57ecc999e664560ef5|8c2b93a608c7163ec9d94d68e0756ff4|
|SHA256|字典未提供|86ccf5df0c67419f8c5f43e93a7ed38d23a080e9f7fde26737290252f3780098|

字典指向的准确资产URL是 https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/banc_888_meta.feather 。**该URL只定位，未下载；上述远端大小/MD5是字典声明，不是本轮读取远端资产响应验证。** 本地与该manifest不匹配；不能断言坏文件、下载时漂移或某次事件造成差异，也不能把发布79列字典直接当作本地81列合同。

本地 `/run/media/sensen/Data2/cell_wound_prototype/data/banc888/b888_to_root.npy` 为3016256 bytes，SHA256 `9eee49a4974edd3a8feb690c07931a23af38a0771f5c4bd4135c11a0b537a7de`，形状188508×2（既有审计）。本轮未找到权威NPY URL或生成代码。

## 3. 本地来源管线核查范围与缺口

读取路线图、tracker、来源报告第7节更正和版本协调协议；检查工作树，保留已有修改。项目内按`.py/.sh/.md/.json/.log/.txt`有界检索feather/NPY、GCS及repo指针；资产目录没有manifest，`git ls-files data/banc888`及资产`git log`无记录。此结论限于已查项目文本/当前git，不声称查遍用户所有shell历史或外部磁盘。

实际本地消费者：
- `/run/media/sensen/Data2/cell_wound_prototype/run_neural_rebuild_11_mn_pools.py`119—124读取feather的root_id；153写pool_root_ids。
- `/run/media/sensen/Data2/cell_wound_prototype/tools_build_design_runtime.py`64—70读取NPY为banc888→root_id映射；注释称“same materialisation”，**这是本地作者假设，不是权威生成/版本证据**，不用于消除不一致。
- 本地研究笔记指向官方字典；未记录资产不可变下载URL/revision。公开管线可解释列设计，但并未给这份本地文件签发生成证明。

## 4. 保留最新纠正后的crosswalk事实

既有最终审计 `/run/media/sensen/Data2/cell_wound_prototype/outputs/p3_motor_version_crosswalk/20261014_004106_179567/result.json`：208唯一声明根，74池，20根至少一版本不同，188全相同；18仅626不同，2为已知工作根/版本根关联。

|声明root_id|同行banc888/root626/850/888/890|冻结P1 tier|
|---|---|---|
|720575941434056663|720575941458992031|**PRESENT，0起索引1849**|
|720575941500775137|720575941622252796|**PRESENT，0起索引10179**|

两版本根不在原声明208根/运动探针202根。旧int对string tier查询是假阴性，不能再说它们缺tier。版本列只读投影为204/67complete/7partial；实际P1仍202/65complete/9partial，未更改。上述计数继承最终冻结审计，本轮没有重跑全crosswalk、连接扫描或模拟。

## 5. 最小具体数据请求（不要求凭据）

1. 提供SHA256为`86ccf5...80098`这份57503026-byte/188508×81 feather的不可变下载URL（含GCS generation或Dataverse file/version）、导出UTC、实际脚本提交和`banc.version`，以及SeaTable查询快照日期与bancr依赖revision。
2. 提供NPY `9eee49...7a7de`的生成代码revision、输入featherSHA、列顺序与缺值/碰撞策略；不能由文件名推断来源。
3. 对两对精确ID提供supervoxel/nucleus和版本化root关联证据、解析timestamp/物化UTC、1:1/1:n/n:1基数及操作历史（如适用），另给原CodexCSV导出version/date manifest。

当前已有权威列设计证据，尚缺**本地资产实例证明**和**逐根lineage**。下一步是上述小manifest/行级证据，不是下载巨型突触文件、不是真实实验、更不是擅自替换旧pool。

## 6. 执行与验证边界

公开搜索3轮，每轮3个查询（<=4轮、每轮1—4查询）；仅按找到的官方repo/字典指针定向读取小文本与API，未批量资产下载。官方代码位置纠正后结束搜索，没有无期限搜寻。哈希/stat/git检查bash退出0，工作树检查退出0；HTTP404与空结果如上透明保留。未新产图；本轮实际调用read_image查看冻结 `/run/media/sensen/Data2/cell_wound_prototype/outputs/p3_motor_version_crosswalk/20261014_004106_179567/projection.png`：标题明确非权威模型、两关联unresolved，original202与版本204/626186标签一致；不将历史图称本轮新动态。交付JSON解析、208=188+20、tier索引1849/10179与两资产冻结SHA/MD5/bytes独立断言全部PASS，bash退出0。机器证据：`/home/sensen/Desktop/engine修复验收/P3-identity-column-source.json`。
