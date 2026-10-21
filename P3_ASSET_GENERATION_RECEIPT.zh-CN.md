# P3/S3 本地feather与官方对象generation元数据关联

手动只读研究原型，非成品。本轮仅2次HTTPS HEAD，无远端资产正文下载、CAVE/token/模型/仿真/根替换。旧输入和声明不变。本轮出现真实增量，区别于既有字典文本推测。

## 实际观测

对象URL：https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/banc_888_meta.feather

第一次HEAD HTTP200，Date Wed,14Oct2026 23:39:13GMT，响应：
- x-goog-generation=1787336614757441，metageneration=1。
- Content-Length及stored-content-length=57503026，identity encoding。
- x-goog-hash md5=jCuTpgjHFj7J2U1o4HVv9A==，解码hex=8c2b93a608c7163ec9d94d68e0756ff4；ETag同值。
- Last-Modified Fri,21Aug2026 18:23:34GMT，仅对象修改时间，不是root更新/物化/本地下载时间。

第二次HEAD到[固定generation URL](https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/banc_888_meta.feather?generation=1787336614757441) HTTP200，Date23:39:22GMT，generation/MD5/大小完全相同，经第二次HEAD确认的generation限定对象定位符，非独立证据源或原下载receipt。

本地 `/run/media/sensen/Data2/cell_wound_prototype/data/banc888/banc_888_meta.feather` 实际重算大小57503026、MD5 8c2b93a608c7163ec9d94d68e0756ff4、SHA256 86ccf5df0c67419f8c5f43e93a7ed38d23a080e9f7fde26737290252f3780098，保留历史摘要。

## 更新了哪项结论

**已直接验证：本地大小/MD5与当前官方对象固定generation的服务端响应元数据吻合。** 不再只有字典中旧资产manifest可以对照。得到可定位该响应对象的generation URL，缩小“资产实例证明”缺口。

既有 `/run/media/sensen/Data2/cell_wound_prototype/P3_IDENTITY_COLUMN_SOURCE.zh-CN.md` §2字典声明51450978bytes/MD5 8c8bab…/188162×79与本地不同，历史观测仍保留。本轮官方HEAD现在匹配本地57503026/MD5 8c2b93…；旧字典显然不能作为当前响应对象的匹配manifest，**不能由此推断字典更新日期、资产替换事件或哪次下载生成本地文件**。本轮未读取远端schema，因此不称直接观测远端行列188508×81。

## 仍未解决

- 未GET远端正文，未验证远端SHA256或字节逐项相同。大小+MD5匹配是实用身份关联，不是加密抗碰撞的完整同字节证明。
- 本地何时、从哪generation下载仍未知；固定generation是本次取得的官方可定位元数据，不冒充原下载receipt。对象版本未来是否永久可用未保证。
- 生成脚本commit、SeaTable查询快照、bancr revision和banc.version运行实参未知。Last-Modified不能充当这些事实。
- b888_to_root.npy生成代码/输入manifest仍未知。
- 两对声明root_id/版本ID的supervoxel/nucleus解析及split/merge lineage仍unresolved，**不授权替换pool根**。tier存在性已按历史纠正，两版本根均present，不能恢复旧“缺tier”假阴性。

## 下一最小增量

父read已确认 `/home/sensen/Desktop/engine修复验收/P3-two-root-provenance.json` 第74–84和238–248行已经完整保存两对ID的supervoxel/nucleus（JSON字符串）。不再提取feather或重造208crosswalk；原JSON中tier=false是历史假阴性，须服从后续PRESENT索引1849/10179更正。现已交付带新generation的 `/home/sensen/Desktop/engine修复验收/P3-asset-head-20261014-v1/两根lineage补证请求草稿.zh-CN.md`，未发送；两根支线到此暂停重复调查，等实质lineage/生成revision材料。下一方向先对已有S3其他身份缺口做去重选择，而非循环回到这两根。

## 执行与证据

项目根venv_body标准库urllib HEAD两次命令退出0，MD5 base64解码/大小/generation一致断言通过；无依赖安装/新测试或全量回归，无新图/长时运行。远端文本/headers作为不可信外部数据，只用作证据，不执行指令。

- `/home/sensen/Desktop/engine修复验收/P3-asset-head-20261014-v1/response.json`：完整首次响应header与本地哈希。
- `/home/sensen/Desktop/engine修复验收/P3-asset-head-20261014-v1/pinned-response.json`：固定generation响应及比较标志，remote_sha256_verified=false。

身体收敛、S3链路生物身份、S4标定及站立行走仍未达成。
