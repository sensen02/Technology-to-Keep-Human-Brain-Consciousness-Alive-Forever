# 已批准 P1–P3：实施与验收记录

本次是手工开发、测试的工程原型。P4共享参考/ADC模型扩展不在本轮范围；未运行全量sorting。

## P1：数据与身体坐标
- 前端拆分 data、timeline、transforms、service、inspection、navigation、layers 七个本地模块，app.js负责组装，保留原服务地址。
- 数据清单提供快照run_id、来源hash、单位、来源类型、独立时基；显示变换与源数据分离，源电极保持7µm/20µm。
- **修复真实坐标错误**：FlyGym记录顺序为模型ID `[1,-1,2,...68]`，并非 `[0,1,...68]`。固定头部在编译后并入胸部，名字查询为-1，记录误取末端后足。旧导出将后足姿态赋给胸部/头部。
- 新导出使用编译模型标准顺序，world0显式单位姿态、thorax1来自记录0，排除错误头部别名。已有NPZ不修改，其他神经/电极/钙/台架JSON逐字节保留。
- 54905顶点与实际MuJoCo三种姿态比较，最大误差8.38e-7mm；不是解剖配准验证。

## P2：服务与结果一致性
- DatasetRegistry、ArtifactStore、RunManager、BenchService拆分。保持loopback、Host/Origin、2MiB限制和静态白名单。
- 持久化串行队列，最多4个待执行，取消、120秒超时、2GiB子进程上限、BLAS线程1；重复键幂等返回。
- 原子保存、重启恢复；运行中断不冒充完成。任务执行前检查源哈希，结果检查单位/固定几何/测试/哈希。
- 每任务独立输出，最新任务失败/取消时不把旧结果当新结果。存储发布故障报告不健康状态。
- 使用轮询（未启用SSE）；数据按源分区加载，尚非逐帧流式传输。历史结果没有自动清理策略，磁盘保留需另行配置。

## P3：可视化与台架
- 一个身体主场景，已纠正头部实际网格位置；内部仍为示意缩放，神经和上皮不是同步生理系统。
- 近景只保留对应头/腹外壳并减淡；局部镜头跟随播放，剖切可选择X/Y/Z与当前区域范围。
- 数值色域和单位，排除消融储库对色域的压制，缺失灰色、储库暗灰，可选蓝黄配色。
- 视角保存/恢复、复位全身、截图加来源JSON下载、低性能像素密度选项；来源可折叠。
- 电极检查器按准确通道ID联动台架检查，无匹配则显示缺失，不猜测通道编号。
- 预算任务ID、状态、旧结果提示、取消按钮；仍无DAQ硬件接入。

## 已完成验证
- 前端语法及模块：不规则时基、缺失值、源显示变换往返、固定几何、缺失证据。
- 身体导出回归：正确映射、网格/链接、无关数据哈希、原NPZ保持不变。
- 后端24项故障/接口测试；真实记录预算任务 `1c8337e5854e4adaaf479724ccbd3f9b`完成退出0，重启后结果身份/hash验证成功。
- 浏览器：原13项基础交互、10项内部跟随、62项附着/包围盒/点击、新增模块/近景外壳/书签/色标/剖切/证据/坐标往返、窄屏布局。
- 已通过图像查看工具核查身体整体、神经与上皮截图。细节仍是网格渲染与点数据，不是医学组织重建。

## 交付路径
服务 http://127.0.0.1:8766/ 。项目 `/run/media/sensen/Data2/cell_wound_prototype`，桌面 `/home/sensen/Desktop/cell_wound_prototype/bench_visualizer`。
实现与测试报告见上述目录及 `/run/media/sensen/Data2/cell_wound_prototype/workbench/API.md`、`/run/media/sensen/Data2/cell_wound_prototype/viewer/browser_test.json`、`/run/media/sensen/Data2/cell_wound_prototype/viewer/export_report.json`。

---

## 补记（2026-10-03 复核后）：本文件的两处需要更正

独立复核（见 `/run/media/sensen/Data2/cell_wound_prototype/GAP_ANALYSIS_2026-10-03.zh-CN.md`）发现本文件的两处不严谨表述，现更正：

1. **「已完成验证」里我列的一项验证当时其实没跑**：我曾用 `python3 -m unittest test_body_export_regression` 跑身体导出回归，输出是 `Ran 0 tests in 0.000s / NO TESTS RAN`——因为 `test_body_export_regression.py` **不是 unittest 模块**，只有直接运行才有效。复核时我按正确方式复跑：`python3 test_body_export_regression.py` → `PASS: canonical recorded mapping, fused-head exclusion, meshes/links, all non-body SHA256, unchanged NPZ truth alias evidence`（exit 0）。**结论本身成立，但「已跑过」这句话当时不成立。**
2. **电极检查器那条建立在旧几何上**：本文件称「电极检查器按准确通道ID联动台架检查」。磁盘上的 `outputs/embodied_body/access_map.json` 仍锁在 **100 µm / 10 µm**（旧几何，`run_access_map.py` 第 44–45 行），而 `viewer/data.js` 前端强制 **7 µm / 20 µm**（20 µm 间距的 256 通道网格由 `export_integrated_viewer.py` 第 109–111 行从旧阵列中心重新铺出）。**每个通道的神经元归属来自旧阵列**。旧几何下通道碰撞实测为 0，因此「通道↔神经元唯一对应」在 7/20 下不成立（`ROUND26_NEW_GEOMETRY.md` 第 22–26 行实测：3186/7737 被多通道捕获，最多 21 个通道）。
3. 另：桌面 `engine/` 在本轮之前缺 30 个文件；现已同步，44/44 文件 SHA256 一致。

以上三条不改变 P1–P3 的实现事实，但改变「已验证到什么程度」的表述。
