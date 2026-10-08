# 身体中心三维项目工作台

使用 http://127.0.0.1:8766/ 。启动命令：`python3 /run/media/sensen/Data2/cell_wound_prototype/serve_workbench.py --port 8766`。桌面入口：`/home/sensen/Desktop/cell_wound_prototype/bench_visualizer/启动工作台.sh`。直接打开 HTML 只会提示进入服务，不再提供分离的离线版本。

这是手工实现的工程原型，不是经实验验证的器件软件。

## 身体内统一查看
- 一个身体主场景；神经与电极放入头部内部，上皮伤口模型放入腹部内部。聚焦切换仅移动镜头。
- 内部图层采用随身体姿态变化的附着变换。**这是统一缩放的示意放置，不是真实解剖配准或真实组织覆盖。**
- 电极源几何保持直径 7 µm、间距 20 µm；神经/电极共同显示变换不改变源空间的距离计算。不能从屏幕像素推算植入尺寸。
- 透明外壳、剖切、图层开关、选取对象、插入深度、50 µm 几何邻域均为检查工具；邻域不是电场截断。
- 拖动旋转、滚轮缩放、Shift 拖动平移。归一化时间轴同时回放独立时基，不代表生理同步或因果耦合。

## 数据和服务边界
- FlyGym/MuJoCo 网格与身体姿态来自模型回放，不是活体扫描。
- BANC 胞体坐标有真实数据来源；活动是独立 ConductanceNetwork 演示，无连接组，也不是实测神经记录。
- 腹部 Ca、ER、IP3 和损伤来自独立上皮伤口模型，**不是脑内钙成像**。消融储库不是存活细胞。
- 服务端可运行真实的记录链预算程序，并评估上传的台架配置 JSON + 原始 CSV；预算不驱动场景物理，未连接硬件 DAQ。
- 外部台架报告仅只读显示，不作为已验证测量。台架协议：`/run/media/sensen/Data2/cell_wound_prototype/bench/PROTOCOL.zh-CN.md`。

## 验证与后续提案
自动交互报告及截图位于 `/run/media/sensen/Data2/cell_wound_prototype/viewer/browser_test.json` 和 `/run/media/sensen/Data2/cell_wound_prototype/viewer/browser_screenshot.png`。自动检查不能替代目视或生物/硬件验证。

P1–P3已批准并实施；实现与限制：`/run/media/sensen/Data2/cell_wound_prototype/WORKBENCH_P123_STATUS.zh-CN.md`。P4未实施。新增视角保存/恢复、复位、快照来源导出、数值色域、蓝黄配色、X/Y/Z剖切、局部外壳和低性能模式；任务可取消且结果具有持久run_id。身体导出修复了FlyGym融合头部记录行混淆，详细证据：`/run/media/sensen/Data2/cell_wound_prototype/viewer/export_report.json`。

BANC：Distributed control circuits across a brain-and-cord connectome，Harvard Dataverse doi:10.7910/DVN/7WTH1N，somas_v1.parquet，CC BY 4.0。Three.js r129，MIT，许可证 `/run/media/sensen/Data2/cell_wound_prototype/viewer/vendor/THREE-LICENSE.txt`。
