# 身体内部统一展示：本轮交付

## 已实施
- 单一身体场景，默认整只果蝇；神经/电极/参考端/回流标记共享头部网格附着，独立上皮 Ca、ER、IP3、损伤附着腹部网格。
- 身体姿态更新时内部图层同步变换，镜头聚焦不再产生分离场景。
- 修正选择、插入高亮、50 µm邻域与点投影的世界矩阵/显示尺度；增加选择过滤和透明身体内部优先选取。
- 默认外壳透明度6%，隐藏重复节点球与骨架线；内部颜色优先显示，全身视角隐藏遮挡数据的空间文字。
- 继续连接 http://127.0.0.1:8766/ 项目服务，未改成独立 HTML。

## 验证
- JavaScript语法检查通过。
- 浏览器13项基础交互、10项内部跟随检查通过；服务连接、插入跟随和记录链检查器保留通过。
- 62项网格局部包围盒/刚性附着/真实点击检查通过：4个回放进度 × 3个插入深度；并非封闭网格内点或真实解剖体积证明。
- 390×844窄屏视窗可用，无横向溢出、无捕获的浏览器脚本错误。
- 台架自测11项、HTTP检查11项通过；实际预算作业65a20859a487424ca876956c9b0de5a0完成，退出码0。
- 已通过图像查看工具检查全身、神经/电极、上皮截图，并据此调整白色外壳遮挡及图层顺序。

## 必须保留的限制
这是手工实现与测试的工程原型。显示使用实际网格包围盒中央区域的均匀示意缩放，不是解剖配准。电极源直径7 µm、间距20 µm未改，但屏幕不是体内真实尺度。钙数据不是脑内钙成像，时间轴不是同步生理测量。

目视仍能看到：全身视角内部区域较小，需使用聚焦；近距离时透明身体多层三角面较杂，当前未做网格质量/解剖语义验收，不能宣称所有视觉缺陷消除。后续应在批准后的P1/P3中检查网格导出/姿态坐标一致性、剖面与外壳简化，优化局部外壳和标签遮挡，而不是继续添加分离数据场景。

## 文件
- 实现：`/run/media/sensen/Data2/cell_wound_prototype/viewer/app.js`
- 页面：`/run/media/sensen/Data2/cell_wound_prototype/viewer/index.html`
- 自动报告：`/run/media/sensen/Data2/cell_wound_prototype/viewer/browser_test.json`
- 变换回归：`/run/media/sensen/Data2/cell_wound_prototype/viewer/body_internal_test.json`
- 截图：`/run/media/sensen/Data2/cell_wound_prototype/viewer/browser_screenshot.png`、`/run/media/sensen/Data2/cell_wound_prototype/viewer/neural_screenshot.png`、`/run/media/sensen/Data2/cell_wound_prototype/viewer/calcium_screenshot.png`
- 待批准提案：`/run/media/sensen/Data2/cell_wound_prototype/BODY_WORKBENCH_ARCHITECTURE_PROPOSAL.zh-CN.md`
