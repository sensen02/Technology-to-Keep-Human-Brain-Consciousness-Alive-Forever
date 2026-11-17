# 视觉实施首项：当前完整身体的真实双眼图像已验证

父托管bash72已实际执行EGL相机probe并收集exit0，随后独立核验源hash/原始RGB数组，实际打开双眼panel。当前结果是视觉输入光学与时序成立，不是视觉行为训练或身体视觉定向通过。

## 实际实现与观察

使用当前已归档完整71mesh、49qpos/48qvel/42有限PD身体的私有可视化Core，从三个已保存真实中性回合姿态恢复。没有重新跑物理，没有用相机测试改写实际身体轨迹。相机通过MjvScene GL视野设置，无XML修改或模型重编译；71mesh全部不透明，真实自遮挡保留。

每眼32×32 RGB，左右两个头部相对工程视点，光轴分别偏左右25°、下俯15°、垂直视场100°。RGB实际由MuJoCo渲染得到，不从目标坐标手算像素。高对比红柱是工程视觉地标，遮挡物也是仅渲染场景几何，不参与碰撞/施力。这不是复眼解剖、生物光谱或真实食物标定。

- 左目标和空场原始RGB平均绝对差1.951171875；右目标差1.5423177083333333。
- 左目标完全遮挡后，与同一遮挡物但无目标的RGB逐位相同，平均差0。
- 实际打开图像可见左右目标像素位置不同，遮挡后目标消失；不同保存头部位姿产生不同视图。
- 光轴±20°反事实扰动改变图像；这项只是光学坐标诊断，不是实际身体或头部动作。
- 原始RGB共9组、shape(9,2,32,32,3)、uint8；物理积分0、尝试积分0、训练更新0、神经模型forward0。物理编译参数在前后验证保持不变。

## 因果时序

相机每20ms采集，延迟20ms发布，行为与运动每5ms查询并保持已发布帧。以实际渲染帧验证采集/发布/使用顺序，禁止未来图像提前使用。通用冷启动前20ms显式invalid/zero；部署预热将维护真实空场帧队列，使5.8s交接可使用已经发布的真实帧。图像/特征缓存与288神经记忆分别记录，不能混淆。

## 证据与交付

源码与预登记：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/camera.py`
`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/probe.py`
`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/PREREG_CAMERA_PROBE.json`

实际结果：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/CAMERA_PROBE/PREFLIGHT.json`
`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/CAMERA_PROBE/RAW_RGB.npz`
`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/CAMERA_PROBE_REVIEW.json`

已实际打开并桌面copy2/哈希核验的眼视图：
`/home/sensen/Desktop/engine修复验收/视觉实施-20261112/首批真实双眼图像-左右遮挡与头部位姿.png`
SHA256 e6297a73437aa4b7922c1d996496441c2bcdc38e9905bd6ffd5364f1a82e3641。

下一步生成真实RGB静态头部帧库，在新目录预登记视觉only CNN+GRU长序列候选。部署仅RGB，不送目标坐标、距离、标签或时钟，不用手写颜色质心产出动作。现有motor789f与嗅觉66f423保持冻结；视觉气味融合、神经减速停驻与身体视觉对照尚未完成。当前无临时无用碎片，所有已有气味失败与证据保持。
