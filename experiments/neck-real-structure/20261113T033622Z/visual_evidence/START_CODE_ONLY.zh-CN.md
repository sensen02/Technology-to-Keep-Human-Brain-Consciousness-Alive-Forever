# 实际视觉—脑—颈部端口—身体可视化：父串行执行说明

当前为ONLYCODE。子代理实际33项标准库检查全true、exit0，未导入数值库／读取回合数组／加载模型／构造Core／渲染／训练／物理。所有执行源码自检查后冻结。本文纯说明不进入科学source清单。

必须等父115的22阶段(10完整物理＋10新进程实际RGB神经重放审核＋汇总＋独立固定帧因果review)全部exit0，才可运行。不是等待功能全PASS：原视觉功能／滑移FAIL必须诚实显示，完整性／模型／来源／真实像素身份不能放宽。

实际准入数据：

`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T033622Z/visual_body/NEURAL_VISUAL_BODY_SUMMARY.json`

实际kind：`ACTUAL_RGB_SOURCE_NECK_VISUAL_BODY_SUMMARY_V1`。

`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T033622Z/visual_body/PARENT_READY_REVIEW.json`

权重身份：新脑2123eeadaffbdcf722e361ad1da26b1816833279918e9231773839ed1cfb924c，冻结视觉996980806ac107137d6d5f4743de207fb3350e84a8a145dc5f52b8807398bb87，真实端口9e88a04dd91aa6f9c1c1c68e0562325205e0f36f5cdd9ae41d1dd46c8171dc23，冻结motor789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd。

先父唯一CPU1绘图：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T033622Z/visual_evidence/plot_sync.py --execute-after-parent-release
```

之后父managed串行4片视频；以下示例先空场，其余依次将condition改为V_left_to_right、V_left_DN_cut、V_left_occluded。不要并行或覆盖已生成VIDEO目录，若实际报错先保存并定位，不重跑物理：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T033622Z/visual_evidence/render.py --condition V_empty --execute-after-parent-release
```

每片输出目录：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T033622Z/visual_evidence/VIDEO_<condition>/`

内含ACTUAL_RGB_BRAIN_NECK_BODY.mp4、VIDEO.json、FFPROBE.json、FRAME_MAPPING.json和DECODED_MONTAGE.png。每片770帧／50fps／15.4秒／1280×720，含稳定、教师预热及自主段。原完整身体qpos／qvel仅重建，禁止积分；逐帧校验COM／head／yaw、71opaque完整体轮廓、编译参数和源前后身份；全770解码后生成阶段蒙太奇，父必须实际打开查看并核桌面副本SHA，不能只看执行日志。

主体显示当前世界的真实红柱／两far闭合遮挡盒；均来自保存的实际CAPTURES、target_centers和cover_rotation，不在主体画伪造眼图。forward2和Z1.2为模型原生长度，SI未建立。右栏为已发布的实际左右眼图、actor/raw captureID与采集／发布／神经消费时间；首20msinvalid显示NO RELEASED RGB，不访问−1索引。当前场景与延迟actor图像分别展示。

右下显示冻结视觉rawturn→brain16→DNheld→神经解码turn、ANheld、端口capture／release／consume和切断标记。304递归状态(256motor＋32视觉＋16脑)，CNNcache16和AN1／DN2另记。所有显示值来自同次保存数据，不运行神经网络，不依刺激标签生成行为。

图表目录SYNCHRONIZED_EVIDENCE包含10条件RGB／端口／身体／帧ID同步图及上下行切断、正常／冻结图像的实际配对差异图。低层步态属于OTHER，原滑移及功能失败保留；这些可视化不是完整颈部电生理、生物眼、一般导航、电极或意识证明。
