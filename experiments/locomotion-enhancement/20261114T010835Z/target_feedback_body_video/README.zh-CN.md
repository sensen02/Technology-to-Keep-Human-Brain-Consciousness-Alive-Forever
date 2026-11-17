# 目标反馈修复：保存身体状态视频工具（仅代码准备）

本目录为独立渲染scope。当前不运行Torch、NumPy、MuJoCo、Core、模型、物理、训练或渲染，也不读取PT/NPZ数组内容。实际final600、模型SHA、身体结果和视频验证均未知，不生成假预览或假PASS。目标是四个source3纯运动ANN组件常量[speed,yaw]诊断，明确不是人工脑—真实颈部部署或成品新脑。

实际身体源目录虽然名为v3，实际CAL为target_feedback_order_repair_v4。原161918 runner/common/audit/Trace结构已只读核对，新工具不导入身体源码、不继承深层AST命名空间。运行工具是独立普通Python函数，json/hashlib等globals由本模块直接拥有；准备检查递归检查全部LOAD_GLOBAL。

父待actual final600、三训练主阶段、最终保存审核及身体全部九阶段完成后，才执行下面唯一入口（四case内部串行，不与训练/物理并行）：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_body_video/render.py --execute-release
```

不带--execute-release只做实际元数据资格检查，仍不导入任何重模块；缺输入会拒绝资格，不会编造结果。READY是代码冻结状态，不表示未来数据通过。默认四case必须都有RESULT/VERIFY和恰九JOBS stages complete exit0，且同一实际600模型SHA；不要求四个行为PASS，早停失败照实展示。

重建从actualIDENTITY绑定的派生XML与资产表编译一个MjSpec，不用Core，不调用神经网络。按原load_full_model设置71mesh及引用材料alpha1，按原Core复现摩擦1.5，然后每个canonical compiled数组shape/dtype/SHA必须与actual完全一致；任何不一致停止渲染，绝不重跑物理。42有限PD元数据及保存实际ctrl限幅核对。mj_step/step1/step2全禁，只有saved qpos/qvel/time/ctrl恢复和mj_forward运动学。重建contactforces不作为承重或指标来源。

每20ms显示该区间末端保存状态，包含最后一个实际qpos；全15.4秒对应770帧50fps。早停只展示actual_ms对应窗口，末尾不足20ms会只缩短最后样本时长，MP4媒体码流不改，逐帧PTS/末样本时长/电影总长由ffprobe明证，再完整解码。1s稳定、4.8s ALGORITHM TEACHER WARMUP、5.8s以后PURE MOTOR ANN COMPONENT均以实际存在阶段为准。初始settle没有保存几何yaw，文字和曲线标N/A；真实足载和非足记录直接读保存trace。不得从视频重新裁定通过。

相机每case固定，用全程COM路径及各保存采样帧mesh球界确定视距，帧帧分割检查全身轮廓在上下文字带之外；不调整真实高度、姿态、根轨迹或转角。960×640 framebuffer仅内存视觉变更，前/中/后声明并恢复；canonical数组前后核验。图片只保留几张解码帧生成montage，不积攒全部帧。

输出目录：/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_body_video/OUTPUT，四case子目录各有ACTUAL_SAVED_MOTOR_COMPONENT.mp4、DECODED_MONTAGE.png、SAVED_DIAGNOSTICS.png、FRAME_MAPPING.json、FFPROBE.json、VIDEO.json。同步图显示同次保存yaw、COM漂移、六足载、actualctrl/target/action RMS及保存周期数；native峰值及原门保存在VIDEO.json，不用重建力取代。

父必须实际read_image打开四张蒙太奇和四张同步图，检查图像/全部视频解码和hash后再执行：

```bash
/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_body_video/deliver.py --parent-reviewed-all4-montages-and4-graphs
```

这才拷贝四个视频、四montage、四同步图到 /home/sensen/Desktop/engine修复验收/运动增强/20261114T010835Z/目标反馈修复-纯运动组件身体验收/，核对桌面SHA并生成中文README和OUTPUT/DELIVERY.json。不会覆盖已有同名文件，不触碰789f/f25b旧对照或其他历史视频。

第一份READY写成后代码不再修改。如果实际资产、schema、视频时轴或取景检查失败，保留失败产物，另开新render scope修复，不重物理、不覆盖封存源。本目录不改tracker，父负责最终记录。
