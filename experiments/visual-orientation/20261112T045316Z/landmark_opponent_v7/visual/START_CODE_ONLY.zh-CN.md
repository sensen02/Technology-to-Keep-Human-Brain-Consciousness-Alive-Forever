# 新opponent身体视觉交付：父执行说明

本目录仅CODE_ONLY工具；未导入Torch/NumPy/MuJoCo/Matplotlib/physics/runtime/model，未读实际回合数组、forward、Core、渲染或绘图。实际只运行标准库check_codeonly.py，exit0/23项全部通过，记录CODEONLY_CHECK.json。执行源已停止修改，不改正在运行的body/policy/camera输入。

## 实际绑定

ROOT固定 `/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_opponent_v7/body/`，VIS固定原视觉根，视觉SHA996980806ac107137d6d5f4743de207fb3350e84a8a145dc5f52b8807398bb87，motor789f。summary实际kind VISUAL_OPPONENT_FAR_BODY_V7，读取NORMAL6_SUMMARY.json与VISUAL_BODY_SUMMARY.json；normal6/control2独立列。

只在所有8actualresult+独立VERIFY+完整summary存在且全部完整性checks通过后运行。模型离线72必需配置和8固定画面描述、savedfinal1000与独立验证必须严格绑定；新identity必须VISION_FIXED_NEURAL_OPPONENT_ARCH_V7及固定受体参数声明，旧79db模型会拒绝。功能门False可诚实显示，完整性失败不可绕。

工具最小复用原visual_body_v2/visual中直接读取/绘图/视频函数，通过完整新globals重绑定，只有summarykind与遮挡装饰必要AST变化。所有继承源SHA、自己源码、实际数据/权重/sourceproof在运行前后核验。

## 父实际运行顺序

先等唯一bodyheavy完成并收集，确认无其他heavy：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_opponent_v7/visual/plot_sync.py --execute-after-parent-release
```

再串行渲染三代表条件，下面只示例左pulse，分别替换为V_right_pulse及V_left_occluded：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_opponent_v7/visual/render.py --condition V_left_pulse --execute-after-parent-release
```

输出visual/SYNCHRONIZED_EVIDENCE的8条件同步图、actualreleased-vs-actorused双眼控制图，以及VIDEO_V_left_pulse等新独占目录下ACTUAL_VISUAL_BODY_WITH_EYES.mp4、DECODED_MONTAGE.png、FRAME_MAPPING/VIDEO/FFPROBE证据。

## 视频真实性与装饰

完整1s稳定+4.8s教师+9.6s ANN，770帧50fps15.4s实时1280×720。保存qpos/qvel恢复到私有可视化Core，mj_forward仅重建几何，mj_step/step1/step2全部禁止；full71meshopaque/身体全帧不裁切、COM/yaw/head核验、source和compiled参数前后不变、所有770帧完整解码。仅内存framebuffer改变并恢复，0模型forward/0物理积分。

遮挡装饰为实际固定的左右两个闭合罩：从保存geometry首个source_anchor_valid行target_centers_xyz(LEFT/RIGHT/CENTER)读取左右中心，Z设为body相同的1.4；用actualCAPTURES.cover_rotation，不依据之后的headpose重算或跟随身体。BOX halfsize[.18,.18,1.4]，仅MjvScene decor/segid−1，无碰撞/施力。纯身体segmentation先于装饰，避免以前segid错误。旧近距离wall不再使用。

右侧眼图来自保存actualRGB，分别显示当前已发布和actor实际使用帧，绝不由目标坐标、opponent值或标签重画。20ms capture/release与source事件分开：普通off3.6，首off发布3.62，恢复4.12—4.62；固定帧4.8锁捕获4.78，rawcamera继续。caption声明工程红选择而非生物复眼/近导航，原slipFAIL照实保留。

父必须实际打开8条件同步图、控制眼图、三片解码蒙太奇，并核VIDEO与桌面副本SHA后才能交付。不因为ffmpeg成功或离线allPASS宣称实际身体功能通过。当前body结果待父真实流程，未知结果不造PASS。
