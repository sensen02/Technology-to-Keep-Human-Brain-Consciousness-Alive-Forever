# 双眼视觉外部重建复验

桌面交付的“外部复算包”包含原八机画面、相机投影矩阵、头部标记与眼位标定、解算脚本。运行 `bash replay_pose.sh /一个尚不存在的输出目录`。可将 `PYTHON` 设为带 NumPy、SciPy 和 Pillow 的 Python 解释器。脚本按 POSE.sha256 核验并保存 estimated.json。只用同步 RGB 和静态标定即可重算头位姿与左右眼位姿。

源实验目录中 `bash replay_full.sh /一个尚不存在的输出目录` 会使用归档 Blender 4.5.16 和本机归档 Python 3.12 复跑。从上一轮 `walking_sensor.blend` 派生标记场景，拍外部 RGB、解算、按解算相机渲染左右眼、用原 FlyGym Retina 生成阵列，最后单独用 Blender 参考位姿绘制评分图。render_eyes.py 默认只打开 estimated.json。EYE_REFERENCE=1 是另启的评分采集。

打开 reconstructed_eyes.blend，切换相机 Optical reconstruction L 或 R。surface.pc2 和 fiducials.pc2 要与 Blender 文件同目录。Blender 时间轴为 500 fps，前 315 帧属于原承重行走。CSV 的 16 行行走样本才是实解帧，其余帧来自 Blender 动画插值。316–319 帧是固定身体的独立头部姿态对照。
