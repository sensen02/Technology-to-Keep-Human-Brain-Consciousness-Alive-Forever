# 复验入口

Blender 4.5.16、归档 Python 3.12 与 MuJoCo 3.9。全流程在源目录运行 bash run.sh /全新输出目录。程序先从原小箱派生仿砂地板和悬空动作，再从 P1 归档回放相同运动驱动，拍摄底视和八机位画面，最后独立计分。setup.sh 中记录本机环境和原始资产的绝对路径。render_checks.py 单独用 Blender 全身网格渲染三个指定时刻。空间泛化台架用 spatial.py，在完成标定后运行。原终端和当前目录的正式结果保留。

桌面交付下的“光学复算包”可只用 NumPy、SciPy 复算。将 PYTHON 指向相应解释器，执行 bash replay_observer.sh /全新输出目录。该包不含模拟器、运动命令、接触标签、关节状态与求解器载荷。EXPECTED.sha256 对照五个输出。

足迹中心使用地板的局部毫米坐标。rig_origin.json 用米记录坐标原点在观察箱标定坐标中的位置。时间从同一次物理运行的开始计，帧 1 为 2 ms。walking_sensor.blend 与 surface.pc2、fiducials.pc2 同目录。small_chamber_sensor.blend 用于原来的 450–520 帧悬空轨迹。
