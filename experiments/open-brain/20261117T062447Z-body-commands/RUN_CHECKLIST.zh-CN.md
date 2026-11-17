# 身体下行扩训运行记录

- P1 / S2–S5 开源脑项目。时间戳 20261117T062447Z。父证据为 20261117T051249Z-bidirectional 和 20261117T055904Z-body-imitation。初态、物理、人为侧推、接收签名及门见 PREREG.zh-CN.md、PHASE/PREREG.zh-CN.md、expansion.py。PHASE/IDENTITY.json 是候选正式运行开始后收集的源与依赖哈希，独立执行后再核验；无法补成开跑前哈希。实际渲染加载同次首课 BODY.mjb，求解用原类构建并加载同姿态快照。
- 环境是归档 Python 3.12.14、Torch 2.5.1 CPU、MuJoCo 3.9.0、FlyGym 2.1.0。Bash drive.sh 设置 BLAS、MKL、OMP 单线程及 EGL。物理、颈部均 .1 ms，命令更新 1 ms，VNC 每五步采样，即 2 kHz。无 PPO 或 GAE。首次语法探查用不存在的 python，退出 127，改用 python3 py_compile 退出 0。两个 cgroup 文件探查在本机未找到，退出 1，仿真与训练无影响。
- bash teacher.sh 退出 0，32 子进程均 0，23/24 教师训练、7/8 教师开发过物理门。./drive.sh train_commands.py BC、batch_commands.py dev evaluate DEV25、DEV45、batch_commands.py correct correct CORRECT、train_commands.py BC_CORRECT、batch_commands.py dev evaluate DEV_CORRECT 均退出 0。三个动作闭环分别 3/8、3/8、5/8，失败回合、保存点及恢复状态保留。
- ./drive.sh PHASE/phase.py、PHASE/phase_rollout.py gait dev、evaluate dev、gait old、evaluate old 均退出 0。两变体均 7/8 新开发、5/5 原门。选择记录 PHASE/SELECTION.txt 落盘先于新留出。./drive.sh PHASE/phase_rollout.py evaluate held 与 evaluate old 各独立新进程组复跑，状态文件都为 0。PHASE/verify_phase.py 退出 0，独立 22 组全匹配。原权重 SHA 位于首课归档，新权重 SHA aee03ac22e167fe1c2269fd694042f115261b6312deb0f7397f87b6f7922a73a。
- ./drive.sh expansion.py teacher 在四个留出转向条件做事后同身诊断，退出 0。原地条件的四份失败均保留，未用于模型选择。新旧验收均使用相同完整 1.7 秒回合。独立验证合计 748,000 原生步。正式、旧和复跑均由存盘模型驱动。无落地轴向接触；四个原地位移超门，没有隐去。完整滑移、附着和 D1–D8 未评。
- PHASE/evidence.py 和首课 render_evidence.py 退出 0。左走弧同次视频 FFmpeg 全帧解码和 ffprobe 退出 0，85 帧 30 fps，播放 2.833 秒。已实际打开 POSTURES.png、COMMANDS.png、VIDEO_ARC/REVIEW.png，检查站姿、起步、运动、撤令后姿态及各方向。
- 下一最小动作是另立低平移转身教师，在同体上同时约束胸部轨迹、足端支撑和 yaw，重新对新相位和扰动验收。两侧相同指令的快走、同向调速和停步可独立复用。DNa01 同功能但本轮未实际投放，飞行与脑侧闭环训练留在独立课。
