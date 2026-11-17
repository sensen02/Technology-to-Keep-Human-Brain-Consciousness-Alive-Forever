# Eon 地面下行交集运行检查表

- 对应 P1/S1–S5。实验根在 /run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261117T064507Z-eon-overlap。父项目为 045901Z 冻结脑、051249Z BANC 映射、062447Z 身体扩训。旧项目产物未重写。
- 跑前 PROTOCOL.zh-CN.md 固定神经刺激和行为门。使用归档 Python 3.12.14、Torch 2.5.1 CPU、MuJoCo 3.9、FlyGym 2.1.0、EGL 和单线程 BLAS。滤波、六根具名轴突对应、最新图、初始状态和体部权重见校准及 IDENTITY.json。前馈两段方便对同一 Eon 发放配对做切断。神经和物理每 .1 ms，VNC 靶点每 .5 ms 更新读出，动作每 1 ms 更新。正式关节无教师干预，无根态写入、无附着，留出种子 362 带 .12 体重侧推 20 ms。无 PPO 或 GAE。
- 首次 run_brain.sh 因将旧 Eon 登记与后加腹部上行后的整图文件哈希直接比较而退出 123，子进程在积分前退出 1，全部失败日志保留在 BRAIN_BINDING_MISMATCH。更正时核对 Eon 记录的归档目录哈希及 25 个实际下行条目的逐字段一致性。新 run_brain.sh 退出 0，21 条脑轨迹均完成。原 Eon LIF 和候选环形缓冲对相同实际激发事件的 100 步全部内部状态相同，58 步有神经发放。稀疏泊松生成使用记录的键及种子。本对照使用相同事件，完整向量随机抽样种子序列未做比较。
- bash run_probe.sh、overlap.py calibrate、六条开发物理、freeze_inputs.py、bash run_physics.sh FORMAL、bash run_physics.sh REPLAY、audit_overlap.py、audit_source.py 均退出 0。三个独立脑重放及两条 PFL3 物理诊断子进程均退出 0。归档 Python 源、模型、真实 SWC 和身体资产列在 IDENTITY.json。正式前 177 输入文件 SHA 冻结，最后及七条校准文件重新哈希一致。独立 23 对，从脑发放到动作的各自存档阵列逐位相同。两档前进速度有横偏，完整低滑移 D1–D8 未评。
- render_overlap.py、FFmpeg 全解码、ffprobe 均退出 0。原回合状态四栏视频共 85 帧，播放 2.833 秒、0.60 倍速。实际查看八格平面轨迹与电影的 20 个关键状态，覆盖开始站稳、活跃转向、单切与停止。视频只读原模拟状态。23 个正式和复跑共 782,000 原生步，另有 6 开发及 2 PFL3 共 136,000 步。全脑 21 条加 3 条独立复跑，共 408,000 步不含基准等价对照。
- 当前结论为跨标本同类型结构绑定下的平地走弧和双速度通过。下一最小动作为用真实接触起因的上行输入接到 Eon，并同步记录感知输入、下行选择和切断后的行为。
