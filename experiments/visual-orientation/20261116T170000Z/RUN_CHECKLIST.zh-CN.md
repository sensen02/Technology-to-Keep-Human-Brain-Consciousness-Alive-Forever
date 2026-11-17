# 平地视觉走停的下行校准（P1/P2、S2/S5）

人工搭建大脑项目。首个脑任务固定平地冻结身体，速度只有单个已验收正值和零，偏航为零。视觉目标出现、持续和撤除分别用于起步、续走、撤令；DNg97 调速和 DNg13 转向另在独立目录校准。实际仿真初态、驱动记录、截止窗口及身体门按各运行身份保存。

- `forward_source.py` 读取 BANC v888 正式元数据、有效颈前截面点、已校核出版 v2 边表及来源定义的两截面。核验同一骨架连续路径、SWC 下载 generation 与 MD5，VNC 目标 exact root ID 联接。左侧官方完整 SWC 和右侧官方 L2 SWC 均过门，右侧完整 SWC URL 返回 404，保留响应头。生成来源记录和绑定脚本退出 0。
- `forward_link.py` 使用归档 SourceTransport，传输为固定来源、有界活动、工程 2 ms 延迟。官方子图投射用按靶点归一的突触计数，并阈值化读出工程单档推进。五项队列预检实际通过并退出 0。两侧的自然放电率和膜动力学仍待独立测量。
- `forward_body_probe.py intact` 及 `forward_body_probe.py cut` 使用归档 Python `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python`、MuJoCo、EGL、单线程 BLAS/OMP/MKL。固定同一站姿、原身体、原传感器顺序和原有限执行器；9.6 秒固定试验波形经实际源队列消费后才调用原型步态。完全切断组相同准备与意图，物理判据量前进、六足周期、滑移和末窗稳定。
- 两条固定波形物理回合各 904,000 原生积分，接触按 1 ms 记录，进程均退出 0。配对审核 `FORWARD_BODY_PAIR.json` 确认意图相同、505 ms 之前身体状态逐字节相同、所有交付延迟 2 ms。完整组实际前速 .0716812、采样滑移 .4808133，六足周期 1/2/2/2/2/2，末窗站稳。切断组交付为零、位移接近零且末窗站稳。物理句柄、源绑定、图、步态数据和停止权重的散列见 `PULSE_INTACT/PROBE.json`、`PULSE_CUT/PROBE.json`。影片与真实眼图课另记。
- 首两条 `walk_pilot.py run` 尝试在元数据中包含 argparse 回调，JSON 序列化拒绝，退出 1。保留 `WALK_TEACH1/2` 和日志。直接用只含命令参数的 Namespace 调用 `walk_pilot.trial`，起点 nominal、视觉出现 500/640 ms、9600 ms 观察窗及两个背景／宽度，输出 `ROOT_TEACH_A/B`，两条均退出 0，原生门通过。保存后 `run_walk_accept.sh fit ROOT_TEACH_A ROOT_TEACH_B` 退出 0。
- `./run_walk_accept.sh run` 的正常、切断、恢复、空场四条在留出的同组设置下各 934,400 原生步，无额外施力，退出 0。随后 `./run_walk_accept.sh audit WALK_HELD_normal WALK_HELD_cut WALK_HELD_restore WALK_HELD_off` 退出 0。离线回放逐张重新计算 2,332 张双眼图、9,344 个动作时点，视觉特征、意图、颈部交付日志和 42 维动作逐位相同，退出 0。渲染实际回合退出 0，视频全解码 585 帧，主线程和子代理实际打开关键帧。桌面复制与源逐字节相同。
