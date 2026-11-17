# DNg13 跨颈来源与行走转向探针

本工作在人工搭建大脑项目中提供 P1/P2 与 S2/S5 候选通路证据。已验证的双 DNa02 定向链路作为比较。转向论文报告 DNa02 缩短内侧步幅、DNg13 增加外侧步幅，两者可按动作和节律阶段组合。因此项目按两种腿部作用分开测量。转向幅度是身体实测结果。 [Cell 转向研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC12778575/)

## BANC v888 来源审计

| 脑侧 | BANC v888 原始 ID | 前颈标记侧 | 完整 SWC 节点 | 同一跨双截面路径距前标记 | 官方 v2 直接 VNC 输出 |
| --- | --- | --- | ---: | ---: | --- |
| 左 | 720575941445859754 | 右 | 47,226 | 147.376 nm | 358 个靶点，1,990 个突触 |
| 右 | 720575941535086424 | 左 | 46,913 | 68.279 nm | 405 个靶点，2,491 个突触 |

两个 SWC 逐件固定公开 GCS generation，尺寸和 MD5 匹配，均为单树。两条原始 ID 的元数据、前截面有效点和该点附近通往后截面的实际 SWC 祖先路径完成核对。前颈标记侧与脑侧相反，符合该型论文所述对侧投射。截面 y 370,000 和 484,000 nm 是数据方的定位截面。VNC 输出以论文 v888 v2 原始表与元数据精确 ID 联接。该结构证据的权威范围是两根来源纤维及直连 VNC 靶点。 [完整源报告](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T172142Z-dng13/DNG13_SOURCE.json) · [BANC 论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13518251/)

## 物理探针与执行范围

冻结站姿、原 42 路有限执行器、原生 12.5 μs 和 5 ms 决策。用同一身体直走的世界系锚足教师，在前进段向左 DNa02、左 DNg13 分别或同时发活动，另设同前缀零活动。两类活动通过具名纤维原路径的 2 ms 工程收发，固定 v2 连边投射后，分别令左侧摆足目标缩短或右侧摆足目标延长 0.08 原生长度。教师参考无预置实际偏航轨迹，根姿态和作用力由模拟积分。探针测量这两种明确工程动作的可行性。突触计数归一化及摆足幅度本身均属人为配置。

使用归档 MuJoCo Python、EGL、BLAS/OMP/MKL 单线程。命令在本目录执行，解释器为 `../../banc-walking-recovery/20261110T125712Z/env/bin/python -B`。先执行 `source_audit.py`、`probe.py prepare`，然后分别 `probe.py off FLAT_OFF`、`probe.py dna FLAT_DNA`、`probe.py dng FLAT_DNG`、`probe.py both FLAT_BOTH` 及 `audit_probes.py`。初批中 NumPy 二维外力数组的布尔检查退出 1，第二批在平地教师继承地形探测设置处退出 1。两批的原始源码快照、日志及路径保留。

## 物理结果

四臂各 16.100 s、1,288,000 原生步，初终站稳、六腿换足、前进、支撑、质心和滑移通过，原生非足与外力峰为零。到首条非零释放时刻 2.902 s，采样的关节、速度、足载与足位记录和原生力前缀均逐元素相同。相同相位的修正先在 3.705 s 影响 DNg13 与双发送动作，DNa02 臂的第一处动作差在 4.505 s。 [相同初态和物理审计](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T172142Z-dng13/STEERING_AUDIT.json)

| 条件 | 9.6 s 活动窗实测偏航 | 相对关闭偏航 | 原生前速（原生长度/s） | 原生滑移比 |
| --- | ---: | ---: | ---: | ---: |
| 零活动 | −0.04693° | 0° | 0.102215 | 0.28709 |
| 左 DNa02 | −0.10123° | −0.05429° | 0.102289 | 0.28337 |
| 左 DNg13 | −0.03739° | +0.00954° | 0.102080 | 0.29023 |
| 左 DNa02 与左 DNg13 | −0.07872° | −0.03179° | 0.102174 | 0.28649 |

全部活动臂均低于预设的 +1° 可用左转响应门。实际前后摆动证据显示，DNg13 臂右后足第一次合格摆脚前移从 0.5263 增到 0.6052，右中足从 0.4617 增到 0.5426。DNa02 臂左后足从 0.4528 减到 0.3737。DNg13 臂同两脚的下一个合格摆脚分别为 0.4481、0.4627，零活动对应 0.4479、0.4615，支撑相的前后行程也近似。单位均为胸部坐标的原生长度。控制器将固定偏置加在每次摆足的绝对落点。同一脚后续两个落点可能共享偏置，因而首次改变单步行程，随后多为保持落脚位置偏移。该分析与实测近零偏航一致。 [肢体行程逐足定量](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T172142Z-dng13/EXCURSION_AUDIT.json)

DNg13 来源通过，可登记为待标定的行走转向候选。这次固定偏置试验保留为物理负例，不据它部署 DNg13 偏航或曲率能力。归档双 DNa02 的视觉转向和独立偏航读出拥有各自的作用证据。最终物理四臂、审计、关键帧和回放均退出 0，同次画面实际查看，视频完整解码 162 帧。 [回放](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T172142Z-dng13/转向探针对照回放.mp4) · [关键帧](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T172142Z-dng13/转向探针对照关键帧.png) · [媒体身份](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T172142Z-dng13/VIDEO.json)
