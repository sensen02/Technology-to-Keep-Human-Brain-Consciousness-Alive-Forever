# DNg97 双侧前进通路调速标定

人工搭建大脑项目身体台架，覆盖单次平地起步、低速、逐渐升速、回落、撤令及站稳。以具名 BANC 子通路交付工程活动，使用上一轮合格的身体侧当前观测调速器。此次发送器按照已验收的台架指令产生波形，脑侧尚未训练，真实动物活动量与速度关系仍未知。

## 来源

DNg97（oDN1）和 DNg100（BDN2）的光遗传激活均可诱发前进。Sapkal 等人的抑制结果仅在 BDN2 观察到自发行走前速下降，并报告 BDN2 活动与前速相关。DNg97 适合作为独立前进通路候选，其调速增益是本轮仿真假设。 [Sapkal 等人](https://pmc.ncbi.nlm.nih.gov/articles/PMC11446846/) · [类型别名和运动研究](https://faculty.washington.edu/tuthill/docs/Dallmann_et_al-2025-Nature.pdf)

| BANC v888 | 来源与颈部连续性 | 官方 v2 VNC 投射 |
| --- | --- | --- |
| 左 720575941519464046 | 官方 L2，2,452 节点。前颈种子与连续路径交点相差 422.3 nm。元数据的另一栏 root_id 为 720575941647118024，明确采用正式 root_888 与 banc_888_id 连接，左侧全分辨率文件未找到。 | 510 条边、2,290 突触 |
| 右 720575941572713886 | 官方全分辨率 SWC，65,058 节点。对应距离 278.8 nm。 | 482 条边、2,479 突触 |

使用经固定 GCS generation 和 MD5 的 SWC 以及正式 v2 边表。前后截面是同一有后裔关系的 SWC 路径。联合 VNC 靶点 973，真实 BANC 投射计数固化。用突触计数归一化的输出投影、2 ms 同纤维延迟和有界双侧共模读出属于工程定标。 [BANC 数据](https://pmc.ncbi.nlm.nih.gov/articles/PMC13518251/)

## 实测

身体调用原当前观察调速器，连续 0.5 s 站立、9.6 s 走动及 1.2 s 末窗。体态固定 71 mesh、42 有界 PD、12.5 μs 原生物理和 5 ms 身体动作，同源初态，自由根及零额外力。相同课程以双侧活动交付、独立重跑、全程切断三种条件，各 904,000 原生步，进程均退出 0。

正常组低速、升速后平台、降速后平台的有符号速度依次为 0.07186749、0.08401535、0.07089115 原生长度/s。实测速升降均大于预登记 0.004 增量门。全程原生滑移 0.47706457，六足各 1–3 个完整周期、起止站稳。释放 11,299 条消息，身体在 2,260 个决策时刻据最近已到达的信号更新。首次非零发送 500 ms，交付 502 ms，身体动作首次改变在 505 ms。切断组无消息交付，身体维持近零推进及站稳。两组前缀在首次不同动作前逐位相同。正常两次的轨迹、决策、原生和消息存档字节一致。 [主审计](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T172138Z-dng97/DNG97_MATCHED_ACCEPTANCE.json)

## 命令和证明边界

使用归档 `../../banc-walking-recovery/20261110T125712Z/env/bin/python -B`。物理/渲染设置 `MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。

- `audit_source.py` 输出官方下行绑定和投射，与两份来源几何报告。左全分辨率对象 HTTP 404 后成功使用该细胞的官方 L2。首个调速准备试验因 NumPy 标量触发传输类型检查退出 1，未完成回合。
- `run_speed.py SPEED_CONNECTED`、`run_speed.py SPEED_CONNECTED_REPEAT`、`run_speed.py SPEED_CUT --cut` 和 `audit_pair.py` 均退出 0。
- `../../body-motor-learning/20261116T075254Z/speed/render_speed.py VIDEO SPEED_CONNECTED SPEED_CUT` 离线生成，566 帧全解码，实际检查关键帧。左为通路完整，右为切断。视频中的未过走动联合门是切断对照的预期表现，切断停稳条件已过。接近静止时相对滑移分母过小。 [录像](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T172138Z-dng97/VIDEO/身体连续调速对照.mp4)
- 结论范围为使用来源身份、轴突连续性和投射图的身体台架工程标定。展示当前姿态调速器通过同一对轴突工程活动实现有因果对照的调速。首个视觉起停训练、独立真实膜模型及电极尚待验证。
