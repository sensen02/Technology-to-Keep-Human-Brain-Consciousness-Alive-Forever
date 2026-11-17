# 开源脑共享下行接入与切断验收

Eon 与 BANC 中的 **DNa01、DNa02、oDN1（DNg97）** 已完成下行连接与平地短程运动验证。三类共六根轴突的 FlyWire v783 和 BANC v888 对应、跨颈连续分支、SWC 和 VNC 靶点可查。原 Eon 全脑冻结为 138,639 个细胞和 15,091,983 条连接，身体使用上轮通过平地扩训的固定权重和同一站姿快照。所有切断均在 BANC 具名轴突上操作。结构来源为真实重建，跨标本配对、2 ms 传导和 VNC 读出为工程模型。 [Eon 公开接口](https://eon.systems/updates/embodied-brain-emulation) · [BANC v888 原始资料](https://doi.org/10.7910/DVN/7WTH1N)

## 接法与标定

冻结全脑接受两档 oDN1 和左右转向细胞组的脑侧输入，模拟原版膜电位、延迟突触和实际脉冲。实际脑发放按审核过的轴突身份进入 BANC。BANC 接入 25 条已匹配下行，读出只从三类下行各自一阶靶点的膜电位获取信息。体内前足接触和负载仍可发放到 BANC 上行端。本轮 Eon 发放采用独立记录并逐拍回放，身体反馈尚未送回 Eon。

新读出允许单独 oDN1 的颈后活动产生左右同向驱动。低档和高档的靶点稳态中位数为 0.955/2.018 mV，分别标定到已合格身体的 0.47/1.00 驱动。DNa01 与 DNa02 各自按左右靶点的差值标定，共同输出冻结身体已学过的左右驱动差。七条开发神经轨迹按时间窗拟合参数，两种留出种子独立。读出只有下游膜电位输入。冻结脑与身体均零权重更新。

## 冻结的正式留出

| 通道与功能 | 完整身体实测 |
| --- | --- |
| oDN1 起步和调速 | 巡航 4.46–4.73 mm/s，快走 10.38–10.53 mm/s。同种子快走为巡航 2.19–2.36 倍 |
| DNa01 走中转向 | 左 +27.34°、+37.37°，右 −34.30°、−30.16° |
| DNa02 走中转向 | 左 +29.91°、+35.46°，右 −39.31°、−35.84° |
| 单束转向轴突切断 | 八个配对中切断轴突各有 89–122 个同一脑发放，颈后对应到达全为 0，其他下行逐拍完全相同。继续前进，偏航绝对值 1.33–4.63° |
| 双侧 oDN1 切断 | 两侧脑发放仍有 171 个，颈后为 0，身体保持站姿 |
| 全部已接下行切断 | 脑发放仍有 355 个，颈后为 0，身体保持站姿 |

23 个预登记条件全通过，包含固定初态、停稳、扰动、双速、无输入、单切及全切门。所有回合轴向部位触地为零，回合末 200 ms 最大平面速度 RMS 为 0.094 mm/s。正式和新进程重放共 782,000 原生物理步，23 对各自的关节目标、位姿、颈后信号和到达逐位一致。另为巡航、DNa01 左转和 DNa02 右转重新运行全脑，所有存档脑状态量相同。调速验收未设置横向偏移门，标称快走的两回合末端横偏约 0.75 和 2.38 mm，航向闭环未实施。

额外做了左右 PFL3 脑侧刺激配对，双侧 oDN1 提供巡航。两类转向 DN 的放电由原脑递归连接招募。脑侧左测试发放 DNa02 左 102、DNa01 左 9 次，右测试发放 DNa02 右 86、DNa01 右 8 次，两类的对侧均为零。过颈后带出左 +32.43°、右 −39.00° 走弧。该对照展示原脑内部能把上游刺激传到转向下行。环境驱动的自主动作选择仍待验证。

本次验收针对有外部脑侧刺激的下行台架和短程平地。原地转向、飞行、连续感知闭环、生理电极和 D1–D8 余项未由此实验验证。

## 复现

在本实验目录用归档 drive.sh。先执行 bash run_brain.sh，再执行 bash run_probe.sh 生成 CALIBRATION.json。运行 ./drive.sh freeze_inputs.py IDENTITY.json，接着运行 bash run_physics.sh FORMAL、bash run_physics.sh REPLAY 和 ./drive.sh audit_overlap.py FORMAL FINAL.json --replay REPLAY。单一用例可用 ./drive.sh overlap.py open BRAIN/271_a01_left TEST_A01_LEFT。

- [六根轴突和输入清单](/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261117T064507Z-eon-overlap/IDENTITY.json)
- [全部正式行为和独立重放](/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261117T064507Z-eon-overlap/FINAL.json)
- [原脑重复和 PFL3 诊断](/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261117T064507Z-eon-overlap/SOURCE_AUDIT.json)
- [运行检查表](/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261117T064507Z-eon-overlap/RUN_CHECKLIST.zh-CN.md)
