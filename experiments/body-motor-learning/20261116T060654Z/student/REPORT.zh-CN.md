# 单右前腿绊脚反射的监督候选

人工搭建大脑项目 S2/S5 的身体台架。本轮沿用同一个 nominal 站姿快照、冻结身体、三球足端、无附着与有限 42 路 PD。每个正式回合连续完成 0.5 秒静立、9.6 秒推进和 1.2 秒撤令停稳。障碍回合为一个固定接地小障碍。政策在首次物理接触前与同场景冻结对照的公共轨迹字段逐字节一致。

## 学习和闭环

唯一示范源为完全通过同体原生资格的 A 障碍教师。1,920 个推进决策中有 61 项包含实际反射修正。冻结平地匹配回合作为零修正锚点。前馈网络的 67 个输入来自当前本体与姿态、执行器目标、足载、右前腿段方向触力、足垫速度和相对其他五腿匹配的右前腿当前目标误差。一个分类头预测是否介入，一个有界输出头预测七路右前腿速率修正，其余腿沿用同一当前观察下的原型动作。架构无循环或历史动作输入。训练 240 步 Adam，教师有效帧动作 MAE 从 0.246900 降至 0.028838，每次更新与散列保存在日志中。B 障碍未进入训练或选择，复验使用同一个保存权重。

| 场景 | 首次触碰／学习介入 | 右前足最高足垫净空（原生长度） | 首次重新承重 | 原生最大地面滑移比 |
| --- | --- | --- | --- | --- |
| A 同场景冻结对照 → 学习 | 2.386 s ／2.390 s | 0.110497 → 0.162221 | 2.789 s → 2.764 s | 0.482149 → 0.482326 |
| B 留出的几何冻结对照 → 学习 | 2.380 s ／2.385 s | 0.110468 → 0.179116 | 2.789 s → 2.889 s | 0.483501 → 0.486287 |
| 无障碍 | 无触碰，零学习修正 | 常规步态 | 常规步态 | 0.481386（位级复现） |

A 和 B 的障碍触碰都局限于右前腿第二跗节。采样首次触碰时右前脚地载为零，其他五脚合计最低在 A/B 分别为 0.995652/0.997978 W。第一次落在障碍前方的地面后，两组连续 200 毫秒的每一个原生步，其右前脚都保持至少 0.02 W 负载。六足各有合格换步周期并通过最后站稳，无额外力或错误腿段碰障。A、B 的原生抵抗冲量略降，障碍接触步数分别 873→877 和 1228→1241。两组冻结对照本来也走完单障碍。已证实触觉引起附加抬足，验收未建立对更难障碍的跨越率改善。

## 独立核验与范围

A、B、平地和第二次独立 A 由新进程从同一检查点加载，共 3,616,000 原生积分。A 独立回合的毫秒轨迹、每 5 ms 策略决策以及每个原生步的接触数据逐字节一致。平地候选和冻结平地的共同数组及原生文件散列一致，学习修正数为零。当前评估范围是同一冻结站姿、同一条右前腿、两个实际静态障碍位置。更换起点、腿、明显改变强碰撞方向、运动阶段与其他身体机制均需对应验证。两组测试的障碍触碰都发生在同一次右前脚摆动。该项是身体侧局部运动能力，继续使用台架推进驱动。

- [训练输出](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T060654Z/student/FIT_RF_A/RESULT.json) · [权重](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T060654Z/student/FIT_RF_A/step_0240.pt) · [每步日志](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T060654Z/student/FIT_RF_A/JOURNAL.jsonl)
- [三情形与独立复验](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T060654Z/student/STUDENT_SUITE.json) · [A 验收](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T060654Z/student/STUDENT_PEG_A/STUDENT_ACCEPTANCE.json) · [B 验收](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T060654Z/student/STUDENT_PEG_B/STUDENT_ACCEPTANCE.json)
- [A 同场景视频](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T060654Z/student/VIDEO_FINAL/绊脚配对回放.mp4) · [同次关键帧](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T060654Z/student/VIDEO_FINAL/绊脚关键帧.png) · [视频及源身份](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T060654Z/student/VIDEO_FINAL/VIDEO.json)
