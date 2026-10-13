# 预定两档身体时钟细化

手工研究原型，不是成品；不引入新的运动逻辑或感觉背景。

12.5与6.25μs两组正常真实神经闭环，各200ms；神经400步、命令40周期固定。对照先前50/25μs，模型仅积分步长不同。观察每细步有限值与激活极值，每0.1ms保存身体状态。

按[0,T]身体窗口、[0,T)命令窗口、(0,T]神经事件分别比较25/50/100/200ms。质心和根位移用mm，铰链角用rad，四元数采用符号不变转角，不使用混合状态范数。三维COM从保存qpos做scratch几何复算，不增加episode。

预先声明COM最大误差0.01mm仅是报告参考，不是生物正确性或站立验收。神经阈值放电可使后期反馈离散分叉，不能仅凭早期轨迹接近声称整个200ms已收敛。

## 实测结果

两组共17.914秒完成，4项测试父任务复跑通过，图已实际查看。12.5/6.25μs各16000/32000身体步，均400神经步、40命令周期，无激活超调和MuJoCo警告。

| 相邻步长 | 0–25ms三维COM最大差 mm | 0–50ms | 0–100ms | 0–200ms |
|---|---:|---:|---:|---:|
| 50 vs 25μs | 0.001738 | 0.003136 | 0.139378 | 0.326361 |
| 25 vs 12.5μs | 0.000484 | 0.002256 | 0.031123 | 0.116298 |
| 12.5 vs 6.25μs | 0.000172 | 0.001701 | 0.012536 | 0.048489 |

质心误差随细化下降，但最细两组100/200ms仍超预声明0.01mm参考；部分关节误差并不单调，不能称全状态收敛。0–50ms的接近不能外推到全程。

12.5/6.25μs正常组200ms COM高0.883282/0.873374mm，根高0.863517/0.851916mm，首次非足部触地96.3/96.0ms，均未站稳。消除粗步长超调后仍然跌落，不能以数值修正冒充自主能力。

## 文件

- `/run/media/sensen/Data2/cell_wound_prototype/run_clock_refinement.py`
- `/run/media/sensen/Data2/cell_wound_prototype/test_clock_refinement.py`
- `/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/clock_refinement/clock_refinement_results.json`
- `/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/clock_refinement/clock_refinement_traces.npz`
- `/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/clock_refinement/clock_refinement.png`

本轮没有为6.25μs重复静默/回放，这两项仅25μs已验证；不扩大声明。
