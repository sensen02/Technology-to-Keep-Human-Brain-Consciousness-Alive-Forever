# 同命令条件下的身体步长敏感性

手工研究原型，不是成品。**本实验是诊断回放，不是自主控制。**

使用已保存6.25μs正常神经组40×90命令，分别在12.5/6.25μs身体步长重放200ms；实际神经仍积分400步，但不使用其新读出来替换已固定命令。6.25μs回放须精确复现源正常轨迹，否则不能解释比较。

对比既有独立正常闭环中两步长各自产生命令的差异。若固定命令仍明显分离，说明身体积分/接触本身也敏感；若固定命令较小，只能说明这个给定输入下误差较小，不能把所有闭环分离定量归因为唯一神经机制。

固定模型、初态、记录时刻和所有参数，除步长外不修改。窗口为25/50/100/200ms，单位分开。## 实测

两组共17.57秒完成，6项测试父任务复跑通过，图已实际查看。6.25μs回放与源正常组所有保存数组精确一致；两组每一个实际身体步应用命令均相同。

0–200ms样本最大差：

| 量 | 相同命令回放 | 各自产生命令的闭环 |
|---|---:|---:|
| 三维COM mm | 0.001198 | 0.048489 |
| 根平移 mm | 0.001260 | 0.059580 |
| 根朝向 rad | 0.001383 | 0.050493 |
| 单铰链最大差 rad | 0.007218 | 0.309410 |

固定同一输入下身体差异明显小于闭环差异；这支持将身体离散误差与反馈后的轨迹敏感性分开研究。但不能用两条路径差的比例断言某一神经机制贡献百分比，也不能证明所有输入和更长时间下机械收敛。

原轨迹和回放依然跌落，结果不是控制修复或站立完成。

## 产物

- `/run/media/sensen/Data2/cell_wound_prototype/run_fixed_replay_refinement.py`
- `/run/media/sensen/Data2/cell_wound_prototype/test_fixed_replay_refinement.py`
- `/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/fixed_replay_refinement/fixed_replay_refinement_results.json`
- `/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/fixed_replay_refinement/fixed_replay_refinement_traces.npz`
- `/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/fixed_replay_refinement/fixed_replay_comparison.png`
