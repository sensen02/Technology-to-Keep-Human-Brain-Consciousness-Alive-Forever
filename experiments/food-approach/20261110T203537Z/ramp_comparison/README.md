# 同一3000步检查点：阶跃与一秒渐变命令的保存物理对照

本任务只分析已有数组，未构造Core、未导入物理或模型后端、未优化或积分。全部代码和报告位于 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/ramp_comparison/`；旧实验源码、追踪文件与当前状态均未修改。

两组使用同一检查点 SHA-256 `17fd103492007dc4de4ae9832fa39e4098cef980c4bebb03425c99f7c2f093a8`。阶跃组来自 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/continuation3000/RECOVERY_EVAL_{condition}/`；渐变组来自 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/ramp3000/EVAL_{condition}/`。中性参考来自 `continuation3000/RECOVERY_EVAL_neutral/`，中性完整工程验收通过。

## 完整自主窗口结果

所有指标保留完整9.6秒窗口，包含命令起始段。没有增加短窗口门槛、删除失败段或改变原滑移分母。

| 条件 | 命令起始 | 最大逐腿滑移 ≤0.5 | signed forward ≥0.05 | COM最低 ≥1.20 | 非足峰值 native | 六腿周期 | 工程结论 |
|---|---|---:|---:|---:|---:|---|---|
| 左转 | 阶跃 | 5.913961 | 0.022631 | 1.248779 | 0 | FAIL | FAIL |
| 左转 | 渐变 | 3.171277 | 0.019683 | 1.243657 | 0 | FAIL | FAIL |
| 右转 | 阶跃 | 8.704047 | 0.035467 | 1.248631 | 0 | FAIL | FAIL |
| 右转 | 渐变 | 7.740867 | 0.034911 | 1.245531 | 0 | FAIL | FAIL |
| speed085 | 阶跃 | 3.355179 | −0.060229 | 0.696679 | 65.982589 | FAIL | FAIL |
| speed085 | 渐变 | 4.568741 | −0.029339 | 0.706253 | 49.886989 | FAIL | FAIL |
| speed090 | 阶跃 | 10.114006 | 0.043129 | 1.250434 | 0 | FAIL | FAIL |
| speed090 | 渐变 | 9.712140 | 0.044592 | 1.247600 | 0 | FAIL | FAIL |

非足值保留原native力；图中为便于对照除以声明的模型重量24.446354755878005，原非足门0.2W与strict零要求分别记录，不把native绝对值声称为生物力标定。

渐变左转的滑移比阶跃低，但仍远超门槛且前进不足、六腿周期未全通过。speed085在两组中均出现质心塌陷和明显非足接触。其余渐变条件也未取得合格运动能力。不能由渐变改善了某个标量推论命令起始是失败的唯一原因，更不能作为食物部署资格。

## 证据核对

`compare.py`对9个回合（8个条件回合和1个中性参考）逐一验证已有独立审计的`integrity_all_pass`及所有检查项，检查同一checkpoint身份，读取每个NPZ一次，核对RESULT绑定的轨迹和神经记录SHA-256。重新计算完整9600端点窗口的signed forward、非负净XY速度分母、全部六腿绝对载荷加权足速与滑移、COM最低及非足峰值，与原报告逐位相同。所有保存数组有限，额外力记录为零。

所有回合与同SHA实际中性参考的前5800个物理端点（time/qpos/qvel/action/target/actualctrl/loads/COM）和前960行神经记录（time/X238/command/before+after hidden/mean/policy+actual action/source）均逐位一致。这证明比较起点和预热轨迹匹配，不是不同重置初态造成的比较。

全部2880个决策的渐变命令经独立公式重算：前960行零；之后`u=clip(elapsed/1s,0,1)`、`blend=3u²−2u³`、`command=固定命令×blend`，最大误差小于1e−15。阶跃命令同样核对。保存文件与分析源的SHA在读取前后保持一致。

首个qpos差异在接管后第一个1ms端点已出现。首1秒qpos/qvel差异仅写作描述统计，不作为任何行为或物理验收，也不替代完整窗口。

## 产物与实际视觉检查

- `METRICS.json`：各回合完整指标、前缀核对、命令误差、描述性起始差异和源哈希。
- `ACTUAL_STEP_RAMP_COMPARISON.png`：六面板实际数据，包含对数滑移及真实0.5线、signed forward及0.05线、COM、非足接触、完整COM路径和高度变化。已用read_image实际打开，塌陷和全部失败标签可见。
- `DELIVERY.json`：桌面副本哈希一致记录。
- `ARTIFACT_SHA256.json`：本目录产物清单。

桌面副本：`/home/sensen/Desktop/engine修复验收/食物趋近实施-20261110/续轮1-阶跃与渐变命令物理对照-均未达标.png`。

运行使用项目既有解释器`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B`、单线程NumPy/Matplotlib；托管任务bash-226完成，退出码0并已收集。没有新增训练或物理调用，没有待清理临时文件。
