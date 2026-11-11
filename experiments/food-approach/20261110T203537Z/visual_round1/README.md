# 续轮1：真实保存轨迹的神经转向物理对照

本目录只读取已完成的七次真实物理轨迹，没有构建物理模型、调用物理积分、训练、联网或修改原证据。本图范围固定为七个已完成的中性/左右转向回合，速度085/090未纳入本图。随后父代理通知速度085五项工程门均失败、速度090滑移约1.458失败；这些速度结果只在此作为父代理后续状态通知，不作为本图已独立读取/重算的证据，详见父级正式报告。

主图：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/visual_round1/ACTUAL_NEURAL_TURN_COMPARISON.png`

机器可读指标与全部输入SHA：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/visual_round1/METRICS.json`

桌面副本：`/home/sensen/Desktop/engine修复验收/食物趋近实施-20261110/续轮1-神经转向物理对照-未达标.png`

## 实际结果

| 回合 | signed forward | slip分母 netXY speed | 最大slip | 实际yaw rad |
|---|---:|---:|---:|---:|
| 算法教师左 | .071620 | .071621 | .474773 | +.061284 |
| 算法教师右 | .072139 | .072471 | .448280 | -.078512 |
| 前代residual500左，神经失败 | .072141 | .072141 | .539664 | +.069252 |
| 前代residual500右，神经失败 | .072306 | .072607 | .515163 | -.080552 |
| 当前fast1000中性，神经通过 | .071319 | .071385 | .492789 | -.006678 |
| 当前fast1000左，神经失败 | -.012406 | .046454 | 23.696004 | +.074551 |
| 当前fast1000右，神经失败 | .032106 | .032304 | 3.473567 | -.030732 |

算法教师与两代神经控制分别用图例、线型和颜色标记，禁止把教师的PASS解释成神经自主成果。当前fast1000左右失败，不是食物趋近成功。所有回合非足力与保存的额外体力/广义力均为0；完整轨迹中仍存在明显滑动/振荡，承重均值约1不能替代稳定协调运动。

## 分母与分子诊断

原slip定义保持不变：先对每腿做载荷加权绝对足端切向速度均值vt，再除以 `norm(COM_end_xy-COM_start_xy)/duration`。此分母非负，**不是signed world-X forward**。因此左转负前进直接导致forward门失败，却不会产生负slip分母。

- fast1000左的净XY分母为教师左的0.648606倍；但各腿绝对vt已增至教师左的8.41—36.08倍，RM为36.08倍，最大slip=23.696。RM滑移分子的0.98722来自load>=.1W时间段。因此这是实质性的承重足高速运动并伴随减少/反向前进，不能归结为负进度或分母接近0。
- fast1000右分母为教师右的0.445742倍；各腿绝对vt增至1.409—5.719倍，LF增至5.719倍且其分子0.86874来自load>=.1W时间段。分子增加与分母缩小共同抬高slip。
- 左转实际COM路径出现大幅回返和环状段；1秒后约有数秒明显前后速度振荡。头向改变方向正确，不意味着完成有效趋近。图保留全部窗口、尖峰和真实比例。

## 图中九面板

同一固定坐标轴的完整9.6秒COM XY路径；因果前1秒平均signed forward；实际yaw；原每腿slip对数轴及真实红线.5；绝对载荷加权足速vt；signed forward与原非负slip分母的对应；每1ms足载/W；每1ms COM高度；全腿载荷加权足速的前1秒平均。足载红线.9表示原均值门参考，图中瞬时值并不自动构成独立判据。高度红线1.2为原COM最低门。

每个NPZ数组只解压读取一次。脚本验证完整15400端点、轨迹SHA与RESULT一致、整个9600自主端点的每腿slip/signedforward/netXYspeed/足载/COMmin与原报告一致，保存root quaternion解析的yaw与原实际yaw差值一致；前后重新计算输入SHA确认未改写。没有调用MuJoCo或物理后端。所示为1ms端点工程测量，不冒称原生D1—D8认证、摄食、生理或生物有效性。

运行命令：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/visual_round1/plot_comparison.py`

托管任务bash-216已收集，退出码0。图像已实际打开：所有曲线、FAIL/PASS标签、.5红线以及大幅左转振荡均可见；桌面副本SHA与主图一致，见DELIVERY.json。必要原始失败证据全部保留。
