# 显式初态范围修复的神经闭环对照

手工研究原型，不是成品。不静默覆盖原始姿态，不改变规范XML、肌肉、神经参数或网络连接。

## 两种初态

原始round37（当前闭环）：5个标量关节越出声明范围，原生软限位仍可积分；通过接触门不代表通过关节范围门。

显式范围内候选：来自已有pose_support_search.json的baseline_exact_source，只改变上述5关节与重新落座后的根高度。**不使用该文件里的优化肌肉激活**。四组全部从零激活/零命令开始，只由网络输出驱动。

比较原姿态正常/输出静默、范围内姿态正常/输出静默。输入角度参考随各自初态改变，须明确记录，不把它说成只改机械几何而感觉完全等价的实验。

## 预设检查

原姿态两组必须逐位复现当前包；两种初态做接触/限位报告，观察运行中超范围和非足部触地，分别记录质心和根部，不以softlimit本身宣布模型无效。原姿态的初始软限位作用可能与后续失稳混合，但需要实测，不预设修复即可站立。

## 实测

4组完成，6项测试通过并由父任务复跑。所有原姿态数组精确复现已保存正常/静默基线。

| 初态/输出 | 越过90%初始胸高 | 50ms根高 mm | 50ms COM高 mm |
|---|---:|---:|---:|
| 原始/正常 | 24.0ms | 1.723613 | 1.498134 |
| 原始/静默 | 14.9ms | 0.997642 | 0.459467 |
| 范围内/正常 | 35.8ms | 1.794155 | 1.505704 |
| 范围内/静默 | 15.9ms | 1.042467 | 0.482524 |

原始初态有5个native limit约束行，范围内初态0个；接触约束行也从48变为56，因此不能把差异单独归因限位力。正常组根高改善明显大于COM改善，依然不能把根部当质心。

尽管范围内初态开始没有越界，正常运行期间最多仍有15个关节超声明范围（原始正常最多14）；原生软约束可继续积分，但初始范围修复并未消除后续动态问题。正常两组50ms无非足部触地，静默组最多分别26/20个接触条目。所有监测状态有限、无警告。

结论：初态确实影响轨迹，但没有治愈下落。四组始终零初始激活，最早原始运动放电均6.5ms，正常最早应用命令均10ms。没有用力学优化激活伪造网络能力。

## 产物

- 实现：`/run/media/sensen/Data2/cell_wound_prototype/run_neural_pose_comparison.py`
- 测试：`/run/media/sensen/Data2/cell_wound_prototype/test_neural_pose_comparison.py`
- 数据：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/neural_pose_comparison.json`
- 数组：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/neural_pose_comparison.npz`
- 已查看图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/neural_pose_comparison.png`

已为历史主入口增加显式初始关节范围报告和警告；保留原初态不静默替换。此观测变更随后需并入全闭环回归。

