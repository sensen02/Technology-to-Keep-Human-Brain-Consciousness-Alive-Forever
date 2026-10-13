# 真实神经闭环200ms持续性检查

**手工研究原型，不是成品。** 不加入CPG、PD、背景电流、优化肌肉预加载，不修改规范XML或神经增益。

## 目的

50ms范围内初态的正常输出将胸高阈值时间从24ms推迟到35.8ms，但质心仍下落。不能据此称持续支撑。本实验只把两种既定正常条件延长至200ms，不做剂量/参数搜索。

## 验证契约

每种条件的前50ms轨迹、命令和放电计数必须逐位复现之前的短窗实验。观测实际身体每步状态、全身COM、非足部触地和原生限位，检查有限值与警告。原姿态与范围内姿态的感觉角度参考分别取各自初态，此差异显式保留。

## 实测

两种200ms真实神经运行完成，前50ms命令、放电、轨迹及501个身体位置样本逐位复现短窗试验。全部监测状态有限且无MuJoCo警告，但两组持续站立均失败。

| 正常输出初态 | 非足部触地首现 | 200ms根高 mm | 200ms COM高 mm | 非足接触最大条目数 |
|---|---:|---:|---:|---:|
| 原始 | 95.0ms | 0.995502 | 0.625227 | 16 |
| 显式范围内 | 85.3ms | 1.123977 | 0.722984 | 6 |

范围内初态虽然末高度较高、接触较少，却更早出现非足部触地。50ms较好的根高不能推出全面稳定优势。

神经活动并非20ms峰后永远消失：原始运动放电35–40ms降到2次/5ms，105–110ms又达88次；范围内初态100–105ms为0，130–135ms达60次。此后放电回升仍未恢复站立；不能将所有失败归因为网络完全熄灭。JSON的全程minimum/recovery标记包括t0零放电，不能单独作为峰后恢复证据，这里引用实际时间段数值。

## 产物

- 实现：`/run/media/sensen/Data2/cell_wound_prototype/run_neural_duration_probe.py`
- 测试：`/run/media/sensen/Data2/cell_wound_prototype/test_neural_duration_probe.py`
- 数据：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/neural_duration_probe.json`
- 数组：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/neural_duration_probe.npz`
- 已查看图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/neural_duration_probe.png`

