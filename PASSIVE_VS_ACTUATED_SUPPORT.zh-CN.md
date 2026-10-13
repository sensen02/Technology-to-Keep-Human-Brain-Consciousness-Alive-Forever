# 更正：63.8%是**被动**支撑，不是"站姿撑不住"

**手工研究原型，不是成品。** 本文档更正我此前表述中的一处框架错误，并给出正确的量化时间线。

## 一、更正内容

- **我此前的说法**："部署站姿只能承担体重的63.8%"。
- **正确说法**：在**声明式零激活初态**下，**被动**结构承担体重的63.8%；随着肌肉激活在最初几毫秒内建立，地面支撑在 **10.03ms** 就达到体重，全程平均为体重的 **0.998**。
- **为什么重要**：初态`act=0`是声明，因此对该状态的静态评估只测到被动结构。**不能用它断言"该姿态在动物驱动下不可支撑"。**

## 二、正确时间线（实测，2001样本，25μs身体步长，同一积分）

| 事件 | 时间 |
|---|---:|
| 初始支撑 / 体重 | 0.638（被动，`act=0`） |
| 平均激活首次 > 0.01 | **2.52ms** |
| 平均激活峰值 0.512 | 6.27ms |
| **支撑首次达到体重** | **10.03ms** |
| 全程支撑均值 / 体重 | 0.998 |
| 全程支撑峰值 / 体重 | 1.512 |

支撑在t=0是被动受限的，10ms后即达到体重；此后支撑围绕体重波动（约95–100ms处一度降到约0.75），而身体继续缓慢下沉，到约95–97ms由身体触地承担主要支撑。**即：这不是自由落体，而是准静态的"缓慢坐下"。**

## 三、两次静态扫描的正确解读

| 度量（均为`act=0`静态） | 支撑 / 体重 |
|---|---:|
| 部署被动站姿 | 0.638 |
| 根高+胫节两参数扫描最优 | 0.788（且需把脚压入地面0.30mm） |
| 六腿对称7关节+根高搜索最优（约束穿透≤0.05mm） | 0.649 |

**这些数字只描述"没有肌肉力矩时"的被动支撑**，说明的是被动结构无法承载体重、需要激活；**不能**用来论证"不存在可支撑站姿"。我在上一轮把两参数扫描的最优0.788当作"可达支撑上限"来讲，属于过度解读，现予以更正。

## 四、由此修正的结论链

1. 站不住**不是**因为地面无法提供支撑力（10ms后支撑已达体重，且平均等于体重）。
2. 站不住**也不是**因为力矩不够或方向反了：力矩审计显示施加/所需比的族均值0.97–1.26，相关性0.78–0.91，**没有任何关节反号**；去趋势后对齐仍为0.70。
3. 但驱动**与关节运动同相**（23个关节正相关，仅4个阻尼型，族均值+0.10～+0.20），即它跟踪"当前姿态的保持需求"，而不阻止身体继续变形——缺少的是**绝对姿态目标/恢复性控制**。
4. 结构上又缺**跗节/股节自由度**（脚部无法重新摆放）与**16/90不可控肌束**，接触/负载感觉通道证据也最弱。

## 五、产物与边界

- 脚本：`/run/media/sensen/Data2/cell_wound_prototype/run_passive_vs_actuated_support.py`
- 测试：`/run/media/sensen/Data2/cell_wound_prototype/test_passive_vs_actuated_support.py`（5项通过）
- 结果：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/passive_vs_actuated/passive_vs_actuated.json`
- 图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/passive_vs_actuated/passive_vs_actuated.png`
- 静态扫描：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/stance_support_scan/`、`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/stance_search/`
- 力矩对齐：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/torque_alignment/`、`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/drive_phase/`

**自查记录**：本审计的跨运行核对最初报false，是我自己的两个bug——先用整数除法算步长比（8001//2001=3），后忽略两端样本的端点关系（应为`(n−1)×比例+1`）。修正后为true：支撑审计的25μs episode重采样到0.1ms网格后，与分支episode的qpos和根高**逐位相等**。

边界：单一episode、单一姿态；支撑由接触力测得、无量纲；本审计只重读既有数组，未运行新动力学。
