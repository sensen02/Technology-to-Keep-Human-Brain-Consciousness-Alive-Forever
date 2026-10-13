# 关节力矩缺口审计：能力不是瓶颈，时机与空间模式才是

**手工研究原型，不是成品。** 本审计只读既有25μs正常分支的保存状态，不做积分、不改参数、不新建控制器。

## 方法与其自检

对每个保存样本重算广义力分解，并在报告任何关节数字前先验证MuJoCo恒等式
`M·qacc + qfrc_bias = qfrc_actuator + qfrc_passive + qfrc_constraint`。

- 修正前后对比：我最初写成`bias+passive+actuator+constraint−M·qacc`，残差**48.9**（恰为2倍重力项），即符号写反；改为上式后最大残差**6.0×10⁻¹⁴**，容差2.4×10⁻⁵，**通过**。
- "保持所需力矩"定义为使`qacc=0`时的致动器项：`bias − passive − constraint`。它是**逐瞬时的静态保持比较**，不是可达性证明。

## 实测

身体步长25μs、真实神经闭环200ms，2001个样本。单位为model unit（读作mN·mm时见单位报告；比值与单位读法无关）。

| 时刻 | 覆盖度 Σ\|实际\|/Σ\|所需\| | 最缺关节（所需 → 实际） |
|---|---:|---|
| 0ms | **0.262** | LFTibia_pitch −2.006 → +0.003；LHTibia −1.811 → +0.004；RFTibia −1.344 → +0.004；RHTibia −1.219 → +0.005；LMTibia −1.204 → 0.000 |
| 10ms | 0.262 | 胫节仍全部未支撑 |
| 24ms（原10%高度阈值） | 1.042 | 出现**过量**：LFTibia −1.916 → −2.909（超需约0.99）；同时RHTrochanter_pitch −3.295 → −4.144 |
| 50ms | 0.870 | RFTrochanter_yaw、RFCoxa_yaw |
| 95ms（身体触地开始） | 0.648 | **LHCoxa_yaw −1.520 → −0.178**（缺1.342）；RHTibia −1.065 → −0.187；LMTrochanter_yaw/roll |
| 200ms | 0.892 | LHTrochanter_pitch、LHCoxa_yaw、LHCoxa_roll、RHCoxa_pitch（后腿髋部为主） |

全程：最大所需10.375，最大实际**14.190**（实际可以超过所需），最大缺口8.927≈体重的36.5%；**没有任何关节的缺口超过整身重量**。

## 结论

1. **t=0几乎没有支撑力矩**：五条着地腿的胫节需要−2.0～−1.2，实际仅0.000～0.005（覆盖26%）。这与"初始激活按声明为0、肌肉被动元件几乎不承担保持力矩"一致，即站立在最初几毫秒内是靠自由落下开始的。
2. **24ms后不是力不够，而是给错地方**：总覆盖度已到1.04，但符号化缺口显示胫节**过量**、同侧转子**不足**。说明力矩在错误的空间模式上分配。
3. **崩塌时刻最缺的是髋部侧向控制**（coxa_yaw / trochanter_yaw / roll），这正是防止身体侧倾与外撇的关节，与95ms身体触地时间吻合。
4. 由于最大实际力矩>最大所需力矩，**当前限制不是肌肉力量上限**，而是命令的模式与时序。这与"缺少经验证的感觉调谐与运动募集映射"这一证据缺口一致，**所以不应靠调增益解决**。

## 边界

- "所需"是在**实际接触力**下算出的逐瞬时保持力矩，不证明该姿态在动力学上可达；某一时刻缺口大不等于该姿态不可能。
- t=0的缺口部分来自声明式零初始激活，不是控制器判断。
- 单一神经分支、单一初态、200ms；不同步长(100μs)结果不同，故只在25μs基线上陈述。
- 单位读法未定，绝对值仅作相对比较。

## 产物

- 脚本：`/run/media/sensen/Data2/cell_wound_prototype/run_joint_torque_deficiency.py`
- 测试：`/run/media/sensen/Data2/cell_wound_prototype/test_joint_torque_deficiency.py`（4项通过）
- 结果：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/torque_deficiency/torque_deficiency_results.json`
- 数组：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/torque_deficiency/torque_deficiency_traces.npz`
- 图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/torque_deficiency/torque_deficiency.png`
