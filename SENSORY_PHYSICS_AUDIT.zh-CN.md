# 感觉测量链路独立物理代码审计（只读、有界）

## 结论与边界

**在本次限定范围内，未发现可以确认为当前 canonical 身体感觉测量、符号或单位实现错误的新缺陷。** 这不是“所有感觉实现正确”的证明，更不是站立成功证据。站立目标仍未达成；不能用代码审计替代行为验证。

本次为人工实施的源码审计及一次轻量 XML 静态解析，非成品验收。未运行 MuJoCo 模拟、未调用 `mj_step`/`mj_forward`、未加载大规模神经网络，未修改核心代码、XML 或 motor_output；唯一交付为本报告。检查的是当前源码路径中的 `observe_quantities()`（本版本没有名为 `sensory()` 的函数）、`prepare_body()`、编码器以及 canonical 肌肉身体 XML。未对生物感受器作校准，也未推断每个细胞的生理正负方向。

## 源码定位

下列为审计时的准确行号，后续源码修改可能使行号漂移：

- 身体初始化、映射与测量：`/run/media/sensen/Data2/cell_wound_prototype/run_neural_muscle_closure.py` 第 37–97 行。
- episode 初始化与采样顺序：`/run/media/sensen/Data2/cell_wound_prototype/run_neural_muscle_closure.py` 第 100–170 行。
- 通道定义、尺度来源：`/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/sensory_channels.py` 第 23–39、68–100 行。
- 编码与带符号电流：`/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/sensory_channels.py` 第 246–288 行。
- canonical XML：`/run/media/sensen/Data2/cell_wound_prototype/outputs/muscles_six_legs/fruitfly_six_leg_standing.xml`。
- 单位声明、canonical 路径与哈希：`/run/media/sensen/Data2/cell_wound_prototype/engine/canonical_physics.py` 第 76–91、139–140 行。

静态解析验证 XML SHA-256 为 `65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6`，与入口要求一致。脚本正常退出。XML 中无 include、无 sensor 块，只有一个 plane，名称为 floor。每腿有 7 个 hinge 关节、唯一一个 Tibia_pitch、5 个 Tarsus geom，共 42 个腿关节与 30 个足部 geom。没有显式非零 geom gap。此处是 XML 结构核验，不是编译后运行测量。

## 逐项核验

### 1. 足部归属完整，当前 XML 不存在漏算 Tarsus1–4 的问题

`/run/media/sensen/Data2/cell_wound_prototype/run_neural_muscle_closure.py` 第 66–74 行按 geom/body 的腿名前缀及 `tars` 子串建立归属，不只取最末端 Tarsus5。

XML 的具体足部锚点：

- LF：`/run/media/sensen/Data2/cell_wound_prototype/outputs/muscles_six_legs/fruitfly_six_leg_standing.xml` 第 251–269 行。
- LH：同一完整路径第 289–303 行。
- LM：同一完整路径第 325–339 行。
- RF：同一完整路径第 371–385 行。
- RH：同一完整路径第 404–418 行。
- RM：同一完整路径第 440–454 行。

每腿实际匹配 Tarsus1–5 各一个 geom，未匹配腿的 tibia、femur 或其他身体部位。XML 中部分足部 rgba 的 alpha 为 0，并不关闭碰撞；相应 geom 仍声明 `contype=2, conaffinity=1`。没有发现因透明外观而丢失接触的代码问题。

**适用边界：** 第 73 行的断言只保证每腿至少出现，不保证任意未来 XML 的全部足部都被识别；但本次锁定哈希的 XML 已核实完整，不能把未来泛化风险当作当前漏映射缺陷。

### 2. 接触过滤与力范数符合目前明确声明的统计量

`/run/media/sensen/Data2/cell_wound_prototype/run_neural_muscle_closure.py` 第 82–89 行：遍历实际 `data.contact`，检查接触双方，只有“已映射足部 geom 对 plane”才进入；调用 `mj_contactForce` 后累计 `norm(f[:3])`。当前 XML 唯一 plane 为 floor，定位于 `/run/media/sensen/Data2/cell_wound_prototype/outputs/muscles_six_legs/fruitfly_six_leg_standing.xml` 第 121 行。因此当前代码的 plane 类型过滤等价于 floor 过滤，既不会把躯干触地计入足部，也没有把腿间接触计入。

所得量为：

`F_leg = Σ_(该腿足部与 floor 的接触点) sqrt(fn² + ft1² + ft2²)`。

这不是世界坐标竖直支撑力，不是多个接触力的合力范数，不是只有法向的接触力，也不是组织应变测量。三个力分量都具有力单位；不混入后三个力矩分量。欧氏范数对正交坐标变换与整向量反号不变，所以无需为了此标量先把 contact frame 旋转到世界坐标，亦不应凭 geom1/geom2 顺序加减该标量。普通足部/plane 接触只命中循环中的一个方向，不会被双计。

**非缺陷但须保留解释边界：** 相反方向的多个接触力可在合力中抵消，而本统计量不抵消；这是“逐点范数求和”的定义，不是求和顺序 bug。把该值称为直接应变或绝对支撑力则超出实现证据。未来新增墙面 plane 时也会被计入，那时需要重新审查 floor 身份过滤。

### 3. 没有发现未 forward 就读取初始接触，或每步读旧接触力的调用顺序错误

- `/run/media/sensen/Data2/cell_wound_prototype/run_neural_muscle_closure.py` 第 43–51 行：施加初始姿态、落座、清零速度/激活/控制后调用 `mj_forward`。
- 同一完整路径第 111–116 行：每个 episode **重新创建** MjData 后，再次填初值并调用 `mj_forward`，不是直接假定 prepare 阶段的数据仍在。
- 同一完整路径第 133–136 行：区间开始才观察和编码。
- 同一完整路径第 162–170 行：每个 `mj_step` 后立即 `mj_forward`，下一控制区间观察前已有完整前向更新。

因此首帧 foot_contact 来源是 episode 的初始前向计算，而不是人为硬编码为 1；初始姿态可以已有非零接触力，即使 `qvel=0`、`act=0`。这不自动等于静力平衡。

有意的时序限制：感觉在 5 ms 控制区间起点采样并保持给该区间的神经子步，不是每个身体子步更新。外力在身体子步设置，因此区间边界的观测对应最近一次 forward 所保留的外力状态；它不是预知本区间新外力的测量。这里未发现 stale-force 调用遗漏，也未据此证明瞬态采样足够精细。

### 4. qpos/qvel 地址使用正确，角度保持有符号

`/run/media/sensen/Data2/cell_wound_prototype/run_neural_muscle_closure.py` 第 59–65 行：每腿按名字收集 7 个关节，唯一 tibia；位置使用 `jnt_qposadr`，胫节速度单独使用 `jnt_dofadr`。第 93–95 行读取时分别使用各自地址。自由根关节使 qpos 与 qvel 地址可能不同，代码没有混用两者。

XML 第 25 行使用 radian；各腿关节静态解析均为 hinge，非 quaternion/ball。因此当前模型下关节 qpos 减参考 qpos 是 rad，hinge qvel 是 rad/s，未将平移或四元数分量混入 7 关节平均。

胫节位置是 `q_tib - q_ref_tib`，多关节位置是 `mean(q_leg - q_ref_leg)`。后者会抵消不同关节的正负变化，是有符号平均定义，不是绝对值平均或关节几何总角度。它的信息损失、是否适合作为 hair plate 输入是模型选择问题，不能仅凭抵消现象判为实现错误。

`/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/sensory_channels.py` 第 263–265 行除以正参考尺度，第 280–282 行直接加到电流中：**未发现 abs、负值归零或正负方向交换**。第 153–161 行限制参考尺度为正，因此合法配置也不会悄悄反转符号。负偏差可产生负外加电流，这是该编码合同；不能未经证据改成兴奋电流或镜像腿翻号。

注意区分旁路 API：同一完整路径第 219–244 行的 `quantities()` 要求调用方传入与参考向量同行的关节角速度向量；不是允许直接传原始浮动根 MjData.qvel。本次闭环入口没有调用该辅助方法，而是明确使用 dof 地址构造 ChannelQuantities。

### 5. 力尺度没有隐含 SI 转换，reference_force 是已声明未校准值

`/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/sensory_channels.py` 第 91–100 行设 `reference_force=1.0`、`reference_angle_rad=0.30`、`reference_velocity_rad_s=10.0`、电流尺度 `0.3 nA/unit`，并明确 normaliser 不是测量值。第 263–267 行归一化，第 282 行乘电流尺度。

`/run/media/sensen/Data2/cell_wound_prototype/engine/canonical_physics.py` 第 76–91 行声明长度为 mm、时间为 s，质量单位仍是未确立的推断。接触力一路保持模型力单位；只要 reference_force 同样解释为模型力单位，维度一致。代码没有把模型力冒称为牛顿后再次缩放的步骤。

所以不能因为 1.0 缺乏生物校准就认定漏乘 1000 或必须改成体重倍数。绝对生物力/电流响应仍未校准，可能影响驱动强度，但本审计不虚构标定值、不建议用站立结果反推“正确生理尺度”。

### 6. XML 没有原生感觉 sensor，闭环使用软件观测而非 sensordata

对 `/run/media/sensen/Data2/cell_wound_prototype/outputs/muscles_six_legs/fruitfly_six_leg_standing.xml` 的完整 XML 结构解析确认不存在 `<sensor>`。所以不存在“某个 XML touch/force sensor 的编号或单位被错读”的当前链路；感觉来自 qpos/qvel/contactForce。肌肉/tendon/site 并不自动成为感觉传感器，外观 marker site 也不是 touch sensor。

`/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/sensory_channels.py` 第 268–278 行将 vibration/nociception 明确输出零并记录 unmodelled。未建模通道、空分组、层级之外细胞，不能靠猜测物理符号补全。第 327–342 行记录 tier 外 lookup 细胞，但遇到不支持的 leg/channel 标签会跳过；本审计未核验全量 lookup 的数据质量，不将此列为已证实的物理测量缺陷。

## 待验证语义与下一步小测试（尚未执行）

1. **接触事件不等于正载荷事件。** `/run/media/sensen/Data2/cell_wound_prototype/run_neural_muscle_closure.py` 第 89 行以接触条目存在置 foot_contact=1，不要求 `force>0`、`dist<=0` 或检查 efc_address。XML 第 34 行存在 0.001 模型长度的 margin。故该二值量应理解为求解器接触列表事件，而非严格几何穿透或有载支撑。建议用极小的离地/接近地面静态样本记录 dist、includemargin、efc_address 与力；在确定合同要求之前，不把 margin 内事件直接判作假阳性 bug。本次未跑这些样本。
2. **合成测量单元测试。** 用伪 contactForce 向量 (3,4,0) 验证输出为 5，交换 geom 顺序结果相同；同腿两接触只累计一次各自范数；非足部/非 plane 接触不进入。无需模拟完整神经体。
3. **地址与符号测试。** 分别在 tibia 的 qpos 地址和 dof 地址注入已知小偏差/速度，确认对应通道值和负电流；7 关节 +δ/−δ 抵消只出现在定义为 mean 的通道。
4. **极短状态快照测试。** 如另行授权实际 MuJoCo 检查，可只编译模型并做初始 forward 快照，验证 30 个 geom、42 个 hinge 的编译后映射及首帧接触。若要查更新时序，再做单步前/后 forward 对照；不需要长跑或参数扫寻。

以上都是下一步有界验证建议，不是已完成测试，也不保证能解决站立。当前无足够证据要求父代理因感觉物理 bug 改动核心实现。
