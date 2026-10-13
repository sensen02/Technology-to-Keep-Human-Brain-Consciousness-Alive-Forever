# 神经—肌肉闭环初始化审计：原始 round37 与显式限位修复不是同一初态

## 范围与结论

本次为人工实施的源码/已有产物审计，不是成品功能，也不是新的动力学实验。仅用标准库 XML/JSON 解析及 NumPy 读取已有 NPZ 的 `seated_qpos`；没有导入 MuJoCo、编译模型、运行模拟、修改 canonical XML 或 runner。早期反馈实验的源代码冻结保持不变。

**已确认：当前闭环使用原始 round37 offsets，接触 gate 确实执行并能抛错，但不检查关节限位；当前已发布初态有 5 个 hinge 超出 XML 范围。姿态搜索的 `baseline_original` 已是显式修复后的姿态，不是原始 round37 姿态。**

这证明初始化有效性的合同需要分层标记，不证明 XML 错误，也不证明初始限位反力一定主导随后失稳。不能把“接触合格”写成“全部限位合格”，不能静默裁剪以掩盖历史基线。

## 1. 实際初始化调用链与 gate

- `/run/media/sensen/Data2/cell_wound_prototype/run_neural_muscle_closure.py:37–70`：加载 canonical；调用 `r39.apply_round37_pose`；`seat_root` 只求根高度；清零 `qvel/act/ctrl`；`mj_forward`；调用 `cp.stance_facts(model, data).as_dict()`。
- `/run/media/sensen/Data2/cell_wound_prototype/run_fix_round39.py:70–86`：从 `model.key_qpos[0]` 开始，逐关节设为 `key[adr] + offset`。没有限位校验或裁剪。姿态 JSON 的 `root_z_mm` 不直接用于初始化。
- 姿态源：`/run/media/sensen/Data2/cell_wound_prototype/outputs/stance_margin_round37/stance_for_reflex.json`。
- `/run/media/sensen/Data2/cell_wound_prototype/engine/canonical_physics.py:344–381`：`seat_root` 求最低跗节到地面的距离为 -0.05 model mm；只改变 `qpos[2]`，不会修复关节角。
- `/run/media/sensen/Data2/cell_wound_prototype/engine/canonical_physics.py:384–438`：`stance_facts` 默认 `raise_on_fail=True`；无足接触、非足接触地面，或非足几何距地面低于 -0.005 model mm 时抛 `StanceGateError`。它检查接触和埋地，不检查 `jnt_range`、`jnt_limited` 或限位约束力。

**没有“算出 gate 却完全不拒绝”的 bug：拒绝发生在函数内部，调用者无需再判断返回字典。** 当前 `StanceFacts` 和 `as_dict()` 都没有 `valid` 字段；不能把其他代码或口头描述中的 `gate.valid` 套到这里。`prepare_body` 不额外检查返回字典，但默认抛错未被吞掉。实际缺口是**限位合同缺失**，不是接触 gate 被绕过。

闭环 provenance 在 `/run/media/sensen/Data2/cell_wound_prototype/run_neural_muscle_closure.py:280` 写 `gated round37 qpos, zero qvel and activation; no LP bias`，只能理解为接触 gate。episode 在同文件 106–111 行再次从相同 seated qpos、零速度/激活/控制开始。零控制不等于严格零肌肉驱动：原生正 ctrl 下界仍保留，此项已有报告明确披露。

另有旧注释问题：`/run/media/sensen/Data2/cell_wound_prototype/run_fix_round39.py:71–74` 仍称模型单位为 cm，与 `/run/media/sensen/Data2/cell_wound_prototype/engine/canonical_physics.py` 已撤回 cm、采用 mm 的说明不一致。此处不影响当前高度计算，因为该字段没有被应用；冻结期间本审计未修改注释。

## 2. 当前已发布 NPZ 的五项越界：无 MuJoCo 加载的独立核对

审计时指针：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/current_run.json`。

本次固定取证运行目录：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/runs/f10c9354-8b21-4faa-b48c-80fee6a5c157`；后续指针更新不改变本报告所指对象。

NPZ：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/runs/f10c9354-8b21-4faa-b48c-80fee6a5c157/closure_traces.npz`。

- NPZ SHA256：`1fc9b255661e3c914377509dafe64f68c3538c9b2bb09846f753793c6c036d36`
- `seated_qpos` shape `(49,)`；原始数组字节 SHA256：`bf34924a0004e6295bd9f30a86cfd65457eca9042f199839c984ac04bf0fad48`
- canonical：`/run/media/sensen/Data2/cell_wound_prototype/outputs/muscles_six_legs/fruitfly_six_leg_standing.xml`
- canonical SHA256：`65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6`，与声明相符。

### 地址与角度的推导方法

XML 第25行明确 `<compiler angle="radian" autolimits="true" />`，所以 XML 范围已是 rad，**不能再乘 π/180**。表中 degree 仅为展示，按 rad × 180/π 转换。

用 `xml.etree.ElementTree` 解析 worldbody；按 body 树先序，先枚举当前 body 直属关节，再递归子 body，而不是直接 `.iter('joint')` 的文本顺序——部分后腿的父关节写在子 body 后面，直接按文本顺序会得到错误地址。该 XML 无 include；一项 freejoint 占 qpos 0–6，随后 42 个 hinge 各占一个地址，合计49。每个 hinge 明确 `limited="true"` 和 range；type 未写时为 hinge。用此映射将 keyframe 的49项加上 round37 offsets，结果与 NPZ **除根 z 外逐位完全相等**，为地址解析提供额外交叉检查。

以下地址均为 **0-based qpos 地址，不是 dof 地址**。越界容差采用既有搜索的 `1e-12 rad`。

| qpos地址 | 关节 | 当前 qpos rad | XML范围 rad | 当前角度 ° | XML范围 ° | 超出量 rad |
|---:|---|---:|---|---:|---|---:|
| 8 | joint_LFCoxa_pitch | 0.754142 | [-0.5783, 0.7375] | 43.209154 | [-33.134149, 42.255637] | 0.016642 |
| 11 | joint_LFTrochanter_pitch | -1.162 | [-3.242, -1.217] | -66.577696 | [-185.752917, -69.728964] | 0.055 |
| 23 | joint_LMCoxa_roll | 0.6355080000000001 | [0.1436, 0.6236] | 36.411926 | [8.227674, 35.729648] | 0.011908 |
| 36 | joint_RHCoxa_pitch | 0.7795000000000001 | [-0.5783, 0.7375] | 44.662060 | [-33.134149, 42.255637] | 0.042 |
| 42 | joint_RMCoxa_yaw | -0.64519 | [-0.597, 0.2745] | -36.966664 | [-34.205580, 15.727691] | 0.04819 |

保存结果 `/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/runs/f10c9354-8b21-4faa-b48c-80fee6a5c157/closure_results.json` 的 gate：根/胸部 z=2.044880332462318 mm，12个地面接触、0个非足接触、`tarsus_only=true`，最深距离 -0.04999999999999996 mm。时长0.05 s。**接触合格与上述五项越界同时成立，不矛盾。** 这些是已有保存数据，不是本次重新测量的接触。

本次未重新编译模型取得 `model.jnt_qposadr`；上述为 XML 树/布局推导和保存数组一致性验证，不应冒充新模型运行测量。当前加载器 `/run/media/sensen/Data2/cell_wound_prototype/engine/canonical_physics.py:267–273` 使用 FlyGym `_load_mjcf`；`/run/media/sensen/Data2/cell_wound_prototype/venv_body/lib/python3.12/site-packages/flygym/compose/fly/musculoskeletal.py:105–114` 表明使用原生 MjSpec 读文件而非重写 main defaults。

## 3. `baseline_original` 实际含义与严格修复链

`/run/media/sensen/Data2/cell_wound_prototype/outputs/workspace_recheck/pose_support_search.json:29419–29442` 明确：

- `provenance.source_label = round37_explicit_limit_repair`；
- 源为 `/run/media/sensen/Data2/cell_wound_prototype/outputs/workspace_recheck/dimensionless_support_search.json`；
- `method.baseline` 写明 exact source best74 **repaired round37 qpos** and **original activation**；
- gate 是 `cp.stance_facts plus strict joint ranges`，另加非足接触禁止、足穿透上限0.06 mm；
- 本轮候选 no reseating、no silent clipping，严格拒绝超范围候选。

对应代码 `/run/media/sensen/Data2/cell_wound_prototype/run_pose_support_search.py:144–166` 明确只接受修复标签，取源 qpos/activation；58–79及182–187行先收集接触和限位失败原因，再拒绝候选，并非仅返回无用的 gate 信息。

修复来源 `/run/media/sensen/Data2/cell_wound_prototype/run_dimensionless_support_search.py:255–270` 的确有 `np.clip`，但它用于**独立、显式命名的候选**，逐项保存 requested/corrected/correction；不是暗改 canonical 或闭环初态。随后293–301行重新求根高度、严格验证范围并拒绝失败项。66–77行验证限位采用1e-12容差。

用同一 XML 映射检查 pose search 的 `baseline_original.qpos`：零越界。相比当前闭环 seated_qpos，仅以下六项不同：

| qpos地址 | 原始闭环 | 搜索修复基线 |
|---:|---:|---:|
| 2（根 z，mm） | 2.044880332462318 | 2.0486760924304157 |
| 8 | 0.754142 | 0.7375 |
| 11 | -1.162 | -1.217 |
| 23 | 0.6355080000000001 | 0.6236 |
| 36 | 0.7795000000000001 | 0.7375 |
| 42 | -0.64519 | -0.597 |

因此 JSON 方法/溯源标签是诚实的，但 `baseline_original` 的短名容易被误读：它指**修复姿态上的原保存 activation**，不是未经修复的 round37 角度。报告应完整标注“原始 round37 接触门控闭环（5项越界）”与“显式范围修复、重新 seating 的机械搜索基线（0项越界）”。搜索还使用优化/预置肌肉 activation，其结果不能直接作零 activation 神经闭环的初态对照。

## 4. 接下来怎样区分限位初始瞬态与持续失稳（本次未执行）

1. 保留冻结运行与原始基线，不改变 XML、原生限位、contact/solver/ctrl 下界，也不暗中裁剪。新增独立结果标签和初态范围报告；只做范围元数据报告时，不应让旧基线莫名无法复现。
2. 做显式2×2配对：原始 qpos/修复 qpos × normal/spike-output-silence。相同网络、权重、通道、时钟、神经状态与刺激协议；两种姿态均从 `qvel=act=ctrl=0` 开始，修复只包含记录的五项角度与相同 seating 规则。**不能加载机械搜索优化 activation 或添加 PD/CPG/外部稳定器。** 新增扰动会改变问题，初始比较宜不用机械 push；若保留人工触觉刺激，须四组一致并明确披露。
3. 在0、首步、0–1、1–5、5–15、15–50 ms 分窗记录初始/后续限位 violation、active limit constraint 数量和按类型分解的约束广义力、接触力、qacc/qvel、胸高和肌肉激活。未来加载模型时再确认 `jnt_limited/jnt_range/jnt_qposadr`；本次静态检查不能给出限位反力数值。
4. 原始与修复间变化若集中于早期，并伴限位约束力/角加速度明显消退，仅支持“初始化限位瞬态影响基线”。仅比较胸高或最终逃逸时间不能证明力的来源，因为修复也改变几何、接触和根高度。
5. 修复后仍快速跌落、仍缺乏 normal 相对 silence 的优势，提示不是只靠修复初态即可解决；须继续排查持续支撑/反馈延迟/映射，不能宣称 XML 不可行。修复后改善也不是自主站立证明；需更长窗口、重复与恢复性验证，但应待当前早反馈任务释放共享运行保护后再启动任何模型工作。

**交付边界：本报告只新增文档；没有行为切换、模型加载、模拟、裁剪或源文件修复。**
