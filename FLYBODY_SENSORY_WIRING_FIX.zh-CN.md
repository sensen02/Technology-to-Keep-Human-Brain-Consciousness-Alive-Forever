# FlyBody 感觉角色接线修正

**手工研究原型，不是成品。工程接线测试，不是生物学验证或站立成功。**

## 已确认的错误

`/run/media/sensen/Data2/cell_wound_prototype/run_flybody_loop.py` 的 observe 在选择 `claw=tibia` 时，曾令 `tib_v=tib_q`。MuJoCo 带自由关节时 qpos 和 qvel 的地址不同，因此该路径会读取另一个 DOF 的速度。修正为按关节 ID 分别取 `jnt_qposadr`、`jnt_dofadr`。未改冻结的 canonical 感觉编码器、连接、极性或增益。

## 实际扰动验证

新增 `/run/media/sensen/Data2/cell_wound_prototype/test_flybody_sensory_wiring.py`，在实际 published FlyBody servo 模型中测试：

1. 六腿逐腿独立注入 tibia 速度1.234 rad/s，验证角色路径读取正确 DOF，与默认 tibia 路径相同；旧的 qpos 数字地址所在 qvel 保持0，因此测试能检出旧错误。
2. 六腿逐腿对所选 coxa 关节偏移0.125 rad，验证 hairplate=coxa 输出0.125、不同于默认多关节平均，并且不串到其他腿。因此该 observation 层角色不是 no-op。**此结果不证明完整 episode 中电流和身体轨迹必然改变。**
3. 仅给左前腿一个 actuator_force 槽写入-2.5，验证 campaniform 模型代理读取绝对力及模型内归一化，其他腿为0。这是输入接线测试，不是实际张力校准。

独立接线测试：3 tests / OK。联合运行 `test_flybody_sensory_wiring`、`test_flybody_branch`、`test_flybody_sourced_sensory`、`test_flybody_anatomy_audit`：**61 tests / OK**。其中既有 artifact 断言只是兼容性回归，不能使旧动力学结果成为修正后的重算结果。

## 科学界限与历史结果

同时纠正 observe 文档：CxHP8 证据不能推广到所有 hairplate；特定 femoral CS 场证据不能推广到所有 campaniform。all-coxa 与 summed actuator-force 均是未验证的诊断代理，不能称为已建立的逐根解剖映射。归一化使用 weight-share × hip-to-claw distance，不是 actuator force limit。

旧的 `/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_loop/flybody_sourced_sensory_run.json` 等结果没有在本轮重新模拟。凡使用 `claw=tibia` 角色路径的旧结果均应视为**修正前速度接线结果，不能拿来验证修正后模型**。已有结果文件保持历史原样，不覆盖或冒称重算。

## 看图状态

用户修复接口后已实际查看 `/home/sensen/Desktop/cell_wound_prototype/channel_attribute_audit/runtime_sensory_coverage.png`。42格计数、标签与原型/局限说明清晰。此图仅验证注释覆盖展示，非动力学证明。
