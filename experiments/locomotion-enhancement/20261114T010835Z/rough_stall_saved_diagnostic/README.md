# 粗地形停滞：保存数组离线诊断 ONLYCODE

交付目录：`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_stall_saved_diagnostic/`。

本目录服务 S2/S5。准备阶段仅执行 stdlib 源码/AST、JSON/XML 和 ZIP NPY headers 检查；没有读取数组 payload、导入 NumPy/Torch、载入权重、构造 Core、运行 NN、积分物理或渲染。没有修改旧实验、权重、`target_feedback_order_repair_v4`、主训练84或主 tracker。92项数值预检通过不能证明身体运动改善，也不能证明已找到唯一卡阻原因。

`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_stall_saved_diagnostic/diagnose_saved.py` 是父以后独占资源时的入口，只计算已有数组。它要求两个显式执行旗标和准确125712环境；只有 NumPy 延迟导入，没有项目 runtime/模型/Core/物理模块导入。父需先完成训练及其关键审核，再串行运行本诊断。当前不得执行。

准备文件：

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_stall_saved_diagnostic/PREREG.json`：窗口、公式、阈值、缺失字段和解释边界。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_stall_saved_diagnostic/SOURCE_SCHEMA_BINDINGS.json`：实际两保存目录完整 NPZ headers、JSON/XML/旧源码 SHA、XML 推导的 actuator→joint→qpos 映射、原地形及归一体长来源。NPZ 的 SHA 来自保存 RESULT，准备时没有重读 payload；父执行前才实际核对整个 archive SHA。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_stall_saved_diagnostic/bind_schema.py`：stdlib 元数据生成源码。FIRST READY 后不重跑覆盖绑定。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_stall_saved_diagnostic/check_codeonly.py` 与 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_stall_saved_diagnostic/CODECHECK.json`：静态编译、source/schema/边界检查；静态通过不等于数值或行为通过。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_stall_saved_diagnostic/INDEPENDENT_STATIC_REVIEW.md`：独立审查发现、修正及剩余验证边界。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_stall_saved_diagnostic/READY.json`：首次冻结的全部文件哈希与准确后续命令。

父后续准确命令（现在未执行）：

```bash
/usr/bin/env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /usr/bin/timeout --signal=TERM --kill-after=2s 90s /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_stall_saved_diagnostic/diagnose_saved.py --execute-parent-saved-arrays --parent-resource-released --output /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_stall_saved_diagnostic/RESULT_SAVED_DIAGNOSTIC.json
```

资源门：单线程环境，90s wall timer，1500MiB address-space hard limit，结束核对峰值 RSS≤1500MiB；全程每 archive 所有字段做 finite/shape/dtype 检查、仅保留所需字段，两 case 逐一处理。成功输出含 actual source hashes、5799 preaction index、native dt12.5µs、0updates/0Core/0NN/0integration/0render、时钟/外力/窗口质量检查以及 wall/RSS。静态检查尚不能认证90s/RSS预算实测成功；预算或输入漂移失败时保留失败事实，不改预算后冒称原门通过。

预期新结果路径：`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_stall_saved_diagnostic/RESULT_SAVED_DIAGNOSTIC.json`。文件必须不存在，使用 exclusive-create；此文件当前不存在。父运行只允许新建此类诊断结果，不覆盖任何历史产物。

窗口为 `(5.8,15.4]`、`(15.4,25.0]`、`(25.0,34.6]`、`(34.6,44.2]`，每窗9600 postsamples与1920 neural decisions。38.4s 数据总44200行、V2参考25000行只覆盖前两窗，后两窗明确 `NOT_RECORDED_FULL_WINDOW`。点位进展用窗口前一帧作边界，首窗 index5799=time5.800；同时报告约9.600s实际边界间隔与约9.599s postsample首末间隔，slip沿原 postsample 首末 COM 公式。原全ANN指标约19.199s/38.399s保存在 engineer reference，不改分母。

实际 XML 证明 Thorax 是 world 的直接 child 且有唯一 freejoint，因此 qpos[:3] 是 Thorax frame origin；`COM_thorax_subtree` 是整个飞行体子树 COM，二者分别报告。Head 原点也独立报告。所有坐标/力/速度为 native；绝对 SI `NOT_ESTABLISHED`。体长使用旧保存七节投影长度2.9024054918026523 native作固定工程参照，已比较保存 compiled mesh/body几何哈希；这不是本次重新测量或 SI 标定。

诊断同时检查净前进/回退/路程和 yaw，足端实际载荷/切向速度/卸载与净空换脚，以及 target边界驻留/target−q。源码 joint 顺序按本体关节先于子体解析，并与保存 `jnt_qposadr/jnt_range/actuator_trnid` 哈希一致，不能按XML文字顺序或 actuator index直接偏移7。保存 `actualctrl` 和 `actuator_force` 是1ms端点样本，仅报告其统计和端点 XML ctrl-bound 驻留；不能用末态q/qvel推导原生子步精确 PD、峰值或cap始终/从未达到。

全时序 component unload→clearance→reload 后再分窗，避免每窗 reset 人为少计。component 采用原数值载荷/净空/10ms加载重载与5ms卸载门，但明确不是原 strict completion（不加advance/timeout门）。原 strict `RESULT.autonomous_cycles` 事件、.03advance/4s timeout/config完整保留，按 completion时刻 `(left,right]` 分配且另给fullycontained计数。两套不可混用或升级原FAIL。

按保存1ms数据可真实分窗的是 registered/plane 总接触数。逐block native身份只有whole/allANN汇总，所以逐窗逐block接触写 `NOT_RECORDED`；足中心/射线高度和足球前表面邻近只能作几何证据，不能分配native接触事件或证明卡边因果。障碍首缘1.35、中心z.02、顶z.04取自实际XML，无移动障碍或调摩擦；当前18球足、无adhesion，不能冒称钩挂已实现。

每脚 registered-slip窗口同时输出 loadweighted绝对切向速度分子、净 COM XY速度分母和路径速度。近零净速度≤1e-9或无载荷则ratio为null并标UNDEFINED，不写NaN/Infinity、不隐藏clamp。停滞或返回压低净速度分母，可以令比值增大而所有脚绝对切向速度变小。原 `autonomous` 的2.086236178710802与 registered的2.074533582371769是不同口径；V2 registered1.414491304等原值及FAIL完整保留为 `orig_autonomous` / `reference_registered`。用户接受更松slip不等于抹除旧FAIL，也不使旧slip.5成为本诊断判断卡阻根因的唯一条件。

FIRST READY 后所有交付源码/协议/绑定/检查文件冻结，不再修改。后续若需改变指标或修正代码，另建版本目录，不修改此冻结包，不扩成教师、训练、肉体新模型或brain/neck集成任务。
