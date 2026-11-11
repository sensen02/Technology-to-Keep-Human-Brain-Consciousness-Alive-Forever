# LEFT 有限 SE2 单变量候选：CODEONLY 交付记录

对应 S4/S5。本次仅代码准备与 CPU1 纯 NumPy 检查；父保持唯一 neural_right heavy 任务。未运行 Core、物理、训练、渲染、子 agent；新轨迹与行为资格均未评估。

- 预登记先写：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_finite_twist/PREREG.json`。
- 单条件固定：LEFT +0.06 rad/native s、speed_scale1，1s settle +4.8s prime +9.6s diagnostic，1s smoothstep。
- 主要变化：仅 sample 水平目标有限共同 SE2 映射及其解析链式 velocity。旧 teacher、.36 stride、.8 step、.12 lift、原 measured order/shape/落点时间、z/velocity_z/stance/phase、IK、物理与全部门不变。新类明确只支持 speed_scale1。
- 速度：theta_dot=kdot*s+k*sdot；平移导数包含 dT/ds 与 dT/dk。theta 绝对值小于1e-3 使用解析 Taylor 稳定项。velocity 只作诊断，真正 teacher qdot 仍是原 IK q_des 差分。k=0 返回整个原 sample，精确不变。
- 假设边界：共同有限 SE2 twist 消除旧 rest 切向一阶近似的几何误差，是数学动机；尚未证实是旧 LEFT 滑移/周期/非足失败的物理根因。
- 真实 rest：只读 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/attempt02/left_yaw_p060/TRACE_1MS.npz` 的 `relativefoot[999]`，物理时刻1s，与 StaticTeacher 构造前同次 settle 后 measure 来源相同；无 Core/FK。原 trace SHA256 `abc43f1af398ecc203a4a034b3b863b37a974214c4a6e2dca039e945245e87ed`。
- 运行前完整冻结：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_finite_twist/SOURCEHASH.json`，1501 文件，SHA256 `1b22a02f29cb319ccc313411ace0a95c19857ae7ae0dcf3756e2a189150f7290`。继承旧 identity 所列源码、XML/资产/环境及新增源码、父诊断/审计、基准模块、实际 rest trace 和测量参考。所有新入口先检查 ROOT 与 hash，物理输出存在后禁止重新冻结。
- 数学预检：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_finite_twist/PREFLIGHT_NUMPY.json`，27/27 PASS，CPU affinity [1]，exit0。零 yaw 全2880样本所有字段 exact；prime exact；全窗非水平字段 exact；s=0 exact；±k 镜像；解析速度包含 kdot 的中心差分；共同 stance 脚距；实际 WaveReference 五条 stance 脚距均通过。
- 数值：h=1e-6s，合成链式速度最大误差6.165111754441455e-10（tol2e-8），实际 reference 240时点最大误差3.5083087546183833e-9（tol3e-6，筛掉 baseline 导数差>2e-7 的插值结点邻近点，本次排除0）。实际 stance 脚距误差4.440892098500626e-16。镜像误差0。常k时不同脚允许固定不同 s_i，脚距随共享 ds 不变；不把不同 s_i 的瞬时脚距错误等同于 rest 脚距。
- 入口与语法：7个 Python 文件 AST parse；默认诊断返回 CODEONLY_READY_NO_PHYSICS；新 audit 无结果返回 NOT_RUN，均 exit0，且无 physics/mujoco/torch/static_teacher/student_wave 导入。没有编译缓存/临时测试文件。
- 额外只读旧格式核对：原 trace root quaternion 重建 yaw 与保存 heading 误差1.2212453270876722e-15；原 foot support 实际体重常量24.446354755878005与纯数组审计相同。exit0，仅证明旧数组接口约定，不是新轨迹审计。
- 诊断适配：新 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_finite_twist/diagnose.py` 不复制父完整源码，绝对路径加载已冻结父 one/轻量 PeakRecorder；显式重绑并检查新 ROOT 与新 finite make_teacher，仅调用一次 LEFT，绝不调用父 main/sweep。默认不运行物理；必须显式 --run-physics；输出已存在则拒绝覆盖。真实运行后另存实际 REFERENCE/rest、完整1ms/5ms、yaw、identity、native摘要及失败/准入结果。
- 新 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_finite_twist/verify_saved.py` 纯保存数组审计，不构造物理；复算 worldX、逐腿 slip/分母、COM、足载、原五门、root quaternion yaw、逐腿周期计数与 all6 标志、严格 nativeNF0/30°/完整窗准入；核对 native摘要与身份/数据hash。完整性PASS和行为FAIL分别如实记录。只有 native peak 摘要，不能伪称独立重建原生轨迹或全D1—D8。
- 门槛保持：worldX≥.05、slip≤.5、COM≥1.20、support≥.90、原非足门≤.20W、严格post nativeNF=0、all6每腿≥1周期、实际LEFT yaw≥30°、无提前停止；不更换投影、不降低门槛。完整性检查通过不代表行为准入。
- 本次所有工具执行exit0；没有后台作业，无需清理任务临时产物。物理/训练/渲染/实际新轨迹审计均未执行。

父稍后在 neural_right heavy 任务结束后顺序启动的建议命令（本次未执行）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_finite_twist/diagnose.py --run-physics > /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_finite_twist/RUN.log 2>&1
```

然后用独立进程审计（本次仅运行无轨迹 NOT_RUN 入口）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_finite_twist/verify_saved.py > /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_finite_twist/VERIFY.log 2>&1
```

若 source hash 漂移立即停止并说明，不静默重锚；若新候选行为失败保留所有证据，不自动扫参。当前代码交付完成，等待父顺序物理。
