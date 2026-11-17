# 气味刺激视频与同步图：CODE_ONLY最小方案

本文件仅设计，尚未运行Torch、NumPy、MuJoCo、模型、渲染或任何重任务。核心worker持有唯一heavy权限；最终schema由父转交确认，不预猜字段，也不修改任何冻结源码或主tracker。

## 复用边界

已只读以下历史实现：

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning/visual/render.py`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/visual_right/render.py`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T072012Z/render.py`

旧实现经过三层source替换已不适合再加一层。新visual采用直接短函数：contracts.py只负责输入契约/身份，render.py只负责保存姿态渲染，plot_sync.py只负责已核验的六条件曲线。公共文件读取/哈希可放在contracts.py，不建立泛化schema框架、不通过可任意映射字段的JSON绕过严格新schema。

重模块全部延迟到父明确允许后的运行入口；默认入口仅打印CODE_ONLY，不读取未来轨迹、不构造Core。只允许stdlib AST/mock契约检查；当前不执行数值代码。

## 已确认基础schema

- trace_1ms.npz和geometry_1ms.npz保留旧combined字段。
- neural_decisions.npz中hidden_before/after为motor256；behavior_hidden_before/after为32；full_hidden_before/after为288。新审核必须显式覆盖完整288链，不能只信旧motor256链。
- 受体输入、filter_before/after、pending响应分开，确切键名和身份布局待父确认。
- RESULT/HANDOVER/IDENTITY/runtime_geometry以及独立VERIFY.json使用新schema。完整性与行为/旧工程门分列，工程失败仍可诚实渲染。

## 视频固定要求

每条件使用该回合同步的15400个1ms保存状态；索引0,20,...15380共770帧，50fps实时15.4秒，完整1s稳定+4.8s教师预热+9.6s ANN。不得拼接不同回合，不能把浓度钳制写成导航。

启动前要求：新schema版本、condition/run绑定、独立审核integrity_all_pass且全部checks真、完整288神经链审核、运动和行为checkpoint身份/哈希与实际identity一致；输入/source/data/runtime物理前后哈希不变；HANDOVER没有身体/神经reset；实际施加动作与神经输出在ANN阶段一致。严格工程FAIL保留，不能因为FAIL拒绝展示或改成成功。

图像只从保存qpos/qvel/time恢复到渲染私有Core的MjData，mj_forward用于恢复派生几何；mj_step/step1/step2全部替换为抛错函数，integration_attempts须0。此处qpos写入是轨迹可视重建，不冒称运行时根轨迹写入。m.nq49/nv48/nu42、71个mesh全部alpha1。原compiled物理数组通过冻结identity.verify_runtime在前后验证。只改内存visual framebuffer至960×640并恢复，不改变XML。

固定相机根据完整保存COM及采样帧所有mesh边界球确定一次lookat与distance，不跟踪移动、不放大yaw。每帧segmentation测全身mask边界，保证不被overlay裁切。教师与ANN控制来源按实际source值显示，早停/不完整回合不可伪造全15.4秒。

所有视频overlay用英文，避免依赖中文字体；另附中文交付说明。顶部显示实验条件/控制来源、实际time/倍率、ANN elapsed=time-5.8s（5.8s交接定义0；交接前显示PRE-ANN而不是负自主时长）、CONCENTRATION_CLAMP / NOT NAVIGATION。底部显示当前实际受体L/R、rawturn及接口命令t、实际yaw相对交接时刻、原滑移和原工程结果。若是高层命令探针则明确COMMAND PROBE / NOT ODOR RESPONSE；不把刺激标签用于产生动作。

受体与turn映射：用decision保存的实际查询时刻，选择time<=视频当前time的最后一个样本；绝不借用下一decision输入。settle尚无神经decision时显示N/A，不能伪造0受体。几何yaw用当前trace时刻，刺激阶段标记来自已冻结刺激包络的环境记录，不从受体推测事件时刻。

编码以单线程ffmpeg rawvideo pipe逐帧写入，不积累770张图。结束ffprobe核验770/50fps/15.4s，并完整解码770帧；从解码帧选稳定、预热末、交接、出现后、撤除/反转、末尾画面构成蒙太奇。保存FRAME_MAPPING.json包括frame→traceindex/time、useddecisionindex/time、qpos/qvel/RGB哈希和实际overlay值；VIDEO.json记录输入/outputSHA、compiled验证、0integration、mask边界、可视buffer恢复。父需实际打开蒙太奇并核验桌面hash后才称已交付。

## 六条件同步图

首批B6为noodor/equal/leftpulse/rightpulse/LR/RL，精确文件id仍以核心schema为准。B6没有真实身体receptor-cut条件，不能把noodor标成已建立响应后切断，不能把offline神经cut称为新身体对照。实际受体切断对照明确列为后续C/D待做；报告不从B6得出完整切断因果证据。待核心给出准确6condition id和结果目录清单后绑定固定顺序。每行一个完整回合，三列：受体L/R，rawturn与实际turn，实际yaw相对handover。yaw列同时用虚线显示匹配无气味基线，并报告差值；baseline与当前time轴必须严格匹配，不能插值或任意平移消去漂移。

以实际环境浓度包络shade显示on/off与方向反转：各事件分别标记包络起点和0.2s包络结束（例如ON_START/ON_RAMP_END、OFF_START/OFF_RAMP_END、SWAP_START/SWAP_RAMP_END），不能把受体开始变化时间冒充刺激起点。曲线用ANN相对时间t-5.8s并保留PRE-ANN，受体保留自然衰减，显示全预热和ANN边界。body yaw明确为相对handover的Δyaw；匹配noodor baseline的Δyaw另画虚线，当前Δyaw-baselineΔyaw单独标注，不能只画差值隐藏身体本身的漂移。浓度钳制面板明确NOT NAVIGATION；同时显示条件原工程FAIL/身体稳定/刺激功能结果，不能把原门失败偷换为刺激成功。

同步图只读通过独立审核的保存输出，不重新执行神经forward，不重算物理。matplotlib/NumPy运行仅在父资源释放后，当前只写代码不导入。

## schema待确认清单

1. 六个条件精确id、trial绝对路径清单、stageA探针id与stageB是否同格式。
2. RESULT schema版本、condition、motor/behavior或整策略身份路径/hash、artifact字段与审核版本/check名。
3. HANDOVER 288状态布局字段；full_hidden前后链是否prime期间behavior32全零（旧规则如此）。
4. receptor/rawturn/command字段确切键，pending/filter时序字段与实际浓度/包络字段；probe是否也记录全288。
5. engineering_pass与stimulus/functional flags确切位置、baseline参照绑定路径。

收到后仅补这些显式接口，先给父审阅，默认CODE_ONLY。当前无渲染、曲线、视频成功声明；无输出重复/临时文件待清理。
