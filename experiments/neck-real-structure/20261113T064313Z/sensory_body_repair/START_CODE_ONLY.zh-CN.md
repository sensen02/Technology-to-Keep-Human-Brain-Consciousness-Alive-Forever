# 嗅觉记忆／融合脑修复候选：同条件身体重测父执行说明

当前仅ONLYCODE，实际86项标准库检查全true、exit0，没有数值库、数组、模型、Core、渲染、forward、训练或物理。所有执行源自最终检查后冻结，本文纯说明不纳source。

唯一变化为已独立训练的嗅觉final250与融合脑final400来源／路径／格式，未知权重SHA直到父实际完成两个训练流水线与全部保存／最终指标审核后才绑定。不能拿本代码准备检查当模型资格。新calibration目录：

`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T064313Z/odor_memory_repair/`

`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T064313Z/fusion_aggregation_repair/`

父需核两实际7stage JOBS exit0、EVALUATION/RESULT与VERIFY的checkpoint_sha256／result_sha256／integrity_all_pass／checks、TRAIN/VERIFY_000250或000400的all_passed／sha256／checks，以及FROZEN全部source／data哈希。融合的PARENT_ODOR_BINDING必须指向同一个实际新odor250。实际输出schema若与草稿约定不同，先read实际source并另做最小准入适配，不能伪造科学JSON迎合body。

1. 实际双权重审核完成后父绑定：

```bash
python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T064313Z/sensory_body_repair/prepare.py --bind-completed-calibrations --brain-sha256 ACTUAL_FINAL400_SHA --odor-sha256 ACTUAL_FINAL250_SHA
```

2. 父唯一CPU1实际零Core预检：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T064313Z/sensory_body_repair/preflight.py --parent-resource-released
```

3. 实际预检通过后父managed后台串行固定30阶段：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T064313Z/sensory_body_repair/orchestrate.py --parent-resource-released > /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T064313Z/sensory_body_repair/PIPELINE.log 2>&1
```

复用的原直接runner／audit／preflight／summary／review／orchestrate AST逐节点完全相同，原数值常量不动；原OdorField、BrainNeckLoop全部方法保持同codeobject但完整重绑定新ROOT／CAL／ODOR／当前实际odorSHA。相机close及capture helper新ROOT已标准库实际检查，不会写旧证据。加载器只调用新的fusioncommon.load_sensory和strictload_checkpoint，finally恢复model／common／steering_behavior别名和sys.path；不沿用66f硬码作为新odor身份。

原14条件、30stage、1+4.8+9.6、336状态／3080神经／15400物理ms／770实际RGB／1232000原生步、Head代理、Gaussian世界场、ClampFilter/pending、20ms视觉发布、2ms颈部发布／5ms消费、预热h0／DN0／teacherprimecmd0、不reset／clearqueue、视觉旧阈值及原odor工程门全部保留。部署不添加截断／清零／减偏置规则；感官切断仍是原声明的工程输入干预。功能FAIL和原滑移FAIL保留，不覆盖旧模型或旧回合。

新case.kind：SOURCE_NECK_MULTISENSORY_BODY_REPAIR_V1。

实际canonical summary：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T064313Z/sensory_body_repair/SENSORY_BODY_SUMMARY.json`

实际summary.kind：SOURCE_NECK_MULTISENSORY_BODY_REPAIR_SUMMARY_V1；pairedreview仍PARENT_READY_REVIEW.json并绑定summarySHA。新可视化以后必须按该实际schema，不将旧kind叙述当权威。

准备过程中一次纯代码source比较误用被继承导出的old变量而指向更老visualmodule，报AttributeError；该失败已保存CODEONLY_FIRST_ATTEMPT.json。改为显式私有_PARENT_RUNTIME来源比较后86检查通过，0数值／模型／物理，无已完成身体重跑。旧所有source和科学结果原字节保留。

长度为模型原生量，不假称SI米。真实神经元身份与工程传输／学习型适配分层；此重测不证明原生感觉调谐、完整颈部、电极、融合优越或食物到达／意识。
