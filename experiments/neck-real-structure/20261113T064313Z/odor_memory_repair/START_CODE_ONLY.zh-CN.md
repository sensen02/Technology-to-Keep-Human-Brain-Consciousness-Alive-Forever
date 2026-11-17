# 嗅觉神经记忆撤除修复：代码就绪，尚未实际训练

实际标准库检查50项全部通过exit0，无数值模块导入、数组payload、模型加载、forward、训练、Core、物理或渲染。本子只写新目录，旧66f、旧D确认失败、045302实际14身体／70诊断及视频全部保持不变。

父唯一CPU1实际执行入口：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T064313Z/odor_memory_repair/run.py --execute-parent-odor-memory-repair > /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T064313Z/odor_memory_repair/PIPELINE.log 2>&1
```

固定7阶段：preflight／train／verify0／verify125／verify250／evaluate／audit。失败保留且停止，不追加步数、不缩有效批、不盲重跑／覆写。父应在嗅觉校准final250独立通过后，才把实际SHA绑定到融合和身体阶段。

公开API位于：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T064313Z/odor_memory_repair/common.py`

- FORMAT=`ODOR_NEURAL_OFFSET_REPAIR_V1`。
- `initialize()`→(odor,parentpayload)，严格加载旧66f实际step500，全部8张量精确一致。旧common.initialize代表更早fc3a，当前代码没有误用它。
- `load_checkpoint(path,digest,final=True)`→(odor,payload)，直接本common实现，严格step0／125／250、final250、curve长度step+1、8Adam更新计数、原类identity和ALPHA及finite权重、源／数据SHA。没有`__import__('model')`依赖，加载器保存并finally恢复common／model／steering_behavior／sys.path。
- 原SteeringBehavior.step／mean_step／initial_state及identity不改。payload中repair_lineage标明学习修复，原identity中历史说明字符串保持来源，不当现有调用禁令或生物有效性证明。

数据：旧B6实际receptors(2880，ANN960:2880)＋最新7actualactor受体(3080，ANN1160:3080)，逐步冷h0重放旧66f后必须与记录h32前后／rawturn逐位相同。再从actualO_left受体史构造.9／2.6／4.8秒后置零的3条离线撤除样本，切断前受体和原神经输出前缀精确相同，不是新增物理。

原66f神经输出作为教师。actualO_left_odorcut(1.6秒)及3增强样本，仅监督目标在cut后.25秒线性衰减到0；B_noodor／O_noodor监督0，其余包括Bpulse自然尾保持原教师。该时间／切断信息不输入NN，不形成运行规则或隐藏门控，不重置记忆。loss为平衡quiet／active／transition turnMSE＋.001原8张量anchor。神经ALPHA／tau100ms不变，受体源／滤波／pending因果边界不改。

固定250，freshAdam1e-4，有效16＝2×微批8，全1920每条不detach，两个loss各/2，整次更新仅一次clip和Adam。实际预检独立Python full16与JIT、manualloss、8梯度、full16 vs2micro8损失／梯度／Adam状态、所有8计数1，以及别名哨兵／源不变／RSS1800通过后才训练。wall1800／RSS1800不扩额度。

保存payload.step0,125,250，curve step+1，TRAIN/RESULT.json提供checkpoint／sha256／updates250；TRAIN/VERIFY_000250.json all_passed／checks／sha256。EVALUATION/RESULT.json含checkpoint_sha256／rows／source_hashes／data_hashes，VERIFY含integrity_all_pass／checks／result_sha256／checkpoint_sha256。JOBS7个exit_code，FROZEN source／data哈希供下游严格绑定。

最终16序列都是已知输入／离线增强，撤除后.5—1秒RMS仅描述，不能据训练内通过就宣布新身体感官恢复。未来actualOcut的RMS≤.05和身体影响仍需独立物理对照；本目录不证明食物到达、停车、天然偏好、完整生物嗅觉或意识。
