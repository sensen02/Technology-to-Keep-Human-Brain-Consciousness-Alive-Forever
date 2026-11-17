# 因果目标首行shape最小恢复：代码准备完成

原122在实际8RGB／12odor重放成功提取之后、训练前被shape门拒绝。实现错误是concat使用1920行zero而非首1行，导致teacher3839。原源、错数据、日志、CODEONLY报告全部保留；不重复已完成感官提取，不重跑任何物理。

新恢复只读取原提取的body111／modalities／current_desired／mean／std，逐位核这些输入未变，重新生成target＝首1行零＋desired[:-1]及同权重函数；新DATASET／RECOVERY_DATA_PROOF／FROZEN均写本新目录。原teacher3839形状与NPZ SHA绑定，新teacher1920／首0／后续移位与source输入精确一致是跑前必查门。

所有模型、扫描、有效27／微9×3累积损失、400更新预算、保存0／200／400、独立重放和指标审核均复用原codeobject，科学参数和目标含义不变。新FORMAT仅记录恢复谱系：SOURCE_NECK_VISUAL_OLFACTORY_FUSION_RECOVERY_V1。

父新入口，勿重跑原122命令：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T045302Z/sensory_fusion_recovery_v1/run.py --execute-parent-fusion-calibration > /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T045302Z/sensory_fusion_recovery_v1/PIPELINE.log 2>&1
```

实际标准库38项准备检查全true／exit0，未导入数值模块、读取数组payload、加载模型／Core、运行训练／物理。检查不仅看字符串：抽出实际causal_target AST用标准库list替身执行，1920个desired生成1920目标，首0／后1919移位exact；另实查arrays／data.main／common.initialize／frozen／load_checkpoint／stage.main及orchestrator的HERE为新恢复目录。data.arrays为显式本地实现，不能读回原错误dataset。父首次review发现路径风险在新scope未执行前即已纠正，不是第二次实际训练失败。

sourceledger包含原所有核心、错dataset、已完成SENSORY_REPLAY、122失败日志及新恢复proof，不修改任何原科学文件。新body若后续获准应指向本CAL和本FORMAT，最终权重SHA仍未知，必须父实际final400及独立审核完成后绑定，不在源码猜填。

API沿用：brain_step(AN,modalities[B,2]=[vision,odor],h)，load_checkpoint(path,digest,final=True)→(bridge,payload)，load_sensory()→(vision,odor,vp,op)，initialize()→(newbridge,parentpayload,old2inputparent)。336神经状态、filter/cache/ports分列。旧D延迟cut确认失败仍保留，无嗅觉食物到达／停车／天然感觉选择或真实电生理结论。
