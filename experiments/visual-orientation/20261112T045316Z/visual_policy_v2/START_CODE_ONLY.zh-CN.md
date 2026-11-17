# 墙体／遮挡视觉域候选2：父执行说明

CODE_ONLY准备完成。真实仅stdlib check_codeonly.py exit0/25checks，全无Torch/NumPy/MuJoCo/model/数组/forward/训练/Core。本文件为运行说明，不纳sourcefreeze；所有已检查核心自CODEONLY_CHECK后停止修改。

父唯一CPU1托管命令：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/visual_policy_v2/run_candidate.py > /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/visual_policy_v2/PIPELINE.log 2>&1
```

流水线完全复用候选1的冻结串行流程：实际preflight（缺DATASET才生成；old24bitexact/18pair/新wallweights检查后才hashfreeze）→fixed500训练→0/250/500独立保存审核→只final500的新12+8controls验证→新进程验证重放。JOBS逐stagePID/log/exit和CURRENT_STATE动态记录；失败停止，先读实际EXECUTION.log，不盲重启、不覆盖已完成数据/权重/结果、不挑250或加500。

新library为 `/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/frame_library_v2/`，kind2，126framepairs=旧112真实RGB原值+14新V2/V3实际RGB。旧V0/V1是development，新common严格区分，不修改旧112库或任何候选1source/hash。核新library独立VERIFY的library/rgbSHA/所有checks/invisibleadmitted/真实physicalidentity，读取RGB后再检查逐帧hash、真实完全遮挡与旧112像素前缀。

同CNN8/8→各眼8feature有序16→GRU32，20ms捕获/发布延迟、5msheldinput推进、9.6s完整历史，目标方位只offline标签，不进入actor。嗅觉不接，运动不加载不改，全部14visual参数从实际f1a...严格初始化，freshAdam1e-4，500更新，batch4从18pair无放回选2展开，36序列每条边际1/9。

原24训练输入/目标/weights/time全部精确复用；新增8左右遮挡和4纯墙序列。遮挡相对第一张已发布墙图做ON/TRANSITION/RECOVERY/BASELINE质量.30/.20/.40/.10；纯墙真实off-before-on与wall-after-on各总质量.5。无runtime墙体检测规则或黑图替换、无减bias、死区或reset。

新验证V2pose5几何.85/±.32、V3pose6几何1.05/±.48，旧V验证不再fresh。原empty≤.02、center/wall≤.05、发布撤除图后.5—1s恢复RMS≤.05、方向≥.9/响应2s/反转门完全不变。固定旧帧与只交换眼顺序只作诊断，不能要求固定帧必turn0或只交换眼必反向。所有功能门全通过且独立审核后父才可新立最多6真实身体视觉回合；本流水线不启动身体/视觉导航或融合。

实际fullbatch4双图比较与RSS≤1800MiB gate沿用，未测不得声称资源通过。训练与生产浮点比较tol1e-10；保存审核相同生产shape要求bitexact，不为过门改变精度或容差。新增数据域和总objective分布共同变化，不声称已证明首次失败唯一原因；保留候选1失败和所有checkpoint。
