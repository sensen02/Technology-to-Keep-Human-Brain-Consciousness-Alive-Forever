# 成对batch单候选：父统一CPU1执行

CODEONLY准备完成，12项标准库检查实际exit0，0numpy/torch/Core/model/forward/训练/数组/物理。本目录source从CODEONLY检查后保持不变，旧候选源/VERIFY/失败结果未改。

父唯一heavy命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/run_candidate.py > /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/PIPELINE.log 2>&1
```

purestdlib调度顺序：

1. preflight.py自动生成/冻结数据：旧000308全部24train数组（含weights）逐位复用，新V4/V5共12验证配置；继承真实生产/scan/8gradient/unitAdam并加pair_checks真实检查。
2. train.py：strictfc3a精确初始化，samebehavior全部8参数/freshAdam1e-4/fixed500/full1920BPTT，save0/250/500，所有checkpoint保存实际pairIDs与展开episodeIDs、Adam/RNG/curve。
3. verify_saved.py --step0/250/500三个独立进程：实际生产turn/latent/hidden32exact、Adam/RNG和全部500调度重构、再生数据exact；不选择250。
4. evaluate.py：final500对12新配置一次验证，原所有RMS/sign/latency/reversal门保持；新增signedmean/std/abspeak描述，不作输出偏置减法或取代RMS。
5. audit_validation.py：独立新进程实际重放12序列并重算所有functionalflags。

12pair顺序：4个同variant左/右pulse，4个LR/RL，4个noodor/equal。每step PCG64seed20261111均匀无放回选2pair，展开4episode，不额外shuffle/不强制neutralpair；每episode边际2/12=4/24不变。方向配对是改变batch协方差的候选，不能由旧结果认定未配对是偏置根因。

真实pair预检核：旧train所有数组exact；12pair覆盖24各一次、同variant、8个方向pair受体交换/目标反号/权重exact（neutralpair仅目标零、不称输入镜像）；四episode loss/所有8梯度等于两个pair平均tol1e-12；计数证明边际保持，实际500pairIDs/expanded4unique记录重构。PAIR_PREFLIGHT.json纳sourcefreeze，检查值显式bool确保JSON正确。CODEONLY不代表上述真实数值检查已经通过。

同oldobjective/actionMSE+.01latentMSE、fourphaseweights/数据标签/架构/alpha/运动789f保持，无新增runtimeifodor/deadzone/reset/gating或bias减法。原fc3a与V2/V3失败转开发历史，新V4/V5在update前冻结。

JOBS记录每stage实际PID/log/exit，CURRENT_STATE动态排除身份，父托管job统一collect；本子无heavy。执行失败先查看已产物，不覆盖已有TRAIN/VALIDATION或盲重启。offline未全pass不自动进入身体B、不加500、不改lr或选中间checkpoint。身体B工具已有父管理候选，当前不重复实现。
