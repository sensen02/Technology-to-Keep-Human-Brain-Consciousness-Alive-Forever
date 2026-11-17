# 有限行为长历史修复：父代理执行入口

当前仅CODEONLY，11项标准库检查已实际通过，0重模块导入、0数据生成、0模型加载/forward/训练/Core/物理。真实梯度/保存/长序列功能结果均尚未产生。所有核心源位于本目录，旧AB、sensorbfd6与motor789f未修改。

父确认当前render/plot全部结束、唯一heavy资源释放后，建议用一条managed后台job启动：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T234150Z/run_candidate.py > /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T234150Z/PIPELINE.log 2>&1
```

该调度器只用标准库，每次最多spawn一个重子进程，顺序：

1. `preflight.py`：自动生成24train/12validation×1920步数据并冻结；实际1920batch4生产/scan/全部8梯度/unitAdam/目标公式/输入边界等预检；motor仅rawSHA、不载入模型。
2. `train.py`：从严格sensorbfd6初始化，不恢复旧Adam；freshAdam1e-4/batch4/固定1000更新/全1920BPTT，initial/500/1000保存，constructor后恢复真实RNG；不看验证结果、不选500、不追加预算。
3. `verify_saved.py --step 0`、`--step 500`、`--step 1000`：三个独立新进程分别actualproduction24train重放turn/latent/hidden逐位一致、Adam/RNG/调度/再生数据/source身份审核。
4. `evaluate.py`：只用final1000对12冻结配置一次长序列验证，各on/noodor/equal/reversal/off门分开，未通过不改门或试另一个save。
5. `audit_validation.py`：独立新进程重新运行12序列，原输出/latent/hidden32逐位核对、每个功能flag与MSE独立重算。

每个stage准确命令、PID、日志、开始/结束/退出码保存在JOBS.json。CURRENT_STATE.json只动态摘要；两者不纳入训练身份。整个managedjob必须由父collect，避免跨session句柄问题。本子不额外启动任何heavy。

也可父逐条同环境python -B运行上述入口。首次preflight会生成FROZEN.json；它包含所有实际核心评分/数据/训练/保存源码、PREREG、生成数据/配置、原sensor/motor哈希与几何证据，初始化还严格核旧sensor自身source/data身份。新产物存在时拒绝盲重启；如发生程序失败，应先检查已完成checkpoint/日志，不重跑训练或覆盖失败。补审只新名/新scope。

零噪声noodor配置重复是有意保持六motif均衡，不是多次独立试验；验证报告同时列配置数和不同受体历史数。oldAB此前失败是开发历史，不冒称freshholdout。

目标latent辅助loss只在优化时使用，不在部署中设ifodor、deadzone、隐藏重置或输出门。离线教师有环境包络/方位标签，实际网络输入仍只有两受体数值和连续32记忆。motor789f完全不动、不加载、不训练。

若VALIDATION/RESULT.json与VALIDATION/VERIFY.json都显示all_targets_pass，仍只是离线长历史行为通过；后续最多6个新sensorSHA的完整B身体回合由父另登记与调度，原A3纯motor可按身份复用，但旧sensor的B_noodor不可当新权重匹配对照。若未全通过，保留所有失败并诊断，不自动进入物理或扩训练。
