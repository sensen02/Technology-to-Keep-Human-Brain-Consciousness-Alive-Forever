# 四段恢复权重单候选：父统一托管入口

状态CODE_ONLY_READY。标准库AST/最小适配检查14/14实际通过；尚未导入Torch/NumPy、生成数据、加载模型、运行forward/梯度/训练/Core或物理。旧234150代码、数据、VERIFY和失败结果全部未改。

父资源空闲后唯一heavy命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T000308Z/run_candidate.py > /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T000308Z/PIPELINE.log 2>&1
```

run_candidate是纯标准库wrapper，复用旧顺序调度，只将保存审核从0/500/1000改为0/250/500。每次一个新process，依次preflight→train500→verify0→verify250→verify500→finalvalidation→independentvalidationaudit。JOBS记录PID/log/exit，CURRENT_STATE写动态状态，不纳训练source身份。父managed后台job负责最终collect，本子没有另启job或任何heavy。

preflight自动调用新generate_data：24旧train输入/target/concentration/time逐位复用，只改8pulse权重；16非pulse权重逐位不变。四互斥集合ON[on+.2,off)、TRANSITION[on,on+.2)∪[off,off+.5)、RECOVERY[off+.5,off+1)、BASELINE其余，各质量.30/.20/.40/.10。新段mask排他/全覆盖/各mass/归一与24train复用在真实预检中验证；不能把CODEONLY检查当作这批数值已通过。

原engine/production/动作MSE+.01latentMSE均冻结复用，无runtime门、死区或reset。新common严格从4be0加载，架构与motor789f保持，freshALL8Adam1e-4、不继承旧moments。fixed500/full1920BPTT/batch4，saveinitial/250/500，只有final500一次12新V2/V3配置验证，旧12明确开发历史。不得选250或在finalFAIL后自动加500。

preflight会冻结新核心、数据和旧实际依赖，包含原common/data/engine/preflight/train/verify/evaluate/audit/run_candidate等source与原权重、原数据和失败身份。所有实际继承适配在adapt_source.py中精确计数或显式数字映射，训练循环AST已核range(1,501)、save250/500，eval固定checkpoint_000500.pt与VERIFY_000500.json。原source不改。模型/数组/梯度/optimizer/保存/长验证真实结果待父执行。

若所有12分项与独立audit通过，仍仅离线行为恢复通过，不代表身体B或空间趋近成功。父后续另安排最多6B同newbehaviorSHA匹配noodor实际回合；本pipeline没有任何物理入口。若FAIL，保留全部checkpoint/日志/validation，不自动训练、物理或修改运行源。
