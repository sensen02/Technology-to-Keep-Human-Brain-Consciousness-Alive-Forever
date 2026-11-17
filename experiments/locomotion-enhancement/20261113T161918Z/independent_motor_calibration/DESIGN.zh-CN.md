# 独立speed/yaw身体motor有限校准：ONLYCODE，未训练

新接口 `step(obs238, independent_command2, state4)→mean42,state4`，命令为forward_fraction[0,1]与signed_turn[-1,1]。严格冻结原789f全部张量和256状态，内部旧命令仍投影 `[relu(turn)/3,turn]`；其输出是新组合模型的基底，不能冒充旧789f已获得独立速度接口。

新ANN用112真实物理通道、旧motor本次afterhidden256和任务2量共370维，Linear96/tanh→leakyGRU96（100ms）→Linear42零初始化；mean=oldmean+6*tanh(head)，action=tanh(mean)。新增encoder2+GRU4+head2共8张量可训练，旧全部冻结。状态[128,64,64,96]=352，未来加感觉脑为432；当前尚未接颈部，不能复用旧336格式或称脑已学会原地转向。没有以速度乘零动作、病例门控、time/phase/世界坐标或教师标签进actor。

8条实际轨迹仅：完整fixedshadowstop、fixedshadowpivot左右、terrain_neural_baseline实际plane_control行走，以及064313四视觉转向回合。全部先用RESULT/VERIFY/配对proof/原始sourcehash准入，不引入失败v1教师或flat_box同轨迹重复覆盖。教师Y为实际42动作，命令[v=0,turn0/±1]来自预登记任务；神经Y为实际applied suffix动作并与storedpolicy_action一致，命令[v=1,storedrawturn]。这些是工程任务，不赋予来源DN已知速度/偏航生理。

只读ZIP NPY头已核全部字段：teacher X(2880,238)/Y(2880,42)；5个NN X(3080,238)、raw_turn、command、mean、policy_action、actually_applied_action、hidden_before/after(3080,256)均存在，所有实际审核true。本子没有读取其数组值。

预热精确分支：teacher取X[:960]；NN取X[200:1160]，前200 settle没有运行motor。old命令0连续运行960步，new96始终零。NN每步storedhidden_before/after、oldmean/action均要求exact，suffix1920同样核对；teacher没有归档oldhidden，只用独立逐步循环与public prime_old双路径核验，不冒称神经实际重放。新身体task当前值直接消费，不人为shift一帧；未来经neck2ms队列接入须另校准。

旧mean和afterhidden特征由冻结789f生成一次缓存并detach，因为不依赖新训练参数，不需要每epoch重复运行旧motor。新GRU在1920完整历史上BPTT，不切短窗口、不隐式reset；micro4×2每次完整1920，等价effective8。目标损失是等权8序列的action MSE+.01仅5个v=1NN序列残差均方+.001全部新参数对初值均方；每个micro按全8/5分母与其batch占比分摊anchor，合起来等于full。停止序列也完整学习，不把quiet样本降权。

preflight首先actualsource/heads与旧priming、head0全部8完整1920生产/缓存路径一致性；临时非零head检查Python/JIT1920 forward，再用32步2参数中心差分作辅助，最后完整1920 B8全图一次损失/8梯度/Adam更新对micro4×2，容差1e-10。full图backward后删除再micro，限制RSS1800MiB。临时参数和RNG恢复，不把优化器探针当实际训练。一次完整micro更新计时×600×1.3，加3次实测setup/cache+完整生产重放的1.3倍作为保存审核开销估计；总计超过1800s输出timingblocked-before-training，父需另立预算/架构，不自动缩步或执行后延时。

训练固定freshAdam3e-4/600/effective8micro4×2，保存0/300/600，仅final600可资格。每保存严格roundtrip model/Adam/TorchRNG比对，另新进程从磁盘装载、全部8完整序列生产与缓存双重重放、旧256状态exact、8Adam moment/counter和RNG/alias保护。壁钟1800s（包括保存校验）与RSS1800MiB限制，超时保留部分记录并失败，不扩epoch。脚本计算fit和残差所需幅度描述，不因低loss称行为通过；atanh目标只为描述，不能改实际目标或幅度6。

公共存储 `common.initialize()→m,oldpayload`；`common.load_checkpoint(path,digest,final=True)→m,payload`；strict格式INDEPENDENT_SPEED_YAW_MOTOR_RESIDUAL_V1。旧load_checked789f前后model/common/sys.path恢复，load_checkpoint保持调用方Python/NumPy/Torch RNG。save排他原子发布同目录临时文件/fsync/hardlink，不覆盖旧checkpoint。

父审阅代码后，先标准库准备来源清单：

```bash
/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/independent_motor_calibration/sources.py --prepare-source-metadata
```

该步骤只JSON/sourceSHA/ZIP NPY头；不导数值库，不加载数组。随后父托管唯一预检：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/independent_motor_calibration/preflight.py --execute-parent-preflight
```

仅actual all_passed且timing READY_FIXED600后，再父独占CPU托管：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/independent_motor_calibration/run.py --execute-parent-calibration
```

流水线记录JOBS/exit/log；train内0/300/600各一次新进程保存审核，之后final评估与NumPy-only独立保存指标审核，0物理。不会在校准阶段加载视觉/嗅觉/脑或MuJoCo/身体。后续须独立身体NN三任务与旧walking回归，行为合格再做来源2DN的独立2输出ANN decoder；本阶段directbody命令只作明确研究诊断，不能当绕颈部署成品。
