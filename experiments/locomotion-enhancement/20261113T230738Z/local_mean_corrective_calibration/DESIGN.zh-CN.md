# 一个有限local-mean纠正候选：ONLYCODE，未训练／未验证行为

本候选位于独立230738目录。所有161918／195245／221508旧科学源、mask、输入清单、权重、辅助采集原始行及失败一个字不改。它是当前局部神经均值耦合与真实掩码数据聚合的联合修复，不是单因素因果实验，也不是brain→neck新任务部署。

## 新输入与迁移

父模型为实际007d0c74118715eacd03d49c7e6847d8349e55db2793dd9d461a4b9c311e0fcd final600。每决策只计算一次冻结789f old.step，得到当前oldmean_direct42与afterold256。新features按顺序为physical112、afterold256、独立task2、directoldmean42，共412；不使用mean−residual兼容分解作该输入。旧均值含冻结target-position prior，这正是已存在Other局部神经输出的因果值，不是教师q_des／clock／理想root。

encoder412→96/tanh、GRU96同alpha .0487705755，head仍concat[h96,encoded96]192→42，residual仍6*tanh，最终action=tanh(oldmean+residual)。old789f全部冻结、旧合法投影不变、8可训练张量和state[128,64,64,96]=352不变；encoder增加4032参数。没有CPG、速度乘零、病例门或教师隐藏输入。

迁移encoder前370列／bias从parent复制，末42列0；old／GRU／head全部复制。宽GEMM舍入可能改变encoded和new96，因此只要求old256逐位一致，new96／mean／residual／action≤1e−12；对12训练及8条实际长度DEV完整历史检验，不误沿用上一版new96逐位承诺。

## 实际12条监督与8条无标签DEV

原8条合格实际源及Y完全沿用：3教师+5NN；不因后续B保守mask把这些原物理合格目标追溯裁掉。另加入221508的old_walk_control和assisted stop／L／R。已实际读取采集collector字段与父PARENT_DATASET_QUALIFICATION：辅助采集.25student+.75teacher不是自主神经通过，只有masked监督资格。

old_walk_control的Y为直接oldmean.tanh／真实应用动作，非007d shadow policy，1920行全有效。新3辅助轨迹的Y必须是完整未混合teacher_action=supervision_target，实际mixed动作只用于provenance核对；label_qualified有效数分别1920／1280／1293。全部1920输入与hidden历史都推进，invalid只令监督loss为0，不删除行、不填新标签、不把masked指标当物理通过。全无效序列直接拒绝prepare，不造fallback。

同类预热严格保持：原teacher X[:960]，NN与采集X[200:1160]，old256真实连续960步、新96为0。所有有NN档的before/after old256及collection directoldmean都精确验证；原teacher没有oldhidden则只双路径priming核验，不假造实际神经重放。compat mean−res仅描述ULP差，不进入412特征。

8个失败学生输入仅DEV：首版4条1920，第二版walk/stop1920、L494、R776，真实长度、不padding、不把旧teacher同clock动作当新state目标。DEV只报告输出／状态分布，不训练、不算teacher误差、不称物理反事实。

## 损失、范围和固定预算

B12／micro4×3／完整1920。每序列base首20tick权重10（0..0.095s，不含.1），其余1，再乘该序列valid；**每序列除其实际有效加权质量**，使总质量各1/12，每动作再除42。不能继续用固定2100当masked后的分母。六个v1序列的residual prior .1按1920×42×6全历史归一；所有8新参数对迁移后anchor的元素均方*.001，micro按B/12分摊。6教师／6v1类别总质量各半，不由有效行较多者吞监督。

首先对每条12实际目标仅在valid行计算closed action interval[tanh(oldmean−6),tanh(oldmean+6)]的保守RMSE下界，工程必要门≤.01；超门立即beforetrain BLOCKED，不改mask、不扩大bound、不删除out-of-range行。下界小不保证共享ANN／GRU可学或闭环稳定。

freshAdam3e−4固定600、save0/300/600，只final600资格。新预算wall3600秒、RSS1800MiB，包含源／加载／缓存、更新和全部保存审核；不自动延期或多候选。preflight实测完整micro4×3更新*600*1.3，加三次fresh验证、DEV和IO余量，预测≤3600才开训，实际超限保失败。

## 数值资格与存储

preflight检12训练＋8变长DEV迁移；Python/JIT/cache-production≤1e−12；完整1920 B12对3×B4的损失、8梯度、Adam／参数≤1e−10；mask质量、first20和父anchor均明确。

输入合同用**数值扰动**而非autograd0糊弄：42:84(target-error)与195:237(previous action)的固定输入扰动不得改变oldmean／old256／完整mean；126:168(target-position offset)可经冻结−gain*Tpos/H改变oldmean，但old256不变。短暂把新增42权重一项置非零，确认encoded路由有响应后恢复全部参数和RNG。该固定输入测试不生成新身体轨迹，不宣wholebranch对target独立。

common.load_checkpoint(path,digest,final=True)继续两返回，strict新FORMAT INDEPENDENT_MOTOR_LOCAL_MEAN_CORRECTIVE_V1、0/300/600及final600、8Adam计数／源／旧冻结摘要／RNG／别名恢复。每个保存先live模型／Adam／RNG roundtrip精确，再新进程12训练＋8DEV完整真实历史生产／缓存≤1e−12，old256精确。final只输出12 masked/unmasked/weighted fit和保守mask统计，以及DEV无Y的输出分布；低fit不行为PASS。

## 父执行接口

父先审源码；标准库实际来源绑定：

```bash
/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T230738Z/local_mean_corrective_calibration/sources.py --prepare-source-metadata
```

独占CPU托管预检：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T230738Z/local_mean_corrective_calibration/preflight.py --execute-parent-preflight
```

仅actual all_passed/timing READY_FIXED600_3600S后，一个有限训练：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T230738Z/local_mean_corrective_calibration/run.py --execute-parent-calibration
```

三主stage为固定600含三次fresh保存审核、final评估、独立NumPy指标审核，JOBS/命令/日志/exit完整，没有自动重试。最终SHA在实际资格完成前未知，后续body4必须同门新SHA绑定；身体组件通过后仍须brainANN→真实DN2 queue→decoder2和完整432感觉回归，不省略。

本子只源码／JSON／ZIP头／字节hash和标准库检查，没有NumPy/Torch数组、模型、Core或训练执行。本候选若仍失败，应据真实状态／误差／数据分析，不转无限架构或epoch搜索。
