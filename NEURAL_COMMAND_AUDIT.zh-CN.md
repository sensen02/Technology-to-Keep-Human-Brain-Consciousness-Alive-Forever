# 真实神经命令与触觉传递审计

**手工实施并测试的研究原型，不是成品。** 不把离线肌肉优化当作神经可达目标，也不添加背景驱动。

## 来源约束

只使用通过`closure_bundle.validate_bundle`校验的完整current_run运行包。历史平铺NPZ/JSON不是当前伴随文件。对比离线命令必须注明姿态、初始激活和命令时变性不同，不能宣称公平能力对比。

## 初步证据

当前LF bristle_touch组有450个细胞。额外+0.3nA感觉刺激在5–15ms内输入该组，编码器/网络入口没有将它裁掉。正常与无外部触觉两组全网络的区间×神经元放电计数有1634处不同，但运动池计数和命令相同。两组总放电数相等，说明只看总数会漏掉时间重分配；不能说刺激没有进入网络。

## 已验证命令与共享池

正常命令与上一完成周期的保存解码率逐项完全一致，误差0。首批运动池放电在5–10ms，首批非零命令在10ms生效；这是实现的因果采样延迟，不是已测生物时延。全部7组形状、有限性、池计数、身份、一致共享命令及切断检查通过；6项回归测试通过。

12000神经元中202个属于映射运动集合；90输出中74可用、16空缺、65完整覆盖。实际细胞集合有66种，8对输出共享相同池。平均映射矩阵秩66只是这一读出定义的代数性质，不能推为网络可控性或生物不可能。

正常10×90命令样本中478项为0、110项饱和，均值0.204949；首两个周期均零。覆盖74肌的时空样本均值约0.249262，反而高于离线候选的覆盖肌均值0.186785，所以不能泛称“神经输出整体太小”。时间分布、共享池、姿态与初始激活不同，禁止据此直接调增益。

- 实现：`/run/media/sensen/Data2/cell_wound_prototype/run_neural_command_audit.py`
- 测试：`/run/media/sensen/Data2/cell_wound_prototype/test_neural_command_audit.py`
- 数据：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/neural_command_audit.json`

## 精确重放与路径诊断

重建相同神经层与感觉电流后，两组10×12000的逐5ms放电计数均精确复现。0.5ms记录显示8738个二值放电样本不同，最早8ms出现；差异全部位于450个受刺激细胞中，运动池202细胞的输入电导和放电均无变化。图遍历只能到达450个起点，没有可到达的运动细胞。两组运动池总放电均405，全网络均48053。

已构建We/Wi中这450细胞输出非零项均为0，输入仍有6个兴奋、1102个抑制非零项。后续原始数据核对发现：450个细胞的ann_nt_verified全部空白；原始输出5832条、29708个突触，其中层内5770条、29479个突触，层外62条、229个突触。当前纯递质标注规则排除了全部空白来源输出。因此这里是递质标注规则导致的实际输出断连，不是原始连接全无，也不是主要由神经层截断造成。

此结论只针对当前450个细胞和建图规则，不推为所有感觉缺失。需要查找可追溯递质依据，不能将空白直接填成乙酰胆碱，不得调高刺激掩盖断连。

- 精确重放实现：`/run/media/sensen/Data2/cell_wound_prototype/run_touch_path_audit.py`
- 测试：`/run/media/sensen/Data2/cell_wound_prototype/test_touch_path_audit.py`
- 数据：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/touch_path_audit.json`
- 完整定向重放：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/touch_path_audit_spikes.npz`
- 已查看图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/touch_path_audit.png`

4项路径/重放测试通过。当前运行代码后续增加定向观测与身份映射，因此对旧实验必须保留记录SHA与当前SHA区别。
