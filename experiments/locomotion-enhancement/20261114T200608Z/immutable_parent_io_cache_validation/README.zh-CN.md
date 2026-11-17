# 固定新600检查点的I/O性能与等价验证包（尚未执行数值）

只服务未来验证，不接入当前144、不改其源码／PID／deadline，不训练、不做资格恢复。输入固定producer `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/state_aligned_action_calibration/wall_replan_9000/` 的 `TRAIN/checkpoint_000600.pt`，SHA f95d759e1b28aee0a4224615d912bdf0e95490acd9629e4752ce8b89108c26ce。入口核实际81项fresh新600证明，不能拿900或别的文件替代。

## 原校验器与缓存临时范围

基准调用原producer.common.load_checkpoint(path,SHA,final=False)一次，原strict config／identity／Adam／RNG／源资格均真实执行。然后只在本验证进程临时将该producer模块自己的load_resume_parent属性指向缓存实例；原load_checkpoint／_validate_payload／_check_config／initialize函数对象及其global代码不替换，也不全局猴子补丁祖先函数。它们通过原模块globals调用替换后的父装载入口，公共m.step等完全不动。finally恢复原属性、八个声明alias、sys.path与callerRNG。

缓存首miss真实调用捕获的原load_resume_parent，完成严格41fa父600装载，只此一次。其返回model/payload经私有deepcopy建立标准；hit继续原producer其他校验，只把父模型反复重建变成经审核模板的deepcopy。postcondition不调用原_validate_payload或initialize，避免递归：逐tensor state／identity／完整payload指纹相等，verify_old／架构、8Adam step600／moments、8名字原fcbd anchor都核。没有alwaysPASS回调。

## 完整清单的实际观测依据

初始清单含producer.INPUT_BINDINGS全部source/data、PREFLIGHT、新600SHA和81项证明，以及本包实现／预登记SHA。基准严格装载期间，thread-local sys.setprofile观察原project verify_hashes／verify实际收到的每个SHA映射和原sha文件函数成功返回，**不替换函数、不跳校验、不伪返回**；再合并实际new600 payload.config source/data和加载的project源码。miss中对原父严格链同样记录并合并，冲突拒绝。entry／miss后／每个hit出口均核全部内容SHA，未用stat／mtime替代。记录实际映射次数、sha返回次数及展开条目，source_before/source_after精确相等。

这里的“完整”限于本次Python严格校验路径实际访问的声明资格映射、成功SHA函数、payload清单及源码。不是对任意C文件I/O或未知外部环境的全知证明；如果原校验器自己不声明的科学依赖存在，需要另审，不能以profile声称补齐未知资格。未来若新增校验函数名／路径，必须更新本独立观察策略并证明没有漏过，不能默认套用。包只在CPU1单Python线程使用；若发现别的profile hook则拒绝，嵌套仅链接本包自己observer，保留原hook finally恢复。项目step／step_details／forward／cache_sequence／prime_old若在观察期间实际调用会立即拒绝，不做新NN重放。

runtime由实际Torch／NumPy／Python／executable／CPU线程／默认dtype与设备读取；完整父内容账本＋runtime＋FORMAT／600组成cache key。不是用户任填字符串。原checkpoint的sourceconfig不回写，本包是独立I/O兼容层，以自身新ledger说明和原producer关系。

## 等价、隔离与负例

比较基准和cached新600的state_dict全部shape/dtype/device/value、identity和完整payload（含Adam／RNG／anchor／curve／config）的类型敏感字节指纹。公共step／step_details／prime/cache等bound函数的定义路径、code和源SHA相等。不调用它们；本次不重复20TF＋12conditional完整前向，也不冒新81项审核或900资格。

Tensor隔离逐模型parameter/buffer和payload所有Tensor核untyped_storage的pointer/nbytes跨pair无交叠（保守按完整storage含views），并核NumPy RNG数组内存区间与可变容器id。只修改缓存返回副本中的训练参数／Adam moment／anchor／curve／RNG，再取一次必须等私有原strict标准；不写checkpoint或live/ref模型。

source拒绝负例仅在独立输出目录创建dummy绑定文件，改变内容后get必须拒绝，再还原核SHA、撤销fixture绑定并删除；不改真实源。异常fixture在原RNG上下文中改八alias/path并生成随机数后抛错，须完整恢复。所有缓存私有/返回模型的精确比较和Tensor隔离仍待父实际执行，ONLYCODE标准库自检不冒这些门PASS。

## 运行方式与边界

父须等144资源释放后，另一次独占运行；当前禁止执行数值命令。预算3600秒、RSS1800MiB、CPU1、0optimizer.step／0NNforward／0Core／0物理。signal硬wall超时，长严格调用后也检查RSS高水位；资源失败明确写FAILURE，不放宽。source校验仍保，缓存未必加速，实际记录uncached／cached装载和单hit时间、原父严格调用次数、RSS。

可随时执行的纯标准库检查（子已执行）：

```bash
python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/immutable_parent_io_cache_validation/run.py --selfcheck
```

仅父将来资源释放并确认执行此验证后：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/immutable_parent_io_cache_validation/run.py --execute-parent-io-validation
```

结果排他写 `RUN_FIXED_F95D600/RESULT.json`、`COMPLETION.json`，失败留FAILURE。报告始终qualifies900=false、new81sequence_replay_performed=false，只有本次固定新600的I/O等价／性能结论。即使通过，也不自动授权缓存postrun、body装载或部署；当前144及900训练结果由父原流程独立裁定。
