# 两DNp09＋一AN：身体侧工程接口与向量颈部边界

本子仅CODE_ONLY准备；新目录独立于已运行的旧1DN／1AN body_interface，不修改旧模型或检查。真实结构包括父取得并核验的两DNp09 v888骨架ID720575941433155799、720575941566493282及原AN08B018 ID720575941351919408。canonical root_id可能不同；本scope用root_888，不强制它等于canonical。

原始ID、版本、骨架与平面来源由build_binding实际读取父已保存的metadata／独立几何检查／下载收据／底层v888文件SHA。结构身份不代表nativeDN固定控制律，AN08B018也不等于任意111身体特征的生理编码。

## 父准确执行顺序

1. 标准库检查，本子已实际14/14通过、exit0：

`python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/dnp09_interface/test_transport.py`

2. 父实际来源绑定：

`python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/dnp09_interface/build_binding.py --execute-parent`

3. 父唯一CPU1数值／真实源向量队列预检：

`env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/dnp09_interface/numeric_check.py --execute-parent`

第2／3步本子没有执行。数值预检不会优化、训练、加载运动权重或构造身体。SOURCE_BINDING、NUMERIC_CHECK、TRANSPORT_JOURNAL皆exclusive输出，已有结果不可盲覆盖。

## 实际API及因果时间

`SourceTransport(path,expected_sha256)`根据有序ports导出DN两维、AN一维，不继承旧ROOTS硬锁。send仅接受from/to/direction/root_ids/values/capture_step/source_bindings_sha256，禁止action／target隐藏字段和方向／ID乱序。每1ms advance返回release事件，2ms后到达；5ms神经consumer应保持release_step=2、另记录consume_step=5，不把2ms发布当成5ms决策已提前消费。cut按方向丢在途，restore无clockreset／旧包复活。

初始化和verify_final核全部源SHA；每tick检查binding文件bytes／SHA，不重复读取57MB源数据，仍需外部父监督整个run末尾身份。未来若并发改源必须停止；本实现不宣称实时性能或安全手术控制。

## 学习模块与边界

api.py直接lazy复用immutable `/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/body_interface/interface.py` 的构造函数：实际N DN→16tanh→learnedturn1；反馈local111→32tanh→M AN。选定N=2/M=1由真实绑定导出，不复制模型代码；视觉cue／脑隐藏状态不会传给身体adapter。向冻结组合motor的[relu(turn)/3,turn]只属既有命令ABI，不是DN原始值的手写比例控制。零turn低层网络仍持续行走，不能归功于人工脑或颈部。

数值检查核反馈mask：0:42、84:126、168:195共111，仅身体特征；排除targetspeed237、previousactions195:237、targetpositions126:168、error42:84，Jacobians为0且排除输入扰动不改变输出。DN两通道到learnedturn梯度均需存在，taskcue对bodydecoder无直接梯度路径。

未来训练预算／人工标定目标由父另预登记，工程映射仍不等于真实细胞调谐；需在source时序、保存重放、DN／AN切断及zeroDN低层基线的范围内核验。当前完整颈部、生理刺激耦合、新三部分行走和意识结果均未建立。
