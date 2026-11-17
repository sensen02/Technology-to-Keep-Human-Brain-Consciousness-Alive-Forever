# 三种地形的当前四NN—真实颈部完整身体基线

本目录为第二阶段代码，使用已经由父代理实际通过的完整身体地形Core。子代理未加载数组、NN或身体模型，未运行物理。已执行的 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_body_baseline` 不改。

父代理在当前实际资源释放后，按下列顺序逐条执行。前两步是stdlib来源检查和绑定；第三步只做四NN与源队列、视觉队列预检，Core／模型编译／渲染／物理均为0；最后一步才是唯一串行重物理任务。

```bash
python3 /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_neural_baseline/check_source_only.py
```

```bash
python3 /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_neural_baseline/prepare.py --bind-completed-core-and-existing-policies
```

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T045410Z/env/bin/python /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_neural_baseline/preflight.py --parent-resource-released
```

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T045410Z/env/bin/python /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_neural_baseline/orchestrate.py --parent-resource-released
```

请由父代理把最后一步放入现有managed background job，并与正在进行的其他重物理互斥；这里不会自行调用或另开服务器／环境。若需只运行一个已登记case，可使用同目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_neural_baseline/runner.py --case plane_control --parent-resource-released`，随后用同目录完整路径的audit.py传相同参数审核；不要同时再启动orchestrate，因为它拒绝已有trial，防止重复。

完整三条件为plane_control、flat_box_control、mild_blocks_seed17，每个1s settle＋4.8s原教师预热＋9.6s纯ANN，总1232000原生步；三条件合计3696000。全部71mesh不透明、自由根、42有限PD和原三球足端保留。模型不新增爪钩、足垫黏附、外力、锁根或位置驱动，不训练任何NN。

`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_neural_baseline/adapter.py` 对实际045302完整runner/audit只做单层显式命名空间改写：删除runtime通配导入、把输出路径条件名改为地形case，以及仅plane要求旧warm逐位复现。数值循环的常数集合不变。064313四NN加载函数保留原globals／ROOT，读取原身份绑定；新目录绑定明确没有新校准。通过局部sw proxy使用已经验证的TerrainCore，旧sw、class和模块global均不改。

真正进入ANN阶段之前，新continuity_snapshot wrapper会核对整段warm实际pad／COM脚下地形高度和全部原生接触地形top均为0；不满足就保留partial并停止，不把旧world-z教师冒充地形自适应。通过后原336神经状态、队列、执行器目标和历史缓冲的连续性检查仍原样执行。

视觉仍为真实双眼RGB。原camera.py已显式显示所有geom group，group5地形会被看见；无红色地标不代表视觉神经输出应为0，地形／身体／地面造成的网络反应照实记录。嗅觉环境为原V_empty零浓度，受体滤波、pending与嗅觉网络偏置／记忆仍保留。脑与身体任务信号继续经原真实颈部源队列，无terrain标签或坐标直接输入脑／动作。

每次trial输出路径为：

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_neural_baseline/trials/plane_control/`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_neural_baseline/trials/flat_box_control/`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_neural_baseline/trials/mild_blocks_seed17/`

原trace、实际RGB、全部336状态／动作／源队列、头部与受体几何产物保留，另存terrain_1ms.npz、INITIAL_TERRAIN.json及WARM_FLAT_CHECK.json。新身份清单绑定**实际派生XML和编译参数**，不沿用旧identity.capture内部固定XML路径。

新审核保留原完整四NN与真实RGB／受体／源队列重放；另外独立用固定轴对齐地形台账检查所有脚球／COM下方高度，逐位重建X238足端高度与速度通道，交叉核对新增trace与原trace的时间、COM、padz、载荷、非足力和phase，并明确把IDENTITY的case/XML/SHA绑定当前ledger。

验收分开记录：原公式的allterrain world-z／XY／world-COM五项；真正plane-only旧口径五项；新的相对COM／法向载荷加权切向滑移五项。旧functional_stable_body值单独保留，新同名功能flag额外要求**原生非足法向力峰也为0**，避免垂直box侧面的纯水平碰撞被world-z漏计。新完整行为和rough标志再要求六腿完整周期。原生峰由已绑定源码的逐原生步recorder统计，未保留全部原生数组，不能声称从1ms trace独立证明12.5微秒峰或完成原D1—D8。

障碍覆盖只认ANN阶段实际原生terrain_block_*接触；坐标到达或ray扫过障碍不算接触。未真正碰到凸台时为NOT_TESTED，rough_terrain_pass必须false。条件位置、高度、seed17全部沿用已经准备的固定ledger，不为制造接触或通过而重抽／移动。

科学行为失败仍保存并完成独立完整性审核；实现、来源或审核失败则保留partial／日志并停止串行编排。不得运行中修改已绑定源码。全部通过后才能进一步讨论神经课程或爪钩接触，实现准备本身不构成粗糙地形行走成功。
