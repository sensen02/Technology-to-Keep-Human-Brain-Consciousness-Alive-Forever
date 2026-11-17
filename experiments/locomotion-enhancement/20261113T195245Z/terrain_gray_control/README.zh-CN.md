# 粗地形环境颜色单因素对照：旧336闭环

此处代码仅准备一个新的灰色凸台环境。子代理只读JSON／源码／NPY头部及普通文件SHA，没有读取数值数组、运行NN、构造Core、渲染或积分。父当前186任务与本控制独立；须等重资源释放后再串行执行本目录，不能争用资源。

匹配参考是已完成的brown163完整回合：`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_start_recovery_v1/trials/mild_blocks_seed17_start_repair_v1/`。它的独立完整性审核全部通过，9.6秒ANN阶段六腿周期、非足法向／world-z原生峰0且实际接触前两个凸台，但slip=1.3391032317139215失败。合法对照资格不要求行为PASS，旧失败原样保留。

唯一世界变化：源XML六个静态terrain_block_00..05的RGB从原三通道值分别取 `math.fsum(RGB)/3`，令三个新通道相等；保留原alpha。实际源是显式 `.45 .40 .35 1`，block没有material引用，默认geom也没有继承material；只有floor引用MatPlane。prepare对材质覆盖／XML include会拒绝待审，不擅自改变材质。恢复六个原rgba字串后，新XML须与原解析序列化逐字相同，terrain ledger全等。

几何仍为已经修正起步区的firstedge1.35原brown场景，**不再次平移0.55**。高度、密度／质量、接触／摩擦、灯光、floor/body颜色、相机光学全部保留。不改图像，不灰化actor输入，不置零视觉特征。真实灰色世界RGB仍经过原视觉99698、odor e21d、brain92dc、motor789f、neck9e88的完整336状态闭环；这不是007d／新352组件的部署。

父逐条执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T195245Z/terrain_gray_control/check_source_only.py
```

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T195245Z/terrain_gray_control/prepare.py --prepare
```

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T195245Z/terrain_gray_control/preflight.py --parent-resource-released
```

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T195245Z/terrain_gray_control/orchestrate.py --parent-resource-released
```

prepare只新建本目录PREPARED下的UTC时间戳XML、GRAY_COLOR_PROOF、单case ledger、bindings和两层FROZEN，拒绝覆盖。预检编译一次brown模型仅用于参数比较，不构造brown Core、不走brown物理；灰色只构造一个完整Core并走20毫秒＝1600原生步。所有编译ndarray除geom_rgba外都须byteexact，且brown新编译与其实际163身份摘要相符。geom_rgba只允许六个block RGB行变化，alpha和所有非block行不变；mat_rgba／灯光数组也不豁免。物理opt额外检查相同。旧Core的MU=1.5覆盖也同样施于只读brown编译模型，避免把XML默认摩擦误当实际runtime摩擦。新初始qpos、18球、71opaque、平坦ray、无初始与短窗block接触、NFnormal/worldz0、无外力均检查。

唯一新完整回合为case `mild_blocks_seed17_gray_visual_control_v1`，固定1s settle＋4.8s教师＋9.6sANN、1232000原生步。旧brown、plane、flatbox均不重物理。完整回合继续在5.8秒接管前检查实际warm平坦区，保存真实770双眼RGB、3080个完整336状态记录、sourcequeue和terrain trace。

fresh audit复用冻结旧完整四NN／RGB／受体／源队列重放及几何／指标检查，同时加入brown匹配：新旧trace前5800个1ms样本的所有身体、动作、目标、控制、载荷等字段逐位一致；前1160决策的X238与motor priming同样检查。视觉编码缓存和RGB允许不同，只记录其差异，不伪称完整光学或全神经warm字节一致。

编排最后一个阶段只统计已保存新旧记录，无NN或物理。`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T195245Z/terrain_gray_control/MATCHED.json` 将报告实际visual_raw_turn／decoded raw_turn／AN-DN的RMS和范围、旧新滑移／COM／接触／周期，以及实际消费眼图上的源固定受体能量：uint8/255后ReLU(R−0.5G−0.5B)。它使用保存的image_capture_id映射真实actor帧；不是重新跑视觉网络，也不声称分割出了凸台像素。独立NN重放由前一审核完成。

阈值与完整六腿／严格非足法向和world-z原生峰0不改；未发生实际post terrain_block接触为NOT_TESTED，不移动障碍。颜色对照可说明本场景／本policy对环境颜色的敏感性，不能把任何改善自动归为全部滑移原因、真实机械hook、一般粗地形成功或新网络能力。所有实现／来源失败和行为失败保存，不运行中改冻结源码。
