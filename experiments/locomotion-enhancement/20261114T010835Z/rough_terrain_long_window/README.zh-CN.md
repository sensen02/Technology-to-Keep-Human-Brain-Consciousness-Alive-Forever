# 旧336真实粗地形：唯一19.2秒自主长窗口（S2/S5）

这是代码准备，不是新行为结果。旧齐平箱体flat_box_control的top_z=0，原本是测量控制，不是上凸障碍；看不到凸起符合其几何定义，不能改名或抬高。旧粗地形9.6秒只接触block00/01，滑移1.339仍FAIL。本trial只检验原控制器多给9.6秒后是否真正接触更多原blocks、能否完整走过；没有训练或爪钩资格声明。

新目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_terrain_long_window`。唯一case是 `mild_blocks_seed17_start_repair_v1_long19p2`，new kind是 `SOURCE_NECK_ROUGH_TERRAIN_LONG19P2_V1`。实际修正XML逐字复用，不移动任何凸台；首缘1.35、高度.04/.04/.04/.04/.02/.04、摩擦、视觉、初态、旧336权重和原来源颈部均不变。1秒稳定+原4.8秒教师预热后完全撤辅19.2秒，总25秒，25000个1ms样本、5000个5ms决策、1250眼图、2,000,000原生步（自主1,536,000步）。不做新物理preflight；准备与静态preflight均0Core/0NN/0数组payload/0物理。旧实际合格preflight与旧rough仅作来源，不冒称本次新物理检查。

`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_terrain_long_window/runtime.py` 只有ANN_MS=19200独立新时长量。新namespace将旧runner、audit、terrain runtime、metrics以及来源颈部finalize、旧stable中的时长节点改为派生符号；所有其他数字、阈值和公式保留。每个替换在DURATION_AST_LEDGER列出原路径/行/列/旧值/新表达式/新值，并反向还原完整原AST逐节点精确相等。旧161918/195245执行源与原数组一个字节不写。旧诊断字符串可能仍含15400等历史字样；真正shape/循环/全窗计量/native门全部使用长窗口变量。remaining_s只进入被丢弃的critic extras，原actor X238完全不读取该值。

全部门覆盖[5800:25000]：原normal-load加权切向滑移<=.5、支撑均值>=.9W、COM相对高度>=1.2、signedX>=.05、原采样NF门、原生NF normal/worldz严格0、extra force0、warmflat/noreset、全部六腿周期和精确全窗。没有教师自主期驱动、根锁、外力、adhesion或CPG。独立fresh336审核重放实际RGB/受体/队列/全状态/四NN、完整5000决策与25000行地形/geometry/trace；不重新积分。

每块只凭自主期实际native solver contact记CONTACTED，零记NOTTESTED；solver事件不是独立试验次数。SUMMARY比较原rough的实际block集合。全场穿越另需所有六块实际接触与保存全身71mesh包围球的末帧最小X越过最后rear的保守充分证明；未满足仅NOT_ESTABLISHED，不把COM到达当全身穿过。

父代理逐条执行绝对命令，严格串行。第一个命令是ONLYCODE零Core预检，可以在轻量进程执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_terrain_long_window/preflight.py
```

它检查旧FROZEN全SHA与旧rough独立审核、XML真实首缘/高度/flat_top0、所有AST反向证明、隔离namespace、政策loader原globals、没有heavy import，创建PREPARED逐字同XML、INPUT_BINDINGS、PREFLIGHT、DURATION_AST_LEDGER、CAMERA_AUDIT、SOURCE_LEDGER、FROZEN与READY。READY后源码冻结，任何修正须新版本，不能执行后追改报告。

父资源释放后只启动下面一个managed heavy，包含唯一新完整case与一个fresh336 audit；不重跑旧brown/gray/controls，没有自动重试，trial目录和JOBS拒绝覆盖：

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_terrain_long_window/orchestrate.py --parent-resource-released
```

编排内部trial的完整命令如下（使用编排时不要再单独执行）：

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_terrain_long_window/run.py --mode trial --parent-resource-released
```

独立fresh336 audit命令如下：

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_terrain_long_window/run.py --mode audit --parent-resource-released
```

成功后父另串行渲染已保存轨迹，0NNforward/0积分（1个Core仅恢复视图）。不能在本子代理执行：

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_terrain_long_window/render.py --execute-after-parent-release
```

新视图保留真实derived XML与.02/.04高度，主画面完整71不透明mesh，低侧角近景只zoom不缩放地形，实际published眼图放侧栏，前20ms明确NO RELEASED RGB。主/近景逐块segmentation像素统计记录，近景追随原blocks，无新增假障碍。逐帧核qpos/qvel/COM/head/yaw、主体无裁剪、禁止所有mj_step、恢复offscreen后核compiled identity，25秒1250帧完整解码。输出 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_terrain_long_window/VIDEO/VIDEO.json`、FRAME_MAPPING/FFPROBE/DECODED_MONTAGE与mp4。CAMERA_AUDIT只是静态真实几何核对，实际视野与.02/.04可见性须父读取像素统计并实际打开蒙太奇；为0的block不可称已可见。主视图遮挡或容纳检查失败只修新可视namespace，不改变科学源或重跑物理。父实际审完再把样本复制至本地桌面并核SHA；本子不交空样本。

结果预计在 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_terrain_long_window/trials/mild_blocks_seed17_start_repair_v1_long19p2/RESULT.json` 与VERIFY；根SUMMARY保存完整长窗行为与逐块覆盖，未接触后续block明确NOTTESTED。原rough FAIL、flat测量control定义、无真实爪钩/SI/全颈生理声明均保留。
