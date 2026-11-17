# 仅修复相机采集上界：V2真实RGB长窗口（S2/S5）

V1父实际trial在15400ms试图继续capture时触发继承visual_body_v1/runtime.py:85的captureexact20msindices硬限制而exit1。已保存15400ms轨迹、3080决策、770真实RGB及1232000native partial；这是采集时长工程FAIL，不能称19.2秒自主长窗完成或行为成功。V1目录、源/FROZEN/FAILURE/全部partial/log均保留，不重跑V1。

本独立V2目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_terrain_long_window/REPAIR_CAPTURE_HORIZON_ONLY_V2` 唯一修复是隔离namespace实际嵌套CameraStream.capture→_capture函数中的15400上界绑定TOTAL_MS=25000，继续原每20ms实际物理头部RGB采集与20ms队列发布。不会在15400后停止采集、置invalid、强清视觉或伪造/只保持旧RGB。原0—20ms尚未发布的RGB无效期照旧。完整capture方法其余AST逆恢复精确相同，新真实摄像模式与V_empty视觉条件不变。

new kind=`SOURCE_NECK_ROUGH_TERRAIN_LONG19P2_CAPTURE_REPAIR_V2`，case仍为 `mild_blocks_seed17_start_repair_v1_long19p2`，输出在新V2 namespace。所有V1预登记科学条件沿用，只有相机guard及输出身份修复：原XML逐字、首缘1.35/.02/.04高度/摩擦/旧336/来源颈部/视觉/初态/1+4.8warm/noreset、ANN19.2秒、总25秒、2000000native、1536000postnative、5000decisions/1250captures和原slip<=.5/support/COM/NF0/all6/fullwindow均不变。flat_box_control top_z0是故意齐平的测量控制，不是上凸障碍。到不了后续block依旧NOTTESTED；无爪钩新资格。

父先执行零Core标准库preflight（子已执行后，重复只验freeze与V1 partial hashes）：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_terrain_long_window/REPAIR_CAPTURE_HORIZON_ONLY_V2/preflight.py
```

CAPTURE_HORIZON_PROOF核原line85/guard唯一替换/完整逆AST/实际CameraStream.capture globals确实路由到新_capture；仅求guard表达式证明0/15380/15400/24980允许、25000及15401拒绝，不调用capture。它还hash冻结V1 source/FROZEN/JOBS/log与全部失败payload原字节；不加载数组。继承1023项预检核XML/政策原globals/36时长节点，准备新V2 XML/ledger/绑定，V2 verify同时核两层来源。源码编译及零heavy import记CODECHECK；未构建Core/相机、未执行NN/物理/数组payload。

父唯一corrected managed重任务（one新长case+fresh336audit；最大2000000native，无自动重试，不跑V1 partial/163/192）：

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_terrain_long_window/REPAIR_CAPTURE_HORIZON_ONLY_V2/orchestrate.py --parent-resource-released
```

如拆阶段，trial和fresh336 audit命令分别如下（不能与编排重复）：

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_terrain_long_window/REPAIR_CAPTURE_HORIZON_ONLY_V2/run.py --mode trial --parent-resource-released
```

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_terrain_long_window/REPAIR_CAPTURE_HORIZON_ONLY_V2/run.py --mode audit --parent-resource-released
```

合格后另串行render已保存同次轨迹：

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_terrain_long_window/REPAIR_CAPTURE_HORIZON_ONLY_V2/render.py --execute-after-parent-release
```

V2渲染复用冻结V1可视代码但r路由严格为V2，实际derivedXML/真实.02/.04低侧近景、71opaque、实际RGB、全1250decode、逐块可见pixels与父目视要求均保留；不是新物理回合。父另外比对V2前15400ms与V1保存partial的可比物理/neural/RGB数据，属于实际后验验证，子静态前缀AST相同不冒称已得到数值parity。若失败保留实际错误/partial，不改视觉fallback/地形/门限、不把短窗晋升通过。

READY完整路径 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_terrain_long_window/REPAIR_CAPTURE_HORIZON_ONLY_V2/READY.json`。结果预期 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_terrain_long_window/REPAIR_CAPTURE_HORIZON_ONLY_V2/trials/mild_blocks_seed17_start_repair_v1_long19p2/RESULT.json`，fresh336审核VERIFY与根SUMMARY各自新路径。所有真实结果尚待父执行，不声称新长窗口/粗地形PASS。
