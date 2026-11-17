# 38.4 秒自主粗地形窗口（S2/S5，ONLYCODE）

独立目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_terrain_long_window_38p4/`，依据 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_terrain_long_window/REPAIR_CAPTURE_HORIZON_ONLY_V2/`。唯一独立变量为自主窗口19.2→38.4秒，观察是否能到达后续障碍；这不是已经证明能穿越。V1 partial/失败与V2完整结果均不改写、不重跑，SOURCE_LEDGER及FROZEN引用其原始哈希。

总44.2秒=1秒稳定+4.8秒原教师预热+38.4秒连续自主，无reset。8840 neural decisions；3536000总native、3072000 post-native；44200条1ms trace/geometry/terrain，2210个真实RGB capture，20ms采样最后44180ms，上界44200ms拒绝，非20ms拒绝。XML逐字相同，首缘1.35、.02/.04真实高度、摩擦、视觉、旧336、来源颈部、初态及全部阈值不变。roughpass仍要求原slip≤.5/support/COM/NF0/六腿周期/完整自主窗口全部成立；未到后续块为NOTTESTED，接触不等于跨越，完整穿越需父渲染保守全身体末端净空证据。不声称爪钩或完整生理颈部。

先执行下列stdlib-only静态命令，禁止科学包、数组payload、Core/NN/物理/渲染。sourcecheck首次创建隔离源码和证据，READY后仅验FROZEN，绝不重写冻结内容。DURATION_AST_LEDGER覆盖runner/audit、trace/geometry/terrain/neural/capture shapes、native recorder、cycles/metrics及真实嵌套相机；视频源码另作反向精确恢复证明，覆盖2210帧映射、44.2秒验收和显示。相机有效V2 25000→44200逆AST一致，其底层原15400常量只绑定TOTAL_MS。CODECHECK记录真实已加载模块零科学导入；没有执行科学入口。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_terrain_long_window_38p4/sourcecheck.py
PYTHONDONTWRITEBYTECODE=1 python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_terrain_long_window_38p4/preflight.py
```

父资源释放后，仅启动一次托管orchestrate（由父使用managed background工具运行此绝对命令并收集job，trial之后串行fresh336 audit，无自动重试）。本代码子不执行：

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_terrain_long_window_38p4/orchestrate.py --parent-resource-released
```

实际trial和fresh336完整审核完成后父另串行渲染同次保存轨迹；渲染不执行积分或NN forward，但加载科学包/数组及一个重建Core，因此只允许父执行：

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_terrain_long_window_38p4/render.py --execute-after-parent-release
```

渲染要求实际derivedXML/71opaque/真实RGB关联/低侧真实高度/无高度夸大；2210帧、44.2秒、全量解码、FRAME_MAPPING及父实际目视。若失败保存真实partial与错误，不能降级时长或改阈值。独立case为`mild_blocks_seed17_start_repair_v1_long38p4`，kind为`SOURCE_NECK_ROUGH_TERRAIN_LONG38P4_V1`。预期结果 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_terrain_long_window_38p4/trials/mild_blocks_seed17_start_repair_v1_long38p4/RESULT.json`；审核同目录VERIFY.json；汇总 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_terrain_long_window_38p4/SUMMARY.json`；视频 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_terrain_long_window_38p4/VIDEO/ACTUAL_LONG38P4_DERIVED_TERRAIN.mp4`。目前所有这些科学结果均NOT_RUN。
