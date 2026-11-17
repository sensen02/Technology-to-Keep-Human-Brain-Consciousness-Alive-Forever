# 唯一起步区修正场景：固定整体平移 +0.55

本目录是新场景版本 `mild_blocks_seed17_start_repair_v1` 的代码。所有已执行足审计、身体Core、三条件神经基线、控制回合和旧凸台partial保持冻结。子代理未加载保存数组／NN／Core，未执行prepare、预检或物理。

实际诊断来自 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_start_area_diagnostic/ACTUAL/RESULT.json`：平面预热0—5.8秒18颗脚球的**实体表面**最前缘为1.2492113609914324，加固定0.05工程余量后必要前界为1.2992113609914324。诊断中的 `all_warm_sphere_material_xy_max` 已含余量，不能再把1.299当raw值重复加余量。

父代理固定本版本唯一修改：六个原seed17凸台沿world X整体+0.55，第一前缘0.8→1.35。高度序列0.04/0.04/0.04/0.04/0.02/0.04、宽度0.25、间距0.38、y范围±5、摩擦、完整身体、初态、四NN及颈部来源、1＋4.8＋9.6秒时序均不变。这个量是依据起步区必要几何包络固定的修正，不根据新回合成绩重抽或继续移动。

父代理逐条执行下列命令。使用实际已运行环境 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python`，全程 `-B`／`PYTHONDONTWRITEBYTECODE=1` 防止生成无用字节码；最后一步仍应放入唯一managed重任务，不与其他物理并行。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_start_recovery_v1/check_source_only.py
```

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_start_recovery_v1/prepare.py --prepare
```

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_start_recovery_v1/preflight.py --parent-resource-released
```

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_start_recovery_v1/orchestrate.py --parent-resource-released
```

source-only检查仅加载stdlib命名空间，验证新旧函数code与globals路由，不调用Core或NN。prepare只读既有来源字节／JSON／XML：验证父实际诊断的before/after来源、两个控制的完整性与物理资格、原五个JOBS退出码0/0/0/0/1、原5.8秒拒绝接管partial，及旧147零Core四NN检查。它新建 `PREPARED/<UTC微秒时间戳>/mild_blocks_seed17_start_repair_v1.xml`；固定索引和证明在新目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_start_recovery_v1/PREPARED/MODEL_LEDGER.json` 与 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_start_recovery_v1/PREPARED/SCENE_TRANSLATION_PROOF.json`。还新建INPUT_BINDINGS与两层FROZEN，均拒绝覆盖。恢复六个原pos字串后XML序列化须与旧XML解析结果逐字相同，证明其余科学属性未动。

新runtime对旧terrain Core和neural context作隔离函数globals绑定：HERE／TERRAIN_ROOT指向新目录，case与KIND用新版本；Core closure读取新ledger实际XML，旧BASE／原XML请求校验仍保留。原四NN加载函数继续读取064313原ROOT和权重；指标模块直接复用冻结旧文件。任何旧module／global／科学文件都不修改。

新场景预检仅构造一个完整身体，20个1ms零增量tick＝1600原生步，检查初态qpos与合格旧plane初态逐位一致、18脚球、71不透明mesh、自由根、新XML时间戳／高度／位置、初始和短窗没有凸台接触、无非足法向力及外力。**旧147四NN证明是明确复用，不声称新做四NN预检。** 新20毫秒不能证明完整预热平坦；新完整回合仍在5.8秒ANN接管前检查4800个warm采样与全部原生warm接触。

唯一新完整回合最多1232000原生步，随后一个fresh实际RGB／受体／sourcequeue／336状态／四NN保存重放审核；不训练、不重复plane/flatbox物理。实际地形RGB不隐藏或强清。新旧指标并列，严格非足法向和world-z原生峰都须0，六腿周期保留，阈值不变。ANN未实际接触terrain_block_*即为NOT_TESTED，不能因为旧plane的必要x可达就记本场景通过。行为失败、无接触或再次guard拒绝均照实保留，不更换起点／位置／种子碰结果。

产物预期位于 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_start_recovery_v1/ACTUAL_PREFLIGHT/`、根PREFLIGHT.json和 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_start_recovery_v1/trials/mild_blocks_seed17_start_repair_v1/`。编排只生成两个JOBS阶段trial/audit；SUMMARY引用旧两控制归档、明确新1232000原生步和先前1600预检步。

本版本没有新增爪钩或pulvilli、没有附着／外力／锁根、没有改变历史原case结论，不声称原D1—D8或生物力学校准完成。
