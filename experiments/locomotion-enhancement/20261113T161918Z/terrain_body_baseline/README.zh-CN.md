# 完整身体地形接触：最小Core与有限父端预检

本目录是父代理授权的**第一阶段代码**，没有运行模型、数组或物理，也没有开放三回合完整神经闭环。不能据此宣称粗糙地形行走、原地转向或爪钩实现通过。

父代理先执行以下两条stdlib准备命令，再在现有身体物理环境中执行第三条。每条单独执行，所有产物新建且拒绝覆盖：

```bash
python3 /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_body_baseline/check_source_only.py
```

```bash
python3 /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_body_baseline/prepare.py --prepare
```

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T045410Z/env/bin/python /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_body_baseline/preflight.py --parent-resource-released --out /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_body_baseline/ACTUAL_PREFLIGHT
```

物理python路径取既有身体环境；父代理应确认该路径仍为项目实际运行环境，不能为本目录另装依赖或替换科学套件。

`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_body_baseline/prepare.py` 将从冻结XML新建三份完整身体派生XML，资源路径改为同一实际缓存的绝对路径，fly geom group=1、静态terrain group=5以支持排除身体的向下ray。plane控制的物理不变；flat_box控制只替换原plane为top z=0的巨大box；seed17条件在原plane前方x≥0.8添加六条低凸台。随机种子、位置和高度范围在预登记中固定，不根据成绩迁移或重抽。

`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_body_baseline/terrain_core.py` 仅在派生类中把 `planes` 解释为显式terrain集合，覆盖旧Core、NativeRecorder和teacher使用同一mask的接触路径；旧科学文件及其模块globals不改。旧模型验证器要求存在plane，因此新隔离AST命名空间只改这一处存在性谓词为“静态terrain_且group5”，保留其他验证及有限PD方法。父端预检检查所有PD方法bytecode和kp/kd/limits未变。

旧 `identity.capture()` 会固定读取旧rc.PAD_XML，不能单独用于本派生模型。这里的 `actual_identity()` 保存实际derived XML路径／SHA、资源源清单及编译数组摘要。

`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/terrain_body_baseline/terrain_metrics.py` 区分：原正world-z支撑、signed world-z、真实接触normal载荷及normal加权三维tangent速度；另保留旧plane-only量。padz和COM高度改为脚下／COM下方静态地形的垂直相对高度，旧world-z量保留。238维宽度不变，但padz／pad速度的语义改变，必须使用新观察版本；AN111下游也会收到相应变化，不能把地形条件当旧输入逐位不变。

父端预检限定四个Core（旧plane、新plane、新box、新blocks），各20个1ms零增量tick，共6400原生步。旧plane与新plane逐tick比qpos/qvel/ctrl/PDtarget/history/脚测量/X238，必须bitexact；box要求新集合接触存在而旧plane-only遗漏；blocks仅验证地形高度查询和有限接触管线，**不执行行走或要求身体已踩到障碍**。输出失败保留。

下一阶段须在预检实际审核后再接当前064313四NN与真实颈部队列的V_empty完整管线，固定1s settle+4.8s教师预热+9.6sANN／无reset／每case1232000原生步。新旧metrics和实际地形接触必须平行报告；若运动没有到达x≥0.8凸台区域，地形跨越为NOT_TESTED。旧StaticTeacher另有直接世界z几何路径，需要用实际暖启动接触区域确认仍在平地，不能默认为已地形自适应。

本次保留源：71个mesh、18个原三球pad、自由根、42有限PD、重力与原时间步；不添加hook／pulvilli／黏附／外力／根部位置驱动，不训练任何NN，不改变历史失败与通过结论。
