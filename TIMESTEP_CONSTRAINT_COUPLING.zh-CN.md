# 时间步长与约束参数耦合：独立只读审计

## 结论与边界

**当前 zero_floor 模型不存在 100→50→25 μs 细化触发的 `refsafe` 接触/限位时间常数变化。** 编译后的所有非空 `solref` 数组均逐行等于 `[0.02, 1]`，而三档 `2*dt` 分别为 `0.0002 / 0.0001 / 0.00005 s`。最小正时间常数 0.02 s 分别是安全阈值的 100/200/400 倍；不是自定义 0.0001 s 接触时间常数。

**这不等于“只改变 activation 而身体数值完全不变”。** 修改 `model.opt.timestep` 是全系统时间离散改变：身体积分、接触检测采样、约束求解时序及轨迹反馈仍可能改变。不能仅凭更细 dt 的站立/支撑/力变化，将物理因果归于 activation overshoot 修复；也不能用“高 dt 没警告、没爆炸”代替收敛证据。本报告无生理/真实昆虫因果结论、无新控制器、无 episode、无身体积分。是手动实施的数值审计，不是成品行为验证。

## 1. 对象、canonical load 与静态 XML

- canonical：`/run/media/sensen/Data2/cell_wound_prototype/outputs/muscles_six_legs/fruitfly_six_leg_standing.xml`
  - SHA256 `65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6`
- zero_floor：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/zero_floor/fruitfly_zero_floor.xml`
  - SHA256 `b4f6b375855420100c4f385bf750ed7d705702095fabd1b9ea13959c5c857008`
- 装载入口：`/run/media/sensen/Data2/cell_wound_prototype/engine/canonical_physics.py` 的 `load_model()` 无 XML 文本参数时调用 FlyGym `_load_mjcf(...).compile()`。
- 实际 FlyGym 源码：`/run/media/sensen/Data2/cell_wound_prototype/venv_body/lib/python3.12/site-packages/flygym/compose/fly/musculoskeletal.py` 第105–132行：`MjSpec.from_file`，修改离屏 framebuffer，解析 mesh/texture 路径；不覆盖 solref、solimp、积分器或时间步长。
- 两份 XML 均已 ElementTree 解析：没有 include，没有任何带 `solref` 或 `solimp` 的属性，没有 option/flag。根 default/main 中 joint/geom/tendon 也没有这些覆盖。因此不存在隐藏在 defaults 类继承中的自定义约束时间常数。显式 geom margin 为 0.001。
- 两份 XML option 为 `timestep=1e-4`、`gravity=0 0 -9801`。积分器等取 MuJoCo 默认值，再由编译字段确认。
- 原有 `/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/zero_floor/model_manifest.json` 记录 canonical/branch 编译数组差异只有 `actuator_ctrlrange`；本轮核对冻结 SHA，并只重新编译 zero_floor 一次。**没有声称本轮重新编译了 canonical**；canonical 的结论来自静态 XML、实际装载器源码与现有冻结编译差异记录。
- 注意 `canonical_physics.py` 还定义了其他命名 ContactSet（例如 upstream 的 0.0002 s），不能把未应用的备选配置当成本次实际模型。

## 2. 编译后的真实字段（MuJoCo 3.9.0）

| 字段 | 数组形状 | 所有行的精确值 / 最小首项 |
|---|---:|---|
| `geom_solref` | 72×2 | `[0.02,1]` / 0.02 |
| `jnt_solref` | 43×2 | `[0.02,1]` / 0.02 |
| `dof_solref` | 48×2 | `[0.02,1]` / 0.02 |
| `tendon_solref_lim` | 90×2 | `[0.02,1]` / 0.02 |
| `tendon_solref_fri` | 90×2 | `[0.02,1]` / 0.02 |
| `eq_solref`、`pair_solref`、`pair_solreffriction`、`flex_solref` | 各0×2 | 空数组，最小值不适用 |

没有显式 contact pair（`npair=0`），但这**不表示没有接触**；运行时由 geom 碰撞生成隐式 contact。`neq=nflex=0`。各非空 geom/joint/dof/tendon 的 `solimp` 行均为 `[0.9,0.95,0.001,0.5,2]`。

其余真实字段：

- `opt.integrator=0`（Euler）；`solver=2`（Newton）；`cone=0`（pyramidal）；`jacobian=2`（auto）。
- `iterations=100`、`tolerance=1e-8`、`ls_iterations=50`、`ls_tolerance=0.01`、`noslip_iterations=0`、`noslip_tolerance=1e-6`、`impratio=1`。
- `disableflags=0`、`enableflags=0`：refsafe 启用、contact override 未启用。虽然 `opt.o_solref=[0.02,1]`，此次并未以 override 替代 contact 参数。
- 所有 geom `solmix=1`、`priority=0`、`condim=3`、`friction=[1,0.005,0.0001]`、`margin=0.001`、`gap=0`。各 geom solref 相同，混合不会生成更短的时间常数。
- 所有 tendon `limited=false`；全部 dof/tendon frictionloss 为0。数组中存在默认 solref 不意味着这些约束当前实际生成了行。
- 90个执行器：`dyntype=4`（muscle），`dynprm` 每行精确为 `[0.0001,0.0004,0,0,0,0,0,0,0,0]`，`actlimited=false`、`ctrllimited=true`、zero_floor `ctrlrange=[0,1]`。
- XML default/muscle 中的 `dynprm=0.01 0.04 ...` 被实际执行器属性覆盖；不可误报为编译后的值。**activation 的 0.0001/0.0004 s 与 constraint solref 的 0.02 s 是不同字段/机制。refsafe 不会把 muscle activation tau 自动改成2dt。**

全量编译数组（非仅 unique/min）、各档 contact 与 efc 行保存在：
`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/timestep_constraint_audit.json`。

## 3. refsafe 在哪里生效，以及本模型为何不触发

对应安装版本的官方源码 `engine_core_constraint.c` 中 `getsolparam()`：先按 `efc_type/id` 读取 equality/joint limit/dof friction/tendon limit/tendon friction 或实际 contact 的参数，再对**局部参数副本**执行：

```c
if (!mjDISABLED(mjDSBL_REFSAFE) && solref[0] > 0) {
  solref[0] = mju_max(solref[0], 2*m->opt.timestep);
}
```

正格式 `solreffriction[0]` 也有对应保护。负值 direct-format 不应用此正时间常数下限；零 friction 参数是特殊默认值，不能当成“零时间常数被夹到2dt”。该保护发生在约束构造，不会把原 `model.geom_solref` 数组重写。因此**只比较模型数组、或只看 contact.solref，不足以观察自动下限的实际影响**；本轮还检查 `efc_KBIP`。

随后 `mj_makeImpedance()` 的正格式系数为：

- `K = 1 / (solimp[1]^2 * timeconst^2 * dampratio^2)`
- `B = 2 / (solimp[1] * timeconst)`

因此如果实际 `timeconst=0.0001`，100 μs时会变成0.0002、50 μs时为0.0001，K/B 可分别变化4倍/2倍。这是有效的审计风险，**但不是本模型的实测情况**。这里三档实际均采用0.02，不能拿 activation tau 代入 constraint 公式。

参考来源：

1. [MuJoCo 3.9.0 实际约束实现：getsolparam / mj_makeImpedance](https://github.com/google-deepmind/mujoco/blob/3.9.0/src/engine/engine_core_constraint.c)
2. [MuJoCo 3.9.0 XML refsafe 文档](https://mujoco.readthedocs.io/en/3.9.0/XMLreference.html#option-flag-refsafe)
3. [MuJoCo solver 参数说明](https://mujoco.readthedocs.io/en/3.9.0/modeling.html#solver-parameters)

已核对本地安装头文件 `/run/media/sensen/Data2/cell_wound_prototype/venv_body/lib/python3.12/site-packages/mujoco/include/mujoco/mjtype.h` 的 `mjDSBL_REFSAFE=1<<12`；同安装的 `sceneAPI.h` 说明默认 refsafe=true，joint/collision/tendon/equality schema 的默认 solref=[0.02,1]。运行库报告版本3.9.0，编译后 flags 与默认值一致；没有宣称安装 wheel 内含完整 engine C 源码，C 实现来自同版本官方源文件。

## 4. 同状态 forward 核验：真实活动约束行

从冻结 zero_floor traces 读取同一 `seated_qpos`，每档新建 MjData，速度、activation、control 均为零，分别设置仅内存中的 `opt.timestep=100/50/25 μs`，调用 `mj_forward`，**绝不调用 `mj_step` 或其他积分函数**。最后恢复原 timestep。每档 `time=0`、qpos完全未变，qvel/act仍为零。

三档均有 **12个contact、53个已实例化efc行**：

- 前5行：type=3（joint limit），joint id 为 `[2,5,17,30,36]`。
- 后48行：type=6（pyramidal contact），contact id 0…11 各4行，来自 condim=3 的摩擦金字塔。
- contact geom id 配对及数量：`(0,26)×3`、`(0,34)×3`、`(0,43)×2`、`(0,53)×3`、`(0,61)×1`。
- 每个实际 contact 的 `solref=[0.02,1]`、`solreffriction=[0,0]`、`solimp=[0.9,0.95,0.001,0.5,2]`；efc_address依次5,9,…,49。
- 所有53行的 K=2770.083102493075、B=105.26315789473685。KBIP 的 impedance/导数组合为 `(0.91528307651657,-78.18715116071168)`、`(0.9452063967686135,-43.78859774592664)` 或 `(0.95,0)`。
- 三档之间 `efc_type/id`、`efc_KBIP`、`efc_R`、`efc_D`、`efc_aref`、`efc_force`、`qacc` 均 **bitwise equal**。全部 warning counters 为0。

这里“活动行”指本次 forward 实例化的约束行，**不等于每行都有非零接触力或处于同一种 solver active-set 状态**。本轮未进行跨轨迹接触模式穷举，也没有证明后续演化相同。同状态观测支持“本次约束参数下限未改变”，不是身体积分收敛证明。

## 5. 执行、保护、复核与交付

新增（没有改既有源文件、canonical/zero_floor XML、模型参数或控制器）：

- `/run/media/sensen/Data2/cell_wound_prototype/run_timestep_constraint_audit.py`
- `/run/media/sensen/Data2/cell_wound_prototype/test_timestep_constraint_audit.py`
- `/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/timestep_constraint_audit.json`
- `/run/media/sensen/Data2/cell_wound_prototype/TIMESTEP_CONSTRAINT_COUPLING.zh-CN.md`

执行环境为 `/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python`。三个轻量测试通过（默认不触发下限、小正值/负格式/关闭refsafe分支、两份XML无隐藏覆盖）；测试不编译MuJoCo。单次重型审计用共享 `heavy_job` memory guard：最多等待60s，总deadline180s，need2200MiB，地址空间cap4608MiB。实际受保护工作约0.145s，VmPeak约1304.84MiB，最大RSS约592.98MiB，已释放共享锁。被审计来源SHA前后全部一致。没有新图或长时运行；交付的是静态/同状态数据审计。

**推荐父任务披露措辞：**“dt细化针对activation显式积分误差；独立审计未发现该模型refsafe约束时间常数随100/50/25μs变化，且同一初始状态的约束构造完全一致。不过dt改变仍作用于整个身体数值系统，因此不将行为差异归结为孤立activation干预，也不据此声称物理或生理因果、行为收敛。”
