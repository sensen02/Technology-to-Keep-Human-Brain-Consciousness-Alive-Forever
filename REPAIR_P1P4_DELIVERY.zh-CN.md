# 全量修复交付报告（P1–P4）

> **手工实施，不是成品。** 本文报告的是本轮按 `PLATFORM_SELFCONSISTENT_PLAN.zh-CN.md` 实施的
> 检查与控制机制。**除"已验收"一节外，任何未列出的条目都未实施。** 全部"真值"仍来自模拟器。

---

## 0. 一句话

**规划的主线已落地：三个曾经真实发生过的静默失败，现在各有一条会自动 FAIL 的不变量，
并且每一条都用"把原 bug 再植回去"的方式验证过它会亮红灯。**

| | 状态 |
|---|---|
| **P1** 体检门 + 统一运行器 + 注入回归 | ✅ 已完成并验收 |
| **P2.2** 物理边界不变量 | ✅ 已完成并验收 |
| **P2.3** 依赖与资产钉住 | ✅ 已完成（结论：**离线不可复现**，已量化阻塞项） |
| **P2.4** 力残留 | ✅ 已解释（见第 3 节；**静态≈1×，动态峰值未解释**） |
| **P3.1** truth/ 越界审计 | ✅ 已完成并验收 |
| **P3.2** 假设登记表 | ✅ 已随 P4 落地（`HYPOTHESES` 内含预注册与证否规则） |
| **P3.3** 结果契约校验 | ✅ 已完成并验收 |
| **P4** 假设检验闭环模板 | ✅ 模板 + 护栏双向验证；H1 实跑（本样本内被证否） |

---

## 1. P1：体检门 + 统一运行器 + 注入回归

### 1.1 记录体检门 `tools_episode_health.py`
判据是**实测分裂**，不是猜的：可用 episode 命中率 93.1–94.0 %，坏 episode 1.4 %，
阈值 50 % 落在两者之间的空带里。**实测结果**：

| episode | 判定 | 3 px 内命中率 |
|---|---|---|
| arena_bare_seed0 | **FAIL** | 1.4 % (20/1440) |
| arena_bare_seed1 | PASS | 93.1 % |
| arena_bare_seed2 | PASS | 94.0 % |
| arena_bare_seed3 | PASS | 93.8 % |

设计上区分四种状态：`PASS / FAIL / NOT_RUN / SKIP`。
**`SKIP`（该检查不适用）与 `NOT_RUN`（该测而缺数据）分开** —— 混为一谈正是
"没测"悄悄变成"没问题"的路径；`NOT_RUN` 阻止 PASS，`SKIP` 不阻止。

### 1.2 统一运行器 `run_all_checks.py`
- 自动发现 **29 个自检** + **6 项额外检查**，全程 35 项。
- **解释器自适应**：源码标记只是快路径；在 `venv` 下因缺 flygym/mujoco 而失败的测试
  **自动改用 `venv_body` 重试并记录该事实**。实测：**5 项需要 venv_body，其中 3 项只靠重试才发现**。
  没有这一步，运行器会报 4 个**假失败**，而"为不真实的原因见红"会训练人忽略红灯。
- 运行器**还抓到了一个真回归**：`tools_test_electrode_payload.py` 在 `venv` 下
  **静默跳过整个 MuJoCo 段**而显示 PASS，在正确解释器下 FAIL 2 项。已修（见第 5 节）。

### 1.3 注入回归 `tools_regression_injections.py`
**这是本轮最重要的文件**：一个从未被看见失败过的检查，和一个不可能失败的检查无法区分。

| 注入 | 植入的真实 bug | 应触发的检查 | 结果 |
|---|---|---|---|
| `unit_1000x` | `MODEL_FORCE_TO_N` 改回 `1.0e-3` | `derived_force_unit` | **捕获**（无故障时 PASS） |
| `gate_no_scaling` | 门限去掉按帧高归一化 | `gate_resolution_scale` | **捕获**（无故障时 PASS） |
| `empty_fiducials` | 指向候选表无基准点的 episode | `episode_health.health` | **捕获**（对照 episode 仍 PASS） |

每条注入都带**对照**：不植故障时同一检查必须 PASS。没有对照，一个硬编码为 FAIL 的检查
看起来也像成功。

---

## 2. P2.2：五项不变量 `tools_invariants.py`

| 不变量 | 判据 | 结果 |
|---|---|---|
| `derived_force_unit` | 代码里的力单位常数 == 由模型自身质量单位推出的值 | **PASS**（推导 = 1e-6 N，代码 = 1e-6 N） |
| `settle_balance` | 驱动模式下腿承担 ≈ 体重 | **PASS**（0.656–1.718 × 体重） |
| `force_channel_order_of_magnitude` | 力通道在体重的数量级内 | **PASS**（均值 6.24 ×，带 = [0.05, 50]） |
| `gate_resolution_scale` | 门限在参考帧高上逐位不变、且随帧高线性放大 | **PASS** |
| `no_bare_unit_constant` | 顶层脚本里没有 `全大写 = 数值` 形状的单位常数 | **PASS**（双向验证过灵敏度） |

**单位推导的定论依据**（写进了 `derived_force_unit` 的 provenance）：
模型质量属性合计 `1.02431e-3`；**按 kg 读果蝇重 1.024 g（真果蝇的 1000 倍），按 g 读重 1.02431 mg（正确）**
→ 质量单位是**克** → 力单位 `g·mm/s² = 1e-6 N = 1 µN`。

---

## 3. P2.4：力残留 —— 已解释，不再是"未解之谜"

本轮的不变量把两种模式**分开测**，机制因此清楚：

| 模式 | 六腿合力（模型单位） | / 体重 | 跗节接触数 |
|---|---|---|---|
| **被动**（致动器关） | 0.70 / 0.72 / 0.80 / 1.14 | **0.070 – 0.113** | 1–2 |
| **驱动**（挂 CPG） | 17.26 / 11.26 / 11.68 / 6.59 | **0.656 – 1.718** | 4–8 |

**结论（重要）**：**准静态下腿承担的正是自己的体重（≈1×），力标度是对的。**
行走回合里"均值 4.3–12.2×、峰值 25.3–27.8× 体重"是**动态瞬态**（刚性接触解算小穿透），
**不是尺度错误**。

**同时这暴露一个真实缺陷**：被动时腿只承担 7–11 %，身体坐在胸/腹上 ——
与肌肉模型的 `stood = False`（5/5 配置）同一性质。**平台能落地只是因为 CPG 在驱动腿。**

**仍未解决**：动态峰值 25× 的量级尚未解释。**因此任何绝对力都不得作为生物载荷引用** ——
这条已写进 `force_channel_order_of_magnitude` 的 note。

---

## 4. P2.3：依赖与资产钉住 `tools_pin_environment.py`

**实测结论：`offline_reproducible = False`**，阻塞项被精确列出：

| 阻塞 | 详情 |
|---|---|
| FlyMimic 体网格**不在本地** | 由 flygym 的 `_resolve_mesh_files` 从 S3 缓存解析，未随仓库分发 |
| `mujoco` / `flygym` / `pyarrow` **未钉版本** | `requirements.txt` 只有 4 个包；实测 `venv_body` 为 mujoco 3.9.0 / flygym 2.1.0 |

**这不是修复，是记录** —— 钉版本与把网格纳入版本控制涉及体积与许可，是决定不是顺手活。
工具本身修过两处自身错误：`flygym` 没有 `__version__`（改用 `importlib.metadata`）、
草纹理路径猜错导致**误报缺失**（路径本身是"在不在"的字段，误报最不可接受）。

---

## 5. P3.1：truth/ 越界审计 `tools_truth_boundary_audit.py`

**扫描方法本身修过一次，值得记下**：第一版剥掉注释与字符串以免误报散文，
**结果把数据键也剥掉了**（`z["truth/thorax_mm"]` 里键就是字符串），报出 3 个脚本而非 **23 个**。
现改用 AST，只排除真正的 docstring。

**结果：24 个脚本、84 处 truth 读取，全部已声明，0 越界。**
并且**非 `simulator` 角色的每一条声明都必须给出理由，否则判 FAIL**（空声明就是能放行越界的标签）。

审计还**修正了我先前一个用词不实的地方**：`tools_marker_coverage.py` 的可见率是
**以模拟器自身标记位置为锚**去数有多少相机能检出 —— 它需要真实检出（所以不是纯预测），
**但无法发现被模拟器放错位置的标记，因此不是独立的图像测量**。
我先前把"trochanterfemur 可见率 6.5–33 %"当作测量引用过，该限制现已写入声明。

---

## 6. P3.3：结果契约校验 `tools_result_contract_audit.py`

契约四字段：`units` / `provenance` / `status` / `what_this_does_not_establish`。

**它对策略做了明确取舍**：只对**本轮新增的 8 个输出**强制（违反即 FAIL）；
其余 91 个历史 JSON 计为**backlog**，因为让四十个文件永久见红只会训练人忽略检查。

**首跑就抓到我自己 6 个输出不合规**，已全部补齐（`units` 对纯审计文件是
"不适用 + 理由"，但**必须写明**，不能不写）。

---

## 7. P4：假设检验闭环模板 `tools_hypothesis_test.py`

模板强制每人事先声明：扰动、对照、观测量、**证否规则**、预注册时间。

**护栏（本项目最有价值的机制）双向验证通过**：

```
空扰动（对照与扰动指向同一批 episode） -> NOT_RUN      ← 必须
真扰动（bare vs full）                 -> PASSED_GUARD ← 必须
SELFTEST: PASS
```

这条护栏不是装饰：本项目正是靠它才发现"受试与完整逐位相同"，进而查出
**90 条肌肉表只含前腿 15 条、另外 75 条从未被驱动** —— 否则会记下一个干净的假阴性。

**首个登记假设 H1**（挂载完整电子载荷是否改变行走的地面反力分布）：

| | 值 |
|---|---|
| 对照（bare，2 episode）均值 | 6.0182 × 体重 |
| 扰动（full，2 episode）均值 | 5.1737 × 体重 |
| 效应 | **−0.8446** |
| 负对照：对照种子间散布 | **3.3771** |
| 判定 | **本样本内被证否**（效应小于种子散布） |

**诚实边界**：两臂各只有 2 个种子，不构成分布；只有 rig 名匹配，无因果；
力标度残留已记录未解释；**与真实果蝇无关**。

---

## 8. 顺带修掉的真回归

`tools_test_electrode_payload.py` 的断言
`len(rep["body_names_added"]) == n_sites_with_mass` **已经过时**：
设计早已从"新增 attachment body"改为"编辑编译模型自身的质量"，
因为原方案**实测会毁掉果蝇**（nbody 69→66、含 "rostrum" 的 body 消失、眼睛从 2 只变 1 只；
增至 5 个 body 时 64 个 body、**0 只眼**）。

现已改为断言那个促成改设计的**安全性质**：`nbody 不变` 且 `body_names_added == []`。
**为什么一直没被发现**：在 CPU 环境下整段 MuJoCo 测试被**静默跳过**并显示 PASS ——
这正是统一运行器存在的理由。

---

## 9. 交付与复现

```bash
cd /run/media/sensen/Data2/cell_wound_prototype
# 一键全量（35 项）
OPENBLAS_NUM_THREADS=1 ./venv/bin/python run_all_checks.py
# 单独跑各项
./venv/bin/python tools_episode_health.py --all
MUJOCO_GL=egl ./venv_body/bin/python tools_invariants.py --all
MUJOCO_GL=egl ./venv_body/bin/python tools_regression_injections.py
./venv/bin/python tools_truth_boundary_audit.py
./venv/bin/python tools_result_contract_audit.py
./venv/bin/python tools_hypothesis_test.py --selftest
MUJOCO_GL=egl ./venv_body/bin/python tools_pin_environment.py
```

| 新文件 | 作用 |
|---|---|
| `run_all_checks.py` | 统一运行器（35 项，解释器自适应） |
| `tools_episode_health.py` | 记录体检门 |
| `tools_invariants.py` | 五项物理/几何不变量 |
| `tools_regression_injections.py` | 三个注入回归 |
| `tools_truth_boundary_audit.py` | truth/ 越界审计 |
| `tools_result_contract_audit.py` | 结果契约校验 |
| `tools_hypothesis_test.py` | 假设检验模板 + 护栏自检 |
| `tools_pin_environment.py` | 环境与资产清单 |
| `tools_fiducial_residual_figure.py` | 基准点残差图（上一轮） |

输出位于 `outputs/`（`all_checks.json`、`invariants.json`、`regression_injections.json`、
`truth_boundary_audit.json`、`result_contract_audit.json`、`hypothesis_tests.json`、
`environment_lock.json`、`arena/episode_health.json`），全部带契约四字段。

---

## 10. 仍**未**做的（明确列出，不含糊）

1. **`truth/` 审计是静态的。** 更强的版本是**运行时**：把 episode 的 truth 键去掉再跑测量，
  要求输出不变。已写进审计的 `what_this_does_not_establish`。
2. **离线复现仍不可用**（P2.3 已量化，未修）。
3. **动态峰值 25× 体重的来源未解释**（静态已对）。
4. **被动腿不支撑身体**（7–11 %）**未修** —— 那是行为底座问题，不在"平台自洽性"范围。
5. **P3.2 的登记表以代码内 `HYPOTHESES` 存在**，尚未独立成可审计的登记目录。
6. **H1 只有 2 × 2 个 episode**，不足以支撑统计结论；需按登记表补齐种子后才可复跑判定。
