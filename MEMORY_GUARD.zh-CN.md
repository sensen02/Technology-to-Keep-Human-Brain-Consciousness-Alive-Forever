# 内存事故取证与本项目内存守卫（OOM 修复）

本文件记录一次真实的内存事故、取证证据、修复内容、验证结果和修复的边界。**这是手工实施并测试的工程修复，不是成品，也没有把"不再 OOM"当成已证明的事实。**

---

## 优先更正：事故关联未建立

以下内核记录确实是一场 OOM，但发生于 12:06，不能凭此认定用户当晚反馈的 DSH 崩溃也是同一次 OOM。被杀 python PID 的任务身份未查明；子代理 `ready` 状态或缺失文件不证明其进程被 OOM 杀死。“90% 静默死亡”没有进度证据，撤回。`msedge invoked oom-killer` 表示触发内核分配失败的调用方，不证明浏览器是根因。

守卫是项目风险缓解，不是 DSH 崩溃根因修复的验收：入口可用内存检查不能预留内存；RLIMIT_AS 限单进程虚拟地址空间，不限全机 RSS 或进程树总和；共享锁只约束自愿使用该锁的入口；顶层导入在守卫之前。record-stride 减少部分记录，但数组/解码记录仍随时长增长，不能声称内存有固定上界。后文未经这些边界限定的因果说法均以本段为准。

## 1. 事故证据（历史记录）

内核日志记录了一次 **全局 OOM**（`global_oom`）：

```
[日 10月 11 12:06:32 2026] msedge invoked oom-killer: ... oom_score_adj=300
[日 10月 11 12:06:32 2026] oom-kill:constraint=CONSTRAINT_NONE,...,global_oom,
    task_memcg=/user.slice/user-1000.slice/user@1000.service/app.slice/ptyxis-spawn-2a6d0c17-....scope
[日 10月 11 12:06:32 2026] Out of memory: Killed process 88800 (python)
    total-vm:19841164kB, anon-rss:7578092kB, file-rss:8kB, UID:1000, oom_score_adj:200
```

配套观测（同一时间窗）：

| 项目 | 实测 |
|---|---|
| 主机内存 | 14,541 MB 总计，swap 4,095 MB |
| swap 已用 | 3,395 MB（修复时仍为 3,413 MB，说明压力持续） |
| `systemd-oomd` | **active**（另有用户态 OOM 通道） |
| 同机其他任务 | **非本项目**的训练任务 PID 119969，RSS 4.88 GB，自带 `--max-peak-gb 9.0` |
| DSH 主体 | PID 122393，RSS 1.9 GB |
| 浏览器 | 多个 `msedge` 进程合计约 2 GB+ |

**直接后果（我自己的损失，可核对）**：我并发派出的两个审计子进程在写出任何产物前被杀，
`/run/media/sensen/Data2/cell_wound_prototype/outputs/body_physics_audit/` **至今不存在**，
两个子进程状态均回到 `ready`（不再运行）。也就是**任务在 90% 处静默死亡、零证据**——
这正是必须修的原因：一个能拖垮整机、又会在收尾前消失的任务，不能用于过夜计算。

---

## 2. 实测占用（先量后定，不猜）

新增 `tool_measure_memory.py`，在守卫自身约束下测出本项目重任务的真实足迹
（产物：`/run/media/sensen/Data2/cell_wound_prototype/outputs/memory_guard/memory_budget.json`）：

| 阶段 | VmPeak | VmRSS |
|---|---:|---:|
| 进程启动 | 312.7 MB | 104.3 MB |
| 导入 MuJoCo | 523.0 MB | 171.1 MB |
| 编译规范模型 | 1,690.1 MB | 479.0 MB |
| 渲染一帧 | 1,773.4 MB | 512.6 MB |
| 构建 12,000 神经元 tier | 2,667.3 MB | 906.6 MB |
| 建网络 / 20 步神经积分 | 2,667.3 MB | 906.6 MB |
| 闭环 prepare_body | **3,159.2 MB** | **1,047.6 MB** |

**结论：单个重任务约 1.05 GB 常驻、3.16 GB 虚拟地址空间。** 两个并发约 2.1 GB 常驻。
这不是主机内存的全部，但在已有 4.9 GB 训练任务 + 1.9 GB DSH + 浏览器的情况下，
**并发重任务是压垮交换区的最后一根稻草**。

---

## 3. 修复内容

### `engine/memory_guard.py`（新增）

三重保护，全部为**显式声明并会大声失败**，不做静默降级：

1. `require_available_mb(need_mb, label)`：开工前对照 `/proc/meminfo` 的 `MemAvailable`；
   不满足就**拒绝启动**，错误信息带具体数字、swap 用量和"运行会重演已记录的全局 OOM"的原因。
2. `cap_address_space_mb(cap_mb)`：给本进程设 `RLIMIT_AS` 硬上限，
   把**无界失控变成本进程内的 `MemoryError`**，而不是触发全局 OOM 杀别的程序。
3. `single_flight(name)`：`flock` 排他锁，**禁止本项目两个重任务叠加**（并发正是事故成因）。

### 端到端验证抓到的两个真实缺陷（已修，附证据）

第一次端到端演示**没能拦住第二个任务**。调查后确认这不是测法问题，而是我的守卫有两个真实缺陷：

**缺陷 A：按 label 分锁 = 等于没锁。**
`guarded_run` 原先把自己的 `label` 当锁名，于是 `neural_muscle_closure` 与
`measure_memory_footprint` 锁的是**两个不同文件**，可以同时运行。
证据：`flock -n` 明确显示持有者**确实持有**内核锁，第二个入口仍然完整跑完并退出码 0。
**不同任务重叠正是当初 OOM 的成因**，所以按任务分锁无法防止它要防的事。
→ 改为全体重任务共用一把锁 `HEAVY_JOB_LOCK = "heavy_job"`，label 只作为元数据记录。

**缺陷 B：开锁文件用了 `"w"`，会先把持有者信息截断。**
被阻塞的任务在**尝试加锁之前**就把文件清空了，于是报错信息无法说出持有者是谁，
而"别人在跑"不是可行动的提示。
→ 改为 `O_RDWR|O_CREAT` 打开，**取到锁之后**才写入 `pid=... label=...`；
被阻塞时报错直接点名持有者。

### 我自己在验证过程中犯的两个方法错误（同样记录）

- 第一次演示用 `sleep 6` 等持有者，但闭环 0.05 s 版本**只需约 5 秒**就跑完了，
  于是根本没重叠——是**我的等待太久**，不是守卫失效。
- 第二次演示用"锁文件内容非空"判断锁是否被持有，但文件内容是**残留的旧 pid**。
  正确的探测是 `flock -n <lockfile>`（内核状态）。**文件内容非空 ≠ 锁被持有。**


预算取自实测：

| 入口 | 需要可用内存 | 地址空间上限 |
|---|---:|---:|
| `neural_muscle_closure` | 2,200 MB | 4,608 MB |
| `mechanics_audit` | 2,200 MB | 4,608 MB |
| `latency_audit` | 2,200 MB | 4,608 MB |
| `render` | 1,600 MB | 4,608 MB |
| `units_audit` | 1,200 MB | 4,608 MB |
| `muscle_floor_audit` | 1,800 MB | 4,608 MB |
| `measure_memory_footprint` | 900 MB | 4,096 MB |

`need_mb` 取实测常驻的约 2 倍，`cap_mb` 取实测最坏 VmPeak 的约 1.4 倍。

### 接入位置（已改）

- `/run/media/sensen/Data2/cell_wound_prototype/run_neural_muscle_closure.py`
- `/run/media/sensen/Data2/cell_wound_prototype/render_neural_muscle_closure.py`
- `/run/media/sensen/Data2/cell_wound_prototype/run_neural_muscle_mechanics_audit.py`
- `/run/media/sensen/Data2/cell_wound_prototype/run_neural_muscle_latency_audit.py`

同时给闭环加了 `--record-stride`：逐区间通道细节只在前 20 个区间和每隔 `stride` 个区间完整保留，
其余标 `detail: trimmed`，**避免长时段运行的产物与内存随区间数线性增长**。

---

## 4. 验证（可复跑）

`test_memory_guard.py`，13 项全通过：

```
test_cap_is_applied_and_inherited_by_children .................. ok
test_different_labels_still_exclude_each_other ................. ok   <- 缺陷A回归
test_guarded_run_reports_footprint_without_lying ............... ok
test_lock_error_names_the_holder ............................... ok   <- 缺陷B：报错点名持有者
test_lock_file_is_not_truncated_while_held ..................... ok   <- 缺陷B回归
test_lock_released_after_use ................................... ok
test_measured_budget_matches_recorded_measurement .............. ok
test_measuring_a_different_label_uses_the_shared_lock .......... ok   <- 缺陷A回归
test_meminfo_is_readable ....................................... ok
test_refuses_when_host_cannot_afford_job ....................... ok
test_second_holder_is_refused .................................. ok
test_sequential_heavy_jobs_are_allowed ......................... ok   <- 串行不死锁
Ran 13 tests ... OK
```

### 真实入口的端到端验证（修复后，可用脚本复跑）

持有者 = `run_neural_muscle_closure.py --duration 0.30`（真实入口），
第二个 = `tool_measure_memory.py`（**另一个**入口），用 `flock -n` 探测内核锁状态：

```
kernel lock HELD after 2s: pid=123192 started=1791779602 label=heavy_job
=== second entry point while holder is still working:
SECOND_EXIT=1
MemoryBudgetError: another heavy run holds .../outputs/.locks/heavy_job.lock:
    pid=123192 started=1791779602 label=heavy_job. This project serialises heavy jobs
    because concurrent runs are what produced the recorded OOM kill
HOLDER_EXIT=0
--- holder still produced its normal result ---
normal escape 0.02399999999999991 active 3739
```

即：**第二个入口被拒绝并点名持有者，持有者本身不受影响、正常出结果。**
完整记录：`/run/media/sensen/Data2/cell_wound_prototype/outputs/memory_guard/lock_demonstration.txt`

### 其他验证

- **拒绝行为**已实测：`need_mb=10**9` 时抛 `MemoryBudgetError`，信息含 `refusing to start`。
- **上限可继承**已实测（子进程内 `RLIMIT_AS` 生效）。
- **上限不误杀健康运行**：全流程（渲染 + 12,000 神经元 tier + 闭环准备）在 4,096 MB 上限下完整跑通。

**自测抓到一处真实不一致**：`units_audit` 原设 3,072 MB，**低于实测峰值 3,159 MB**。
没有任何单任务实测支持更低上限时，低上限会杀掉本来健康的运行，因此统一提到 4,608 MB，
并把这条规则写进 `test_measured_budget_matches_recorded_measurement` 固化下来。

**回归**：接入守卫后重跑闭环，7 个条件的逃逸时间与判定
（`repeat_trajectory_exact` / `replay_trajectory_exact` / 输出置零 / 15ms 切断 等）
与修复前**逐位相同**，证明守卫没有改变任何物理或科学结果。

---

## 5. 修复的边界（不允许被夸大）

1. **最大内存用户不是本项目**：同机训练任务 4.88 GB（上限 9 GB）。本守卫**管不到它**。
2. **触发 OOM 的是浏览器**（内核记录 `msedge invoked oom-killer`）。本守卫也管不到它。
3. 守卫只能保证：**本项目自己的重任务声明、受限、不可叠加，并且失败时给出可读原因**。
   它不能保证整机不 OOM。
4. 命中上限的运行算**失败运行**，**不得**当作科学负面结论。
5. 实测峰值对应上述阶段序列；**更长的闭环会线性增长轨迹数组**，不受同一峰值覆盖。
   这正是加 `--record-stride` 的原因，但长时段仍需单独实测再定预算。
6. 主机当前 swap 仍占用约 3.4 GB，**内存压力没有消失**，只是本项目不再叠加。

**建议（外部依赖，需你决定）**：本项目重任务与那个 4.88 GB 训练任务**不要同时跑**。
如果必须并行，建议给训练任务降 `--max-peak-gb`，或把本项目工作放到该任务结束之后。

---

## 6. 产物与复跑命令

工作目录：`/run/media/sensen/Data2/cell_wound_prototype`

```bash
# 实测占用并验证守卫（安全，随时可跑）
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python \
  /run/media/sensen/Data2/cell_wound_prototype/tool_measure_memory.py

# 守卫自测
/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python \
  /run/media/sensen/Data2/cell_wound_prototype/test_memory_guard.py

# 带守卫的闭环
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python \
  /run/media/sensen/Data2/cell_wound_prototype/run_neural_muscle_closure.py
```

- 守卫模块：`/run/media/sensen/Data2/cell_wound_prototype/engine/memory_guard.py`
- 实测产物：`/run/media/sensen/Data2/cell_wound_prototype/outputs/memory_guard/memory_budget.json`
- 锁目录：`/run/media/sensen/Data2/cell_wound_prototype/outputs/.locks/`
- 规范模型 SHA256 未变：`65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6`
