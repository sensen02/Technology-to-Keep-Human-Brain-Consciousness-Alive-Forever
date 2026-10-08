# 清理清单（本次自动生成）

- 生成时间：2026-10-03T21:05:58-07:00
- 工作区：/run/media/sensen/Data2/cell_wound_prototype
- 桌面交付：/home/sensen/Desktop/cell_wound_prototype

## A. 已删除：Python 字节码缓存（`__pycache__`，可自动重建）

```text
```

## B. 已删除：零字节文件

```text
./outputs/f20.log
./outputs/f15.log
./outputs/physicell_run/cell_rules_parsed.csv
./outputs/workbench/a25b435f842241e59de670cb1f031a94/measurements.csv
./outputs/workbench/bench/6517455578fd45bfaf58f2c45f20c5b0/measurements.csv
./outputs/workbench/d213adf626f245aaa1097922b4cecc31/measurements.csv
./outputs/workbench/ed01a2098f894820bebe3060fb4a3764/measurements.csv
./outputs/workbench/manager.lock
./outputs/workbench/2c0d6274cbb341e38e62b9388adc38d3/measurements.csv
```

## C. 保留：陈旧日志（经确认，不删）

决策：用户选择保守方案。`outputs/*.log` 共 89 个、合计 172,655 字节（约 169 KiB），
是 10-01 至 10-02 的扫描/探测记录，体积可忽略且属于证据链，全部保留在本目录。

## D. 保留：大件（各自仍有用途）

| 路径 | 大小 | 为什么保留 |
|---|---|---|
| /run/media/sensen/Data2/cell_wound_prototype/data/flywire/fafb_skeletons_swc.zip | 12.9 GiB | 139,273 个 FAFB 骨架；被 download_swc.py、run_real_morphology_demo.py、run_real_subtree_demo.py、audit_brain_regions.py 引用 |
| /run/media/sensen/Data2/cell_wound_prototype/vendor/pyVertexModel | 3.3 GiB | 项目脚本不引用，但带 Zenodo DOI，重下需时间；按用户选择保留 |
| /run/media/sensen/Data2/cell_wound_prototype/vendor/pvm_data | 1.8 GiB | 同上，且与 pyVertexModel/Input 有重复文件 |
| /run/media/sensen/Data2/cell_wound_prototype/vendor/engines/PhysiCell | 201 MiB | 血管生成那一层的引擎，`outputs/analysis_vasc.py` 第 31 行按路径读取 |
| /run/media/sensen/Data2/cell_wound_prototype/outputs/workbench/* | ~1 MiB | 工作台任务状态（jobs.json、bench_latest.json）是活的服务状态，删了会丢任务历史 |

## E. 已做：桌面 engine/ 同步（按用户选择）

- 目标：/home/sensen/Desktop/cell_wound_prototype/engine
- 同步前：缺 30 个文件（da_protocol.py、electrode*.py 全部、embodied/ 全部、graded_vision.py、injury_tissue.py、isolation_scenario.py、local_tissue.py、multirate_scenario.py、neck_cut_data.py、neural_active.py、neural_cond.py、neural_hist.py、neural_hybrid.py、synapse_map.py、touch_damage_scenario.py），cable.py 与 receptors.py 内容不同。
- 同步后复核：**44/44 文件 SHA256 一致，0 处不匹配**。

## F. 已做：轮次报告索引（按用户选择，不删文件）

- 新增：/run/media/sensen/Data2/cell_wound_prototype/outputs/embodied_body/REPORT_INDEX.zh-CN.md
- 原因：26 份轮次报告被 `deliver_embodied.py` 第 17–178 行按路径引用，**移动或删除会打断交付链**，因此改为建索引说明哪一份被哪一份取代。

## G. 磁盘现状

```text
2026-10-03T21:17:18-07:00
/dev/sdb1       620G  256G  364G   42% /run/media/sensen/Data2
19G	/run/media/sensen/Data2/cell_wound_prototype
```

审计明细（机器可读）：/run/media/sensen/Data2/cell_wound_prototype/outputs/disk_audit.json

## H. 意外发现并已修复：删掉空的 manager.lock 后的并发风险

- 现象：`outputs/workbench/manager.lock` 是 0 字节，被 B 步当作空文件删除。
- 后果（实测确认）：`workbench/runs.py` 第 56 行用 `open('a+')` + `flock` 做单实例保护。文件被 unlink 后，正在运行的服务**仍持旧 inode 的锁**（文件描述符有效），但**新进程会创建新文件并成功加锁**——我实测 `flock(LOCK_EX|LOCK_NB)` 在新文件上**成功获取**，即第二个服务可以同时启动，两边各持一份任务状态。
- 处置：`touch` 重建该文件（0 字节，与 RunManager 创建方式一致），并**重启服务**让锁落到当前 inode。重启后 `/`、`/api/status`、`/api/manifest`、`/api/jobs` 全部 HTTP 200，任务历史保留（`1c8337e5854e4adaaf479724ccbd3f9b`，exit 0）。
- 教训（写入规则）：`manager.lock` 属于**服务运行时状态**，不是可回收空文件；后续清理脚本的排除名单已改为包含 `*.lock`。
