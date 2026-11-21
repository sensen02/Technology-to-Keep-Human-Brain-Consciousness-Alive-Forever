# P1 单针颈部损伤与嗅觉绕墙复跑

从研究项目根目录运行。已有回合按目录归档，复跑时使用新的输出目录。主项目读取本机已安装的两个 Python 环境、BANC 数据与 Eon 模型。

```bash
cd /run/media/sensen/Data2/cell_wound_prototype
project="$PWD"
base="$project/experiments/open-brain"
joint="$base/20261119T202650Z-neck-odor-integration"
py="$project/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python"
runner="$base/20261119T012530Z-survival-orchard/run_env.sh"
out="$(mktemp -d)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
(
  cd "$joint/omnisim_source"
  sha256sum --check --quiet SHA256SUMS
  "$py" -B -m omnisim.scenes.neck_implant "$project" --runtime-replay --diagnostic-offset-um 2.43 --output "$out/blocked.json"
  "$py" -B -m omnisim.scenes.neck_implant "$project" --runtime-replay --diagnostic-offset-um 3.3 --output "$out/partial.json"
)
cmp "$joint/profiles/blocked.json" "$out/blocked.json"
cmp "$joint/profiles/partial.json" "$out/partial.json"
"$runner" "$joint/run.py" "$out/live_intact_left" --profile "$out/blocked.json" --mode intact --side 1
"$runner" "$joint/run.py" "$out/live_target_partial_left" --profile "$out/partial.json" --mode target_only --side 1
"$runner" "$joint/run.py" "$out/live_target_block_left" --profile "$out/blocked.json" --mode target_only --side 1
"$runner" "$joint/run.py" "$out/live_shaft_left" --profile "$out/blocked.json" --mode shaft --side 1
"$py" -B "$joint/audit.py" "$out"
"$py" -B "$joint/probe.py" > "$out/receiver-probe.json"
cmp "$joint/RECEIVER_PROBE.json" "$out/receiver-probe.json"
```

`shaft` 按具名轴突将同一针位在所有地面模型端口上的损伤一起作用。`target_only` 只把选中根的损伤施加在同一运行器，用于单通道干预。`intact` 走同一入口并保留健康动力学。另一个镜像用 `--side -1`；配套审计当前匹配归档的左绕试验。每个回合输出 APPLIED.json、NECK.npz、BODY.npz、SENSE.npz、ARM.json、RESULT.json。

`omnisim_source` 收录本次单针场景加载的 OmniSim 模块、原生后端源码与本机编译库。上游源码在 `/run/media/sensen/Data1/omnisim`。若需在兼容 ROCm 主机重建，可在该快照内用 `cmake -S native -B build/native -DOMNISIM_HIP=ON` 和 `cmake --build build/native` 编译，重建库的文件哈希须另记。嗅觉实验和果园迁移的源目录位于 `/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain`。主项目保留原训练和复验清单、选用权重、原始轨迹及数据校验单。RUNTIME_SHA256.json 记录部署依赖原位置和两处改为相对位置的导入。
