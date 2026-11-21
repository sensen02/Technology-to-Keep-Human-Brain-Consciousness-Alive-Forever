# 复跑与数据身份

在主工作树运行。物理入口会创建新的回合目录。下列命令用新目录归档单次复跑，参考权重、图和各阶段训练数据保存在原实验目录。

```bash
cd /run/media/sensen/Data2/cell_wound_prototype
base="$PWD/experiments/open-brain"
run="$base/20261119T205558Z-orchard-curriculum"
body="$base/20261119T012530Z-survival-orchard/run_env.sh"
cpu="$PWD/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python"
out="$(mktemp -d)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$body" "$run/rollout.py" CLEAR "$out/baseline" --seed 37 --duration 6.4
"$body" "$run/rollout.py" CLEAR "$out/head" --seed 37 --duration 6.4 --policy "$run/S3/POLICY.npz"
"$body" "$run/deploy_body.py" CLEAR "$out/combined" --seed 37 --duration 6.4 --policy "$run/S3/POLICY.npz"
"$body" "$run/deploy_body.py" CLEAR "$out/replay" --seed 37 --duration 6.4 --policy "$run/S3_REPLAY/POLICY.npz"
"$cpu" -B "$run/learn_stage.py" 1 "$base/20261119T190005Z-odor-sac/TRAIN_DETOUR/0110000.pt" "$out/S1" "$run/R1_CLEAR" "$run/R1_DEBRIS"
"$cpu" -B "$run/learn_stage.py" 2 "$out/S1/checkpoint.pt" "$out/S2" "$run"/R{1,2}_{CLEAR,DEBRIS}
"$cpu" -B "$run/learn_stage.py" 3 "$out/S2/checkpoint.pt" "$out/S3" "$run"/R{1,2,3}_{CLEAR,DEBRIS}
"$cpu" -B "$run/terminal_replay.py" 3 "$out/S2/checkpoint.pt" "$out/S3_REPLAY" "$run"/R{1,2,3}_{CLEAR,DEBRIS} "$run/NEAR_REACH" "$base/20261119T190005Z-odor-sac"/LIVE_DETOUR_ODOR_{L,R}
```

将场地参数 CLEAR 改为 DEBRIS 可复跑草枝入口，同一位置的四个模型使用相同种子、步长、气味和提前结束规则。rollout.py 保存原 14 维头侧观测、实际执行的探索动作、口器距离、六足支撑标记、摄入库存和逐拍颈部记录。开发回合的距离、时限和探索噪声见各回合 CONFIG.json。奖励由 rewards.py 按物理时间对齐。S1、S2、S3、S3_REPLAY 各有检查点、部署权重、逐更新日志和全新进程重训验证。S3 与 S3_REPLAY 都从 S2 出发。

身体搜索通过 body_search.py 在同两处原三角面上重放健康镜像墙的具名颈后驱动，依次运行两代五组并在 BODY_SEARCH 保存每组实测回报。deploy_body.py 使用该目录选中的四项增益并恢复活动脑与实时过颈。在整个项目原目录复跑分析会重新写入派生审计和图；重做完整搜索应先在同级派生实验目录运行，保留原样本。
