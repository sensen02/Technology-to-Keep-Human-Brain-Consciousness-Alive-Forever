# P1 果园身体与局部摄入复跑

从主工作树调用既有环境。物理运行将写入新的空目录。下面的命令使用保存的权重及身体参数。

```bash
cd /run/media/sensen/Data2/cell_wound_prototype
p="$PWD/experiments/open-brain/20261119T205558Z-orchard-curriculum"
d="$PWD/experiments/open-brain/20261119T213721Z-orchard-recovery"
run="$PWD/experiments/open-brain/20261119T012530Z-survival-orchard/run_env.sh"
out="$(mktemp -d)"
for site in CLEAR DEBRIS; do
  "$run" "$p/rollout.py" "$site" "$out/base-$site" --duration 6.4 --seed 37 --policy "$p/S3/POLICY.npz"
  "$run" "$d/reflex.py" "$site" "$out/new-$site" --duration 6.4 --seed 37 --policy "$p/S3/POLICY.npz"
done
for mode in on off; do
  "$run" "$d/live_food.py" CLEAR "$out/taste-$mode" --duration 4.8 --seed 37 --policy "$p/S3/POLICY.npz" --reflex "$mode" --assay-config "$d/FOOD_CONFIG.json"
done
```

局部摄入入口沿用宿主程序的 CLEAR 参数，在建立物理环境时换成 FLANK.json 记录的同源果实上侧网格。口器半径、库存、脑和过颈记录使用原逻辑。足端识别原生碰撞几何，读取原生口器和腿关节位置。站姿初始化只在回合开始时设置自由根，随后由重力、接触和原生执行器驱动。

回放训练和审计脚本写入脚本所在目录。独立复训时在 experiments/open-brain 下复制本实验目录后运行 stance.py、train.py、ablation.py。stance.py 使用仓库既有 CPU Python 环境；其余物理脚本使用上方 run_env.sh。train.py 用两处原起点各 1.4 秒的真实颈后驱动评价八个候选。inspect_replays.py 读取保存的活动脑原始记录与参考，重新生成审计及图。

第二组近果配对在经独立物理预检的果皮面上进行。身体收到实际摄入回执达到 0.18 后释放进食姿态。将上段最后一组中的 live_food.py 换为 release_food.py、FOOD_CONFIG.json 换为 MEAL_CONFIG.json、4.8 秒换为 3.2 秒，并使用两个新的输出目录即可复跑。原入口比较可将神经种子设为 53 独立复验。

实际整合入口为 run_recovery.py，使用上文相同环境和 S3 头侧权重。例如在前述 shell 变量下执行以下三条。开阔和草枝从原地面起点出发；第三条从已验证的同源果皮面近处起步。

```bash
"$run" "$d/run_recovery.py" CLEAR "$out/full-clear" --duration 6.4 --seed 37 --policy "$p/S3/POLICY.npz"
"$run" "$d/run_recovery.py" DEBRIS "$out/full-debris" --duration 6.4 --seed 37 --policy "$p/S3/POLICY.npz"
"$run" "$d/run_recovery.py" CLEAR "$out/full-near" --start-on-fruit --duration 3.2 --seed 37 --policy "$p/S3/POLICY.npz"
```

最终的口器待机版本以 run_passive_mouth.py 为入口。将上述三条中的 run_recovery.py 换为此名称，并使用新输出目录，即可复跑。两个版本均保留，便于核对启用口器执行器带来的物理变化。每回合 ORAL_ACTUATORS.json 记载新增执行器的名称、原始增益和偏置，以及两种状态的实际切换时刻。
