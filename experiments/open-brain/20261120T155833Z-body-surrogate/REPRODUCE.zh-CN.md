# P1 身体替代模型使用和复跑

在本机 Bash 中使用以下环境。直接预测只依赖 Python、CPU PyTorch 和 NumPy，输入字段与根编号次序见同目录 IO.json。weights.pt、skeleton.json 和 Python 模块放在同一目录。

```bash
cd /run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261120T155833Z-body-surrogate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
py=/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python
physics=/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python
"$py" predict.py example_input.npz /tmp/body-surrogate-prediction.npz
```

每个批次是同一回合的 N 个连续决策时刻。previous_mm 和 current_mm 均为 [N,21,3]，在统一毫米坐标系中相隔 5 ms。touch_uN 为 [N,2,6]，从旧到新排列两个足压样本。dn_counts 为 [N,4,17]，按旧到新排列四个已完成的 5 ms 电极读数窗，启动时更早的窗补零。标记顺序、腿序和根序固定。输出 next_mm 是再过 5 ms 的预测，an_prob 和 an_body_source 是该时间段五个 1 ms 小窗中两侧前足上行的概率及源端发放。以固定种子采样，可复现同一回合。逐拍流式调用见 run_loop.py，跨包保存采样器的不应期状态。新回合重置采样器。闭环每个事件以 1 ms 窗中点注入具名 BANC 身体侧上行源。

以下顺序重建当前目录的派生结果。已登记输入由 SOURCE_SHA256 和 SHA256SUMS 校验。

```bash
"$physics" observe.py
"$py" electrode.py
"$py" calibrate.py
"$py" train.py
"$py" floor.py
"$py" skeleton.py
"$py" evaluate.py
"$py" verify.py
"$physics" independent.py external
"$py" independent.py electrode
"$py" evaluate.py --seed 11903
"$py" run_loop.py
python3 plot.py
```

run_loop.py 的活动源是冻结 Eon 当前输出和替代身体预测的双侧前足上行。三个条件共用相同外部驱动、初始两帧、阵列噪声与采样随机数。第一个条件连续给定同回合足压，第二个条件只用预测姿态计算足压，第三个条件在第二个条件基础上关闭双侧前足上行源。替代模型、平地足压估计器、冻结脑和具名骨架传导分别保存。关闭双侧源只用于对照。
