# 原训练中性拟合诊断：只推理，不开新候选

本诊断回答“原训练的空场/中心/双罩零目标是否也有较大误差，还是仅留出失败”。固定812f0bf409f39550eed6e47d2dbafb0287ce894e725a632c0ffc48afc009f02b最终1000权重，不训练/反传/优化，不改旧source/FROZEN/结果/门。

## 实际输入与计算

从原policy/train.npz和DATASET.json选择240条真实原训练序列：empty80(index0..79)、center80(80..159)、occluder_only80(640..719)。data.batch原frameID及其完整on/off历史，20ms采集发布延迟/5msGRU/full1920step；不是把中心替换成always-on新history。生产路径batch1，实际RGB→CNN→GRU，no_grad/eval/所有参数requires_grad=false。图像、环境label和metadata不被临时改写。

输出whole9.6RMS/原门、scene-activeRMS、mean/std/peak/末.5RMS、原weightedobjective及实际pair取样次数，每motif和背景逐项保存。只读取已失败的validation JSON作开发对比，不再跑验证模型/不碰freshval。参数tensor SHA与所有source/data/checkpoint字节前后核验。

## 父单线程命令

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_coverage_v5/diagnostic_neutral_fit/evaluate_train_neutral.py --parent-resource-released > /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_coverage_v5/diagnostic_neutral_fit/RUN.log 2>&1
```

预登记PREREG.json、原始输入INPUTS_BEFORE.json、逐20条进度RUN.log、全部240行及分组小摘要RESULT.json、OUTPUTS.npz(turn[1920,240]/time/source_train_episode_indices)。首运行exclusive输出，错误先看产物，不盲重跑。

CODE_ONLY脚本实际标准库检查exit0/13项true，Torch/NumPy/MuJoCo/physics/flygym未导入；证据同目录CODEONLY_CHECK.json。实际神经推理由父唯一heavy执行，本子不运行。

## 可证伪解释边界

若原训练中性误差也大于同一门，排除“仅留出泛化差”作为全部解释；仍可能欠拟合、损失分配、优化采样、图像编码与状态历史等，不据一项诊断定唯一根因。若训练过门而留出失败，支持train-heldout gap，不等于证明架构不行或唯一过拟合机制。pair采样计数只描述，不因此补训未取pair。80静态背景来自同一条身体轨迹，不是240条独立运动轨迹。

本脚本绝不启动训练、修改checkpoint、加载motor、构造Core或运行身体；不再申请盲加图/加步的新候选。
