# 中性目标重平衡唯一候选6：父执行说明

依据父实际train-neutral诊断，原训练本身empty52/80、center18/80、box4/80未过门，不能仅归咎留出泛化。此项只改变训练目标中neutral episode的贡献，保持原720输入历史和图像不变。固定812f父权重继续、freshAdam1000预算为联合候选，不声称已证明唯一根因。

## 已固定

- 根目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_neutral_v6/`。
- 库672实际RGBpairs=旧616像素精确前缀＋56新V10/V11×4实际新姿态×7场景，父已报告bash88生成exit0、2112审核通过且实际开图。旧V8/V9在新库是development，旧库/source不改。
- policy/train.npz的原frameIDs/targets/pointweights/capture/decision/usedindex均从coveragev5/train.npz逐位复用。新增独立episode_scale[720]：240个empty/center/boxonly为4，其他480为1。原pointweights.sum1照旧；不乘后重新归一化抵消。
- 新目标 `mean(episode_scale * original_per_episode_weighted_actionMSE_plus_.01latentMSE)`。neutral objective总质量960，其他480，总1440。这是离线loss，不是运行时检测neutral或强置输出。
- 原360uniformpairs不变，每步2pair展开batch4，边际1/180；实际paircount/unseen如实保存不补训。
- strict812f0bf409f39550eed6e47d2dbafb0287ce894e725a632c0ffc48afc009f02b初始化，freshAdam1e-4/fixed1000/save0/500/1000仅final，新实际V10/V11验证48+控制32，所有RMS/sign/latency门不变。

## 已完成CODE_ONLY检查

`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_neutral_v6/check_codeonly.py` 实际exit0/37项全PASS，未导入Torch/NumPy/MuJoCo/physics/flygym，无模型/数组/forward/训练/Core。原库适配首次仅标准库检查发现旧panel文件名出现两处，冻结前明确expected2修复后通过，无实际失败渲染或训练。

数值预检保留实际全9.6batch4 production/scan/all14gradient/unitAdam/RSS≤1800MiB，使用pair[0,160]混合direction两样本(scale1)与empty/center两样本(scale4)，独立manualscaledobjective和两个pair平均梯度等价tol1e-10。避免只方向样本造成scale4未经检查。

## 父CPU1统一命令

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_neutral_v6/policy/run_candidate.py > /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_neutral_v6/policy/PIPELINE.log 2>&1
```

顺序为实际preflight/data生成与beforeupdate冻结、fixed1000训练、0/500/1000新进程RGB生产重放、final48+32一次验证、独立功能门审核。所有batch入口显式4元组(x,target,pointweights,episode_scale)，verify/evaluate/audit只取RGB时仍正确解包4。验证episode_scale全1，不改变原评分。

严格loader `/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_neutral_v6/policy/common.py` 的 `load_checked(path,digest,require_final=True)`，格式VISION_NEUTRAL_REBALANCE_POLICY_V6，仅final1000；最终policy/TRAIN/checkpoint_001000.pt，SHA由实际保存产生。

## 边界

当前本子仅CODE_ONLY，父实际训练结果待运行，不自动启动身体。所有旧近域、远域、镜像、coverage失败不变。无runtime偏置减法/归零/图像质心控制/状态reset，无新网络，无额外训练图，无数字镜像。失败不择500/追加1000/改原阈值。任何执行错误先查JOBS及已有阶段产物，保留失败后最小新scope恢复，不盲重跑完成的训练或物理。

同一条neutral轨迹的静态背景不算720独立运动轨迹。离线全PASS仍不等于真实运动头部视觉、视觉导航、融合、停车摄入或生物复眼资格；身体由父另外事先登记。
