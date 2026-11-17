# 镜像增强唯一候选：父代理执行说明

本子只CODE_ONLY写代码与标准库检查，没有导入Torch/NumPy/MuJoCo、读取数组、运行forward/渲染/训练/Core。父已报告library生成bash83exit0、260项独立审核与实际开图完成，这些是父执行结果，不是本子测试。

## 固定范围

- 根目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_mirror_v4/`。
- library/RGB.npz只有56组真实双眼RGB：旧farv3原42像素逐位精确前缀＋14新增实际渲染V6pose9(8599,1.7/.36)和V7pose10(11999,2.3/.42)。旧V4/V5在新metadata成为development，旧source不改。数字镜像不计新增相机曝光。
- policy/train.npz存72条序列描述符，原far36的frameIDs/targets/weights/时钟逐位复用为前36；后36只镜像flag与目标取反，同frameIDs/weights。R为同时左右眼交换和每眼水平翻转；batch输入F,B,E,H,W,C的torch.flip维度(2,4)，R²精确。整条序列保持同一反射，不逐帧随机造运动。
- 36pair=[original_i,mirror_i]，每步PCG64seed20261112无放回取2pair展开4，每条序列边际1/18。原36目标权重不变，不加镜像惩罚、不改在线网络或归零规则。
- strict69c808023470c8460da73a17df83f8bc5afea2f6e42acda0bfc8d8e6bd5fad5d初始化，freshAdam1e-4/fixed500/full1920BPTT，save0/250/500仅final500验证。原14参数架构/相机时序/motor789f/全部验收门不改。
- 数值预检和checkpoint重放样本选mirrorpairIDs[8,16]，包含原leftpulse和原LTR及其镜像；不再用全零目标对作唯一数值覆盖。该变更只为使现有检查有意义，不增加训练步数或改变目标。

## 已完成的标准库检查

入口 `/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_mirror_v4/check_codeonly.py` 实际exit0/29项全PASS，无heavyimports。证据同目录CODEONLY_CHECK.json，exclusive文件不要盲重跑覆盖。所有已检查执行源停止修改。

## 父统一单线程训练命令

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_mirror_v4/policy/run_candidate.py > /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_mirror_v4/policy/PIPELINE.log 2>&1
```

父唯一heavy依次执行实际fullbatch4 production/scan/全14梯度/Adam/RSS≤1800MiB与镜像/旧数据精确复用检查、冻结输入、fixed500训练、初始/250/500新进程保存重放/采样RNG审核、final新actualV6V7十二配置及8控制一次验证、独立CNNGRU输出和门重放。JOBS/日志/退出码持久保存，模型训练pipeline不启动身体。

严格新loader `/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_mirror_v4/policy/common.py` 接口load_checked(path,digest,require_final=True)，格式VISION_FAR_MIRROR_POLICY_V4，仅final500可准入。最终checkpoint路径policy/TRAIN/checkpoint_000500.pt，SHA必须由实际运行产生，不能猜测。

## 证据边界与失败处理

数字反射只是明确训练归纳偏置，不证明非对称果蝇身体/场景/相机在真实世界镜像下严格等价。实际新V6/V7图像才是验证，镜像本身不是新的真实曝光/独立身体回合。旧nearbody与farv3失败全部保留。没有runtime减bias、ifwall归零、死区、记忆reset或图像质心steering。

任何actualpreflight/训练/审核错误先看JOBS与已有产物，不盲重启或重跑已完成训练；功能FAIL正常保留，不选250/追加500/修改RMS或符号门，不自动身体准入。即使离线全PASS，真实移动头部视觉身体确认仍由父在独立新scope安排；本子没有实现body或visual交付框架。
