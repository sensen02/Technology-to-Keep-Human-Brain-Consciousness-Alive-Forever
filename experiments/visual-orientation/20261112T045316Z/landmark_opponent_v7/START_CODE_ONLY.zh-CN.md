# 固定神经颜色对立编码候选7：最后单项有限机制实验

本子仅CODE_ONLY源码与标准库检查，无Torch/NumPy/MuJoCo/数组/模型/forward/Core/渲染/训练。父已报告库真实bash90exit0、2288项独立审核及实际开图完成。实际训练结果待父运行，不能把代码检查当功能通过。

## 冻结机制

实际RGB/255先进入固定1×1神经层，权重[1,−.5,−.5]、bias0，经ReLU成为单通道，再进入新Conv1to8。其后conv2/eye_head/adapter/GRU32/head保持原结构。固定receptor以registered buffers表示，不加入Adam；14个可训练参数张量保持。

初始化明确分两部分：新Conv1权重和bias按seed20261112独立fresh；其余12个可训练张量逐位复制be62264966789c59a82e27dc696d710d0751f4c9ac8c1b575d26b0a1920a7651。没有把旧RGB卷积权重求和假装等价；新receptor+局部初始化+后续训练是联合机制候选，不作唯一因果主张。

该层是工程红选择神经前端，不是已验证的果蝇视网膜。它不计算颜色质心、轮廓模板、目标坐标、bearing或动作决策；没有ifwall、输出归零、减bias、死区或clearhistory。原始actualRGB始终保存，网络内部前端可完整重放。

## 图像与数据

库728组actualRGB=旧672像素精确前缀＋V12/V13×4新姿态×7场景=56新验证图。TRAIN_OPPONENT_DIAGNOSTIC只对训练560actualframe的raw像素测relu(R−.5G−.5B)分布，没有模型或新验证前向，不强制背景/双罩为零。父实际发现空场/灰罩对立响应0、红目标非零只是信号前提，不等于策略成功。

原720条训练frameIDs/target/pointweights/episode_scale/所有时钟逐位保持neutralv6值；240neutral×4、480other×1，原pointweight.sum1，损失仍显式mean(scale×persequenceobjective)。采样360uniformpairs、batch4、PCG64seed20261112、不补漏pair。旧失败全部保留，未再增训练图或数字增强。

## 实际父执行命令

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_opponent_v7/policy/run_candidate.py > /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_opponent_v7/policy/PIPELINE.log 2>&1
```

先实际full9.6batch4 production/Scan/全14梯度/Adam/manualscaledloss/pairedmean/RSS≤1800MiB；额外仅fixedreceptor buffers/独立numpy像素等价≤1e-15、newconv1freshseed/shape、12parenttensorsexact、原720数据与672prefix实际审核/sourceproof。mixedpair[0,160]覆盖scale1/4。全部证明在输入freeze之前生成。

再freshAdam1e-4/fixed1000/save0/500/1000onlyfinal，新48validation+32controls一次，保存与验证分别新进程从实际RGB生产路径重放。原空场.02、center/box/off.05、符号.9等门完全不变，不挑500、加1000或再开候选。离线全通过后才可由父另行严格准入真实移动头部身体，pipeline不自动启动。

严格接口：policy/common.py `load_checked(path,digest,require_final=True)`，FORMAT=VISION_OPPONENT_POLICY_V7，仅final1000。新model.py独立不依赖common；父旧权重加载期间临时basename model映射原RGB类，finally恢复，之后构造新opponent类，避免旧checkpoint身份错配。网络identity为VISION_FIXED_NEURAL_OPPONENT_ARCH_V7，body适配必须真实绑定它而非旧identity。

## 已实际标准库检查

`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_opponent_v7/check_codeonly.py` 已exit0，34项全PASS/无heavyimports。记录CODEONLY_CHECK.json，全部执行输入现停止修改。只本START纯说明不入sourcefreeze。

所有原近域/远域/mirror/coverage/neutral失败、气味切断未过项及滑移失败保留；新结果不能追改旧FAIL。不是近视觉导航、全视域能力、嗅觉融合、停驻、摄入、生物有效性或意识证据。
