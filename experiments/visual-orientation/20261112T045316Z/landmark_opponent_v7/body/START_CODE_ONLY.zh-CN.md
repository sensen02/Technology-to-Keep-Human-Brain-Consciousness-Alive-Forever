# 固定神经对抗色感受器视觉策略：far身体父执行说明

新body目录：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_opponent_v7/body/`

实际标准库检查41项全PASS/exit0，0numpy/torch/mujoco/physics/model imports、0数组/模型/forward/Core/渲染/训练/物理。所有执行源已停止修改，无INPUT_BINDINGS/FROZEN/PREFLIGHT/trials。本说明不纳源码freeze，实际执行与job收集由父统一CPU1。

## 实际新策略身份

新loader已read：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_opponent_v7/policy/common.py`
FORMAT=VISION_OPPONENT_POLICY_V7，load_checked(path,digest,require_final=True)严格final1000。

新model.py无common导入依赖，identity VISION_FIXED_NEURAL_OPPONENT_ARCH_V7。actualRGB/255经固定神经Conv1x1权重[1,-.5,-.5]bias0/ReLU，再learnedConv1→8/CNN/GRU32，不是颜色质心/几何方向规则。感受器buffer与conv1输入通道严格核，部署原始实际RGB不镜像或强制归零，source坐标/visibility/time不入actor。父已报告新48validation+32controls独立审核全部required72通过，准备仍会核actualSHA和全部证据，不能仅凭文档准入。

## 准确父执行顺序

```bash
python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_opponent_v7/body/prepare.py --bind-completed-offline-pass --visual-sha256 996980806ac107137d6d5f4743de207fb3350e84a8a145dc5f52b8807398bb87
```

标准库prepare验证新final1000全部必需离线功能、独立VALIDATION/VERIFY、TRAIN/VERIFY_001000、输出/源hash及原运动接受证明，之后冻结新/复用source、模型、物理与相机输入。不构造Core。

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_opponent_v7/body/preflight.py --parent-resource-released
```

真实0Core预检加载newmodel/newcommon（嵌套父旧model alias在common.initialize内恢复），再加载COMBINED模型，finally恢复sys.modules['model']与sys.path。真实原始RGB批量1encode、延迟/缓存、faranchors/闭罩眼外几何/fixedframe以及新opponentbuffer精确值/新架构/最终1000均核验。额外严格model重载使报告model_loads=4，仍无Core或物理。

预检通过后父使用托管后台job，保存job_id并收集：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_opponent_v7/body/orchestrate.py --parent-resource-released > /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_opponent_v7/body/PIPELINE.log 2>&1
```

8unique=empty/center/Lpulse/Rpulse/LR/RL/leftoccluded/flipfixedframe，每完整1+4.8+9.6s后独立新进程实际rawRGB→newencoder→visionGRU32→motor256重放，288chain/cache16分别记录。Capture770真实双眼帧、15400body、2880decision。oldmotor789f/42boundedPD/71opaque/noadhesion/0extra及原指标周期不变，livecapture证明接触求解数据只读，无额外mj_forward。功能失败诚实收齐，执行/完整性错误停诊断，不盲重跑已完成身体。日志独占/JOBS逐PIDexit，最终字段all_declared_targets_pass/normal6_all_functional_pass/control2_all_pass保持已修复版本。

## 场景/控制/门槛保持

farengineering source在实际handovereye-midpoint+水平headbasis一次锚定forward2、左右±2tan(.4)、center0；pillar半径.12高2.4centerZ1.2，仅renderdecor不collision，world固定。onANN.6、pulse/centeroff3.6/reverse4.8；camera20mscapture+20msdelay、5msGRU/运动，实际publishedoff3.62→恢复4.12..4.62。

遮挡3.6在左右targetcenters同时出现双闭合BOX罩，halfsize[.18,.18,1.4]/centerZ1.4/rotation固定handover水平headbasis。真实target+双罩唯一actorRGB，CF同双罩无target独立render，像素exact且所有眼在所有罩外clearance>.005才有效；首规定capture3.6..3.8/release≤3.82设置origin，+.5..1整个窗口持续遮挡/眼外、turnRMS≤.05，否则FAIL，不把眼入黑盒冒充成功。

fixedframe4.8保持actualreleasedleftfid529(ANN4.78capture/4.8release)，raw相机/queue继续，cache16固定而视觉/运动记忆持续，无turn0/right要求，不计新闭环成功。

原empty.02/center.05/off.05/sign.9/response2s/body1deg/reversal.25deg及全部稳定/六腿周期/原生NF0/extra0门不放宽，slipFAIL独立。旧near、far、mirror、coverage中失败与未准入bodycode原样保持。本scope不证明任意视觉导航、近距修复、停车、摄入、气味融合或复眼生物有效性。
