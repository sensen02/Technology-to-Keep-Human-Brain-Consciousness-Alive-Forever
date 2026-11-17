# 镜像增广训练权重的far限定身体适配：CODE_ONLY

当前仅标准库检查48项实际exit0、heavy_imports全false、0数组/模型/forward/Core/渲染/训练/物理。没有INPUT_BINDINGS/FROZEN/PREFLIGHT/trials；尚未知最终权重和离线资格，不得预先绑定失败模型。所有运行源从CODEONLY_CHECK之后停止修改，本文不纳源码身份。

根目录：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_mirror_v4/body/`

真实新loader已文本核验：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_mirror_v4/policy/common.py`
FORMAT=VISION_FAR_MIRROR_POLICY_V4，load_checked(path,digest,require_final=True)严格最终500，原视觉架构保持。

## 父独占资源后的准确顺序

仅在实际最终500全部必需离线功能及独立validation/saved审核通过后，将下面ACTUAL_MIRROR_FINAL500_SHA替换为真实SHA。prepare检查新路径、FORMAT、source/data/输出SHA、原motor接受证明和全部实际资格，不允许用near或far3失败模型冒充。

```bash
python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_mirror_v4/body/prepare.py --bind-completed-offline-pass --visual-sha256 ACTUAL_MIRROR_FINAL500_SHA
```

此步只绑定和冻结，不加载模型或物理。然后父唯一资源执行实际0Core预检：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_mirror_v4/body/preflight.py --parent-resource-released
```

预检真实加载新视觉/motor、核model别名恢复、RGB批量1编码、far锚点与闭合罩眼外几何、20msrelease/5msheld、fixedframe真假输入区别和明确globals。实际预检尚未运行，不能将标准库48项当模型/身体通过。

全部预检通过后，父使用托管后台job运行并负责记录/收集job_id：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_mirror_v4/body/orchestrate.py --parent-resource-released > /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_mirror_v4/body/PIPELINE.log 2>&1
```

8unique完整身体回合(V_empty/center/Lpulse/Rpulse/LR/RL/leftoccluded/flipfixedframe)，每次物理后独立actualRGB→CNN→GRU→motor重放审核再下一回合。功能FALSE如实收齐；完整性/执行错误停后诊断，不盲重跑已完成身体。JOBS和日志独占，不覆盖；最终状态用真实all_declared_targets_pass、normal6_all_functional_pass、control2_all_pass，保留原修正，不重带旧wrapper KeyError。

## 仅训练增广，线上不镜像

本新scope唯一执行差别为ROOT/POLICY/新FORMAT/新实际权重与来源。原far3 runtime函数和CameraStream/_OpticalProxy的方法codeobjects保持相同，所有globals完整绑定到新ROOT/POLICY，sys.modules['model']临时视觉类/COMBINED类别名严格finally恢复。运行/审核都只处理实际capture RGB，不调用训练mirror变换，不先平均左右或减bias，不读目标坐标/visibility/time作为神经输入。

目标仍handover实际eye-midpoint+水平headbasis一次锚定forward2.0/angles±.4/center0、柱半径.12高2.4centerZ1.2，固定world、仅decor不collision。20mscapture+20msdelay、5ms神经，vision32+motor256共288，held16cache另记，完整1+4.8+9.6。

遮挡3.6出现左右两个worldfixed闭合BOX罩，halfsize[.18,.18,1.4]/centerZ1.4，rotation交接水平headbasis，与库相同。实际target+双罩图是actor唯一输入；同双罩无target反事实图必须pixel exact，且每eye在每个BOX外clearance>.005，第一规定capture3.6..3.8/release≤3.82确证origin后.5..1整窗仍遮挡且眼外/RMS≤.05，否则FAIL。眼进入黑盒不算遮挡成功。

固定帧4.8锁capture529(ANN4.78采/4.8发布)而raw相机/队列继续，视觉/运动记忆不重置；不强制turn0或right，不计新闭环成功。所有原正常门、原slipFAIL和near失败保留，本次不是修复近域或任意视觉导航/停车/融合/生物有效性。

输出仍NORMAL6_SUMMARY.json与VISUAL_BODY_SUMMARY.json，完整770pairs/15400body/2880neural/实际CF图与独立审计，不需要新视频框架。
