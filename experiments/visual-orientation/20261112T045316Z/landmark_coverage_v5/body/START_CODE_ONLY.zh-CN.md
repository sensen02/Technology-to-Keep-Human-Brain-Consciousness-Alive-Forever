# 数据覆盖候选final1000的far身体适配：父执行说明

当前CODE_ONLY，最终标准库51项实际通过/exit0，numpy/torch/mujoco/physics/student_wave/flygym/model导入全false，0数组/模型/forward/Core/渲染/训练/物理；无INPUT_BINDINGS/FROZEN/PREFLIGHT/trials。所有执行源已停止修改，本文不纳冻结身份。

根目录：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_coverage_v5/body/`

实际新loader已read确认：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_coverage_v5/policy/common.py`
FORMAT=VISION_COVERAGE_POLICY_V5，load_checked(path,digest,require_final=True)严格最终1000，权重路径TRAIN/checkpoint_001000.pt；不是500模型。新架构与原类相同，实际运动/相机/场景/8条件不变。

## 父实际通过后的顺序

仅新最终1000全部required离线功能、独立validation审核、独立saved replay PASS后，替换下面唯一ACTUAL_COVERAGE_FINAL1000_SHA为真实已完成权重SHA。未知或失败权重不绑定；prepare本身是标准库，只创建身份和冻结，不加载物理。

```bash
python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_coverage_v5/body/prepare.py --bind-completed-offline-pass --visual-sha256 ACTUAL_COVERAGE_FINAL1000_SHA
```

父独占资源后执行真实0Core预检：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_coverage_v5/body/preflight.py --parent-resource-released
```

检查实际coveragefinal1000/motor加载别名恢复、真实RGBbatch1编码、20msrelease/5msheld、far锚点、双闭罩眼外clearance、fixedframe保持、newROOT/POLICY/函数和类methods完整绑定。尚未实际执行，不把51项代码检查当模型或身体通过。

预检通过后父以托管后台job串行运行，记录并最终收集job_id：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_coverage_v5/body/orchestrate.py --parent-resource-released > /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_coverage_v5/body/PIPELINE.log 2>&1
```

全部8unique完整回合，每次physics后freshprocess实际RGB→CNN→GRU→motor重放及原物理指标审核。功能FALSE仍如实收齐；执行或完整性失败中止，只诊断不盲重跑已完成身体。每日志独占、JOBS记录PID/exit，最终CURRENT_STATE按all_declared_targets_pass/normal6_all_functional_pass/control2_all_pass读取，避免旧wrapper字段错误。

## 只定向改final1000，不动身体时序

新小adapter仅映射实际FORMAT/任务kind/检查点文件字符串，以及特定vp['step']比较。代码检查逐项证明runner/audit/normal6summary所有numericconstants保持原值。没有全局把500替为1000，不改1000ms稳定段、15400端点、1232000原生步、20ms相机、5ms神经、固定帧529或目标forward2。

首轮标准库检查exit1捕获继承路径字面量带TRAIN/前缀而初版映射遗漏的错误，冻结前已用exact字符串AST定向修复，最终51项通过。保留CODEONLY_FIRST_ATTEMPT.json，不影响任何训练或身体数据，未生成准入。

## 固定实验边界

正常6条件：empty/center/leftpulse/rightpulse/LR/RL；2对照：leftoccluded/flipfixedframe。原1+4.8+9.6、视觉32+motor256共288持续记忆，held16输入缓存单列，真实暖空场队列、20msdelay、实际moving-head RGB，无线上mirror/augmentation或坐标/visibility/time输入。

源world固定handovereye-midpoint/headbasis forward2、左右angle±.4，半径.12/高2.4/centerZ1.2，仅decor不collision；onANN.6/off3.6/reverse4.8。双闭罩在3.6同时包左右世界柱，halfsize[.18,.18,1.4]/centerZ1.4、交接水平headbasis固定rotation。只有target+两罩实际图进入actor，CF是同罩无目标实际render；像素exact且眼在每罩外clearance>.005才有效，第一规定release≤3.82后.5..1完整窗口持续遮住/眼外/RMS≤.05，否则FAIL，不把眼入黑盒作成功。

fixedframe4.8只保持actor已release左帧529(ANN4.78capture/4.8release)，actualcamera/rawqueue继续、GRU/motor无reset；不强制turn0/right或计新闭环成功。

原empty.02/center.05/off.05/sign.9/body1deg/reversal.25deg及支撑/COM/推进/非足0/all6/额外力0条件全不变，原滑移FAIL单列。NORMAL6_SUMMARY与VISUAL_BODY_SUMMARY保留完整分项。旧近域和far/mirror失败不改，不宣称任意视觉导航、停车、摄入、融合或复眼生物有效性。
