# Far bounded真实身体视觉验证：父执行说明

当前仅CODE_ONLY。实际运行标准库检查40项全通过，exit0；numpy/torch/mujoco/physics/student_wave/flygym/model均未导入，没有数组、模型、forward、Core、渲染、物理或训练。全部运行源已停止修改；本文不纳冻结来源，仅提供准确命令。

新scope权威根：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_far_v3/body/`

政策输入来自实际新路径：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_far_v3/policy/common.py`
实际文本已确认FORMAT=VISION_FAR_LANDMARK_POLICY_V3，load_checked(path,digest,require_final=True)仅最终500。

## 1. 未知权重不准入

只有父已经得到far最终500的真实SHA、全部required离线功能PASS、独立validation和saved replay PASS，才运行以下命令，替换唯一的ACTUAL_FAR_FINAL500_SHA。不要把旧79db或失败near权重塞入新准入；prepare会拒绝。不要先生成假INPUT_BINDINGS。

```bash
python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_far_v3/body/prepare.py --bind-completed-offline-pass --visual-sha256 ACTUAL_FAR_FINAL500_SHA
```

此命令标准库绑定真实证据、源码、相机、底层物理依赖和权重，创建INPUT_BINDINGS.json/FROZEN.json，不构造Core。旧相机probe与离线来源全部必须当前hash相等。

## 2. 父独占资源时执行真实0Core预检

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_far_v3/body/preflight.py --parent-resource-released
```

真实加载strict视觉和motor、核临时model别名恢复、770采样路径所用20msrelease/5msheld、batch1实际farRGB编码、原命令投影、闭盒外clearance和inside拒绝、fixedframe原released左图保持、完整newROOT/POLICY/CameraStream/anchor绑定。没有Core、积分或训练。这些尚未实际运行，不把40项标准库检查当数值通过。

## 3. 父托管唯一单线程8回合

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_far_v3/body/orchestrate.py --parent-resource-released > /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_far_v3/body/PIPELINE.log 2>&1
```

必须由父托管后台job并记录/收集job_id。每次真实物理后独立新进程审核再下一条件，8unique=6normal+closed-box occlusion+fixedframe control。功能FALSE仍完整保存并收齐，执行/完整性失败停止后只诊断，不盲重跑已完成身体。JOBS.json独占新建、每stage独占日志，CURRENT_STATE是动态状态不纳输入身份。最终状态读取真实all_declared_targets_pass/normal6_all_functional_pass/control2_all_pass，已修正旧wrapper obsolete字段KeyError；没有body2最终状态错误的回退。

## 场景与边界

交接真实eye-midpoint+basis一次锚定forward2、左右angle±.4(rad)，柱半径.12/高2.4/centerZ1.2。坐标固定world，不跟头；仅渲染decor，不碰撞/施力。sourceonANN.6，pulse/center off3.6，reversal4.8；相机20msdelay，所以正常first released on约.62、off约3.62，普通恢复窗口约4.12—4.62。

V_left_occluded保持左柱、在3.6同时生成包住左右柱的两个闭合BOX罩，halfsize[.18,.18,1.4]/centerZ1.4，朝向为交接水平头基与far库一致，之后world固定。实际RGB与同两罩withouttarget反事实RGB必须完全相同，且每eye在每box外、欧氏表面clearance>.005，才可作为有效完全遮挡。眼入盒、near裁剪或画面全黑不能冒称成功。第一规定范围内released遮挡帧≤3.82设origin，origin+.5..1整个窗口都须持续遮住且眼外，RMS≤.05；否则诚实FAIL，不后选新时刻/阈值。

V_flip_fixedframe实际场景4.8换右但actor只保持当刻已发布左帧capture529(ANN4.78 capture/4.8 release)，raw真实相机和队列继续，16featurecache固定，vision32/motor256连续推进。该干预不要求turn0或转右，不计新闭环成功。

原normal6门未改变，原运动滑移FAIL独立保留。此为FAR工程地标限定任务，不宣称旧近域center/RTL已修复，不是任意视觉导航、停车、摄入、视觉嗅觉融合或生物复眼验证。

## 文件与schema

每trial保存RESULT/VERIFY/IDENTITY/HANDOVER、INITIAL_STATE、trace_1ms.npz(15400)、geometry_1ms.npz、neural_decisions.npz(2880)、runtime_geometry.npz、capture_rgb.npz(770,2,32,32,3)与CAPTURES.json。visual32+motor256完整288链，held16feature/inputqueue分开。occlusion另存occlusion_counterfactual_rgb.npz及每capture RGBexact/MAE/眼盒距离/clearance/cover_rotation。audit重新从实际batch1RGB计算视觉/运动并核全部身份、时间、连续状态、物理指标/周期、真实头部光轴以及视觉渲染前后contact求解数据只读。

结果汇总NORMAL6_SUMMARY.json与VISUAL_BODY_SUMMARY.json分别包含normal6/control2及总门；失败通过都依实际记录。无需新视频框架，父按结果另协调实际看图与视频交付。
