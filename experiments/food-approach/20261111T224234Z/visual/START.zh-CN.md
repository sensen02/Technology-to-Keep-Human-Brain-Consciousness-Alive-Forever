# 气味刺激视频与同步图：父代理执行说明

状态CODE_ONLY。实际仅运行stdlib语法/导入/小契约检查，exit0/18项全部通过。没有Torch/NumPy/MuJoCo/Matplotlib/physics/runtime/model导入，没有模型加载、forward、Core、积分、渲染、绘图或读取trial结果。本委托不拥有heavy执行权限，后续所有运行和job收集由父统一调度。

检查：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z/visual/CODEONLY_CHECK.json`

实现均为直接函数，不执行旧renderer源码或叠加source字符串替换。核心schema字段已按当前runner/audit文本核对：command/raw_command都是(2880,2)，raw_turn(2880,)，motor256+behavior32=full288，完整审核不能由旧motor256代替。

## 单条件视频

先等该trial完整RESULT、IDENTITY、HANDOVER、四NPZ和独立VERIFY.json齐全；若B条件，匹配B_noodor也必须完整审核通过。功能或原工程门False允许诚实展示，但完整性审计必须全True。下面以左脉冲示例，替换--condition为已注册九个ID之一，不能运行另一个heavy时并行执行：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z/visual/render.py --condition B_left_pulse --execute-after-parent-release
```

右脉冲同命令仅改为`--condition B_right_pulse`。注册ID：A_left_half、A_right_half、A_switch、B_noodor、B_equal、B_left_pulse、B_right_pulse、B_left_to_right、B_right_to_left。

输出新独占目录：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z/visual/VIDEO_B_left_pulse/`

包含ACTUAL_STIMULUS.mp4、DECODED_MONTAGE.png、VIDEO.json、FFPROBE.json、FRAME_MAPPING.json。已存在目录会拒绝覆盖；失败碎片保留给父诊断，不静默重跑完整仿真。

视频：原全1s稳定+4.8s教师+9.6s ANN、15400保存端点取每20ms、770帧/50fps/15.4s实时。仅在可视私有Core恢复保存qpos/qvel/time后mj_forward，step/step1/step2禁止，原模型compiled physics前后验证/仅visual buffer960×640声明恢复。71mesh全部不透明、固定相机全程mesh边界fit、每帧分割mask确保全身未被overlay遮挡。每帧复核qpos/qvel、COM、胸部yaw、实际头部xpos/xmat，以及唯一HEAD_AUDIT.localxyz重建的双采样点；数值派生误差≤1e-10，不增加新的物理回合。

overlay英文：ANN相对时间t-5.8、PRE-ANN、真实控制来源、clamp不是导航/A命令探针不是刺激成功。受体z来自已保存前区间pending，C held只影响next；query选time<=frame的最后样本，settle缺query显示N/A。bodyΔyaw相对handover及minus no-odor差值分列。Original5 PASS/FAIL从autonomous原五门读，Functional单列，不把30°端点门用于刺激成功。B_RESULT.functional_pass仍为None且不修改；正式B视频/图现在强制绑定 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z/B_FUNCTIONAL_SUMMARY.json` 中对应trial行，核六行顺序/flags与判定一致以及全部24个RESULT/VERIFY/neural/geometry源哈希和审核绑定，再显示最终Functional标志。summary及全部source_hashes加入可视输入前后核验，manifest另记summary来源、functional_flags与metrics。summary不存在时B正式交付拒绝；A探针仍可独立展示其RESULT或PENDING。事件起点/0.2s包络结束明确标注。蒙太奇10帧覆盖稳定/教师末/交接/onset/持续刺激2.8s/off开始末/反转开始末/最终。

结束完整ffprobe/full770decode、输入源hash/物理不变确认后，父必须实际read_image打开DECODED_MONTAGE.png，再核manifest与桌面copy hash，才可宣布交付。

## 六条件同步曲线

所有六个B trial独立审核齐全后，只读已保存输出，不能在另一个heavy活跃时执行：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z/visual/plot_sync.py --execute-after-parent-release
```

输出：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z/visual/SYNC_SIX_CONDITIONS/SIX_CONDITIONS_SYNC.png`

及同目录PLOT.json。六行按固定B_order、三列分别receptorL/R与heldC、raw/appliedturn、bodyΔyaw/noodorΔyaw/二者差值。共享实际time-5.8，baseline时间轴exact匹配、不插值。ON .60/.80、OFF3.60/3.80、反转4.80/5.00为起点/包络结束。全部图明确CONCENTRATION_CLAMP / NOT NAVIGATION；B6没有真实身体receptor-cut，不把noodor或offlinecut写成建立响应后受体切断。父实际打开图、核PLOT.json与副本hash后交桌面。

## 核心代码文本审阅结论

已只读新runtime.py、runner.py、audit.py及原计划：2受体输入边界、行为prime保持、288连续记忆、因果pending、A仅高层command override、B真实behavior、无postteacher、唯一HEADlocalxyz几何链和原工程/功能分列均已核对。无执行实证声称。核心实际preflight/物理/审核由worker独立完成；本工具在运行前再次强制绑定独立VERIFY到run/condition/RESULTsha/motor与behavior两个SHA/288layout及必需checks，不能只信文件名或旧256检查。

所有修改仅新visual目录，旧冻结源码、权重、结果、主tracker和Git快照保持。已按 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z/summarize_B.py` 精确schema增加最小summary绑定，不调用该模块（顶层runtime会重导入）。补充stdlib检查仍18项全部通过、exit0、重模块全未导入；新记录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z/visual/CODEONLY_CHECK_B_SUMMARY.json`，原CODEONLY_CHECK.json保留历史。未读取实际summary或trial输出，未绘图/渲染。
