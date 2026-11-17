# 视觉真实身体六条件：父执行说明

当前仅CODE_ONLY。`check_codeonly.py`标准库实际exit0/20checks全true，未导入Torch/NumPy/MuJoCo，未运行数组/模型/forward/Core/渲染/训练。9核心文件及预登记现停止修改，父掌握所有实际重任务。图像库静态通过不代表身体通过。

准入必须是visual_policy_v1最终500完整离线功能PASS、独立validation审核及saved500审核均通过；未知最终SHA不得猜，不得改门。若离线失败，不执行本目录prepare或身体。当前尚无INPUT_BINDINGS/FROZEN/PREFLIGHT/trials。

## 准入（真实权重通过后）

```bash
python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/visual_body_v1/prepare.py --bind-completed-offline-pass --visual-sha256 ACTUAL_FINAL500_SHA
```

prepare纯标准库核验实际离线结果、原相机成功预检/输入SHA、旧用户运动视觉接受，冻结现有源码和科学输入；不构造Core或渲染。

## 实际0Core预检（父唯一资源）

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/visual_body_v1/preflight.py --parent-resource-released
```

实际加载视觉/运动模型，核临时model别名恢复、无physics先导入、真实库RGB batch1编码重复一致、20ms release/5ms hold、32+256/16cache形状与投影。无Core/积分/训练。这里传感器图像是预检用已归档帧；真正身体运行每20ms实时RGB，不复用静态库图。

## 六回合托管流水线

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/visual_body_v1/orchestrate.py --parent-resource-released > /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/visual_body_v1/PIPELINE.log 2>&1
```

父managed job记录ID并收集退出码。内部纯stdlib顺序6physics+各新进程audit+summary，JOBS/CURRENT_STATE动态不纳科学freeze。若出现部分结果/已有JOBS，拒盲重跑；先查实际stage、日志、存盘状态，完整物理不能因后续报告失败重跑。功能FAIL全回合仍保存并独立审核，不自动重训或择权重。

六条件固定V_empty、V_center、V_left_pulse、V_right_pulse、V_left_to_right、V_right_to_left；世界固定三个柱坐标在5.8s真实头部基准锚定，只有环境开关/换侧，不跟头。1s稳定+4.8s教师+9.6s ANN，capture0..15.38s共770，queries1..15.395s共2880；同原42动作/71完整mesh，不隐藏身体，不接嗅觉。

## 输出与数据schema

每条件根 `/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/visual_body_v1/trials/<condition>/`：

- `capture_rgb.npz`：唯一rgb uint8[770,2,32,32,3]，真实在线采集，含settle和teacher空场。
- `CAPTURES.json`：capture_id/capture_time_s/actual_physical_time_s/release_time_s、head位姿/eye光学pose、active刺激与fixedworldcenter仅环境诊断、每RGB SHA、每capture物理qpos/qvel/ctrl/qacc/qacc_warmstart/qfrc/efc_force beforeafterSHA与contactpipeline unchanged。初始capture0另有INITIAL_STATE.json。
- `neural_decisions.npz`：time、X238、held_features16、encoder_cache_updated、image_capture_id/capture/release时间/valid、used_stimulus_code（只诊断）、vision_hidden_before/after32、hidden_before/after motor256、full_hidden_before/after288、raw_turn/latent、command2、mean42/policy_action42/actually_applied_action42/control_source。编码形状每帧1×1×2×32×32×3，缓存每4次query最多更新一次，GRU32每5ms连续更新；teacher期h32保持初始但缓存/queue真实维护。
- `trace_1ms.npz`：原完整15400真实身体端点；`geometry_1ms.npz`：实际head/thorax及targetworld环境记录；`HANDOVER.json`：原连续快照288hash；`IDENTITY.json`：实际导入源/资产/物理/执行器compiled身份；runtime_geometry保存71opaque证明。
- RESULT原工程五门、stable功能身体条件与刺激功能分列，未以新功能替代原slipFAIL或用户运动基线qualifiedfalse。VERIFY独立读真RGB，batch1逐帧重CNN/heldcache→GRU→motor全精确，复算1ms原指标/六腿周期/交接和前5800endpoint基线exact。

最终 `VISUAL_BODY_SUMMARY.json` 只在全部6独立audit完成后写，按新V_empty同SHA基线比较左右4s航向/反转3s、20ms已发布目标消失后的.5..1s恢复窗口；实际图像遮挡/动态OOD导致失败仍严报。center网络输出不强置0、没有deadzone/ifvisible/归零/hidden reset。

本目录不预制大量视频。实际回合后由父再选择有证据的展示/故障图，不能用静态帧库播放冒充身体视觉闭环。
