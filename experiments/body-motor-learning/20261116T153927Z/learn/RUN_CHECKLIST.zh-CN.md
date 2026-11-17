# 缓坡及共同反射入口的运行记录（S2/S5）

- 冻结求解环境 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python`，原 12.5 μs 原生步、5 ms 决策。`drive.sh` 固定 EGL 及单线程 BLAS/OMP/MKL，锁定物理初态。所有命令从 `/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/learn` 执行。
- `bash build_slope_contact.sh` 退出 0。C++ 原生测量扩充坡面法向力与切向滑速，独立 Python 逐 4,031 步交叉核对且平地与旧口径对照。
- 原始整片坡试跑 `SLOPE_2_UP`、`SLOPE_LEVEL` 和 `SLOPE_2_UP_B` 在 0 个物理步因平面变换缓存、重力单位断言及旧平地初态观察断言退出 1，修复后实际恢复同一 qpos/qvel。`SLOPE_LEVEL_B` 保存原生数据但源身份改变，拒绝记为通过。`SLOPE_2_UP_C` 原生全记录及身份核查完成，最终 JSON 因 NumPy bool 退出 1，历史日志和原始源的精确副本保留。`verify_slope.py SLOPE_2_UP_C SLOPE_2_STOCK_RECOVERED.json --retained-failure` 退出 0，原物理结论为失败。
- `slope.py SLOPE_LEVEL_C --angle 0` 和 `slope.py SLOPE_08_UP --angle .8 --walk-ms 14400` 退出 0，前者走停通过，后者坡上倒退。`slope_anchor.py ANCHOR_LEVEL --angle 0` 和 `slope_anchor.py ANCHOR_SLOPE_2 --angle 2` 退出 0，前者通过，后者虽有六足周期，仍丢失坡上站稳及上坡进度。退出码和行为门独立。
- `compose.py stumble COMPOSE_PEG` 及 `compose.py gap COMPOSE_NEAR` 退出 0，完整组合重放与各自合格证据数组逐位相同。`compose.py stumble COMPOSE_PEG_OFF --ablate stumble` 及 `compose.py gap COMPOSE_NEAR_OFF --ablate gap` 退出 0，对比 `audit_assembly.py` 两次退出 0，两组关闭分支保持同场景同前缀。
- `compose_far.py COMPOSE_FAR_PROBE ../gap/P2 --stop-at 3700` 退出 0，匹配旧短窗实际远侧复载。完整新远侧版本另列通过范围。
- 使用旧 `render_reflex.py` 与 `render_gap.py` 的无积分录像流程，分别全解码 669、436 帧，退出 0，实际查看关键帧。两组录像都绑定此次组合开启和关闭分支的原始轨迹。
- `compose_far.py COMPOSE_FAR_CLEAR ../gap/CLEAR_FULL --safe` 退出 0，完整 17.25 秒四档原生、轨迹和动作逐字节一致。`compose_block.py COMPOSE_ENGINEERING_BLOCK` 退出 0，完整触撞探针 5.5 秒，原始三档逐字节一致。两者都经 `compose.py` 的同一响应选择进入有限执行器。
- `ramp_course.py RAMP_PHYSICS` 与 `ramp_course.py RAMP_MATCHED_FLAT --flat` 退出 0，盒体初版两个科学结果失败；前者有实际坡面负载但过渡和末态未达到门。单高度场 `ramp_hfield.py RAMP_CONTINUOUS` 及 `ramp_hfield.py RAMP_CONTINUOUS_FLAT --flat` 均退出 0，实际均在 225 ms 中止。两步的地形、锁定源码与原始数据保留。
- `ramp_hfield.py --over-original-plane --compile-only --plan ramp_overlay_long.json` 配对有坡／无坡均退出 0，体姿和冻结原感官完全一致，原始站姿最前足垫中心 x=0.890119，埋入坡材未接触。原生射线核上表面前、中、后的实际抬升 0、0.012049、0.024098。长程与配平是否过门另见各自结果。
- `ramp_hfield.py RAMP_OVERLAY_ENTRY --over-original-plane --plan ramp_overlay_entry.json` 退出 0。走停联合门、质心跨越及上升过，短程全足过线未过。
- `ramp_hfield.py RAMP_OVERLAY_LONG --over-original-plane --plan ramp_overlay_long.json` 和 `ramp_hfield.py RAMP_OVERLAY_FLAT --over-original-plane --flat --plan ramp_overlay_long.json` 都退出 0，各 3,336,000 原生步。平地过，在实际全程坡仍有一条抬升门失败。`audit_ramp.py RAMP_OVERLAY_LONG RAMP_OVERLAY_FLAT RAMP_OVERLAY_AUDIT.json --short RAMP_OVERLAY_ENTRY` 退出 0，比较最早分歧前状态、动作和原生逐位性。
- `probe_plateau_height.py` 的 1.8 秒无修正／有界抬高两组均退出 0，保存末窗稳定不合格。`split_plateau_load.py PLATEAU_LOAD_SOURCES` 退出 0，从保留末态同一末动作实际积分 800 步，分开高台和原地板的竖直支撑。
- `render_slope.py` 当前同次录像三条依次保留失败三盒、覆盖原地平面的短课、完整长课。全部退出 0，实际打开关键帧。使用新组合后的远侧与碰撞配对录像各全解码并已查看，分别为 636 和 280 帧。
