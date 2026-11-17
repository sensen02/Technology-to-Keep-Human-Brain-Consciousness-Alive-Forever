# 两种倒地恢复站姿运行记录（S2/S5）

- 预登记 `PREREG.zh-CN.md` 与 `LEARNING_PREREG.zh-CN.md`。固定原三球身体、1 ms 关节目标积分、2 rad/s 限速和原生 12.5 μs 步长。两条指定倒地全态的原始哈希与父实验一致。
- 归档解释器 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python`，用 `-B -u` 及环境 `MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。
- `rollout.cpp` 用 g++17、`-O3 -ffp-contract=off -shared -fPIC`，链接同环境实际 MuJoCo 3.9.0。 `search.py preflight PREFLIGHT` exit0。 `search.py search SCREEN_50US --width 4 --count 40 --depth 6 --ticks 330 --dt .00005 --workers 16` exit0。
- `verify.py` 两基线、三候选及两通过教师独立物理重放均 exit0。正文所列体轴、足载与非足承重按原生步记录，质心水平速度从毫秒记录计算。完整 200 ms 站稳窗口采用预登记。通过的范围为两条真实仿真倒地起点。
- `learned.py train FIT --teacher NATIVE_SIDE_D4 SCREEN_50US/side_fallen_depth04.npz --teacher NATIVE_SUPINE_D6 SCREEN_50US/near_supine_depth06.npz` exit0，实际 800 个 Adam 更新。两次主评估 `learned.py trial FIT_<seed> FIT/step_0800.pt <seed>` exit0。两条独立重复结果完整且逐位匹配。侧倒工具返回 143，近仰卧返回 0。 `audit.py --repeated RECHECK` exit0，模型及全部 801 个检查点核对通过。
- `render.py VIDEO FIT_side_fallen FIT_near_supine` exit0，30 fps × 351 帧各两段，完整解码，实际目视三阶段。新增物理积分 0。
- 总量 3,764,000 次原生步和 8,559,600 次粗筛步。PPO、步态全验收和新颈部接口未执行。
