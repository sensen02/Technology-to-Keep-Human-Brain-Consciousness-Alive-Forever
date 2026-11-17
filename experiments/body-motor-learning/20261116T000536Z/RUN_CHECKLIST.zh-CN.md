# 站立—直走—停稳实施检查表（S2/S5）

- 预登记顺序 `PREREG.zh-CN.md`、`FIT_PREREG.zh-CN.md`、`RESCUE_PREREG.zh-CN.md`、`PROTOTYPE_PREREG.zh-CN.md`、`STOP_WINDOWS.zh-CN.md`、`PUSH_PREREG.zh-CN.md`。身体、初态和训练资料散列见各回合 `RUN_IDENTITY.json` 与训练清单。固定 12.5 μs 原生步、5 ms 决策、1 ms 限速目标积分、三球足端和干摩擦。
- 解释器为 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python`，参数 `-B -u`。环境 `MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。
- `walking_bench.py <初态> TEACH_<初态>` 四组退出 0，物理首步与停步记录通过数据准入。起步完整回合从现有归档初态开始。
- `fit_walk.py FIT_INITIAL` 及 `fit_walk.py FIT_AGGREGATED --predecessor FIT_INITIAL/step_1200.pt` 均退出 0，均实际做 1200 次 Adam。初训与补采续训的三组联合实测各程序退出 0，各行为未达走路门。两组权重为 81c27c…6f36 和 99914e…4db，全部权重与 Adam/RNG 逐次保留。
- `rescue.py {a,b,c,d,e,f}` 全退出 0。同态分支三组合格、三组不合格，保留完整数据。合格动作 960 条。PPO 未用。
- `prototype.py --build` 及 `prototype.py --start <初态> --out <目录> [--walk-ms …] [--light] [--native]` 退出 0。七条非原生对照包括一个全程静立和六组联合，另有标称全原生重跑。真实侧推两次 `push_walk.py` 各退出 0，施加扰动前实际状态逐位匹配对照。
- 第一次合并动作和权重审计工具返回 143，无最终产物。拆成初训、续训、原型三次独立 `replay_saved.py`，均退出 0，合计 2402 个保存检查点哈希和 13 个对照回合，动作误差 0。
- `render_walking.py PROTO_nominal VIDEO` 退出 0，11.3 s 同次原生物理轨迹的毫秒端点渲染。566 帧均完整解码，实际查看静立、直走和停稳。新增物理积分 0。
- 正式物理回合及台架准备共 18,903,200 原生积分。原生单独复跑的结果与无原生记录器的名义轨迹文件逐位相同。完整身体71 mesh。身体侧候选，尚未作颈部接口或生物步态验收。
- `native_score.py PROTO_native` 和原生结果的独立 `replay_saved.py` 退出 0，904,000 帧／442块原生接触，原五门、六腿循环和两端站稳全通过。D3/D5 同回合辅助窗缺失，保留未知。胸部侧推两回合动作已在新进程倒序逐位核对。
