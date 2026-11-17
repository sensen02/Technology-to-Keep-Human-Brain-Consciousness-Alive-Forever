# 人工脑—来源颈部端口—完整身体：同次轨迹交付工具

本目录仅CODE_ONLY准备。实际标准库codecheck.py exit0／28项检查通过，未导入Torch／NumPy／MuJoCo、未读回合数组、未构造Core、推理或渲染。执行源已停止修改。

## 何时运行

必须等父唯一重任务body102全部6物理回合、6独立审核、最终BODY_SUMMARY共13阶段完成且exit0。contracts会拒绝缺审核、缺完整时段、源SHA漂移或未绑定实际eed1脑／9e88颈部／789f运动权重。原工程／功能失败不阻止诚实展示，完整性失败必须拒绝，不能以视频播放代替审核。

## 精确父命令

先绘图，不启动Core：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/body_visual/plot_sync.py --execute-after-parent-release
```

再串行生成两片视频，每片完成并收集后才开始下一片：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/body_visual/render.py --condition P_left_pulse --execute-after-parent-release
```

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/body_visual/render.py --condition P_left_DN_cut --execute-after-parent-release
```

父用managed后台job保存stdout日志并收集退出码，不另起服务器。输出exclusive，已存在部分或完整产物先检查，不盲覆盖或重复物理。

## 内容与核验

- 每片1280×720、770帧、50fps、15.4秒；1秒稳定＋4.8秒教师预热＋9.6秒ANN段完整。
- 只恢复保存的49qpos／48qvel到私有Core，用mj_forward重建同次几何；mj_step／step1／step2被禁止并计attempt。逐帧COM、yaw、head核对，71mesh全部opaque且分割bbox不被标题／页脚裁切，compiled／源SHA前后不变。
- 不加载人工脑／运动网络，不新增任何动作或物理积分；谱图和视频都读取已审核3080条真实神经记录与15400条身体记录。
- 无相机RGB存在，因此没有眼图、视觉目标或遮挡装饰。侧栏只显示实际接收AN1／DN2活动、capture／2ms queue release／5ms neural consume、cut状态、人工脑cue、brain16范数和实际learnedturn。
- 神经帧索引为nominal physical_ms//5，等价floor(globaltime/.005)，0..3079，保存time不得超当前视频时刻。不是旧2880行／288状态视觉schema：本次272记忆＝brain16＋motor256，端口held buffers另记。
- 切断ANN1.6／global7.4为实际源通道静默＋接收held基线0的显式工程干预，不reset脑／运动记忆，不按条件硬设turn0。低层自主步态属于OTHER，不能说脑控制起步或停步。
- VIDEO.json保存全帧映射、artifact SHA、逐帧完整身体bbox；ffprobe770／50fps／15.4s以及ffmpeg逐帧全解码必须通过，再生成10阶段DECODED_MONTAGE.png。

## 父实际审阅与交付

先打开：

`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/body_visual/SYNCHRONIZED_EVIDENCE/SIX_CASES_SOURCE_BRAIN_BODY_SYNC.png`

`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/body_visual/SYNCHRONIZED_EVIDENCE/MATCHED_DN_AN_CUT_COMPARISONS.png`

以及两个VIDEO目录的DECODED_MONTAGE.png，核对全身、预热交接、实际端口信号与cut标注。桌面复制视频／图／报告后逐源和副本SHA核对，保留所有原滑移FAIL、稳定性失败及零cue时OTHER仍步行的证据。

这些是来源约束端口上的人工耦合，非完整真实颈部、生物DNp09转向功能、AN关节调谐或意识证据。报告不能把离线校准MSE或同次运动展示扩大为上述生物结论。
