# 真实身体视觉8条件交付：父执行说明

当前CODE_ONLY。实际仅标准库check_codeonly.py exit0/13项全部通过，0Torch/NumPy/MuJoCo/数组/模型/Core/forward/渲染/绘图/运行trial读取。本目录不在身体科学输入freeze内，不修改正在运行的任何源或记录。所有重任务由父在body2八条件及独立审核、最终汇总全部完成后单线程执行。

## 绑定与失败展示

contracts.py严格核验每trial独立VERIFY全部checks真、RESULTSHA/condition/run/visual79dbcdcd.../motor789f.../288state/770actualRGB/CAPTURES与sourcehashes；功能FAIL允许诚实展示，完整性FAIL拒绝。真实汇总文件为NORMAL6_SUMMARY.json和VISUAL_BODY_SUMMARY.json，后者含normal6/control2两组，未创建猜测的CONTROL2_SUMMARY。

## 八条件同步图与真实眼图对照

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/visual_body_v2/visual/plot_sync.py --execute-after-parent-release
```

新独占输出目录：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/visual_body_v2/visual/SYNCHRONIZED_EVIDENCE/`

包含EIGHT_CONDITIONS_SYNC.png、CONTROLS_ACTUAL_VS_ACTOR_EYES.png和PLOT.json。八行三列显示真实发布／策略使用图像capture时间、neuralturn／appliedcommand、实际yaw／匹配V_empty／差值。普通6功能与控制2分列，原滑移／工程门和功能门分列，不把静态帧库叫真实身体视觉。控制眼图在ANN4.82/6/9.5三时刻展示实际已发布LEFT/RIGHT与策略所用LEFT/RIGHT，直接从保存capture_rgb复制放大，不从坐标画像素。

## 代表左右脉冲视频

下面左脉冲，右脉冲仅改condition为V_right_pulse；父必须顺序执行，不能与身体或其他重任务并行。

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/visual_body_v2/visual/render.py --condition V_left_pulse --execute-after-parent-release
```

新输出目录：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/visual_body_v2/visual/VIDEO_V_left_pulse/`

视频ACTUAL_VISUAL_BODY_WITH_EYES.mp4为1280×720，主画面左侧960×640固定相机包含全部71opaque身体，右侧独立320宽两栏实际发布／策略使用的双眼真图，不遮身体。底部声明原滑移门、功能或控制门及未生物验证；保留完整1s稳定+4.8s教师+9.6s ANN，取保存轨迹0,20,...15380共770帧，50fps实时15.4s。只有私有可视Core恢复保存状态、mj_forward；step/step1/step2禁用，qpos/qvel精确、COM/yaw/head派生1e-10检查，物理compiled与source前后不变，framebuffer恢复、770帧完整解码/ffprobe核验。

右眼图／左眼图均来自actual capture_rgb，注释独立写在边栏，不写入眼图。实际已发布图按release_time≤当前画面时刻选最后一帧；actorused按query_time≤当前画面时刻取真记录。settle期可有真实已发布图，但首次neuralquery前策略栏N/A；最初20ms未发布显示N/A，不假称无图或提前用未来曝光。固定帧控制若需以后额外渲染，实际与actor两栏清楚分离，capture4.78/release4.8不会冒充实时新帧。

主世界柱从实际CAPTURES环境metadata重建，只作MjvScene spectator装饰，不影响物理或policy。分割在decor前执行、decor.segID=-1，避免先前已知装饰分割错误。source/capture/release分开：环境off3.6→capture3.6→release3.62，正常恢复窗4.12—4.62；不是气味旧4.1—4.6。

附带DECODED_MONTAGE.png、FRAME_MAPPING.json、VIDEO.json和FFPROBE.json。输出目录存在会拒绝，首次运行API或渲染错误须保留日志并只修visual，不重跑物理；不预先声称运行成功。父完成后必须实际打开两视频蒙太奇、同步图与控制眼图，再核manifest和桌面copySHA才交付。
