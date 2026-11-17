# C6空间源可视化：父执行说明

仅CODE_ONLY准备完成，未读取trial数组、未导入NumPy/Torch/MuJoCo/Matplotlib、未构造Core或运行forward/绘图/渲染。stdlib检查exit0/11checks PASS，证据 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/spatial_C/visual/CODEONLY_CHECK.json`。

直接按真实C runtime.py/spatial.py/summary.py schema实现，不借用B_FUNCTIONAL或clamp标题。所有输出仅新CROOT/visual，不修改已冻结C源码。父先等全部6C实际/独立audit/C_DEVELOPMENT_SUMMARY齐全后释放唯一heavy，逐命令串行执行。各输入/source/checkpoint/全部288独立审核、固定worldsource/actualhead/1msdistance/arrival重算检查必须真；工程门False可诚实展示。

## 优先六布局路径图

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/spatial_C/visual/plot_paths.py --execute-after-parent-release
```

输出 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/spatial_C/visual/C6_PATHS/SPATIAL_C6_PATHS_DISTANCES.png` 与PLOT.json。

六行固定C_left_near/right_near/left_far/right_far/front/noodor；三列为实际头部采样中点XY路径、固定气味源与patch圆和新C_noodor matched路径；当前中心/边缘距离与matchednoodor相对该行source的距离；真实causalreceptors和raw/projectedturn。每个source单独以同一新noodor路径计算baseline距离，不用noodor自己的源距离冒充匹配对照。进入、201连续端点且真实跨度≥.2s的确认、穿出/过冲分列，阴影仅endpointinside，不证明全连续时间都inside。气味on.6/riseend.8明确；前向C_front单独不构成感觉因果证明。

## 代表近源视频

左近源：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/spatial_C/visual/render_source.py --condition C_left_near --execute-after-parent-release
```

右近源同命令，仅改`--condition C_right_near`。输出 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/spatial_C/visual/VIDEO_C_left_near/` 或对应右源目录，含ACTUAL_SPATIAL_SOURCE.mp4、DECODED_MONTAGE.png、VIDEO.json、FRAME_MAPPING.json和FFPROBE.json。

保存真实qpos/qvel/time只用于visualdata恢复，mj_forward恢复几何，step全部禁止。完整1+4.8+9.6秒/770帧50fps/15.4s实时、71opaque全身分割入镜、固定相机并包含source位置、源/compiledruntime前后核验、framebuffer960×640恢复。每帧实际COM/yaw/head/sensorlocalxyz与保存值核验。图上patch仅`MjvScene` usergeoms的32段ring，`mjv_initGeom`/`mjv_connector`，objid=-1/decor，不改MuJoCo模型或碰撞；源从真实5.8s anchor起固定世界坐标，不随身体走，不在预热泄漏未来anchor。无目标平移/身体轨迹修正。

英文overlay明确SPATIAL_GAUSSIAN / DEVELOPMENT、phase/ANNelapsed、真实causalz与heldC/turn、sourcefixed/distancecenter/edge/inside/endpointconfirmation/afterentry、全回合pass-through、Original5/实际slip。不把通过目标区称停车、摄入或D留出成功。C_left_near已知mincenter.003516/停留endpointspan1.153s后穿出、strictslip1.9029失败，未来视频必须保留FAIL，不用到达掩盖滑移。

图像与视频生成后，父必须实际read_image打开PLOT和两个montage；检查完整decode770、50fps15.4s/metadata、sourcebeforeafter/桌面copySHA，再发布桌面。当前本子未运行任何重任务或产生正式样本。

## 运行期风险范围

本轮只做AST/stdlib核验，无MuJoCo usergeom API运行验证；父若首次marker接口或mask出现错误，应保留失败日志/空staging说明，仅修新visualscope，不能为画面调整冻结C物理或重跑实际body。图与视频各独占输出目录防覆盖。既有C来源身份/指标完整性必须通过，原工程与功能资格不更改。
