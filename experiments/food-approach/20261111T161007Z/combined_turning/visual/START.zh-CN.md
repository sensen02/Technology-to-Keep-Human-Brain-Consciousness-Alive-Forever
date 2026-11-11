# 组合单SHA双向条件渲染（CODEONLY）

只完成语法、精确源码派生和18项纯mock负例检查，退出0；没有import Torch/Numpy/MuJoCo/Core/model/model_io，没有读未来eval、加载权重、训练、物理或渲染。父已代转确认接口为 /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning/COMBINED.pt，load_checked(path,digest,require_final=True)。组合没有训练curve。

必须等待父放行、唯一heavy任务结束，以及同一COMBINED SHA的两方向完整RESULT和独立VERIFY全部完成，才能逐条串行启动（不要同时运行）：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning/visual/render.py --run /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning/EVAL_left --audit /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning/VERIFY_left.json --condition left
```

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning/visual/render.py --run /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning/EVAL_right --audit /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning/VERIFY_right.json --condition right
```

分别输出 /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning/visual/VIDEO_left 和 /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning/visual/VIDEO_right，含ACTUAL_MOTOR_HANDOVER.mp4、DECODED_MONTAGE.png、VIDEO.json、FFPROBE.json、FRAME_MAPPING.json。已存在输出目录会拒绝覆盖；实际失败应保留后按原因处理。

运行时严格load_checked、COMBINED_TURN_V1、128+64+64=256、全部审核true、两方向actual同SHA；直接核对saved(2880,256)全hidden chain。engineering_pass为False可正常制作，保持数值与FAIL标记，不修改轨迹、不放大yaw、不拼组件视频。每个视频固定相机，先由全程保存COM和所有采样帧mesh球界确定视距，再逐帧校验全身轮廓。770帧/50fps/15.4秒包含完整1+4.8+9.6秒；0积分，仅forward。内存framebuffer960×640声明并恢复；原compiled物理参数前后核验，stream CPU1编码、完整解码与源hash验证。

父实际打开两张解码montage、确认VIDEO.json/ffprobe/全解码退出码及源hash后，再拷贝并hash核验：

- /home/sensen/Desktop/engine修复验收/食物趋近实施-20261110/大角度左转-同一神经模型-待你判定.mp4
- /home/sensen/Desktop/engine修复验收/食物趋近实施-20261110/大角度右转-同一神经模型-待你判定.mp4
- 对应montage使用上述完整路径basename加-蒙太奇.png。

现有f25b桌面视频保持明确历史；本准备未删除任何历史或科学证据、未发布新视频、未改tracker。已按5dac确认固定要求full2880_threehidden_decisions、full256_hidden_chain、actual2880_checkpoint_threehidden_exact三审核key为true，并直接验证256 shape/chain及HANDOVER.json的state_dims=[128,64,64]；HANDOVER也加入源hash账本。已实际重读最终脚本，AST确认无整数192旧hidden维度或dualhidden残留（1920仅ANN决策数）。
