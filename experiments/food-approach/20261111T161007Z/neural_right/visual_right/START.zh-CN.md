# 条件渲染启动说明（仅代码准备完成）

当前仅语法与纯mock契约检查通过（退出码0）。未import torch/numpy/MuJoCo/Core/model，未读尚不存在的eval，未加载权重、未做物理、训练或渲染。没有视频成功声明，没有桌面发布，也未改tracker。

父代理等唯一heavy任务及完整EVAL_right和VERIFY_right.json完成后，串行启动：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/visual_right/render.py --run /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/EVAL_right --audit /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/VERIFY_right.json --condition right
```

预期输出目录：/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/visual_right/VIDEO_right。目录必须尚不存在；如实际运行失败则保留失败证据，勿静默覆盖。

脚本严格接受新final2000 RIGHT_ONLY_V1 / yaw/.06身份与同一run完整审核；实际失败数值照实展示，30°只是目标。全部770保存状态/50fps实时、全阶段、全71不透明mesh、固定相机、无轨迹或yaw修正。相机先只mj_forward扫描保存帧的mesh球界和COM全边界确定视距，再逐帧分割断言不裁切；只变内存visual framebuffer并恢复，compiled物理参数前后核验。

父在真正渲染完成后仍须实际read_image打开DECODED_MONTAGE.png，再核验VIDEO.json、全解码退出码和hash，才可拷贝并hash比对桌面。建议桌面：/home/sensen/Desktop/engine修复验收/食物趋近实施-20261110/大角度右转-待你判定-同次神经物理轨迹.mp4 和 /home/sensen/Desktop/engine修复验收/食物趋近实施-20261110/大角度右转-待你判定-同次神经物理轨迹-蒙太奇.png。

轻量检查入口：/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/visual_right/check_codeonly.py；结果：/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/visual_right/CODEONLY_CHECK.json。旧renderer通过SHA256固定并只读复用，未修改。
