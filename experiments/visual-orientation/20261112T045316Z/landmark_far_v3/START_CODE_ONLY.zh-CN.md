# 远地标视觉唯一候选：父代理执行说明

本子仅CODE_ONLY准备与标准库检查，没有加载数值模块、模型、数组，没有forward、渲染、训练或物理。父代理已报告真实42帧库生成及253项独立审核通过、实际打开PANEL；这属于父执行证据，不能把本子代码检查当实际数值预检。

## 已固定范围

- 新目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_far_v3/`。
- 相机原32×32×2实际RGB、20ms捕获/20ms发布延迟、5msGRU和原14参数视觉架构、motor789f保持。
- 新farT0..3 forward1.4/1.8/2.2/2.6、angle.35/.40/.45/.30；新V4/V5 forward1.6/2.4、angle.375/.425、真实保存姿态7299/14799。
- 6config×7scene=42实际双眼帧。遮挡场景始终同时放左/右两个目标中心闭合BOX，halfsize[.18,.18,1.4]、Z1.4；同双罩无目标像素counterfactual实际相等，眼在两个罩外clearance>.005才可不可见标签准入。坐标只环境与离线标签，不作为actorinput。
- 36新far长序列，原近域图不混入；18pairs每步uniform2pair展开batch4。所有empty/center/offset/occlusion/sign/response门原值不改。
- strict79db最终权重精确初始化、freshALL14Adam1e-4、fixed500/full1920BPTT/save0/250/500仅final500验证。所有失败保留，不选250、不加500、不在runtime归零。

## 已实际运行的标准库检查

```bash
python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_far_v3/check_codeonly.py
```

exit0，27项检查全部true；Torch/NumPy/MuJoCo/physics/flygym均未导入。记录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_far_v3/CODEONLY_CHECK.json`。

该检查已运行，文件为exclusive，不应盲重跑覆盖。只在源码确实变更且另立记录时新名检查；当前所有执行输入停止修改。

## 父统一训练命令

先确认父运行的library生成、独立审核与实际开图完成，且无别的heavy任务，再托管执行：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_far_v3/policy/run_candidate.py > /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_far_v3/policy/PIPELINE.log 2>&1
```

纯标准库orchestrator依次：真实preflight(从42实际RGB生成36/12/8帧ID、核14梯度/Adam/RSS和新域段权重)→fixed500train→独立新进程0/250/500保存生产RGB重放→final新12验证+8控制→独立验证输出/门重放。阶段命令/PID/日志/退出码写policy/JOBS.json，动态CURRENT_STATE明确功能与完整性分开。没有自动启动身体。

最终严格loader接口：`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_far_v3/policy/common.py` 的 `load_checked(path,digest,require_final=True)`，格式VISION_FAR_LANDMARK_POLICY_V3，仅final500可验证/准入。最终文件 `policy/TRAIN/checkpoint_000500.pt`，SHA必须实际训练完成后获取，不能预猜。

如果任何阶段报错，检查实际JOBS/stage日志和已完成文件，不盲重启全pipeline或重训练；先前物理/近域失败权重与source一字不改。功能失败正常保存且不进入新身体。离线全PASS仍不等于真实身体远地标定向通过；body由另一个子/父独立新scope处理，不在此处实现。

## 证据边界

这是限定工程远地标视觉定向，不宣称已修好近距视觉、自由视觉导航、障碍绕行、减速停驻、摄入、生物复眼、嗅觉融合或原连接组能力。更远的源避免了本窗口直接走过地标，但这不构成任意距离控制证明。旧nearbody中心RMS/右到左/遮挡失效保留，任何本新域通过均不能回写旧FAIL。
