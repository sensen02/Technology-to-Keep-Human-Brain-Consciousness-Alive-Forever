# 实际视觉经来源限定颈部的10条件身体试验：父执行说明

当前为ONLYCODE。子代理实际标准库检查53项全true、exit0，没有数组反序列化、模型加载、forward、Core、渲染、训练或物理。初始50项报告与额外namespace检查中的一次ModuleNotFoundError已原样保留；最终CODEONLY_CHECK记录当前源码SHA。旧模型、旧物理、相机源码和结果未改。源码READY后冻结，本文不进入科学source清单。

父须先完成以下真实最终300校准全部7阶段及保存／指标独立审核，不能凭未知SHA启动身体：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T033622Z/visual_bridge_calibration/`

1. 用实际final300权重SHA绑定：

```bash
python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T033622Z/visual_body/prepare.py --bind-completed-calibration --brain-sha256 ACTUAL_FINAL300_SHA
```

2. 父唯一CPU1实际零Core预检：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T033622Z/visual_body/preflight.py --parent-resource-released
```

3. 上述实际通过后，父managed后台串行执行固定22阶段：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T033622Z/visual_body/orchestrate.py --parent-resource-released > /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T033622Z/visual_body/PIPELINE.log 2>&1
```

22阶段为10完整物理＋10新进程实际RGB／身体观察源队列神经重放＋summary＋独立fixedframe配对review。每回合1+4.8+9.6秒、1232000原生步／15400物理ms／3080神经rows、304递归状态(256motor+32vision+16brain)，CNNcache16与AN1/DN2另记。没有新训练或自动追加候选，已经完成的物理不能因报告bug盲目重跑。

视觉真实20ms曝光／20ms延迟，5msCNNcache/GRU查询；first0/5/10/15ms invalid只settle，不encode假零图。AN真反馈从t0，vision32/brain16/DN0预热，teacher960次motorprimecmd0，5.8交接不reset/clearqueue。sourceDN释放2ms、下一5ms决策消费。fixedframe首次新右图在ANN4.82可改变视觉与脑状态，DN4.822发布、motor4.825消费分别记录。

新camera类及全部helper／opticalproxy已完整重绑定到此新ROOT，尤其occlusion_counterfactual写入新试验目录，不会覆盖旧视觉冻结数据。远地标forward2和z1.2均模型原生长度，SI NOT_ESTABLISHED；角度±.4rad。所有condition／坐标／刺激码只用于渲染和审核，不能进actor。脑只读已交付AN＋真实RGB派生的连续视觉rawturn，不缩放／取sign／加deadzone；身体解码器只读已交付DN。

canonical summary：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T033622Z/visual_body/NEURAL_VISUAL_BODY_SUMMARY.json`

其实际kind为`ACTUAL_RGB_SOURCE_NECK_VISUAL_BODY_SUMMARY_V1`。最终保存重放／配对复核：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T033622Z/visual_body/PARENT_READY_REVIEW.json`

原正常6视觉阈值、遮挡和固定帧条件未放宽。功能失败与完整性失败分开；原五工程门／滑移逐条件保留，不能用限定视觉功能PASS当全运动／完整颈部／原生视网膜／一般导航或意识通过。嗅觉仍断开，本代码没有恢复电极刺激。
