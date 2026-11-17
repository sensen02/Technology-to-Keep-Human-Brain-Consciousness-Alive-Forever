# 刺激首批A/B运行检查表

目标S4/S5，版本STIMULUS_AB_288_V1；原运动用户视觉接受仅实验准入，不改严格工程失败。

## A 启动前

- 先读150行完整计划、95行协议与71行检查表，已完成。
- 开始git工作树clean，父已push d5c8ce8fb08e52bdf677dfc2e9b8631a6f949e21，GIT_PREIMPLEMENTATION.json另绑定。
- 新ROOT /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z。
- 权重motor789f/sensorbfd6严格原loader，256+32神经状态=288，2受体滤波状态分开；神经ReLU投影[relu(t)/3,t]。
- 首A3，独立每项完整实际+新进程审核，所有functional+integrity通过才B6；无训练/无C/D/无场源趋近声明。
- 原身体71opaque/49qpos48qvel42finitePD/12.5us不变，1s稳定+4.8教师+9.6ANN，撤辅/刺激变化不reset，无actualroot/pose写入或额外力。
- 原ALL5分数与FAIL保留，30deg端点指标另列不用于弱刺激资格。刺激functional门在ROOT/PREREG.json已先冻结。
- B严格0噪声首批浓度钳制，按旧头几何和训练域高斯源算高低C；不用[1,0]超域受体强迫、不将浓度标签直接变动作；实际behaviornoodor偏置保留。
- 每段source/input/compiled身份冻结，CURRENT_STATE/JOBS/REPORT/visual与主tracker排除。补审新名字，旧VERIFY不改。
- PPO/GAE/训练保存测试不适用（不训练），实际模型前向/时序/投影/重放/输入边界与原生峰值计数适用。

## B 实施与资源

- 所有重进程CPU1，OPENBLAS/OMP/MKL/Torch1；物理和模型审核顺序，不常驻大数组orchestrator。
- 子只执行零Core预检并收集，父接管orchestrate.py后台job及最终收集（本次明确所有权交接）。
- 当前physics job状态未运行；最终状态见JOBS.json与CURRENT_STATE.json，不把本模板当通过证明。

## C 独立审核

- 每trial新进程audit.py：strict输入/源/资产、model exact means/actions/256motor+32behavior chain、5ms因果受体滤波、所有15400数组/实际姿态yaw/头部采样几何、完整2880输入/命令/actual action、交接hash、原native计数/NF摘要、原五门与六腿周期重算。
- Native摘要不能从1ms独立重建全部高频力，明确非完整D1—D8。
- B匹配noodor终判只写B_FUNCTIONAL_SUMMARY.json，不回写既有trialRESULT的PENDING状态。

## D 交付

- 完整trace/geometry/neural schema已发父供新renderer。
- 视频尚未制作，父负责实际渲染/全decode/目视与桌面hash；不能把模型或函数检查当刺激成功。
- 新科研失败保留，不重复物理；分析报错先从已保存payload恢复。

## E 当前状态

PREREG/核心代码已准备；真实preflight与A/B结果尚待执行。无无用临时文件。动态阶段报告写ROOT/CURRENT_STATE.json，父负责主ROADMAP/TRACKER。
