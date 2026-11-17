# 首批气味刺激与真实身体响应结果

当前结论：9个完整物理回合和独立审核均完成；左右刺激与两向反转产生正确方向身体效应，但左右脉冲撤除后神经转向未按预登记及时回落，因此首批刺激功能并非全部通过。尚未进行空间导航验收。视频正在从同次保存状态生成，目视核验与桌面交付另写DELIVERY。

## 实施与身份

用户要求先Git推送，已完成分支food-approach-20261110提交 `d5c8ce8fb08e52bdf677dfc2e9b8631a6f949e21`，远端同SHA核验exit0。新实验根目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z/`。父托管bash-54运行orchestrate_v2，全部收集exit0；逐trial物理与独立新进程audit的命令/PID/日志/退出码见 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z/JOBS.json`。全程单CPU单线程重任务，未训练任何新权重。

组合motor权重SHA789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd，感觉权重SHAbfd65a3896d259496114695d7f7f3550272cf4c369895a4ddb34f3ddc3352c06。运动256状态加行为32状态，共288神经状态；受体滤波2状态单列。接口为神经ReLU投影 `[relu(t)/3,t]`；感觉网络只读两个受体，无气味位置、距离、方位、时钟、条件标签输入。当前浓度只影响下一5ms的受体，t动作使用previous interval pending response。预热行为状态保持初始，撤辅及刺激变化均不reset。

每回合完整1s稳定+4.8s教师+9.6s ANN，1232000个12.5us物理步，15400个1ms身体端点和2880个5ms决策。全部完整71mesh/49qpos/48qvel/42有限增量PD，原生非足峰0、额外力0、六腿周期完成。运行时无根姿态写入。身份/连续交接、实际动作、神经记忆、受体因果与原指标周期在新进程重放审核；A各208项，B各209项全部true。未保留完整原生力数组，不能据此声称本批完整D1—D8或生物验证。

原v1零Core预检20项通过；跑前补充身份预检12项通过，保留v1冻结字节与FROZEN_SUPPLEMENT。原运动严格资格仍false，用户限定视觉接受不转写为数值PASS。

## 命令探针

| 条件 | 实际全ANN航向 | 相对中性效应 | 最大滑移比 | 新功能门 |
|---|---:|---:|---:|---|
| 左半幅 | +11.917376° | +12.299974° | 2.130314 FAIL | PASS |
| 右半幅 | −12.671116° | −12.288518° | 1.027195 FAIL | PASS |
| 左→中性→右 | +.043778° | 左窗+3.498833°、右窗−3.758211° | 1.832038 FAIL | PASS |

新功能门只含预登记正确方向、稳定/支撑/COM/推进、非足0、六腿周期；原滑移0.5门不放宽。半幅表现不能由端点插值得到；命令会改变GRU记忆和身体闭环。该批不能称连续幅度运动质量全部合格。

## 六个受控气味条件

这些是CONCENTRATION_CLAMP，不是空间源趋近。左右强弱浓度固定为C=.4510848855472039/.34581507678322543，工程转导C/(1+C)、tau50ms，5ms因果延迟。脉冲在ANN .6s起、.8s完成上升；3.6s起撤除、3.8s完成；反转在4.8—5.0s。无气味偏差保留，未人为置turn0。

| 条件 | 实际全ANN航向 | 预登记关键效应 | 新功能判定 | 原最大滑移 |
|---|---:|---|---|---:|
| 无气味 | −.513721° | turn RMS .008586≤.02；相对原neutral额外−.131122° | PASS | .485282 PASS |
| 双侧等强 | −.536204° | 相对无气味额外−.022483°；撤除RMS .017138≤.05 | PASS | .502503 FAIL |
| 左强脉冲 | +26.466129° | 4s相对无气味+8.845956°；合格末段符号均正确 | FAIL：撤除回落 | .962415 FAIL |
| 右强脉冲 | −11.020545° | 4s相对无气味−9.521917°；合格末段符号均正确 | FAIL：撤除回落 | .763723 FAIL |
| 左→右 | −1.169335° | 切换后3s窗口相对无气味−7.665318°；前后符号正确 | PASS | 1.177690 FAIL |
| 右→左 | +2.417993° | 切换后3s窗口相对无气味+8.953523°；前后符号正确 | PASS | .843565 FAIL |

左右脉冲在有效受体输入首次满足预登记阈值的同一5ms决策已具正确转向符号；这是离散符号判据，不能称生理反应延迟为零或已达到满幅。持续末段200个相关采样符号正确，不是200次独立重复。每条件只有1个完整回合。

关键失败：脉冲撤除后ANN4.1—4.6s的turn RMS，左 .9997665142、右 .1145374107，都高于.05门。左侧尤其接近饱和持续转向；不能由左右刺激方向正确就声称完整动态感觉功能通过。该失败不是身体停驻/刹车指标，而是神经转向命令没有按时撤回。

## 证据

权威完整匹配汇总与24个输入hash：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z/B_FUNCTIONAL_SUMMARY.json`

A阶段冻结准入：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z/A_GATE.json`

全部运行日志入口：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z/PIPELINE.log`

左右脉冲实际结果：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z/trials/B_left_pulse/RESULT.json`
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z/trials/B_right_pulse/RESULT.json`

对应独立审核：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z/trials/B_left_pulse/VERIFY.json`
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z/trials/B_right_pulse/VERIFY.json`

## 后续最小动作

先交付这批真实视频和六条件同步曲线，明确撤除恢复FAIL。随后从保存的受体/神经状态诊断持续输出，优先独立有限行为网络长序列训练/验证，冻结运动权重，不用手写deadzone、状态reset、刺激标签动作或距离门控修补。旧静态数据仅约1.2s，不足以验证9.6s历史后的撤除；循环吸引态只是待验证候选原因。动态接口修复后再开始固定空间源开发布局，先用无气味真实头部路径检查可达性并排除本来就会经过的目标。

当前无空间趋近、到达停驻、摄入、饱腹、原连接组或独立SI/生物有效性成功结论。首批没有建立反应后受体切断的真实身体回合，无气味对照不能冒称该干预。历史失败/权重/完整轨迹均保留；本轮无确认无用的临时碎片待清理。
