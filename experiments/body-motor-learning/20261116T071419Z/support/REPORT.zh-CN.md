# 未承重足换足补撑的同身消融

本轮属于人工搭建大脑项目身体台架 S2/S5。使用冻结名义站姿、原腿碰撞几何、42 路有限 PD、自由根和实际静态浅缺口。前期轨迹自然出现右后足应尝试着地时左前足仍在空中。配对两组使用同一个右后足踏空触发与其他支撑足有限保持规则，比较有无左前足额外落脚动作。全程逐原生步记录 12.5 μs 法向载荷和两足位置。

## 实际缺口和对照

A 的地面接缝影响了左后足第一次承重，填平和缺口早期在 1.892 s 同样偏离原地面，已保留但未入资格。B/C 的 x 向分界避开了左后足早期承重范围。C 的实际缺口与填平对照到 4.258 s 姿态逐字节相同，然后才因右后足试落而分开。相同填平布局与原平面到 5.581 s 的姿态差最多 7.27e−15，后来同一铺面接缝影响完整走动，所以填平臂仅用于这一触发窗口的归因。 [几何分离证据](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T071419Z/support/SCENE_PREFIX_C.json)

4.280 s 真实缺口内右后足为 0 W 且触地点低于上表面。填平时同一时刻右后足已有约 0.003263 W，属于正常的新支撑刚开始增长。左前足此前连续 20 ms 载荷为零且最低净空仍约 0.00888 原生长度。主动和关闭补撑的原生力、位置、全体姿态和直到第一个辅助动作前的指令均逐字节一致。

## 命名辅助足确实提前承重的诊断

D 的首个不同动作只涉及左前足关节。主动臂左前足在 4.3070 s 接触真实上表面，连续 39.1 ms 承受至少 0.02 W，首 20 ms 最低 0.020193 W。匹配关闭辅助动作，左前足该 20 ms 载荷全为 0，首次持续承重在 5.1212 s，相差 814.2 ms。D 的独立物理重放中完整 TRACE、DECISIONS、NATIVE 散列都逐字节相同。D 随后出现持续非足接触并停机，整项稳定恢复不通过。这个早期时间窗口支持从空中落脚补撑的因果归因。 [D 原生配对与独立复验](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T071419Z/support/DIAG_D_FOOT_ONLY.json)

C 的固定幅度较低，两臂都完成 8.000 s 并站稳，主动左前足峰值约 0.01813 W，未达到持续承重门。增大幅度的 D/E 与初次负载触发后回退的脉冲方案均未同时完成稳定恢复。保留这些实测和所有关节限速，独立复验只说明其对应物理效果可重现。

D 干预后第一段 500 ms 的总足载最低 0.99604 W，质心最低 1.23591 原生长度，地面非足力为零。停机前末 200 ms 的质心最低降至 0.86716、非足力峰值 0.63339 W。关闭辅助足动作的对应 8 s 回合末窗稳定。该差异要求把实际早期补撑和稳住后的持续回归分开报告。 [对齐诊断录像](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T071419Z/support/VIDEO_DIAG_D_RETRY/换足补撑配对.mp4) · [关键帧](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T071419Z/support/VIDEO_DIAG_D_RETRY/换足补撑关键帧.png)

## 较小幅度与载荷反馈

固定下探 0.055 原生长度的候选保留了末窗站稳。左前足瞬时峰值 0.02277 W，在 0.02 W 以上的最长连续时间仅 1.7875 ms。最后三个受限对照依次采用 0.056、0.057 的固定目标以及以真实左前足载荷为反馈且下探上限 0.058 的目标。它们均未通过末窗站稳，最长持续承重依次为 1.8125、1.8375、2.3875 ms。把较强落脚成功后改回 0.050 的温和目标，可得到 30.2875 ms 的早期连续支撑，但仍在 5.940 s 停机。固定 0.050、0.090 和脉冲方案的匹配辅助关闭臂整条物理轨迹、动作和原生力完全相同。

因此，本轮已验证指定未承重足可以主动建立一个较早的真实新支点。尚无同一方案同时保持规定持续承重和最终稳定，也未验收继续走动、训练学生权重或跨颈功能。下一最小动作应保持这样经过归因的早期落脚，只用当前实际足载与躯干倾角对几条支撑腿的目标做有界恢复，从同一物理状态下验证末窗稳定，避免仅对左前足进一步增加下探幅度。

## 实际调用与交付

解释器为 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python`，环境设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl`。每个 `.exit` 文件记录该独立命令的实际退出码。各切换用有限执行器运动，姿态和求解器原生步闭合校验，无外力或根重置。

- 几何物理对照命令 `support_experiment.py control FILLED_A --case CASE_A_FILLED.json` 及同配置的 on/off。开头错误 JSON 参数退出 1，保留 `PLAN_RH_LF_PRECHECK.json` 与原日志。修正后 A/B 物理实际运行退出 0，实验结果各自按资格检查。
- D 主臂、消融和独立重放分别用 `support_hold.py on ON_D --plan PLAN_HOLD_D.json`、`support_hold.py off OFF_D --plan PLAN_HOLD_D.json`、`support_hold.py on REPLAY_D --plan PLAN_HOLD_D.json`，均退出 0。C 和后续上限、反馈、回退变体均使用各自新的计划和输出目录。原生审计初次因不等停机长度退出 1，按时间交集复核退出 0，未改变任何原始数组。
- 通过 `render_pair.py VIDEO_DIAG_D_RETRY ON_D OFF_D DIAG_D_FOOT_ONLY.json --diagnostic-only` 实际回放存档，纠正了首轮渲染入口的导入名后退出 0。视频全 330 帧独立解码，绘制时没有新的求解器积分，已打开关键帧检查。同步锚点是主动臂实际负载时刻，左右数值显示两组的实测区别。
