# B门末端诊断结果入口

**后续资格警告：** 离线复查发现本批baseline内部act最大2.0、965个采样值>1。精确重放/命令层因果保留，但未取得归一化肌肉活化的数值资格，不能宣布完整B门通过。完整限制与下一停止线见 `/run/media/sensen/Data2/cell_wound_prototype/B_LOCAL_FORCE_DECISION.zh-CN.md`。

手动研究非成品，当前仅输出命令层因果。已执行4x50ms body-only重放：baseline精确匹配旧P1端点，repeat全部数组逐位一致；flex13/ext14单命令cut、其余89命令固定、全无push。LF tibia角最大差0.015010949/0.502121121rad，不能解释为生物功能强度或静态支撑成功。

执行及完整限制 `/home/sensen/Desktop/engine修复验收/B-local-replay-20261014T213817/单通道末端因果诊断报告.zh-CN.md`，原始4NPZ/report/source ZIP/图同目录，local-cuts.png已实际查看。

重要更正：此前normal与旧motor_cut/output_silence差带不同push条件，不是匹配单干预效果；本轮新对照无此混杂。当前run_neural_muscle_closure.py原本就在每step后forward，新采集只读额外力字段，不修改轨迹，且端点精确重现已实证。

下一项限离线首次差/目标moment符号与力项解释，不重跑四案例，不调gain或延长倒下，不称完整神经闭环或站立。A门当前姿态静态资格仍失败；站立路线需有来源初态/身体分支决策，生物标定/整体收敛未完成。
