# 前足触觉经颈上行首课

复用人工搭建大脑项目的平地视觉走停。冻结双眼、推进网络、身体和局部执行器。只训练两侧前足外侧受触的读出，并用明确的停步任务验证它能否给人工脑提供新的信息。

双 T1 AN06B002 采用 BANC v888 官方完整 SWC，核验下载 generation 和 MD5。左、右前颈标记至同一跨两截面路径的距离为 44.49 和 38.27 nm。官方 v2 定向边表精确联接到 85、71 条前足感觉输入以及 94、92 个脑侧靶点。颈部经过固定的 2 ms 工程传递，脑侧只读取到达后的真实来源投射。已识别生理结构，触觉力代理、权重比例和传递动力学采用声明的工程实现。 [来源图](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T180500Z-touch/AN_GRAPH.json) · [双向绑定](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T180500Z-touch/TOUCH_BINDING.json)

## 触觉读出

训练 15 条、留出 36 条，共 2,754,000 原生步。按指定足段与探针的实际碰撞力做物理标签。身体端仅使用通用的侧向接触力代理，将活动按已核验的纤维发给脑。脑端从官方靶点工程投射的实际到达帧学习两个侧别阈值。留出事件左 12/12、右 12/12，虚触零误报（0/12），端到端延迟 p95 为 4.0 ms。一个留出回合独立重跑 54,000 步，原始数组逐字节一致。范围为同一初始姿态、受控侧向触碰、虚触及所测的短时正常步态。地面滑移、承重强度、主动重启和腹部内感受尚待各自验证。 [留出验收](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T180500Z-touch/DATA/TOUCH_READOUT_ACCEPTANCE.json) · [复验与输入散列](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T180500Z-touch/READOUT_EVIDENCE.json)

## 走停因果验证

视觉地标保持可见时，左触在 2.399 s 由实际求解器记为首次接触。上行完整组在 2.405 s 读出左侧并撤令，身体在 2.410 s 收到停止，之后 610 ms 通过站稳窗。切断同一 AN 组后，同一碰撞仍发生，但脑没有触觉可用，继续推进直到视觉地标撤除。恢复组在 2.408 s 重开上行，新的活动在 2.410 s 被读出，身体在 2.415 s 停止推进，之后 610 ms 通过站稳。右前足组正确读出右侧并同样停稳，虚触组继续按视觉指示前行。两条左触配对使用相同身体起点，首次运动意图变化前的姿态、速度、质心、足载及原生探针数据逐字节相同。

五条各 424,800 原生步，完整、切断、恢复、右触和虚触全部达到各自任务门。首次执行报告保存遇到 NumPy 类型错误，实际轨迹与消息已写出。保留源码和错误日志，修复后五条独立积分退出 0，原始轨迹和收发日志各自均逐字节复现。 [因果验收](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T180500Z-touch/BEHAVIOR_REPEAT/TOUCH_WALK_ACCEPTANCE.json) · [独立复跑](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T180500Z-touch/BEHAVIOR_REPLAY.json) · [回放](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T180500Z-touch/VIDEO_FULL/触觉上行停步对照.mp4)

触觉两侧阈值来自监督校准，触碰触发停步是显式任务映射。后续补充承重、滑移和恢复状态的可用上行及进退安全范围。SAC 阶段可测继续、减速、等待与绕行的自主决策，仍用同源上行切断与恢复量化其贡献。 [运行记录](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T180500Z-touch/RUN_CHECKLIST.zh-CN.md)
