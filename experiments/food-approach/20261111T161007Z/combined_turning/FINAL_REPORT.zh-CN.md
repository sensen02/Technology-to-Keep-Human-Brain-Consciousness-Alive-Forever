# 同一神经模型的大角度左右转：实际样本交付

已用同一组合神经模型分别完成完整直行、左转、右转真实物理回合，并交付左右视频供用户裁定。当前严格工程验收仍未全部通过，未放宽门槛或把失败改成通过。

| 同次9.6秒ANN指标 | 直行 | 左转 | 右转 |
|---|---:|---:|---:|
| 实际航向变化 | -0.382599° | +29.651288° | -30.632932° |
| 最大逐腿滑移比 | 0.492788833 | 0.578096809 | 0.503296114 |
| 六腿完整周期 | LF1，其余2 | LF1，其余2 | LF1 LH2 LM2 RF2 RH2 RM1 |
| 原生非足接触、额外力 | 0 | 0 | 0 |
| 支撑、非足、COM、推进四项工程门 | 通过 | 通过 | 通过 |
| 独立记录完整性检查 | 193项通过 | 169项通过 | 169项通过 |
| 严格工程结果 | 通过 | 角度不足30°、滑移超过0.5 | 滑移超过0.5 |

左转缺少约0.3487°，滑移超过原门0.07810；右转滑移超过原门0.00330。数字不四舍五入为合格。记录完整性通过只说明实测可信，不代替行为门通过。用户视觉裁定尚待给出。

两片均为15.4秒、50fps、770帧、实时：0—1秒稳定，1—5.8秒算法教师预热，5.8—15.4秒纯ANN。撤辅不重置身体或神经状态。渲染只读取同次保存的姿态和速度，0积分/0轨迹修正；全部71个mesh不透明、全帧全身入镜。两片已完整解码，父代理实际打开六阶段蒙太奇查看，另独立ffprobe及源/桌面哈希核验通过。

桌面左视频：
`/home/sensen/Desktop/engine修复验收/食物趋近实施-20261110/大角度左转-同一神经模型-待你判定.mp4`

桌面右视频：
`/home/sensen/Desktop/engine修复验收/食物趋近实施-20261110/大角度右转-同一神经模型-待你判定.mp4`

对应蒙太奇：
`/home/sensen/Desktop/engine修复验收/食物趋近实施-20261110/大角度左转-同一神经模型-待你判定-蒙太奇.png`
`/home/sensen/Desktop/engine修复验收/食物趋近实施-20261110/大角度右转-同一神经模型-待你判定-蒙太奇.png`

统一权重：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning/COMBINED.pt`

SHA256：`789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd`。

组合共享已验证的直行GRU128，左右各有GRU64神经分支，共256状态。分支由命令的神经ReLU门调制，均持续更新、不在交接时reset。它由两个分别训练的分支组成，不冒称自然联合训练；左右不是不同权重视频拼接。右组件b4ed的可视选择为事后名义比较，已在组装前AMENDMENT04声明，严格工程失败保留。原直行权重及其验收档案没有覆盖；组合的实际直行全共享数组仍与原基线一致。

原始实际结果与审核：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning/EVAL_neutral/RESULT.json`
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning/EVAL_left/RESULT.json`
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning/EVAL_right/RESULT.json`
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning/VERIFY_neutral.json`
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning/VERIFY_left.json`
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning/VERIFY_right.json`

最终合取与交付哈希：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning/FINAL_SUMMARY.json`

两方向各为一个固定标称初态9.6秒ANN回合，属于替代ANN／已声明派生完整身体／有限PD实现。左转命令伴随较快前进，当前仅验证端点模式与一秒渐变，未证明任意命令切换、任意时长、持续趋食、原连接组或生物有效性；本次原生记录为全步非足峰值等摘要，不是新的完整D1—D8认证。所有研究失败、编排错误及身份恢复说明保留。最终子代理仅在整理阶段中断，所有物理与独立审核已收齐，父未重跑物理，独立接手最终合取和视频验证。
