# 共享人工脑的动作与感觉联合训练

## 同一个脑

视觉前端和原视觉走停读出作为冻结的子模块收入同一份模型权重。真实双眼特征、经颈部到达的两侧 AN06B002 侧碰和两侧 AN19B004 承重、上行有效位以及脑自己的上一条命令进入共享 GRU。GRU 的一条输出选择继续走或撤掉推进，经已登记的 DNg100 下行送给物理身体。辅助读出使用同一个 GRU 的状态，并在动作选定后使用该动作的副本。

辅助目标覆盖未来 20/50/100/200 ms 的左右侧碰和前足承重，以及未来 50 ms 的六腿股胫角和角速度、六足载荷、身体前速、直立、各足支撑和是否运动。关节与运动真值仅作物理监督，当前没有将原始关节量直接传入在线脑。静止、支撑与运动状态来自真实物理测量。动作标签来自视图保留、真侧碰撤令、虚触继续、上下行干预等原型课程。运行时侧碰阈值只用于记录参考标签，网络动作不读取该分类结果。

先采 9 个新真实双眼/身体示范，全部通过物理和末端停稳检查。另取旧档案中训练组的 36 条走动历程，按原始分组隔离，只为联合版增加辅助监督，动作示范两版一致。两版同容量、同起始种子、同批次、同 440 次 AdamW。独立加载检查两项损失均能更新共享 GRU；随后逐项检查触碰、承重、股胫角、股胫角速度、全身前速与支撑/移动标签的梯度均非零。 [脑结构](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T232400Z-joint-brain/shared_brain.py) · [逐项梯度](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T232400Z-joint-brain/MODEL/SHARED_GRADIENTS.json) · [正式权重](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T232400Z-joint-brain/MODEL/joint.pt)

## 走动留出预测

16 条旧实验留出物理走动中，100 ms 前足承重 MAE 为 .01463 倍体重。提前 50 ms，六腿股胫角 MAE .01919 rad，相同样本的训练集均值基线 .04416 rad；股胫角速度 MAE .08039 rad/s，基线 .10427 rad/s。身体前速 MAE .03350 原生长度/s，基线 .04966。实际移动状态平衡准确率 .830。侧碰提前 100 ms AP .0231，仍处于很低的绝对值。变更未来输入后，已经输出的脑状态和动作逐位一致。 [独立离线评估](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T232400Z-joint-brain/MODEL/joint_OFFLINE.json) · [原脑联合训练设计](/run/media/sensen/Data2/cell_wound_prototype/RECOVERY_DESIGN.zh-CN.md:375)

## 新闭环配对

联合版和仅动作版在相同的新目标时程和 2.295 s 左／右探针刺激下各跑 7 个物理条件，共 6,059,200 原生步。联合版和仅动作版均 7/7 通过。联合版左右侧碰实测始于 2.294 s，颈后在 2.305 s 撤令，延迟 11 ms，并在约 640 ms 后站稳。上行切断组仍发生同一物理接触，行走持续到视觉目标消失后的 4.165 s。恢复组实际在 2.310 s 撤令。虚触继续走，无目标保持站立，下行切断保留脑意图并且身体未推进。实际起走的回合中，推进段没有中途空档。 [全部闭环结果](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T232400Z-joint-brain/PHYSICAL_ACCEPTANCE.json)

两版各自的真实接触、上行切断和恢复组，位姿、速度、力及双眼图像至首次颈后动作分歧 2.305 s 均逐项一致。虚触与真触在物理碰撞前一致，2.294 s 后因真实碰撞已发生而物理分离。仅动作版右侧撤令 16 ms，联合版 11 ms，单一配对不足以断定行为改善。接下来用软侧缘、承重变化和不同触碰时机，在匹配的真实任务内评价误停、漏检和撤令时间。具名关节感觉上行仍待测。 [虚触前缀](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T232400Z-joint-brain/SHAM_PAIR_AUDIT.json) · [同次物理回放](/home/sensen/Desktop/engine修复验收/共享人工脑联合训练-20261116T232400Z/共享脑触觉停步与切断对照.mp4)
