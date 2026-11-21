# 本轮执行记录（P1，S1/S3/S4/S5）

- 实验目录：/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261117T234120Z-ascending-paired-adapter
- 父方案：/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261117T232237Z-brain-adapter-plan/PLAN.zh-CN.md
- 预登记：PROTOCOL.zh-CN.md；实际输入/代码/库/参数/轴突与源记录SHA见SOURCE_AUDIT.json及DATA各RESULT.json。
- 身体/脑/颈部权重修改：0；物理资产不改。新增真实颈部积分160000tick，新物理0，复用80000份已归档物理样本；没有全脑积分、任务控制或训练完成宣称。
- 审计：audit.py首次exit1；audit.log保留。audit_v2.log对应修正后exit0。不是重复成功运行。
- 首例：bash /run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261117T234120Z-ascending-paired-adapter/drive.sh prepare.py train_left /run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261117T234120Z-ascending-paired-adapter/DATA/train_left，exit0，train_left.log。
- 后7例实际完整命令/退出码/用时：BATCH_COMMANDS.json，全部exit0。
- 训练资格：bash /run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261117T234120Z-ascending-paired-adapter/drive.sh train.py /run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261117T234120Z-ascending-paired-adapter/FIT_GATE，exit0，更新0；实际监督梯度、Adam更新及新权重重放未执行。
- 验证：bash /run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261117T234120Z-ascending-paired-adapter/drive.sh verify_data.py，exit0；PREPARATION_ACCEPTANCE.json各门全true。
- 独立加载：held_left_replay、held_left_cut_left_replay两新进程，7数组逐值一致。
- 同次图：ASCENDING_PREPARATION.png实际打开；观察说明见报告，不制作替代身体视频。PPO/GAE不适用。
- 没有教师生成、Eon感觉响应、DN选择或身体闭环验收；不能把准备通过等同翻译学会。
- 当前问题：共同原生教师缺失、跨样本感觉边界未核对。未修改旧记录或覆盖通过身体。
