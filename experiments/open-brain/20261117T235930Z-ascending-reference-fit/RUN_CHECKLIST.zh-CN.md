# 执行记录（P1，S1/S3/S4/S5）

目录：/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261117T235930Z-ascending-reference-fit

- 用户授权：先补齐，再训练。预登记PROTOCOL.zh-CN.md；切断补充CUT_PROTOCOL.zh-CN.md在切断前冻结。非成品，P2/身体/飞行范围不扩张。
- 输入身份：BINDING.json/PUBLIC_SOURCES.json、FIT_INPUTS.json、DATA各RESULT.json、各科学运行source_sha256。实际源码/库/SWC/表/物理归档均哈希固定，旧文件不改。
- binding首exit1（AN互连假设错误，未积分），修正边界声明后binding_v2 exit0。未偷改连接真值。
- 两训练参考命令/退出码：TRAIN_DATA_COMMANDS.json，全0。fit.py exit0，fit.log；250更新＋独立恢复资格影子1更新、251完整状态保存。
- 冻结后两留出参考/四重载/一预测重放：EVALUATION_COMMANDS.json，全0；留出标签文件生成时间晚于FIT_RESULT，未用留出挑模型。
- brain_probe.py BRAIN/BRAIN_REPLAY各exit0，日志brain_probe.log/brain_replay.log，6000批tick各4lane；14回馈边明确排除、原脑矩阵不改、DN全0。
- cut_restore.py CUT_RESTORE/CUT_RESTORE_REPLAY各exit0；CUT_COMMANDS.json，实际40000颈tick；无切断标签输入、因果尾迹保留。
- verify_and_plot.py exit0，verification.log；FINAL_ACCEPTANCE.json全部预声明工程门通过；原脑7数组/切断5数组/预测4数组精确。
- 神经/物理/训练数量见FINAL_ACCEPTANCE.json；新身体积分0，身体权重不改。原脑学习0、颈可塑性0。
- 两PNG已实际打开；MEDIA_REVIEW.json。没有新身体视频，不复用旧视频声称新行为。
- 未执行：其余上行、连续压力、生理感觉验证、脑自主决策及完整AN脑回馈闭环；PPO/GAE不适用。
- 下一最小动作：检验感觉驱动DN的已有通路并处理必要的原脑有界局部适配；补AN回馈状态所有权。
