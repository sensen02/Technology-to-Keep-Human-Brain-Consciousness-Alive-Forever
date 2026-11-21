# 本轮运行检查表（P1 / S2、S3、S5）

权威跑前协议：/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261117T192322Z-terrain-body-learning/PROTOCOL.zh-CN.md；观察扩充：/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261117T192322Z-terrain-body-learning/OBSERVATION_EXTENSION.zh-CN.md。报告及FINAL_ACCEPTANCE列全部结果、限制与下一动作。本表为收尾如实汇总，不冒称额外跑前审批或预登记。

- 教师：20261117T184709Z冻结terminal，同体42+6、有限PD、局部CPG；同快照，不改脑/颈；已归档真实颈后两维驱动。
- BC/纠偏：`bash /run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261117T192322Z-terrain-body-learning/drive.sh batch.py train TEACHER_TRAIN`；dev_teacher TEACHER_DEV；`fit.py FIT0 --epochs 60`；`batch.py dev DEV0 --checkpoint FIT0/epoch030.pt`；dagger DAGGER1 beta .5；fit_dagger.py FIT1 --epochs 40 --initial FIT0/epoch030.pt --correction DAGGER1；dev DEV1；dagger DAGGER2 beta .2。输入均绝对路径由脚本HERE解析。
- 扩观察：首fit_plus.py预处理被停止130，0更新0积分；fit_plus_fast.py FIT2 --epochs 40 --initial FIT1/epoch050.pt --correction DAGGER1 --correction DAGGER2退出0。初fit_plus语法多余括号预检失败，在启动前修正；无该版本训练结果。
- 开发/留出/独立：batch_plus.py dev DEV2；screen_checkpoints.py仅开发筛查已有10/20/40轮；batch_plus.py held HELD、replay REPLAY均`--checkpoint FIT2/epoch030.pt`，全exit0；两套最终5/6不冒称通过。原物理门和完整六足跨越要求未放宽。
- verify.py实际运行exit0，后续一次重复只读核验同0；原生物理总1,020,000，新神经tick0，训练7,320updates。PPO/GAE不适用，监督梯度、Adam/RNG和保存重载实核。
- media_student.py、failure_states.py退出0，无新增积分；ffmpeg全解码/ffprobe退出0。340帧/.125倍，同次状态含失败；实际打开STUDENT_STATES、STUDENT_METRICS、FAILURE_STATES。
- 身份：每阶段INPUTS、最终source_sha256；继承教师PREPARE_INPUTS逐项终局重算。缺项：每batch完整checkpoint未存、新运行完整导入跑前清单不足、FIT1父hash仅跑后绑定。见报告，不伪装全规范符合。
- 结果：TRAINED_PARTIAL_NOT_QUALIFIED；原BODY_CURRENT不覆盖、旧产物不删；无后台训练或真实服务/动物副作用。
