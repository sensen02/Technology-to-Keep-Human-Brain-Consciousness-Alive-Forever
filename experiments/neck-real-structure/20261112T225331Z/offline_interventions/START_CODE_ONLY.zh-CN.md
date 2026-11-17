# 最终200模型的离线通信干预：父执行说明

标准库codecheck实际exit0／17项通过，子未导入Torch／NumPy、未读取数组、未加载模型或运行任何干预。旧训练／接口／源数据完全不改；本目录独立预登记并绑定最终eed1权重及真实9e88端口来源。

## 父唯一CPU1命令

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/offline_interventions/run_interventions.py --execute-parent-interventions > /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/offline_interventions/RUN.log 2>&1
```

不带flag仅显示CODE_ONLY。所有输出exclusive，已有INPUTS_BEFORE／RESULT／OUTPUTS时拒绝盲重跑，失败先保存证据与查具体错误，不能覆盖旧结果。

## 精确范围

6条归档自主序列×5条件＝30次完整1920决策重放；每次必须通过实际SourceTransport推进9600个1ms tick，保留2ms队列发布、5ms神经消费两个时钟。没有代理快路径代替实际源队列。baseline的turn／DN／AN／重建／brain16隐藏状态必须与原final200 EVALUATION逐值一致(≤1e-10)。

条件为baseline、AN_cut_from3s、DN_cut_from3s、AN_allzero、brain_cue_allzero。

- 中途切断在ANN3.0s＝tick3000＝决策600执行source.set_cut，拒未来发送并丢在途；同时接收held活动置基线0且清除旧包身份。这个“切断＋接收端基线”的工程代理是显式假设，不称真实神经去传入机制。恰在5ms决策网格切断时在途可能0，actualdropcount如实记录。
- 不reset脑状态或时钟。DN切断后仍调用原学习decoder(heldDN0)，它的输出可能不是0，报告实际decoder_zero_output，不能硬设turn0。
- AN_allzero仍计算原反馈编码／重建，只把实际上传队列输入换为0，cue保持；brain_cue_allzero保留实际AN／两队列，仅人工任务指令变0。
- 归档body111固定不随干预演化。结果是给定旧身体观察下的通信敏感性，不是身体闭环物理因果试验。

每条件保存8类数组与实际队列日志、cut／消费记录。RESULT记录配对全程和3s后turn RMS／mean差、teacheractive符号、DN／AN／脑状态差、反馈重建误差不变和原参数SHA前后精确不变；完整旧源和输入SHA复核。输出不预设效应阈值、不声称脑必然利用AN、真实DNp09转向规律或AN关节调谐。无优化器、反传、新训练、Core或新物理积分。

parent必须先读日志退出码、RESULT.integrity_all_pass和baseline匹配，再解释干预效应。此目录不修改神经网络或旧checkpoint，也不擅自扩试验条件或运行身体。
