# 后续最小动作备忘（不是新训练或预登记）

当前goal `goal-d3ce9a34-ed60-4720-9769-da757b6e602e` revision2 active/armed。先读本目录`CURRENT_STATE.json`与`correction_fit5000/FINAL5000_SUMMARY.json`最终实际结果，不依赖会话叙述。若234仍运行先由直接子e6b5转其子9311收集，不重复物理。

本轮通过之处：原已验证模型和中性身体/神经轨迹保住，实际教师转向与温和减速可通过；气味场/受体接口完成。未通过之处：所有当前非中性神经转向/减速候选仍未满足全部原门，未开始食物行为优化/到达/摄入/饱腹。

本轮已实际检验：旧慢残差500满幅转向滑移.539664/.515163失败，半幅/切换4项.519–.523失败；即时残差1000及稳定低lr续3000均大幅滑移/前进不足；1s渐变并未解决；新增两条实际失败状态纠正标签后5000仍转向失败。不能把原9.6秒直行通过扩展成可控运动已完成，也不能把这些有限失败当作身体不可能。

**如果最终5000五项仍未形成可用运动接口，建议下一目标轮先考虑回到较保守的慢残差谱系，做一次明确固定预算的纠正标签续训，而非继续无界延长即时MLP拟合。** 原慢残差实际滑移距门槛较小，只能作为值得检验的候选理由；两架构的数据、训练预算和残差界限不同，没有证明额外慢GRU是唯一稳定性原因。

可用旧谱系：

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/residual_candidate/residual_motor.py`，冻结base151636参数、可训练43,242参数slowGRU64，explicit双状态128+64，gate=tanh(sum(command²))、latentbound.5，neutral输出严格保留。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/residual_candidate/TRAIN_RESIDUAL_500/checkpoint_000500.pt` SHA`040b26b7ff0e48329c6f6f8f3461f5ced783f108db3d48c264cdf78c5e35e7e3`。保存Adam/RNG/完整曲线在payload config内，先实际审计，不猜字段，不修改旧文件。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/residual_candidate/cached_training.py` 和 `verify_saved.py` 提供旧连续全序列监督与身份验证。

若做新学习实验，必须新时间目录/新PREREG，事先固定父500→例如1500的具体预算/唯一物理检查点/优化器政策，明确只做转向以先取得可用接口，速度未训练不能开放。旧Adam/model/RNG/curve若称精确接续必须真实记录RESUME_STATE，改lr或数据需提前单列；不要复制状态后重置随机或默默清空优化器。

纠正标签来源：

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/recovery_data/COLLECT_left/` 和 `COLLECT_right/`。1920实际观测/command2/teacher_action；前960为真实neutralprime，后960为25ms学生/175ms教师恢复混合控制。
- 轨迹dataset_safety_admitted必须保持FALSE（slip.5495/.5519），不能改成通过示范。`VERIFIED_SAMPLES.json`27项实际记录审计通过，`LABEL_REPLAY.json`全部1920教师标签/私有IK顺序历史逐位一致，maxrootresidual<既有.02诊断值；新预登记可以明确作为失败状态上的CORRECTIVEEXAMPLES学习响应，不代表最优/因果恢复，更不算行为成功。旧4.8s匹配纯教师对照本身可通过，排除了窗口独自解释。

慢残差训练可能较慢。可以在新实验中将4条变长序列（原2turn2880＋纠正2序列1920）按batch4做缓存BPTT、valid padding冻结短序列状态及lossmask，保持逐轨迹post平均等权；实际生产输出/全部双状态/每个可训练梯度需对照原逐序列计算<=1e-12后才采用。若做TorchScript或编译优化，也要先相同验证，不引入时钟/相位/来源输入，不将padding/schedule搬入部署。不为省计算截掉真实连续hidden历史。

备选只读分析已完成但未实际采用：`feature_conditioning/`显示输入尺度异质而非广泛饱和，函数保持标准化需要谨慎处理Adam坐标，不能原样拷贝moment称精确；`command_gate_review/`显示有符号线性神经头可扩大可达范围/改善零命令导数，但同时改变残差范围，不能混同为纯梯度修复。保持本次数据实验变量不变，若未来选这些须另立比较项。

最终验收始终原完整身体/真实接触/42有限PD/native12.5us，无附着/外力/根轨迹写入，完整1s稳定+4.8s教师+9.6s纯ANN/continuoushandover，原五门/六腿/strictNF0/方向与减速要求。实际条件通过后才做命令切换/留出初态和食品行为训练，正式主样本要独立保存原生审计及同次视频，不能靠训练误差或标签审计替代运动结果。
