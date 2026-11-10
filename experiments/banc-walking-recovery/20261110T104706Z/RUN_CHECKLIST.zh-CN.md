# S2/S5 执行器目标记忆成对标签：启动登记

新目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T104706Z`，原始假设与参数见PREREG.json。与物理身体/执行器不改、actor raw238不改、原FF层宽不改，训练时增加物理一致目标记忆±扰动配对损失。只作受控候选，不是最终行走成果。

A. 诊断来自 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T103413Z/TARGET_FEEDBACK_AUDIT.json`：T影响观测target−q和target−q0两块，因此两块必须同时扰动。局部冻结身体的反馈矩阵不能当完整物理稳定性证明。教师在未触界时动作对T偏导−10I，由原.05秒目标追赶与2rad/s接口决定。

- 派生身体/原生12.5微秒/1ms积分/5ms动作、有限PD/力矩与速率均不改。所有真正评估只由ANN输出42维，不增加外部目标反馈控制器。
- 目标扰动只在离线输入/教师偏导fixture中，不对真实步行轨迹写姿态/根部/目标；fixture无物理step，必须单列。教师每个探针使用相同新私有历史，避免action推进状态造成假差分。
- epsilon.002rad、pairweight.1、固定随机方向，训练仅原train0/1、验证2不参与。原始BC全行保留，配对只安全rate/target余量足够的样本，源配置/合法样本数与H记录。
- 准备检查包括成对通道一致、完整教师FD、eps敏感性/夹紧排除、梯度方向与下降、holdout/身份、fullcheckpoint/RNG/Adam重载。未通过不启动长训练。
- 计划FF模型仅载parent权重、freshBCAdam/RNG/PPO，1000步lr.0003，seed20261331，父路径 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T102045Z/BC_FF_HALF_RECOVERY/final.pt`。eval同次1sstand4.8steacher then9.6spureANN，所有原门槛不变。

B/C/D/E. 29检查通过：8完整教师fixture三挡eps偏导最大误差2.12e-11、无物理step，3142训练/1543验证配对，checkpoint/BC Adam/RNG/nextupdate重载一致。bash101实际训练1000步exit0，方向误差.999→.934，目标正反馈对角约13→1但局部记忆谱半径仍>1；完整物理未由此证明。bash103新进程真实连续纯神经评估exit0，COM最低.51129/非足38.7149/滑移8.1486，失败。source/hashes与原体保持。最终原始验收和成功交接视频未完成。临时检查fixture清理，必要失败保留；goal继续。
