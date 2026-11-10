# S2/S5 同任务BC精确续训：固定总5000步

本目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T113454Z`，父保存模型 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T111240Z/BC_FF_TARGET_1000/final.pt`。预登记PREREG固定，不是无界加预算。

A. 理由是目标位置反馈FF已在实际9.6s稳定足部承重，但1000BC步仍MSE.0010且下降；本次保持全部模型/输入/原BC损失/数据/seed/物理，仅恢复原Adam矩/计数/lr和RNG续到5000。不能model-onlywarmstart冒充resume。

- 检查必须覆盖完整保存后下一优化步与未中断路径模型/BCAdam/RNG逐位一致，critic/std不变，PPO状态独立为空；真实父curve和所有Adamstep=1000只读核对，篡改与数据身份拒绝。
- 数据原SMOOTHtrain0/1,val2，验证不训练/不选择中途“高分”。global2000/3000/4000/5000保存为独立checkpoint，final新进程重载；父目录不可改。
- 同完整身体/原有限PD2rad/s/力矩/重力摩擦/无附着/12.5us物理与1ms积分，只有ANN输出，原actor不见相位/脚本时钟。

B/C. 48精确续训检查通过后bash125实际1000→5000相同任务BCAdam/RNG续训完成exit0；全部Adamsteps5000/criticstd不变/最终model+Adam+RNG/metrics独立重载逐位一致。heldoutMSE.001007→.0001526但bash130真实1sstand4.8steacher9.6spureNN仍forward−.03178/slip15.911/body三门通过COM1.25272/nf0，完整周期0，失败。没有按中途loss选择，父目录/全部globalcheckpoint保留，source/输入identity核验。

D/E. 不以BC拟合或站稳代替全身步行。原始完整验收及最终成功视频未完成，旧负结果与必要权重保留、临时fixture自动清理，goal继续。
