# 固定ef768纠正样本head2730新候选

- [x] 先PREREG，初始CODE_ONLY；父随后明确释放heavy，代码ready后才执行。
- [x] 单个private Core/0mj_step，完整5800prefix几何tol1e-10及960warm动作tol1e-10；实测两类最大误差均0；初始语法错误bash28在执行前停止，修复后bash29完整0积分通过，均collect。
- [x] 实际trace999+5k完整49qpos48qvel/ctrl/target恢复，privateq_des连续且从actual1s初始化；所有写入仅诊断snapshot copy，禁止reset_from/label积分；新保存私有链不冒充原未保存private数组bitexact。
- [x] CORRECTIVE_EXAMPLES来源/finite/unbacked/unsupported/rateclip等记录与新进程保存审核9214项全通过；postunsupported/unbacked/hypothetical各252/1920=.13125，rateclip97/1920=.0505208333。
- [x] 从固定ef768分片head精确重组2730 densehead，两个source同state初始2880means/192hidden bitexact；base/encoder/GRU冻结。
- [x] 两完整2880轨迹均衡post1920loss，production/cache/gradient/unitAdam/frozen/neutral预检≤1e-12；最大mean差1.333e-15、梯度6.723e-18、unit head参数1.943e-16。
- [x] freshAdam.0003/fixed4000/save2000/4000和独立新进程重放各13项全通过；最终SHAe92b591df32d8f40dccc8e1c23f54ce22b896c199b1cd27bfab29b9b2633cfcb，训练26.86s，均衡MSE.001003028→.0006739593；qualified误差增大、CORRECTIVE误差减小，非行为通过。最后4000唯一物理模型。
- [x] 完整RIGHT独立audit163项完整性全通过，实际yaw30.134122°但slip.73749452显著恶化、严格FAIL，其余门/all6/nativeNF0通过；依事前规则不执行neutral、不扩budget。
- [x] 全job收齐、无运行任务、heavy已交还父LEFT NN；报告 /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/dagger_head_refit/REPORT.zh-CN.md，tracker父负责。
