# RH承重加权读出8000：可行性成立，实际左转仍滑移FAIL

ROOTPREREG在分析和优化前固定RH承重样本RH7损失权重4。只读feasibility满足预登记条件：RHloaded误差能量占.0918612，weightedgrad5.2538e-5有限非零、与uniform梯度cos.37072；真实动作前载荷idx999+5*k/2880clock/RH列/关节排序/W验证，手工/numpy/训练weightedloss精确一致。没有给部署网络加入loadmask/phase/source/食品坐标。

精确head2000→固定8000，原2730正向head参数训练，原194878motor和128+64记忆冻结，数据/神经界限/身体均不变；lr.003→.001和loss同时改变，不能作weighting-only因果结论。6000更新20.305s、总27.866s，childbash13exit0收集；5000/8000各23独立保存检查PASS，actualmodel/Adam2/RNG/全curve/固定mask/negativezero2880一致，livepresavefullstate比较和构造后liveRNG恢复正确。只实际测试最终8000。

最终checkpoint：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T031057Z/train/TRAIN/checkpoint_008000.pt`
SHA：`65e3635ec966323f22ce30b18f74cef0ccb6445499dbed0e6d1264dce75d15ba`

| 实际条件 | 最大滑移，要求<=.5 | yaw rad | 原工程结论 | 审计检查 |
|---|---:|---:|---|---:|
| neutral | .49278883265 | -.00667761 | 全5门/all6/NF0 PASS |128|
| left | .5172079401750383 | +.06092919134 | 仅slip FAIL，其他4/all6/NF0/方向PASS |104|
| right | .49501104818 | -.07642541452 | 全5门/all6/NF0/方向PASS、实际完整父right保持 |147|

parentOWNbash14三回合和独立审计exit0并收集。评估46项实际finalmodel准备核验及源码哈希无漂移，所有actualsource_paths在Core前验证，负载权重sidecar身份纳入证据。实际full1s稳定＋4.8教师＋9.6ANN、全71opaque/42有限PD/native12.5us，无附着、外部体力或广义力、根脚本，双记忆连续，无删窗/改分母/降低门。

本轮提高RH承重误差训练权重的实际结果没有达标，也未优于head2000 .514841585。故不将该模型交给食品行为，不增加本计划权重系数/步数。教师拟合改善不能代替完整物理能力；不作身体不可行推断。

下一项独立、提前登记的有限物理校准位于`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T033228Z/`：固定原head2000，只改RH两个屈伸输出神经元的scalar bias正/负.05，共4候选。它研究偏置工作点而非重复统一RH连接缩放；全部其他权重/物理门保持，任何结果前模型和hash全冻结。该计划尚无实际结果，不能提前声称修复。

当前goalactive/armed rev4/round4，神经左转、神经减速、命令/初态资格、正式原生审计及食品闭环/切断重放仍待完成。原行走和固定rightPASS/所有失败数据完整保留。
