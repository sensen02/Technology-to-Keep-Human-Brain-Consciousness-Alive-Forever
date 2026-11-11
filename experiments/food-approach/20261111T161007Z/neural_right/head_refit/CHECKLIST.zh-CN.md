# S4/S5 uniform head-only候选检查表

- [x] 新PREREG先落盘；旧RIGHT30.1385°/slip.52244 FAIL和全部旧源/权重不改。
- [x] 初始CODE_ONLY阶段未执行模型/训练/physics；父完成左物理并明确释放资源后授权全部预检/4000训练/最终两物理顺序执行。
- [x] 新strict format/head2730参数/frozen encoder+GRU+base/freshAdam两状态实现。
- [x] 实际production/cache输出与全部head梯度/unit Adam≤1e-12、frozen无grad/权重不变、neutral2880严格一致；最大前向差1.777e-15，head梯度差2.365e-18，unit head参数差1.685e-14。
- [x] 固定4000 uniform post960动作MSE；save2000/4000完整状态，live/saved exact与新进程生产重放各13项全通过；最终MSE.0001921895446，训练11.09s。
- [x] 最终4000 neutral/RIGHT两个完整回合及独立审核，原门和30°同窗；neutral168项完整性PASS/严格原基线，RIGHT144项完整性PASS，actual右转30.689918°但LHslip.51066597 FAIL，其余门/全六腿/nativeNF0通过。
- [x] 如实保留失败、收齐job、报告父并交由父更新tracker；报告 /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/head_refit/REPORT.zh-CN.md，已无运行任务。

本表未勾选项均未执行。PPO/GAE不适用，只有head-only BC。不能从当前stance拟合诊断预先宣称能修复滑移。
