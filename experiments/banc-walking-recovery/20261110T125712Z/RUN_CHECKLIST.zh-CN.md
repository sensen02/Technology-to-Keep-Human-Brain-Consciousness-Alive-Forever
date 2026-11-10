# S2/S5 1000→固定2000步同任务精确续训

目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z`，父模型 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T120531Z/BC_GRU_ROLE_1000/final.pt`。原始滑移.5门槛不改，父1000实际6腿前进/COM1.25596/nf0/足载1，唯一超门RH=.534655；其余5腿<.5。此前300→1000确实改善动作MSE和滑移，PPO3u/剪枝未改善，因此本次固定增加学习预算，不变更结构或物理。

A. 只恢复同神经actor、训练专用角色head、13组BCAdam矩/计数1000和RNG，继续原lr.001/roleweight.001/seed20261110/fullsequence/同train0/1val2至2000。不能redo300→1000，不model-onlyreset、不改PPO/神经时间常数/感觉输入。所有source/head/auxconfig/parent checkpoint+RESULT+IDENTITY/provenance/compiledH/tau校验。

- 新检查应覆盖真实父全计数1000与初始metrics逐位一致、零优化预检；小独立直接第三步和完整恢复第三步对比actor/head/Adam/RNG逐位相同；辅助head不进入部署，critic/std冻结，PPO状态独立空。
- 预登记保存1250/1500/1750/2000和final，实际物理评估只做1500与2000，不评估1250/1750、不按中途loss挑模型。任何全部门及六腿周期通过的候选需独立原生复核；1500若通过不打断固定2000训练。父检查点不修改。
- 同完整71mesh身体/无附着/重力摩擦有限PD2rad/s/力矩/12.5us物理与1ms目标积分，部署ANN42输出/实际感觉/连续hidden，无角色标签、计时、腿序输入或ANN外脚本控制。

B/C/D/E. 40精确恢复检查及81学生检查通过后bash155实际1000→2000固定预算完成exit0，所有13Adam计数2000、model/head/Adam/RNG/metrics最终独立恢复逐位一致，heldoutMSE.00041999→.000171667。只评估预登记1500(bash156)与2000(bash157)，均exit0，六腿周期/前向/body三门保持，但滑移分别.5163826/.5040259仍>.5，严格失败；未评估1250/1750，不四舍五入过门。新进程同次1sstand4.8steacher9.6spureNN，无后段teacher/head查询/状态重置。RH_SLIP_DIAGNOSIS显示96.3%积分来自真实承重≥.1W、最后半窗约51%，不能删启动窗规避。原始完整验收与最终成功视频未完成；全部job已收集、失败保留，临时fixture清理，goal继续。
