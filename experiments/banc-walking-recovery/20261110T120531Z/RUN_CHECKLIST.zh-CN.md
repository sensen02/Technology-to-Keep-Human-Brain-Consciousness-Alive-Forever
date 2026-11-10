# S2/S5 角色辅助GRU精确300→1000续训完成记录

父模型 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T114148Z/BC_GRU_ROLE_AUX_300/final.pt`，新目录本目录。预登记固定actor、训练专用head、13组BCAdam/RNG精确恢复，原train0/1 val2/roleweight.001/lr.001/seed/SlowGRU结构/物理不改；22精确检查＋81学生检查通过，真实父300零更新metrics与Adam计数逐位核验，tiny直接3步/完整恢复3步相同。

实际bash137续训到1000exit0，所有13Adam计数1000，model/head/BCAdam/RNG/metrics重载一致，critic/std冻结、PPO状态独立空。heldout动作MSE.003263→.00041999，权重和head配置/源/数据/编译物理身份绑定。

bash144新进程同次1s站稳4.8steacher实际感觉hidden预热＋9.6spureANN，无reset/后段teacher/head查询；实际forward.07222696/COM1.255956/nf0/足载1.000052/六腿周期LF1其余2，RH滑移.534655>.5，未通过。其余五腿滑移<.5，不以接近达标改阈值。OFFLINE_ROLE_DIAGNOSIS只事后读取实际神经动作并逐位重放，角色预测随参考阶段一致率.98698，分类器不在控制中，不能把准确率当步行成功。

全部任务退出0并收集、身份独立verify通过；长任务由managed后台运行，记录保留。未来预算扩展在独立125712目录预登记，不改父输入。完整原始验收与最终成功视频尚未完成，goal继续。
