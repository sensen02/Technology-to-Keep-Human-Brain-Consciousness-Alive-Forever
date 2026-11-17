# 前足承重、侧碰分离与预测首轮

人工搭建大脑项目。本轮在同一脑、颈部工程接口和身体范围内，沿用双 T1 AN06B002 接收侧向碰撞，新增双 T1 AN19B004 承载前足地面法向力代理。BANC v888 官方 v2 边中左右 AN19B004 有 21、18 条来自同侧前足钟状感器的直接突触和 16、47 个直接脑侧伙伴边。发布完整骨架的轴路连续通过两处颈截面，前切面距离相应标记约 13、41 nm。4 根具名上行投到 193 个图中脑侧靶点，采用互不混淆的实际靶点池作工程读出。2 ms 传递和载荷编码仍属工程标定。 [官方数据](https://doi.org/10.7910/DVN/7WTH1N) · [通路与来源证据](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T225900Z-separated-afferents/SEPARATED_GRAPH.json) · [固定绑定](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T225900Z-separated-afferents/SEPARATED_BINDING.json)

## 现实物理分离

以左、右前足各做承重和侧碰的 2×2 试验。目标脚取消平面承载许可作为卸载干预，仍能与独立侧向探针相碰。其余五足继续使用平面。改变的是受控地面相碰关系，回合内无姿态重置或外力注入。承重看脚与平面的真实正常载荷，侧碰看脚与探针的原生求解接触。16 组共 64 条实验，3,481,600 原生步。所有案例通过物理和读出门，留出组 16/16，每种组合 4/4，测得侧碰最迟 3 ms 到达已阈值化的脑读出。32 条触碰对照在第一次实碰前的 5 ms 抽样位姿与速度逐值相同；另一完整四条件物理复跑逐字节相同。重放时仅静默两根侧碰来源，承重读出逐值一样；仅静默两根承重来源，侧碰读出逐值一样。整个上行切断后返回缺测，恢复后 2 ms 有效。 [分离验收](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T225900Z-separated-afferents/SEPARATION_ACCEPTANCE.json) · [物理配对](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T225900Z-separated-afferents/PHYSICAL_PAIR_AUDIT.json) · [四条件曲线](/home/sensen/Desktop/engine修复验收/承重侧碰分离-20261116T225900Z/承重与侧碰四组对照.png)

## 预测模型与留出结果

验收通过后，使用上述 64 回合与前一轮 80 个走停和侧碰回合。旧 80 条按各自输入哈希把传感器活动经新的四根已登记纤维实际重放，旧触觉脑端信号复现至 2×10⁻¹⁵ 容差，负载从最近一次 5 ms 实测样本保持。前向因果 GRU 从四条到达活动、有效位和 DN 指令副本预测未来 20、50、100、200 ms 两前足承重及左右侧碰概率。完整、有源负载消融、指令副本消融各训练 420 次，工作分组隔离，独立加载检查前三个因果输入前缀。身体真值仅用于评分和标签。 [训练日志与权重身份](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T225900Z-separated-afferents/MODEL/full_TRAIN.json) · [训练权重](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T225900Z-separated-afferents/MODEL/full.pt) · [留出指标](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T225900Z-separated-afferents/MODEL/FORECAST_EVALUATION.json)

- **走动承重。** 提前 100 ms 的平均绝对误差为 .0151 倍体重，直接保持当前载荷的基线为 .0164，移除承重上行为 .0353。实际高载/卸载判别的平衡准确率 91.5%。提前 200 ms 的误差 .0241，比直接保持的 .0311 低约 22.3%。短 20 ms 以及对照台架的稳定承重区间仍以当前载荷持值更准。
- **侧碰提前预测。** 走动留出在 100 ms 触前 AP 为 .0083，事件发生率 .0076；去掉指令副本为 .0104。该批数据仍缺乏对碰前可靠判断的证据。具名侧碰上行在实际碰撞后的检测已经验收。此次没有用预测改变运动策略。

下一最小动作是保留分离上行，在走动中加入可提前观测的侧缘与同外观无侧碰对照，使用新的留出场景验证提前触觉对闭环行为是否有增益。 [运行检查表](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T225900Z-separated-afferents/RUN_CHECKLIST.zh-CN.md)
