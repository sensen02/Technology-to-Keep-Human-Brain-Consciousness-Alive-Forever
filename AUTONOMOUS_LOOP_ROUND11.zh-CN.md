# 第十一轮：神经步长敏感性与上游肌肉来源

目标未完成，飞行暂缓，程序不增加行为控制。手工仿真原型。

## 实际身体回归

独立入口新增 --neural-dt-ms {.5,.25,.1}，用于数值精度检查。默认仍为.5，不依据站立结果选步长。神经连接权重、细胞身份、96club感觉转导、44候选肌肉、5ms接口和0.1ms物理步长不变。神经源码哈希纳入记录。

0.1ms神经步长、0.3秒四分支运行结果：完整138364脉冲，感觉切断0，读出切断138732，突触传播切断823。相比0.5ms完整125018增加10.675%；同一反馈环中的身体轨迹会分化，因此不是逐时刻同电流比较。

完整身体最终高0.73236mm，仍塌陷；减少步长没有解决站立。240区间命令来源检查通过。感觉切断和传播切断的逐步身体轨迹与原0.5ms运行完全一致；传播切断仍仅感觉细胞823脉冲且无主动输出。图已打开核看。

结果 `/run/media/sensen/Data2/cell_wound_prototype/outputs/autonomous_dt_round11/`。

## 独立数值检查

孤立神经元/小网络29项实现检查通过：nF、uS、mV、ms、nA换算正确，矩阵post/pre方向、2ms延迟环、不应期与声明的后向Euler公式正确。没有证据支持反转连接符号或改权重。强电导例中相对解析电压最大误差在dt .5/.25/.1ms分别约5.887/3.438/1.563mV，证明存在一阶离散精度问题，不等于公式写错。

固定同一12k连接与同一96club脉冲输入（50ms .05nA+100ms释放）的独立测试中，三种步长释放期网络脉冲47964/48596/47694，持续活动在更小步长仍存在，不能归咎为仅.5ms步长伪影。未宣称达到生理或完全数值收敛。

数值诊断 `/run/media/sensen/Data2/cell_wound_prototype/diagnostics/diag_neural_numerics.py`，输出 `/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_numerics_round11/`。图已核看。本轮不因需要“修复”就凭空改正确的单位或连接方向。可下一步增加精度更高的电压积分方法作独立对照，不能按站立效果挑算法。

## 上游肌肉数据

配套公开FlyMimic仓库README指向原模型开发库 https://github.com/gizemozd/neuromechfly-muscles 。目前GitHub API与raw主README返回404；不能区分私有、迁移或未发布，不能声称已检查原始分割。已获取公开FlyMimic固定commit的4个小文件约1MB，确认其MuJoCo XML与安装包资产字节完全一致。OpenSim中a/b为独立肌肉条目，有不同最大力、纤维长度与收缩速度参数，没有共同组ID或神经支配表；不能因此推断一定共享或一定不同支配。具体源文件和参数在 `/run/media/sensen/Data2/cell_wound_prototype/outputs/upstream_muscle_round11/`。

原始分割/参数优化仓库仍404。下一步需要其代表纤维选择表、lff_thco_stta命名表与volume/PCSA优化输入，而非继续在最终XML中猜名字。公开树还有一份约1.45MB OpenSim PDF尚未读，可作为后续窄检查来源。
