# 细身体时钟真实神经分支

手工研究原型，不是成品。原规范和零下限100μs模型均保持冻结。

## 依据

单位阶跃独立诊断确认100μs欧拉身体步长相对肌肉初始有效50μs时间常数产生act=2，50/25μs误差下降但尚未证明收敛。本分支不是改变肌肉增益或裁剪状态，而是显式减少积分步长。

## 固定协议

原100μs正常回放须重现零下限已存完整200ms结果；50μs正常与25μs正常只改变身体积分步长；25μs增加神经输出静默和正常命令回放。所有条件神经0.5ms、感觉/命令5ms，零初始act/ctrl、相同感觉参考。使用新显式body_dt调度器，保留主入口严格时钟检查不绕过。

## 实测

五组共15.15秒完成，3项测试由父任务复跑通过，图已实际查看。原100μs复制调度器结果与已保存零下限命令、放电及身体状态精确一致，排除调度器复制本身改变实验。每组均400神经步和40个命令周期；身体步数分别2000/4000/8000。

| 正常组身体步长 | 最大激活 | 200ms根高 mm | 200ms COM高 mm | 原胸高阈值 |
|---|---:|---:|---:|---:|
| 100μs | 2.0 | 1.014438 | 0.640406 | 24.0ms |
| 50μs | 约1.0 | 0.948579 | 0.876890 | 23.35ms |
| 25μs | 约1.0 | 0.831522 | 0.907106 | 23.15ms |

100μs出现2223个肌肉×步样本act>1；50/25μs均0。25μs输出静默始终零激活/命令，正常与回放8类数组精确一致。细步长改善积分超调，也显著改变晚期轨迹，但根高与COM趋势不同，所有条件仍下落/身体触地。25μs相对50μs有差异，不能宣称已收敛，更不能称站立成功。

独立约束检查发现solref均0.02s，远大于2dt，同一初态的约束量在三个dt完全一致，未触发步长相关的refsafe改值。但变细仍改变全身体积分，不能把闭环变化唯一归因激活积分。

## 产物

- 显式时钟副本：`/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/branch_episode.py`
- 实验：`/run/media/sensen/Data2/cell_wound_prototype/run_fine_clock_branch.py`
- 测试：`/run/media/sensen/Data2/cell_wound_prototype/test_fine_clock_branch.py`
- 结果：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/fine_clock/fine_clock_results.json`
- 图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/fine_clock/fine_clock_comparison.png`

