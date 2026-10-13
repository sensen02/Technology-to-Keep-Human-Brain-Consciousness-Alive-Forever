# 激活积分步长核查协议

手工研究原型，不是成品。零下限真实神经分支观察到act最大2而ctrl≤1；这是数值诊断线索，不是调整增益的依据。

三个独立步长0.1/0.05/0.025ms，以同一零下限模型、同一初态和固定控制阶跃做5ms诊断，使用MuJoCo原生激活函数和高精度连续ODE参考。直接控制只用于积分器单元实验，**不当自主运动成果**。不裁剪act，不改变动力学参数。

若较细步长确有收敛改善，下一步另建真实神经闭环分支，保持神经0.5ms、感觉/命令5ms的物理采样周期不变；不能仅把身体步长变细同时偷偷改变神经输入更新频率。原规范及零下限基线保留。

## 实测结果

3个真实模型5ms诊断完成，4项测试通过并由父任务复跑，图已实际查看。

| 身体步长 | 最大激活 | 相对连续ODE最大绝对误差 |
|---|---:|---:|
| 100μs | 2.0 | 1.381377 |
| 50μs | 1.0 | 0.559825 |
| 25μs | 约1.0 | 0.209985 |

三组均有限、无仿真警告，但误差显著。初始有效激活时间常数50μs，100μs欧拉步直接产生2倍输入激活。50μs不超范围仍瞬间跳到1，不能称已收敛；25μs也有明显瞬态误差。5ms末误差很小不能掩盖早期力脉冲错误。

已启动显式细身体时钟真实神经分支（100μs重现、50/25μs比较、25μs静默与回放），保持神经0.5ms/感觉命令5ms不变。未改主入口的严格时钟检查，不预报站立改善。

- 实现：`/run/media/sensen/Data2/cell_wound_prototype/run_activation_timestep_audit.py`
- 测试：`/run/media/sensen/Data2/cell_wound_prototype/test_activation_timestep_audit.py`
- 结果：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/activation_timestep/activation_timestep_results.json`
- 图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/activation_timestep/activation_timestep.png`

MuJoCo激活动力学参考：[API](https://mujoco.readthedocs.io/en/3.9.0/APIreference/APIfunctions.html#mju-muscledynamics)。注意接触约束solref时间常数规则不是肌肉激活规则，不能混用。
