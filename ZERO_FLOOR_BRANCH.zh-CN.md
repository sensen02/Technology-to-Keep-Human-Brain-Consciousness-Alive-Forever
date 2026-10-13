# 零控制下限独立模型：零驱动通过，站立仍失败

**手工研究原型，不是成品。** 按用户新授权建立独立模型，不修改原规范XML或旧证据。

## 改动和控制

只将90肌肉ctrl下界从0.0001改为0。模型资源指向与规范加载器相同的已解析mesh，资源哈希记录；编译参数对照只允许控制范围变化。神经网络、增益、原始初态、感觉参考、零初始act均保持；没有每步清零act、CPG、PD或额外背景。

200ms三组：正常输出、输出静默、正常命令回放。7项测试通过并由父任务复跑，图已实际查看。

## 结果

| 分支条件 | 200ms COM高 mm | 首次非足部触地 | 原胸高阈值 |
|---|---:|---:|---:|
| 零下限正常 | 0.640406 | 95.0ms | 24.0ms |
| 零下限静默 | 0.520239 | 27.2ms | 14.9ms |
| 零下限回放 | 0.640406 | 95.0ms | 24.0ms |

静默组2001个身体状态的ctrl、act、act_dot及同状态激活依赖肌肉力全部严格为0，被动力允许存在。正常和回放的命令、完整身体状态、肌肉力、COM轨迹精确一致。

原规范正常200ms COM为0.625227mm，零下限正常为0.640406mm，均倒下；首非足触地均95ms。解除残余驱动问题没有解决站立，不能把消除一个验收障碍称为整体完成。

新增数值发现：正常组原生激活最大达到2.0，尽管命令上限1。正在独立检查0.1ms积分步长与肌肉快速激活时间常数的相容性；不直接裁剪状态掩盖超调，先做有参考解的步长收敛试验，再决定是否建立更细身体步长的真实神经分支。

## 路径

- 模型：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/zero_floor/fruitfly_zero_floor.xml`
- 模型差异：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/zero_floor/model_manifest.json`
- 结果：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/zero_floor/zero_floor_results.json`
- 数组：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/zero_floor/zero_floor_traces.npz`
- 已查看图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/zero_floor/zero_floor_comparison.png`
- 实现：`/run/media/sensen/Data2/cell_wound_prototype/run_zero_floor_branch.py`
- 测试：`/run/media/sensen/Data2/cell_wound_prototype/test_zero_floor_branch.py`
