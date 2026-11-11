# S4/S5 新RIGHT_ONLY大角度神经训练与物理评估检查表

- [x] 读规范、当前状态、路线/台账开头；父负责tracker；旧证据不改。
- [x] 检查现有进程：没有本项目训练/物理进程；其他项目进程不动。
- [x] 先落盘独立PREREG，明确新yaw/.06与旧yaw/.008区别，组合改动不作单因果。
- [x] 模型与strict loader、新右门/界、迁移encoder+GRU/zerohead/freshAdam；迁移固定SHA c6b395f29676ef933e2f55ca5a92b7c5b55183a2d432226487b39e916b51acdb且检查旧payload source/data。
- [x] 实际production/cached/scan输出、状态、全部梯度与unit Adam更新<=1e-12；最大前向状态误差1.555e-15、梯度1.076e-16、unit参数1.381e-15；/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/PREFLIGHT.json。
- [x] 中性全2880/baseexact/无base梯度，右zerohead初始baseexact。
- [x] 固定2000更新；保存1000/2000模型、Adam、RNG、curve、实际重放；独立新进程审核各11项全过。
- [x] 固定最终2000 neutral+RIGHT两个完整顺序真实回合；保存全轨迹与轻量原生NF。neutral通过且全数组严格原基线；RIGHT实际30.138533°，LH .5224375/LM .5031382滑移FAIL，其余门通过，未追加训练。
- [x] 新进程审计身份、连续hidden/动作、handover、物理完整性、同窗五门/NF0/六腿/30度；neutral156项/RIGHT132项完整性全过，行为FAIL不改写。
- [x] 收集所有job、完整报告结果/限制/下一动作并回报父；/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/REPORT.zh-CN.md。

视频按本次明确授权暂不做；不能据此声称完成视觉验收。

纯BC动作MSE，没有PPO/GAE/critic/logstd或概率比，模板相关项目不适用。1000保存的live模型/Adam/RNG/curve严格相同，中性2880继续保持；新进程审核等待固定2000训练退出后顺序执行。保存校验的load_checked构造临时模型会消耗live torch RNG，未在训练中改源码；训练本身无随机操作，另做实际1000→2000 RNG转移重构审计，不声称任意随机过程精确续训。
