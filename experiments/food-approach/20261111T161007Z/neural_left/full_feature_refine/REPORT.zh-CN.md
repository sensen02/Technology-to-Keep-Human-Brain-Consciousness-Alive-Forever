# LEFT全特征微调实际结果：严格FAIL

独立目录从HEAD4000 SHA9c58e2a8772c7ef034f2fc0e18d01d5e4e59ace8d7cfb89b31e6f8569b6ae799精确继承全部权重；原baseGRU128冻结，原encoder+GRU64+head共43242参数/8张量解冻，freshAdam .0003固定2000额外更新，full2880 BPTT、prime960无loss但状态链保留。没有新actor输入、部署教师反馈或改物理/准入门。

最终checkpoint：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_left/full_feature_refine/TRAIN/checkpoint_002000.pt`

SHA256：`47e2ec012c016d4c912518ad1d7a326c080d0c65ecf8778a2016c937e58cf838`。

教师分布实际生产postprime动作MSE从.0017393248057010965降为.001003827564677095。实际预检76项PASS，两个保存审核各15项PASS，live-save model/8Adam/RNG/curve exact，准备12项PASS。原base权重无梯度且精确不变；neutral2880离线mean/action/basehidden exact不等于新的neutral物理回合。

完整LEFT实际1232000原生步（1s稳定+4.8s教师预热+9.6s ANN，无交接reset）：

| 指标 | 实际 | 原门及结论 |
|---|---:|---|
| signed LEFT yaw | +29.65128828979569° | ≥30°，FAIL，不四舍五入 |
| 最大滑移 | .5780968089711384 | ≤.5，FAIL |
| 六腿完整周期 | LF1；LH/LM/RF/RH/RM各2 | PASS |
| support/nonfoot/COM/forward | 四项均通过 | PASS |
| 1ms及每原生步非足力峰值 | 0 | 严格PASS |
| 额外力 | 0 | PASS |

每腿滑移：LF .29755110074455293；LH .5189071371553164；LM .350244925644701；RF .3857419211543555；RH .5780968089711384；RM .5599231067874955。

结果：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_left/full_feature_refine/EVAL_left/RESULT.json`。

独立审核：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_left/full_feature_refine/VERIFY_left.json`，140项完整性全通过、行为false。较原HEAD4000的28.648846869932036°/.7631680044518852改善，但未达准入；本实验不能隔离证明旧冻结特征是失败根因。

按登记跳过neutral，停止该训练family；固定47e2仅作为组合LEFT实际能力组件，不能称qualified PASS。原教师/base/失败历史不修改。bash-38训练、bash-39准备、bash-40实际LEFT、bash-41独立审核均exit0并已收集，所有heavy结束，资源已释放父。源码运行后未改，无额外训练、物理重跑或渲染；后续无新指令不继续重活。
