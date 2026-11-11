# 大角度LEFT HEAD_ONLY检查表（S4/S5）

- [x] 先登记PREREG；独立LEFT格式、coupled命令域、bound6、freshAdam .0003固定4000。
- [x] 只读确认合格教师与独立AUDIT路径；旧教师/训练/右侧证据均不修改。
- [x] 标准库AST与准确计数源码适配检查完成；check_codeonly.py两次exit0，最终7项静态检查全真，CODEONLY.json记录最终源/教师JSON身份（不等于实际模型预检）。
- [x] 父会明确释放唯一heavy资源并授权实际实施；进程均CPU affinity1及OPENBLAS/OMP/MKL1。
- [x] 实际模型预检43项全PASS：缓存/生产/梯度/unitAdam、zerohead/baseexact、neutral2880exact、冻结2730之外所有参数；最大比较误差2.5673907444456745e-16。
- [x] 固定4000更新；2000/4000 live-save全部exact，新进程保存审核各15项全PASS；仅4000物理评估。bash-34 exit0已收集。
- [x] prepare_eval12项全PASS，sourcepaths提前存在及hash，最终4000身份绑定。bash-35 exit0已收集。
- [x] LEFT实际完整1232000原生步，保存Trace1ms/neural192/geometry/HANDOVER/每原生步NF峰值，无部署教师反馈。bash-36 exit0已收集。
- [x] 独立actual审核132项全PASS；完整性true、行为false分开。bash-37 exit0已收集。
- [ ] 行为未通过：yaw+28.648846869932036°<30°；maxslip .7631680044518852>.5。六腿LF1/其余各2，1ms/nativeNF严格0、extra0，其余support/nonfoot/com/forward四门PASS；保留严格FAIL。
- [x] 因LEFTFAIL跳过neutral；未扩大结构/预算或重跑物理。离线neutral2880exact仅为预检，不冒充实际neutral回合。

当前状态：TRAINED_AND_FULL_LEFT_EVALUATED_STRICT_FAIL，独立完整性132项通过。checkpoint_004000 SHA=9c58e2a8772c7ef034f2fc0e18d01d5e4e59ace8d7cfb89b31e6f8569b6ae799，实际教师分布postprime动作MSE=.0017393248057010965。全部heavy已结束收集并释放；源码自运行起未改，无渲染。Tracker由父会负责；本目录保留失败全部证据。
