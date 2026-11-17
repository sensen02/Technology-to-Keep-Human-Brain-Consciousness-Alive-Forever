# 独立motor首版失败：阶段A保存数据／固定观察神经诊断

实际checkpoint SHA：454581a41034f92e253a2cb31215b5ddd1e09f8795ae54fb89053fd3187d1398。四条baseline352状态／mean／residual／action完整保存重放均精确一致；0物理积分、0优化器更新、0教师状态查询。

所有干预固定原观察X，仅解释神经输出/状态敏感度，不预测新身体yaw或成功。所有8条来源都是训练数据；同一时刻teacher动作不是学生实际状态的纠正标签。完整性通过不改变四项身体任务实际FAIL。

|条件|实际yaw°|漂移/体长|支撑均值|COM最低|native NF峰|行为通过|
|---|---:|---:|---:|---:|---:|---|
|walk|-1.888954|0.204652|1.000053|1.244320|0.000000|False|
|stop|-5.553935|0.147273|1.000009|1.253831|0.000000|False|
|left_pivot|21.516389|0.489415|0.775844|1.111564|73.720005|False|
|right_pivot|-6.843234|0.076100|1.000014|1.242285|0.000000|False|

## 预热／交接与训练来源
机械qpos/ctrl/targets与神经old256 priming分别核查；prefix不预设完全相同，逐字段最大差与首次差时间保留。教师没有归档old256 hidden，不伪造其重放资格。训练缓存来自冻结789f实际8条源，new96从零起，不依赖训练目标进入actor。

### walk

初始匹配与接管输出：
```json
{
  "X238_first_exact": true,
  "command_first_exact": false,
  "old_after256_first_exact": false,
  "new96_input_first_zero": true,
  "actual_first_action_vs_source_target": {
    "RMS": 0.037026114887989424,
    "absmax": 0.10616921640082302,
    "mean": 0.004190032625856494
  },
  "actual_first_residual_vs_traincache": {
    "RMS": 8.178999331447514e-07,
    "absmax": 2.402302596148098e-06,
    "mean": -4.6387620094253005e-08
  },
  "trained_zero_h96_readout_residual": {
    "RMS": 0.038454564050699615,
    "absmax": 0.10821447265929261,
    "mean": 0.004384869402238436
  },
  "warm_last_to_first_autonomous_action": {
    "RMS": 0.038489853284462416,
    "absmax": 0.10560451066372445,
    "mean": 0.0053283721888553225
  }
}
```

最早神经/域阈值（ANN秒，null表示本窗口未达）：
```json
{
  "actiondelta_above_p1": 0.0,
  "physical_domain_threshold": 0.225,
  "oldhidden_domain_threshold": 0.0,
  "task_domain_threshold": null,
  "new96_saturation_above_p01": 3.54
}
```

物理onset、四固定窗口原始统计、机械/神经prefix、分组训练域与干预量见STAT.json该case；全部是实际保存数组统计。

### stop

初始匹配与接管输出：
```json
{
  "X238_first_exact": true,
  "command_first_exact": true,
  "old_after256_first_exact": true,
  "new96_input_first_zero": true,
  "actual_first_action_vs_source_target": {
    "RMS": 0.03661060364254457,
    "absmax": 0.11584728674525391,
    "mean": 0.006236514145951166
  },
  "actual_first_residual_vs_traincache": {
    "RMS": 2.3631357381593988e-17,
    "absmax": 1.249000902703301e-16,
    "mean": 9.680376540049147e-20
  },
  "trained_zero_h96_readout_residual": {
    "RMS": 0.038454564050699615,
    "absmax": 0.10821447265929261,
    "mean": 0.004384869402238436
  },
  "warm_last_to_first_autonomous_action": {
    "RMS": 0.03869119918702335,
    "absmax": 0.1010230064667163,
    "mean": 0.005256841213411216
  }
}
```

最早神经/域阈值（ANN秒，null表示本窗口未达）：
```json
{
  "actiondelta_above_p1": 0.0,
  "physical_domain_threshold": 1.0,
  "oldhidden_domain_threshold": 0.0,
  "task_domain_threshold": null,
  "new96_saturation_above_p01": null
}
```

物理onset、四固定窗口原始统计、机械/神经prefix、分组训练域与干预量见STAT.json该case；全部是实际保存数组统计。

### left_pivot

初始匹配与接管输出：
```json
{
  "X238_first_exact": true,
  "command_first_exact": true,
  "old_after256_first_exact": true,
  "new96_input_first_zero": true,
  "actual_first_action_vs_source_target": {
    "RMS": 0.19973977849549457,
    "absmax": 0.731991027333477,
    "mean": -0.005622256049657162
  },
  "actual_first_residual_vs_traincache": {
    "RMS": 1.3002851659341615e-17,
    "absmax": 3.469446951953614e-17,
    "mean": -2.2097072848752184e-18
  },
  "trained_zero_h96_readout_residual": {
    "RMS": 0.038454564050699615,
    "absmax": 0.10821447265929261,
    "mean": 0.004384869402238436
  },
  "warm_last_to_first_autonomous_action": {
    "RMS": 0.20244040183093834,
    "absmax": 0.7442030694844844,
    "mean": -0.006601928982197107
  }
}
```

最早神经/域阈值（ANN秒，null表示本窗口未达）：
```json
{
  "actiondelta_above_p1": 0.0,
  "physical_domain_threshold": 0.005,
  "oldhidden_domain_threshold": 0.0,
  "task_domain_threshold": null,
  "new96_saturation_above_p01": 5.14
}
```

物理onset、四固定窗口原始统计、机械/神经prefix、分组训练域与干预量见STAT.json该case；全部是实际保存数组统计。

### right_pivot

初始匹配与接管输出：
```json
{
  "X238_first_exact": true,
  "command_first_exact": true,
  "old_after256_first_exact": true,
  "new96_input_first_zero": true,
  "actual_first_action_vs_source_target": {
    "RMS": 0.06891461982564379,
    "absmax": 0.17082236130620446,
    "mean": 7.65348751374641e-05
  },
  "actual_first_residual_vs_traincache": {
    "RMS": 1.6845272831571202e-17,
    "absmax": 5.551115123125783e-17,
    "mean": 2.4523620568124507e-18
  },
  "trained_zero_h96_readout_residual": {
    "RMS": 0.038454564050699615,
    "absmax": 0.10821447265929261,
    "mean": 0.004384869402238436
  },
  "warm_last_to_first_autonomous_action": {
    "RMS": 0.06347747732639039,
    "absmax": 0.15760106558386403,
    "mean": -0.0009031380574024881
  }
}
```

最早神经/域阈值（ANN秒，null表示本窗口未达）：
```json
{
  "actiondelta_above_p1": 0.0,
  "physical_domain_threshold": 0.4,
  "oldhidden_domain_threshold": 0.0,
  "task_domain_threshold": null,
  "new96_saturation_above_p01": null
}
```

物理onset、四固定窗口原始统计、机械/神经prefix、分组训练域与干预量见STAT.json该case；全部是实际保存数组统计。

## 解释边界与下一步
先比较四固定窗口0–.05/.05–.5/.5–2/2–9.6s的动作跳、基底/残差抵消、新96状态和机械失稳先后；域外比例或高z只是相关描述。任务v翻转保持turn与old256连续，不能据其输出推测身体会停转。残差置零和每tick新96输入归零均未改变任何真实身体。
原生NF峰摘要无法定位首次原生子步，报告onset只来自1ms记录。控制器动作/target/ctrl触边不等于已证实唯一根因，完整时窗也不等于安全门通过。
本阶段不增加训练或候选，不改任何科学来源。教师在学生状态上的查询需要完整姿态/速度/执行器历史与FK/支撑重建，是尚未执行的阶段B；必须单独授权并验证教师目标可行性后再讨论有限纠正学习。当前仍是身体组件，不是经真实neck的新模型部署。
本次wall=35.493s，峰RSS=801.016MiB；源/权重前后SHA、RNG和全部重放证据见VERIFY.json。
