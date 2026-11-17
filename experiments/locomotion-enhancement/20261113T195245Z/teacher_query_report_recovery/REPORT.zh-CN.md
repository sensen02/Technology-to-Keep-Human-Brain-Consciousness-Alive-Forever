# 报告导出恢复与保存数组独立审核

原主任务数值查询已保存，但原程序退出码为1：report_text缺少json全局导致报告导出失败。本恢复没有重跑查询；保留原FAILURE、STAT、全部查询数组及日志。原0字节报告不作为需保留输入，父可按清理规则删除。

以下正文只从原report_text AST抽取，并显式绑定json；不是原主任务exit0的证明。

# 阶段B：真实学生保存状态上的教师离线查询

**LABEL_QUERY_ONLY_NOT_PHYSICAL_SUCCESS**。这是保存状态重建和教师局部标签可行性分析，不是物理复跑、恢复证明、神经控制成功或新训练。

每次查询前状态来自trace[physical_ms−1]，40ms目标缓冲来自target[ms−40:ms]；5.8s用第5799行，不借下一物理步。离线复制qpos/qvel/ctrl/target/buffer不属于实际运动控制；mj_forward只重新求解当前约束，不是原生接触力逐步重放。

Core=9；原生积分=0；NNforward=0；优化更新=0。

## 教师自身轨迹资格

qualification_stop：reproduction_pass=True
```json
{
  "action_max_abs_error": 0.0,
  "private_qdes_max_abs_error": 0.0,
  "desired_target_max_abs_error": 0.0,
  "IK_error_max_abs_error": 0.0,
  "contact_count_max_abs_difference": 0.0
}
```
qualification_left_pivot：reproduction_pass=True
```json
{
  "action_max_abs_error": 0.0,
  "private_qdes_max_abs_error": 0.0,
  "desired_target_max_abs_error": 0.0,
  "IK_error_max_abs_error": 0.0,
  "contact_count_max_abs_difference": 0.0
}
```
qualification_right_pivot：reproduction_pass=True
```json
{
  "action_max_abs_error": 0.0,
  "private_qdes_max_abs_error": 0.0,
  "desired_target_max_abs_error": 0.0,
  "IK_error_max_abs_error": 0.0,
  "contact_count_max_abs_difference": 0.0
}
```

## 学生状态查询
|cohort／条件|实际查询行|合格标签行|合格比例|全部有限动作差RMSE|合格动作差RMSE|
|---|---:|---:|---:|---:|---:|
|v1_stop|1920|200|0.10416666666666667|0.6335950200025451|0.20964680662674381|
|v1_left_pivot|1920|219|0.1140625|0.6948994562705768|0.275048419112151|
|v1_right_pivot|1920|91|0.04739583333333333|0.6386609500665278|0.27274064454814817|
|v2_stop|1920|40|0.020833333333333332|0.7148794547485963|0.1686205466039845|
|v2_left_pivot|494|52|0.10526315789473684|0.6963103317940903|0.16051713212374627|
|v2_right_pivot|776|285|0.36726804123711343|0.4913452967844552|0.2883000065401146|

## 实际body寄存器与actor访问差异
已读BASE/sources/proprioception.py:38-43：42:84为(target−q)/halfspan目标误差，126:168为(target−q0)/halfspan目标位置偏移，均感知裁剪[-10,10]后float32；不是effort。physics.obs238将这些当前body寄存器原样编入观测。
新残差／fast encoder和old256 recurrent编码器的physicalmask排除42:84、126:168及195:237(previousaction)。但整个旧motor并非完全不读target：BASE/models.py:140-142在base输出均值显式减去gain()*obs126:168，故最终oldmean/newmean已有冻结的身体本地target-position prior；只是可训练新分支和recurrent memory没有该当前寄存器的直接通道。不能据此断不可观测／不可能，原walking曾通过。直接encoder的autograd梯度为0仅说明该被测分支的连接／冻结detach，不能证明整个mean对相应body寄存器数值不敏感，尤其不能忽略已存在的冻结target-position prior。
每行保存实际target、query desiredtarget及源X238的两个归一化target字段；每case统计它们与教师纠正动作的描述关系。教师本身的rate律明确依赖当前target，所以相关性不构成唯一根因。后续若考虑body-local efference／target寄存器显式观测，那是另立未授权候选，真实body寄存器不等于教师privateq_des，不得将理想教师目标塞actor。

标签mask在跑前固定：实际COM≥1.2／源1ms NF0、queryIK≤.01、无unsupported、unbacked计划载荷≤.01W、qdot/rate/target/tau不裁剪、有限42、恢复exact。全部invalid及算法错误行保留，不用mask掩盖标签失败；1ms源NF不冒称该时刻native峰为0。
每个case的私有q_des、clock、世界anchors从同一handover初始化后连续推进，不逐tick冷reset。重解的足载/NF/速度/actuator_force与归档差异在每case JSON中明确保留。老师自身轨迹复现不合格时不会继续给学生标签。
walk两cohort只比较保存oldmean的tanh与原NN动作，没有生成walkingteacher纠正标签。
所有标签仍是算法教师的条件性局部建议；合格mask不证明真实身体施加后能恢复。没有Dagger训练、新权重、真实root轨迹、脑neck部署。今后actor只准真实physical112/old256/task2/new96，不把teacherclock/anchor/rootstate当隐藏输入。
wall=69.209s，峰RSS=786.035MiB；源/RNG/编译参数与禁止step/NN调用证据见VERIFY。


## 保存数组独立审核附录

以下只重算已保存数组，不重新执行Core、教师、NN或积分。原积分／NN0依据原traps及STAT记录；本审核不冒称独立重演了该现场。

|case|有效标签|有效比例|最长起始有效前缀行|最长连续有效段行|最常拒绝项|
|---|---:|---:|---:|---:|---|
|qualification_stop|1920|1.0|1920|1920|None|
|qualification_left_pivot|1231|0.6411458333333333|157|252|target_bounds|
|qualification_right_pivot|1240|0.6458333333333334|39|251|target_bounds|
|v1_stop|200|0.10416666666666667|200|200|rate_clip|
|v1_left_pivot|219|0.1140625|68|72|rate_clip|
|v1_right_pivot|91|0.04739583333333333|40|45|rate_clip|
|v2_stop|40|0.020833333333333332|40|40|unbacked_plannedload|
|v2_left_pivot|52|0.10526315789473684|52|52|rate_clip|
|v2_right_pivot|285|0.36726804123711343|40|114|rate_clip|

全部有效区间／每mask失败计数及目标寄存器范围RMS、与纠正动作的描述相关见AUDIT.json。失败项可重叠，不将各mask失败数相加冒充总无效数；invalid NaN和错误行保留原数组。
合格标签只对应预登记局部条件，不能掩盖原身体任务失败，也不证明施加后可恢复。若有效覆盖仅短前缀，下一步数据策略需父先评估；本恢复不授权或实施Dagger／teacher mixing训练、第三模型或新物理。
actualtarget／target-error／target-position offset与教师理想q_des访问层级不同，相关不构成因果证明；冻结oldmean含target-position prior，不能误称wholemean独立。
