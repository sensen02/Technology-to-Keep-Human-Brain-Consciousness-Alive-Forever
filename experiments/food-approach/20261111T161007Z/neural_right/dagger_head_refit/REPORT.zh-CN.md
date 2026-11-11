# 固定双轨迹CORRECTIVE head候选：oracle重放正确，真实RIGHT滑移显著恶化

本阶段所有已授权任务已结束，全部job已collect，heavy资源已交还父。零积分oracle前缀/标注审核通过，固定双轨迹head2730训练与保存审核通过；最终实际RIGHT转30.1341222667°，但最大滑移.7374945181，显著高于固定父ef768的.5098287866，整体FAIL。按事前规则未运行中性回合，不追加此纠正family预算。Oracle标签重建正确不等于实际控制改善。

最终失败模型：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/dagger_head_refit/TRAIN/checkpoint_004000.pt`。

SHA256：`e92b591df32d8f40dccc8e1c23f54ce22b896c199b1cd27bfab29b9b2633cfcb`。

固定父模型为`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/left_hind_stance_refit/TRAIN/checkpoint_004000.pt`，SHA256 `ef768c788c94aeecbb49f0bd1ccb5c424333af36ec514941b72e49ab0d4917ff`；没有后验挑三bias最佳模型作为父。

## 零积分oracle：实测与边界

仅一个private diagnostic Core，mj_step/mj_step1/mj_step2全部禁止，实际调用0。对保存状态进行8681次私有snapshot恢复，绝不称该诊断为physicalrootwrites0或新物理轨迹。完整49qpos/48qvel/ctrl/actuator.target/time/零外力按actual trace index999+5*k恢复，teacher后label不积分到target，不使用reset_from。uniform负载偏好oracle只用接触几何/Jacobian和qfrc_bias/passive，不使用实际接触力解作为偏好；未保存的solverwarmstart每snapshot置零且明确记录。

先恢复并mj_forward核验全部5800个原1ms前缀端点，然后从实际1s保存settle qpos/rot/origin/rest初始化ConditionalTeacher(speed1,yaw-.06,prime4.8)，private q_des沿全2880步连续IK递推，private shadow root保持初始根，实际snapshot root逐步恢复原NN轨迹。前960warm动作与实际保存teacher动作最大差0，foot/padz/COM/head位置/Head旋转/root-yaw最大差全部0，满足提前声明1e-10容差。原始private q_des链没有直接保存，因此不冒称与未保存原私有数组逐位对比。

证据：

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/dagger_head_refit/ORACLE/PREFIX_AUDIT.json`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/dagger_head_refit/ORACLE/RESULT.json`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/dagger_head_refit/ORACLE/VERIFY.json`：独立零Core/零积分保存审核9214项全通过，检查来源/完整恢复数组/全部warm动作/5800前缀几何/连续private IK差分/每个postlabel与保存rate/全source和artifact SHA。

后1920个labels只标为CORRECTIVE_EXAMPLES，未实际执行教师回合，不称qualifiedteacherPASS。unsupported/unbacked/hypothetical各252/1920=.13125，rateclip97/1920=.0505208333，全部保存未过滤。实际输入X238/command/previousaction保持ef768真实轨迹，teacher target每次恢复真实NN actuator.target，不把拟议label推进为合成目标。

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/dagger_head_refit/ORACLE/CORRECTIVE_EXAMPLES.npz`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/dagger_head_refit/ORACLE/POST_DIAGNOSTICS.jsonl`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/dagger_head_refit/ORACLE/ORACLE_REPLAY.npz`

所有实际source/import/raw measured-reference/compiled body/input身份before-after均绑定。初次bash28因oracle.py括号SyntaxError在执行前停止，0Core/0label，已记录原错误日志；修复且AST通过后bash29执行上述唯一成功oracle，没有重跑任何physical。

## 新NN与固定训练

从ef768的LH参数分片精确重组完整42x64+42 dense head，仅2730参数可训练；base/encoder/GRU严格冻结。两个source各2880初始生产means及128+64hidden均bitexact父模型。部署仍原RIGHT_ONLY新yaw/.06域、bound3、relu(-turn)，不使用oracle/phase/time/foodcoord或训练label。与旧yaw/.008域区分。

两条完整轨迹：原qualified attempt02/right_yaw_m060教师轨迹与本次ef768实际NN状态上的CORRECTIVE轨迹。各保留960连续prime但不计loss、各1920 post行均衡.5/.5、全42动作均匀MSE，缓存每条独立全history，未把oracle输出替换到实际X中的上一动作。冻结history缓存与实际生产head梯度等价，未detach可训练历史。

实际production/cache means最大差1.333e-15，全部base/residualhistory严格一致；head梯度差≤6.723e-18，unit Adam head参数差≤1.943e-16，满足1e-12。冻结权重/无梯度/neutral2880严格原基线。证据`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/dagger_head_refit/PREFLIGHT.json`。

freshAdam.0003、seed20261111/float64/CPU1，固定4000更新，wall26.86秒；没有旧Adam继承，没有按最低loss换档。实际均衡loss .001003028026→.000673959309；原qualified MSE .000193249200→.000460278431（变差），CORRECTIVE MSE .001812806853→.000887640188（变好）。这种训练目标tradeoff如实保留，不能用平均loss下降冒充闭环通过。

save2000仅完整性存档，SHA256 `4f0430b05b3bf0aaf6d7149a510c87d88f408b4ea74dc6a2fc56f69ebcb34083`，未物理。两保存档live模型/Adam2/RNG/curve/frozen状态严格一致，临时strictload后恢复live RNG；两个新进程独立保存审核各13项全通过，全两轨迹实际生产means/basehistory/residualhistory逐位相同、cache SHA/两轨迹均衡身份/neutral严格原基线。

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/dagger_head_refit/TRAIN/VERIFY_002000.json`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/dagger_head_refit/TRAIN/VERIFY_004000.json`

## 最终真实物理RIGHT

固定final4000，一个完整1s稳定+4.8s原straightteacher prime+9.6s纯ANN回合，接管不reset，RIGHT命令1s smoothstep至-1。保存全部15400个1ms端点、2880个5ms实际神经决策/192hidden/动作/geometry/handover。1,232,000个原生12.5us步，ANN768,000步；teacher仅warm960调用，ANN阶段0次。原完整71opaque mesh/49qpos48qvel/42有限PD、gravity/friction/全部compiled数组与执行器参数不变，noadhesion/extra0/物理运行rootwrites0；此项真实部署与前述oracle snapshot恢复明确分开。

| 同窗指标 | 固定父ef768 | 本Dagger e92b | 判定 |
|---|---:|---:|---|
| 实际RIGHT yaw度 | -30.6311514969 | -30.1341222667 | ≥30°通过 |
| LF slip | .4098881260 | .4962626388 | 通过 |
| LH slip | .5098287866 | .7374945181 | FAIL且恶化 |
| LM slip | .4740876590 | .5964252140 | FAIL |
| RF slip | .3274102588 | .3697123651 | 通过 |
| RH slip | .4664102720 | .5578432212 | FAIL |
| RM slip | .4486675386 | .6308187567 | FAIL |
| 足载均值/W | 1.0000645498 | 1.0000649676 | 通过 |
| COMmin，1ms原生长度 | 1.2551137298 | 1.2507277000 | ≥1.20通过 |
| 原world-X signed速度 | .08179755890 | .08006052614 | ≥.05通过 |
| 原生ANN NF峰值 | 0 | 0 | 严格0通过 |
| 六腿LF/LH/LM/RF/RH/RM周期 | 1/2/2/2/2/1 | 1/2/2/2/1/1 | all6通过 |
| 整体 | FAIL | FAIL | 不放宽slip≤.5 |

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/dagger_head_refit/EVAL_right/RESULT.json`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/dagger_head_refit/VERIFY_right.json`：新进程独立实际审核163项完整性全通过，行为slipFAIL保持。全actual2880生产输出/192state逐位重放，全部时钟/动作持有/上一实际动作感觉/来源/连续handover、1ms门/周期/姿态yaw/几何、原生计数与NF/extra、全部source/input/artifact SHA、compiled完整身体和有限PD以及无physicalrootwrites路径均验证。

RIGHT未通过，因此依事前规则没有新中性physical，未把此前其他checkpoint的neutral或yaw拼成此模型通过。当前neutral只具备本模型离线2880输出严格原基线的实测，未声称本模型做了完整neutral实际回合。

## 收尾和下一动作

bash28执行前语法失败/29 oracle零积分/31训练/32两保存审核+prepare/33完整RIGHT及独立audit均已收集结束；无运行任务。所有source、oracle不安全标记、训练tradeoff与失败模型/完整轨迹保留；没有重跑已完成physical或扩此family。没有待清理无用临时文件。父已明确接管heavy资源，优先LEFT新NN；本子不再扩RIGHT搜索。

后续展示可使用既有ef768实际30.631°/.509829、all6/NF0的明确能力对照，但strictslipFAIL保留，不称formalqualification。双向组合若再授权，仅先CODE_ONLY，不启动新模型/物理。当前没有执行额外候选或新组合。

限制：单原qualified轨迹+单actualNN纠正轨迹、固定冻结特征、固定head候选/标称初态9.6s，只属已授权替代ANN/原声明派生完整身体/有限PD工程结果。Oracle重建通过不能证明postlabels物理可执行，13.125% unsupported保留；观察到qualifiedloss增加和实测slip恶化，不足以隔离唯一根因。不是完整原生D1—D8/原生slip重新认证、原连接组/SI/生物有效性/任意扰动或食物趋近成功。未为本模型制作视频，旧f25b同次视频不能冒充本e92b模型。
