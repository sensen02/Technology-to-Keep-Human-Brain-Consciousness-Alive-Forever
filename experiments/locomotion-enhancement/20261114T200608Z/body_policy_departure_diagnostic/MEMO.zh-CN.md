# 保存身体轨迹偏离诊断与有限on-policy标签收集建议（ONLYCODE，未执行）

当前900模型具有较好的真实源状态拟合及严格数值／保存资格，但四身体实际任务仍失败：walk功能性推进／支撑／周期可以过，原slip失败；stop与左右出现非足／COM下降或方向不符。不能据此继续盲目加相同loss epochs，也不能把已有source-state fit改进当身体稳定。此前固定身体one-step target敏感性没有放大证据，不以此宣唯一反馈不稳定原因。

## 本次诊断的实际可见字段与资格

只读900身体目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/state_aligned_motor_body/` 的四case RESULT／VERIFY、完整9stage JOBS／SUMMARY和存档数组。每case trace含1ms time／qpos49／qvel48／target42／loads6／nonfoot／COM／padz6／speed6／control_source；decision含X238／command2／mean／action／hidden_before352／hidden_after352／physical_step_ms。程序只取所需字段，不加载模型、Torch、Core或接触求解器。control_source候选NN值6来自冻结原组件common.py:18，不猜成2。

训练清单来源 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/state_aligned_action_calibration/wall_replan_9000/INPUT_BINDINGS.json`：index0 stop、1左、2右使用原teacher_samples_5ms X238/Y42，2880记录中warm0:960、candidate960:2880；walk**只用index8 old_walk_control**，neutralcommand[1,0]实际记录3080含settle200、warm200:1160、自主1160:3080。原walk rawturn不是这个中性对照，不替换来源。

所有source RESULT／VERIFY及实际trace/samples artifact SHA绑定，body四fresh独立42NN replay审核要求原VERIFY checks全部TRUE且resultSHA对应；9stage exit0才准描述。脚本SELFcheck仅JSON／NPY头／完整文件SHA验证，不读取数组payload，当前子可执行。父才可用NumPy执行实际诊断。

preaction物理5.8s的q/target来自trace[5799]；first5ms后trace[5804]。本诊断网格取**相应ANN动作之前**的身体状态；时间点0就是handover，.005是下个decisionpreaction而非重复首动作。按实际time／physical_step_ms精确对齐，无插值或外推。stop/L/R的lastpartial神经动作保留，物理轨迹比完整5ms短时只按真实available时间报告。8秒等不可用窗口明确complete=false，不能填0。

## 比较项目及不能推断的事情

先分别核四case全warm X／qpos／target与初preactionX/q/target逐位一致。若handover匹配，初始body action对sourceY的差可称同状态初动作差；source teacher没有old256完整记录，不能把未知teacherhidden伪造为匹配证明。neutralcontrol的源hidden虽记录，控制来源不同，程序也不把其new96等同900新96。

固定网格0、.005、.02、.05、.1、.2、.5、1、2、4、6、8、9秒，报告42动作RMS距离／有符号均值、关节q/速度/target距离、X分组距离、6足载／support／nonfoot／COM／最低padz以及352/new96幅度。padz_min是爪垫高度，不是身体最低几何高度；正载荷腿数是载荷代理，不冒接触计数。独立geometry/contactcount未读，输出contact_count_available=false；保存的actuator target是执行寄存器，不冒teacher内部理想q_des，后者当前未纳入数组契约。

.1／.5／1／2／4／8秒完整累计窗报告动作RMS、每通道signedbias及q/target/速度距离。.1秒正是first20×5ms窗口，便于检查虽然有10倍权重，早期残余误差及机械轨迹差何时出现；不能从该相关性证明10倍权重不足或该权重造成崩溃。

先注册描述阈值actionRMS .01/.05、q/target RMS .01/.05rad、qvel .1/.5rad/s、相对自己handover的COM下降.05/.1native、nonfoot/weight .01/.1，以连续4个5ms动作保持bin或20个1ms样本确认。忽略1e−17首次浮点差作为因果触发；这些不是新任务验收门、不修改原slip或COM判据。事件时间顺序最多提供“轨迹动作距离先变大／机械差后变大／非足或COM下降更晚”的证据，**不建立唯一根因或全身体因果反事实**。

teacher Y在teacher自己的X/T上计算。后续student状态不同，跨轨迹action距离不是“teacher在student当前状态会给的正确动作”，也不能用旧Y给新bodyX做监督。比较q/target／足载只是两个真实轨迹之间的距离，正常转向本来也会走不同几何路径。脚本不算未经定义的最近邻coverage分数；要证明“真实状态覆盖不足”仍需新实验。

## 下一最小受控路线：有限DAgger标签收集设计，不执行

若父实际诊断显示早期机械状态已偏离原source邻域、而原source状态fit较好，可以只提出一次有限on-policy标签采集，不能宣布覆盖不足已证。

建议先限定三个失败任务stop／左／右各一次短自主rollout，使用当前900固定checkpoint和原完整身体/接触/关节映射/5ms NN及1ms执行器，warm与handover不改。时窗优先选诊断中安全持续偏离阈值触发前后和早期.1/.5/1秒几个可比点，仍提前固定每task上限和总labels数，不事后挑好看的行。窗口必须终止于原非足／COM安全闸，失败／不可解也存。walkneutral保为独立监督回归控制，不能把不同rawturn来源混进matchedcounterpart。

在**实际student当前物理状态、当前target、当前previousaction／sensor**上调用已声明的完整身体教师，沿用其真实当前Jacobian／support优化，只出label和资格诊断。query不替换自主动作：student仍应用自己的NN动作推进真实身体，teacherlabel只进新数据档案。每条记录绑定实际X、完整状态／target、teacher原source SHA、solver约束／ratecap／支撑／可行性mask和时间；不能用teacherclock／理想root偷偷进actor。teacher不可解则明确invalid，不补旧Y或零向量。原teacher在自己的轨迹停步／±45度合格只是来源资格，不保证在失败student状态可解。

保存数组本身不足以重算当前Jacobian／contact/support，不可只从qpos几何数组伪求教师标签。若已有可信零积分restore工具能完整恢复原Core状态、target history／derived fields并经actualrestore核验，可另授权0integration状态查询；否则必须新真实on-policy rollout同步采样。仅qpos/qvel相同不足以确认完整状态恢复，尤其约束／contact／执行器寄存器／时间与派生感觉一致性。当前脚本没有做restore、teacherquery或新physics。

收集完另预登记一个有限同架构监督状态配对实验，先审核mask和原source／neutral回归，再决定是否训练；不是自动增加epochs。即使示范标签资格通过，最终四项body验证必须完整撤辅、纯NN、无teacher calls afterhandover，在原门下验；不得用混合teacher动作过程冒自主通过。更远realDN2／decoder完整接口和粗地形主线保持，不用新架构、CPG、hook/rootforce绕过。

## 执行接口与预算

本包总3文件：PREREG.json、diagnose.py、本MEMO。父可先只读／标准库检查：

```bash
python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/body_policy_departure_diagnostic/diagnose.py --selfcheck
```

仅父独占保存数组诊断：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/body_policy_departure_diagnostic/diagnose.py --execute-parent-saved-departure
```

90秒／RSS1500MiB，0NN／Core／物理／更新／教师query，产出STAT／REPORT／VERIFY，old文件只读。新900四FAIL和原144timeout124／postrun保存资格完全保留。本ONLYCODE未读取任何数组payload或执行数值；实际诊断结论等父执行。
