# 同架构有限实验：动作监督与真实保存状态配对，600续到900

本候选只改变action标签的监督状态配对。parent41fa1e2232f5143fec0a3ebb5d506c819af0ba084b3a05041e829fd1477b906b已完成总600更新与保存／数值一致性审核，但四项身体行为全部FAIL，旧结果保持。actualX与conditional动作距离的改善不同，提出“在另一predT状态上回归真实X/T教师动作，可能形成训练捷径”的受控待检验假设，不认定唯一身体失败原因。

实际局部target敏感性六个stop／walk时间点两epsilon未见放大／positive mean diagonal，固定身体one-step结果不支持当前“target正反馈发散”叙述；它也不证明全身体稳定。本scope不转target-free、gating、CPG或加架构。old_walk_control明确有neutral[1,0]训练覆盖，机械七hash／full352handover与bodywalk匹配，不能声称无中性数据；原walk rawturn命令与bodyneutral差异仍标NOT_CONTROLLED。stop/L/R预热物理／输入／首输出匹配证据保留，不伪原teacherold256已记录。

## 同部署网络，双训练分支

部署412encoder96／GRU96／head192／bound6／state352／old789f冻结／8张量完全复制actualparent600，生产step／prime／cache不加任何门、输入、root／clock或控制器。新FORMAT INDEPENDENT_MOTOR_STATE_ALIGNED_ACTION_V1只标训练身份。初始20条实际观察mean/state迁移必须与父逐位一致。

TF分支读取实际保存features412，new96每序列从0完整1920展开，动作与actualX/T对应的Y配对。actionloss仍12seq、21773有效标签、首20tick×10／其他×1与valid相乘，各seq实际有效质量归一后42／12等权。ActualSourceScan使用每tick每序列B1的相同encoder／GRU／head运算，避免生产值门因batch kernel差异跨量化阈；它与conditional扫描共享同8参数。

conditional分支保留父exact三冻结组件／B1学习核／5×1ms目标积分／float32感觉编码。它不读action标签求loss，只有原9条真实执行postT半span归一均方权重1、6条walking的conditional residual prior*.1、原最初fcbd8参数anchor*.001。3assisted真实mixedtarget不能监督unmixedteacher累计T；其动作标签只在actualX分支原合格mask上使用。两支全部1920历史均推进，不删invalid行，不改old256外生真实历史。

每个micro4先TF_action.backward并释放TF图，再conditional_target_prior_anchor.backward并释放cond图。三个micro完成后一次Adam.step；期间参数不变，两分支独立new96都从0展开。完整B12参考也分支依次反传，不能同时持两张图或换成截短BPTT。global分母12／9／6与anchorB/12保持，微批梯度应同fullhistory参考。

## 600到900的严格恢复

原600全部权重、Adam8的moments／step600、RNG与config.initial_anchor（仍最初fcbd）一起恢复；不fresh optimizer状态、不reseed、不把600参数重作anchor。只更新601..900共300次，总targetcal900／global旧4350+300=4650。新保存600初始资格、750、900，final只接纳900；原parent600四FAIL不升格。初始化包装newidentity后才发布，避免保存与loader身份不同。

common只重绑_validate_payload/load_checkpoint/save_checkpoint三项科学函数，保留contextlib装饰器及11purehelper原对象／globals。旧validator硬final600只对确切final guard做AST修改为900，记录变换次数，不能全文件replace。strictload_parent真实旧v4final600途径与四alias／path finally恢复；每次检查source、canonicalAdam、原anchor和RNG，不绕过原失败或哈希。

## 跑前数值与资源门

完整preflight要检查：20actualX部署迁移exact；TF全12／micro4／B12 mean/action/new96对public生产≤1e−12、old256exact；conditional原12 B1／micro4／B12各六输出对完整生产替换内部target感觉≤1e−12。TFonly canary中改Y影响TFactionterm但不影响conditionalterm，改conditional输出mean/action不改变动作监督，改TF输出改变动作损失。13条真实actuator target／float32编码在新预检重复数值重放，不只继承口头资格。

完整1920分支顺序B12参考和micro4×3的loss／全部8梯度／Adam／参数差≤1e−10；optimizer两探针各载canonical600独立deepcopy，恢复后原parentpayload／权重／RNG／oldfrozen完全不变。FD32两个坐标epsilon1e−4、绝对1e−3或相对.05，保留量化近似解释，不放宽阈值。target gradient canary、Python/JIT两支全部输出门和所有源哈希照常。

新训练wall7200秒，只因两支前后向；训练和保存审核RSS1800MiB，fullB12参考预检RSS3072MiB。实测双分支完整micro4×3更新×300×1.3，加三次fresh20actualX＋12conditional、setup／finalEVAL余量≤7200才READY，不自动延期或加epochs。资源未过是beforetrain阻断，不假完成。

## 保存与最终指标

每600／750／900 live model／Adam／RNG精确roundtrip后fresh进程20actualX cache/public重放、12conditional fullproduction替换target感觉重放，conditional数组有限且模型权重不变，0Core／物理。评估把actualX_actionfit作为主要监督指标；conditional动作对sourceY只记COUNTERFACTUAL_DISTANCE_NOT_SUPERVISION，没有条件动作label MSE。9target normalizedRMSE／绝对peak／末值另列，assisted3无target监督分数；DEV8没有标签，不补teacherY。独立NumPy只审核已保存指标，不NN／物理。

## 父独占执行接口

先绑定metadata（标准库）：

```bash
/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/state_aligned_action_calibration/prepare.py --prepare-state-aligned600-to900
```

再实际完整预检：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/state_aligned_action_calibration/preflight.py --execute-parent-state-preflight
```

仅actual全部通过且READY_STATE_ALIGNED600_TO900后：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/state_aligned_action_calibration/run.py --execute-parent-state-resume
```

本子及所有编码子代理ONLYCODE，无Torch／NumPy导入、NPZ值／PT反序列化／模型/Core／训练／物理。900最终模型与行为未知；离线拟合、保存一致性都不能代替四项无教师身体与realDN2decoder/fullintegration、粗地形后续主线。
