# 十四条件实际嗅觉／视觉身体试验：ONLYCODE已就绪

父最终设计批准后实现，子没有导入NumPy／Torch、读取数组、加载模型、运行Core／物理／训练／render或网络。初次43标准库检查通过后，静态审阅发现继承视觉summary的exec需要独立非main命名空间，已在任何实际执行前修正；原CODEONLY_CHECK.json保留，最终CODEONLY_CHECK_FINAL.json实际44项全部true、exit0。prepare严格只用最终报告的sourcehash，不把旧已漂移检查冒作当前通过。

## 准入与实际命令

未知新权重不准入。CAL实际指向新恢复目录：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T045302Z/sensory_fusion_recovery_v1/`

严格新FORMAT为SOURCE_NECK_VISUAL_OLFACTORY_FUSION_RECOVERY_V1，APIbrain_step(AN,modalities2,h)／load_sensory()→vision,odor,两个payload／load_checkpoint(path,digest,final=True)→bridge,payload已按实际源读取。

只有父实际固定400全部7stage、0／200／400保存重放及最终审核通过后，才可用真实finalSHA：

```bash
python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T045302Z/sensory_body/prepare.py --bind-completed-calibration --brain-sha256 ACTUAL_RECOVERY400_SHA
```

随后父单CPU零Core预检：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T045302Z/sensory_body/preflight.py --parent-resource-released
```

通过后父唯一托管后台job：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T045302Z/sensory_body/orchestrate.py --parent-resource-released > /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T045302Z/sensory_body/PIPELINE.log 2>&1
```

30阶段＝14实际完整物理＋14新进程独立审核＋summary＋配对review。JOBS记录PID／命令／log／exit；失败即停，保partialrecord，不自动重跑已完成物理。功能FAIL不改阈值，也不影响保存真实结果，完整性FAIL则不发布通过。

## 输出身份（可视化不得猜）

每case RESULT.kind：SOURCE_NECK_MULTISENSORY_BODY_V1。

总summary文件：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T045302Z/sensory_body/SENSORY_BODY_SUMMARY.json`

summary.kind：SOURCE_NECK_MULTISENSORY_BODY_SUMMARY_V1。

每case3080神经rows／15400物理1msrows／770双眼RGB／1232000native，递归state[motor128,64,64,visual32,odor32,brain16]＝336，cache16／filter2／pending2／AN1DN2队列另记。真实RGB、Head位姿、完整precision触角proxy、世界场浓度、滤波前后／previouspending／实际actorreceptor、双感觉神经raw、全部336前后state、AN/DN采集释放消费时钟、policy与实际动作全部保存。独立audit不构造Core，而从保存actualRGB＋真实Head几何重采样field／pending／receptor＋X111，重走全部4模型和来源队列，输出／状态／动作逐位核验，原指标／周期重新计算。

## 范围与新门

十四条件：6视觉alone＋O左／右／无味＋F左右同侧一致＋F左下行切断＋O左感受切断／交换。视觉alone依然向嗅觉NN输入真实零receptor并自然输出偏置，不条件清odorraw；嗅觉alone视觉仍处理真实空场RGB，不清visionraw。条件／源位置只进环境或明确感受干预，不进脑网络。

F是“同侧一致”，不是同源：visual2native/.4rad，odor.6native/.3rad，各按实际头／各自midpoint独立锚定世界。Gaussian sigma.5／amp1.25／ANN.6起.2rise持续，actor receptor为旧ClampFilter.advance返回的previouspending，tau50ms／dt5ms。Ocut仅actor receptor0，field/filter/pending继续，olfGRU不reset；swap只actor左右序交换，ENV不交换。DNcut只源消息静默／接收基线零，不能强turn0。

嗅觉和F方向门是父运行前新工程定义：early ANN.82..1.82符号fraction≥.9、2.6秒相对O_noodor有符号bodyeffect≥.25°；O_noodor RMS≤.02与yaw相对V_empty≤2°；odorcut后.5..1秒输出RMS≤.05且配对2.6角差幅度≥.25°；swap反向同门。视觉六条沿旧阈值，因融合权重改变须重新测。所有原五项／滑移／周期／NF分列，不把稳定当全部运动通过。

.25°是父以旧视觉反转工程分辨尺度为参考新声明的最小效应门，不是旧空间C或生物嗅觉验收，更不覆盖旧Dcutdifference失败。没有arrival／停驻／摄食／脑走停／生物膜电生理／多模态优越或冲突决策结论。
