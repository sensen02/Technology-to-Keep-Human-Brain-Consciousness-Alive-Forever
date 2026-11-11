# 续轮5最小动作备忘（不是已登记新试验）

先读本目录CURRENT_STATE.json、STATUS_POINTER_SEAL_CORRECTION.json和两个最新结果：035953感觉预训练PASS（仅静态合成感觉）及041122半幅身体FAIL。goalactivearmedrev4/round4，所有本轮任务已收集，无后台进程。用户原授权允许参数搜索和ANN训练，不要求逐次审批；不要把每项已完成有限试验的预算当全项目禁令，也不要改写原有限计划后追加隐藏候选。

## 现在实际成立的事实

1. 原walking限定实施完整验收已PASS，原参数/归档保持。fullright-1在c6b1500和eeced2000上固定工程PASS/slip.495011048。eeced中性实际PASS/.492788833。工程PASS不等于nativeD1-D8或任意状态泛化。
2. 原head2000fullleft+1 slip.514841585FAIL。续10000 .515626812FAIL，新增MLP .568578724FAIL，RH承重加权8000 .517207940FAIL。不能继续以更低teacherMSE推断运动成功。
3. RH统一输出连接增益0/.5/1.5/2全FAIL；单个输出偏置±.05，RHtrochanter_pitch+ .05是已试中最好.5094544387，但仍FAIL。身体/脚承重已验证能行走和teacherleft能过，不作不可行结论。
4. 不改网络权重的HALFLEFT+.5 slip.5142855863且yaw+.0092595511<.01双FAIL；HALFRIGHT-.5 slip.5276619023FAIL/yaw-.02696264PASS。原fullrightPASS不能泛化半幅，不能降低yaw到.005给其资格。完整原5门/all6/NF0/全9.6s保持。
5. 独立SENSOR_PRETRAIN_ONLY平衡预训练global3000 SHAbfd65a3896d259496114695d7f7f3550272cf4c369895a4ddb34f3ddc3352c06，44新静态头感觉测试全部4目标PASS：totalMSE.0056079793/food-onlyMSE.0076232500/signfraction.9990403071/nofoodRMS.0151462587。全部10560正常+10560交换输出/337920hidden值独立最终权重重放bitexact；11数组指标及17实际模型重放检查PASS。方向样本是相关时间点，不报告独立置信区间；4无食物配置仅2独立噪声流。
6. 感觉模型actor只有2受体及32记忆，source/bearing/clock不入actor；forward结构0；0Core/物理，不是食物到达、摄食、动态头/odoroffset泛化。payloaddeployment_qualifiedFALSE不可直接部署。

## 建议下一项动作：身体真物理目标的小维神经参数优化

不再任意追加上一轮单bias同网格。下一轮可新时间目录在优化前登记“3维耦合神经输出偏置的有限物理目标优化”：原eeced全部权重冻结，只调RH Coxa_pitch/Trochanter_pitch/Tibia_pitch三个readout.bias神经参数，实际关节index29/32/34需sealedmapping核验。目标为完整物理回合maxslip和原门槛的强约束，所有动作仍ANN自身输出+原有限PD；优化器在完整回合之间改变神经参数，绝非每步外置控制器。

一个可审查的有限算法是小种群cross-entropy/CMA或预声明seed的SPSA：固定最多16LEFT完整回合、max2CPU1，算法/seed/initialmean/bounds/step sizes与早停条件在PREREG写明并保存每个候选NN权重/sha BEFORE该回合。可以以已测试bias[32]=+.05附近作为明示起点，其他bias0，初始扰动scale.05、三参数额外bias界限±.25，具体数值要新预登记，不在本备忘先启动。回合奖励/排序只能用真实完整全5门/all6/NF0/yaw.01，不能删窗/改变分母/以MSE代替slip；其他门失败显著惩罚并保留，只有实际全PASS候选才允许选择，不能拿优化器未运行mean当通过。

可预声明首次全PASS早停或固定全部预算后选择，必须在新PREREG先定。选出的同NN权重还需真实N/R确认、完整negativezero分支不变核验及native正式原生记录。若16仍FAIL，应报告并另作可辨识数据/新正向记忆网络的问题，不能把有限搜索说成身体极限。原owner允许调参与表现搜索，所以不用为这项正常项目内工作索要批准。

## 后续动作不可跳过

- 取得full±1固定回合也不能自动接受±.5或连续域。当前half实测FAIL表明命令条件分布可能需要额外qualified半幅teacher数据或范围训练，或另立明确的神经离散命令接口（需新神经图/因果/切换资格），不能靠外置规则/CPG/动作脚本补动作。
- 020958/steering_holdouts目录67项离线准备只有CODEONLY且绑定失败非线性模型，不能原样运行在不同checkpoint；新的loader/资格绑定和实际全量证据必须注册。切换/初始motorpulse（禁止root写入）、原生D1—D8方法沿用原规范，工程/原生/生物结论分层。
- 减速教师.85/.90有合格数据，但神经减速未合格。可以另立门控神经支路冻结已过转向功能学习forwardnegative输入；不可用非神经减速控制器或未经验证的u值接食品。
- 之后才将感受器真实头几何和行为神经网络接实际身体，预登记留出source布局、输入交换/切断/时间重放、无源及同次视频，检查到达并保持所有物理门；感觉静态PASS不是物理趋近PASS。

## 封存与可变指针纪律

三旧ARTIFACT清单误封CURRENT_STATE后父更新了状态指针，导致仅3状态摘要漂移。STATUS_POINTER_SEAL_CORRECTION.json记录原清单与旧新hash，verify_status_correction.py实际验证其他181文件和10运行IDENTITY全部不变、原清单本身不变。不要再修改024558/033228/041122中的任何文件，包括已更正当前状态摘要；新状态写到新目录，明确exclude mutable CURRENT_STATE.json from seal。旧034309目录已封29文件不能改（其摘要可能保留运行中历史标记，不代表当前状态），新035953正确exclude CURRENT_STATE且28-filemanifest+SUPPLEMENT全部保持。

感觉报告/模型在 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T035953Z/`；桌面图 `/home/sensen/Desktop/engine修复验收/食物趋近实施-20261110/续轮4-平衡气味神经反应-纯感觉新测试.png` 已由子与父实际打开。所有绝对力/SI/生物问题仍NOT_ESTABLISHED，不扩展连接组能力。
