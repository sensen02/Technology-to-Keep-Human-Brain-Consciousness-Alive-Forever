# 嗅觉—视觉—真实身份颈部端口：身体设计审阅，待父裁定功能门

仅只读旧source／JSON与嗅觉源审计，尚未写运行代码或加载模型／数组。视觉114/115完成的实际结果以父报告为背景，当前嗅觉／融合尚无新身体执行。

## 空间语义：同侧一致不等于同一物理源

视觉v7远域使用实际眼mid及头水平forward/side：世界固定目标约为mid＋2forward＋2tan(±.4)side，render-only红色远地标。旧C Gaussian使用实际头触角proxy左右sensor midpoint_xy：center_xy=mid_xy＋.6forward＋.6tan(±.3)side；sigma.5、amplitude1.25、ANN.6开始.2秒smoothstep爬升，采样只有XY。两个目标距离、初角和锚点均不同，首轮保持旧域应把F_left_agree／F_right_agree叫“同侧一致的两种刺激”，不能写同源、同一食物或到达。

在全局5.8秒handover，分别按各自实际Head位置／姿态一次锚定世界目标，之后都固定世界。不将sourceXY／bearing／条件标签送入任何神经网络。如果将嗅源移到视觉2m／.4角点实现真正共源，则Gaussian双侧幅度与梯度明显变更，需另立场参数与可观察性实验；不是首14条件的最小恢复。

## 推荐固定14条件

六视觉alone：V_empty／V_center／V_left_pulse／V_right_pulse／V_left_to_right／V_right_to_left；嗅觉场实际off，嗅觉GRU仍每ANN神经步接零receptor并自然输出偏置，不能if条件把odorraw清零。

三个嗅觉alone：O_left／O_right／O_noodor，视觉环境实际无红地标，但相机／CNN／视觉GRU继续处理真实空场RGB，不把visionraw gate0。

两个同侧一致双刺激：F_left_agree／F_right_agree，同时真实RGB与Gaussian场，不是离线配对；视觉可沿L/R pulse .6..3.6，嗅觉Gaussian沿旧C持续rise后不关场，两个刺激时程不同应明确，不把融合cue时间假定相同。

一个F_left_DN_cut：同正常Fleft，ANN1.6消息通道cut＋DN接收基线零，无脑／视觉／嗅觉／motor reset，不强行turn0。

两个感觉因果对照：O_left_odorcut在ANN1.6仅actor receptor2置零，真实field／filter／pending连续，嗅觉GRU自然演化不reset；O_left_swap整个ANN交换actor LR顺序，环境两个采样点／field不交换。swap是声明反事实通道干预，不把场source标签反射或用turn翻号。

## 接线与时序

脑侧同时有冻结视觉99698GRU32和嗅觉66fGRU32，rawvisual／rawodor每ANN5ms更新，不按modalitypresence规则选择输出。融合brainGRU16输入[已交付AN1,visualraw1,odorraw1]；新weight_ih旧两列copy、第三列初零，其余2123桥全statecopy。全部任务动作从DN2→9e88来源队列2ms→bodydecoder→789f；禁止恢复旧ABproject直接接motor。

总递归状态[motor128,64,64,visual32,odor32,brain16]=336。cache16、filterstate2、pending2、AN1／DN2接收缓存和两个队列另记。相机20ms捕获／20ms发布／5ms神经消费；嗅觉每5ms真实Head local proxy (.571317656,±.459157176,0)变换到world并采样C，target=C/(1+C)，tau50ms的离散alpha=-expm1(-.005/.05)；本步输入为previouspending，新filter/pending更新只供下一决策。脑／嗅觉／视觉warmh0、teacherprimecmd0、真实AN从t0发布、交接不reset。warmfieldoff、filter/pending0；ANN后两个感觉GRU同时运行，包括模态实际不存在时的自然输出。

## 功能门建议（先供父裁定，当前不发明新通过阈值）

视觉六条因为融合脑已改变必须重测原视觉门，不能继承115成功；旧视觉感知门与原五项运动／slipFAIL分开。

旧Gaussian空间C是development没有方向合格门，旧D cutdifference3<4失败保留。因此首轮嗅觉方向可固定早窗ANN.82..1.82(在.6+.2rise+.005pending+.005DN之后)，报告相对matched O_noodor的有符号yaw差／速度、turn符号、receptor差和rawodor，不用任意1°阈值把它宣布生物orientation通过。父若需要工程最小效应门应在运行前依据实际旧B门及本次任务语义裁定；不能把B的人工浓度夹持门直接冒作C真实空间保证。

F左右报告相对matched V_empty／O_noodor和各单模态的有符号yaw效应，但两cue不同距离／时程不应用“比任一单模态更强”作无依据成功门。冲突取舍不在本14首轮，不推断原生感觉偏好。

Ocut/F_DNcut前缀必须实际物理／336状态／动作相同至全局7.4；所有消息release/consume与filterpending无未来，全完整1+4.8+9.6／1232000native／15400bodyms／3080neuralrows／770actual双眼RGB保存，源SHA／全部参数固定。独立审核应重新实际RGB→visual、真实receptors→odor、实际X111→AN、来源queue→fusionbrain→DN→decoder→motor，完整336状态／动作精确；环境field另据真实Head／sensor点重算，而不是只信保存receptors。

原五工程门单列，身体稳定／完整周期／NF／extra力单列。当前空场原五门PASS/slip约.4793和转弯原slipFAIL保持，嗅觉／融合不意味着到达、摄食、停驻或脑走停。当前66f只是既有turnNN、deploymentfalse、旧D失败，不叫实测嗅觉神经编码。
