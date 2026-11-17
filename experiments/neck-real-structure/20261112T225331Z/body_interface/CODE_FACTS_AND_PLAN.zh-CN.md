# 身体侧接口：实际源码事实与最小可训练工程实现

本目录是ONLYCODE准备，当前没有加载Torch或模型、forward、反传、优化器、Core或身体仿真。代码导入默认仅标准库。人工脑—真实结构颈部—其他分层不变；身体侧学习映射属于明确工程适配，不称真实DN运动功能或AN调谐。

## 旧行走并没有独立下行命令入口

`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/models.py` 88—143行：SlowNeuralFeedbackGRU.step(obs,state,reset)仅238观察和128递归状态，输出42关节latent，没有descendingcommand参数。

`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/physics.py` 95—100行：obs238由本体特征、clearance、previous42action和末位target_speed组成。`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/student_wave.py` 29和103—108行将TARGET_SPEED固定.075。它只是一个输入字段，不是已证明能调控速度／启停的颈部端口。不能把旧42输出改名真实DN。

`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/recurrent.py` 22—52行：replay_mean只回放actor.sequence(obs)，既有PPO loss按完整episode动作分布和优势训练，不包含新脑—颈部边界；不能原样声称穿颈梯度已验证。

## 已有组合转向模型具有一个受约束外部控制自由度

`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning/model.py` 34—45行：step(obs,command,state)接受两元素command但要求command=[relu(turn)/3,turn]，turn∈[-1,1]，所以独立控制仅signedturn一个自由度。base128每次始终前进，两转向分支64＋64也始终演化，零turn仍保有旧自主步态。

因此首范围可检验脑经neck调制已有转向，不能据此宣称脑经neck启动、停止或从零产生行走。旧运动ANN的持续步态贡献必须单列，zeroDN／zero turn低层基线不能少。

## 反馈mask实际修正

旧physical_mask排除42:84、126:168、195:237后共112特征，但仍包含index237的target_speed常量。它不应混称纯身体感觉。新AN工程encoder只用111个原始索引：0:42、84:126、168:195；排除targeterror42:84、targetposition126:168、previousaction195:237和target_speed237。local frozenmotor本身仍可用原完整obs238接口，不向人工脑暴露这些旁路。若将来需要改变允许模态，必须另立显式接口版本与证据。

## 最小学习模块

`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/body_interface/interface.py`：

- source_ports读取明确来源与空间审查文件SHA，按实际有序DN／AN清单导出N/M；不凭全候选3699或3229计数猜维度，也不把动态canonical root_id等同root_888。sourcebinding中的root_id是父选择并锁定材料化版本的原ID，版本证据由引用文件保留。
- make_neural_interfaces仅在父明确执行准入后lazy import Torch：BodyDecoder N→16tanh→1tanh产生learnedturn。motor_command只适配既有冻结组合运动ABI，不形成固定DN→turn手写增益。
- FeedbackEncoder仅将local111身体特征经32tanh→M tanh编码；输出必须再走真实neck的AN身份队列返回脑，不允许脑直接取localraw238。该映射明确工程，不是nativeAN调谐。
- validate_delivery拒绝脑／身体直接跨界、隐藏action/target字段、身份错序、超界/NaN或未来feedback，只接受neck已发布的指定方向packet。

人工脑GRU由父另实现，仅接发布后的AN及明确实验taskcue。taskcue为人工训练任务指令，不冒称原生感觉或电极记录，不允许共享brainhidden／cue直送身体adapter。

## 来源限制

原两样本DNpe054／AN08B018因几何覆盖被选，不因行走／本体感觉功能证据被选。父连接审计发现所选DN无直接legmotor边，AN输入以clubchordotonal振动相关为主；不能把工程111感受编码说成该AN的真实生理功能。若改选DNp09等纤维，必须另真实材料化ID与几何绑定，不能默默沿用旧两sample的root映射。结构真实性和训练得到的工程控制接口是不同验收层。

## 父可执行的最小预检顺序

1. `python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/body_interface/test_interface.py`：本子实际12项标准库检查通过exit0，0heavyimports。
2. `python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/body_interface/prepare_binding.py --execute-parent`：将前一真实2纤维绑定转换为新工程body接口binding；不自动扩大完整neck准入或nativefunction。
3. 单CPU环境设置OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1，再由既有env Python运行 `/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/body_interface/numeric_check.py --execute-parent`。数值预检由父唯一执行，本子未运行。

数值检查实际构造NN并检查mask Jacobian、排除输入扰动的精确不变性、DN路径梯度、无cue直达路径、有限参数梯度、motorABI以及真实sourcepacket经2ms运输。该脚本不优化或训练、不载入组合motor权重、不构造Core、不证明已能控制身体。

之后父另冻结有限校准预算：先工程bodydecoder/ANencoder校准后冻结，再brain-only训练与DN/AN切断、置乱、延迟、回放和恢复。必须独立保存重放、原完整身体门与zeroDN基线，不能以无生理依据的损失或teacher关节直驱让新架构挂名通过。
