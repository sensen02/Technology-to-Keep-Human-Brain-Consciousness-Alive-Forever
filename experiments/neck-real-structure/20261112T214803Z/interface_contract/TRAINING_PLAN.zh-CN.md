# 三部分模型：接口准备与后续人工脑训练计划

## 身份与当前结论

本目录只实现标准库接口契约和拒绝不合格绑定的验证器，不实现神经动力学、不载入模型、不启动训练／物理／真实电极。架构为：

**其他（虚拟完整身体、身体侧神经系统及明确工程适配）⇄ 真实结构依据的果蝇颈部神经通路 ⇄ 人工训练的脑ANN**。

当前MODEL_CONTRACT.json状态 **NOT_ADMITTED_NECK_SOURCE_PENDING**：DN/AN端口为空、维度null、时间参数null，没有真实结构准入或新身体行走通过。颈部是跨颈轴突／通路边界，不假定每根轴突对应颈部独立胞体，不在无证据时添加突触或稠密RNN。人工脑是用户明确允许训练的模块，不冒称活体脑；电极记录＋刺激和意识问题仍是未来独立实验。

## 已检查旧接口，不能无条件继承

1. `/run/media/sensen/Data2/cell_wound_prototype/engine/neural.py`：LIF状态／稀疏连接接口存在，但文件明确动力学参数illustrative、结构不等于功能；旧递质到符号映射不能直接作新颈部真实生理证据。
2. `/run/media/sensen/Data2/cell_wound_prototype/engine/isolation_scenario.py`：明确为虚构graph＋理想经典squid-HH颈部电缆，无重建解剖，不能替代本次真实结构要求。
3. `/run/media/sensen/Data2/cell_wound_prototype/engine/neck_cut_data.py`：旧BANC缓存可由注释导出crossing候选、outgoing-edge class removal；文件明确无soma/arbor坐标、单细胞上下行极性未独立验证、不是空间切平面。旧注释数量不能自动成为新脑输出维度。数据子代理单独审计，接口侧不重复全库或采纳未经审计数字。
4. `/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/descending_command.py`：有两个source-ID DNp09和显式短时外置神经启动电流，不等于完整下行接口、更不等于上行感觉或自主人工脑。
5. `/run/media/sensen/Data2/cell_wound_prototype/engine/sensorimotor.py`：分类端口可用于边界参考，但端口名称本身不构成视觉／感觉系统；不强迫脑侧直接视觉／听觉走颈部，实际通路必须有来源。
6. `/run/media/sensen/Data2/cell_wound_prototype/WALKING_CURRENT_STATE.json` 与 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/models.py`：旧已验证替代ANN使用238感觉特征、42执行器输出；已通过的权重SHA246f9fab7d21e2a1332a745fafd4a343b80ad5a70d2cafeeb900e876f7933809只属于原限定完整身体行走。该维度不是颈部神经元数，不能直接作为人工脑DN端口或新架构验收。

## 数据准入后需绑定的内容

源bundle必须记录每个实际文件绝对路径／SHA256、dataset/version/sample/species/role、crossing边界的原记录定位。每个DN/AN port必须关联原始神经元或纤维ID、identity evidence、crossing evidence、信号单位和活动范围。维度由实际port清单计算，不手填固定数。候选catalog可先单独引用，但不能凭ID清单把NOT_ADMITTED改成已实现。

neck_links允许同一原始纤维在边界两侧的身份保持传输，或确实有数据依据的source_synaptic_mapping；并不要求颈部内部必须存在突触。不允许invented denseRNN、不允许训练添加连接。拓扑身份绑定实际source文件SHA；连接映射的科学正确性仍需独立结构审计，不由验证器把哈希存在等同为解剖真。

dynamics应记录ENGINEERING_ASSUMPTION或MEASURED_AND_CITED，前者不冒称生理真实。未测时延、递质、膜参数和电极映射保持未知。结构来源审查、crossing/cut定义审查、动力学预登记和runtime binding均需独立证据引用，缺一拒绝训练准入。不得以schema验证替代独立科学审查。

## 允许的信息通路与运行包

脑输出只能是已登记DN端口活动：brain→neck→other。身体侧反馈只能走已登记AN：other→neck→brain。禁止脑输出直接关节动作、目标坐标／理想状态泄漏到body低层网络、共享脑隐藏状态作为未声明旁路。完整身体的局部感觉闭环可在other内部保留，不能以此冒称人工脑贡献。

脑侧直接感觉单列brain_side_sensory，必须有真实不经颈部解剖依据及明确模态来源；不将身体载荷等反馈伪装为直接视觉。当前清单空，不虚构听觉或视觉通路。

每个运行包固定from/to/direction、原序port_ids、有限有界values、capture/release/consume时间和topology SHA。不接受额外隐藏字段、重复／错序端口、未来反馈。通信包的时序验证只约束包本身，无法证明完整程序没有旁路；未来还需源审计、状态/队列连续日志和因果干预。当前无运行包可接纳。

## 后续三阶段训练（本次不执行、不盲定神经维数或预算）

### A. 接口响应与反馈校准

先固定来源拓扑和已预登记动力学，用限幅、可重放的DN活动探测身体侧响应；检验AN对身体状态变化的响应与因果发布时间。控制范围、噪声、时延、秩／饱和、闭环稳定性是待实测项，不能用一组标签成功代替。身体侧工程解码器如需校准必须单独scope，校准完成后在脑训练期冻结。

### B. 经DN蒸馏已通过的行走教师

旧walking教师仅产生离线身体运动／感觉参考。人工脑ANN学习从允许的感觉及历史，生成指定DN端口活动；这些活动必须实际经过neck和other产生物理轨迹。教师不能向关节直接投喂动作完成新架构验收，不能将42关节输出改名DN。颈部拓扑冻结；优先只训练脑ANN，身体侧若仍需变更须独立预登记和回归。loss需同时检查身体运动匹配与接口活动范围，不用目标轨迹作隐藏在线输入。

可微neck使用时间展开梯度；不可微或脉冲系统采用明确的策略梯度或训练辅助代理模型。代理模型只帮助优化，最终回到原颈部和实际身体核验。训练方法、实际损失／优化器／梯度预检和资源预算必须在数据门通过后另立预登记，本次不凭空定维数、epoch或生理活动幅度。

### C. 完整身体RL与独立因果确认

撤掉教师，以原完整身体物理和原行走门评估，持续记录脑输出、neck两侧DN/AN、other状态、神经/物理时序与身份。脑only训练；neck连接拓扑、身体低层冻结。完整预热和撤辅段连续不reset，旧walking通过不自动继承。

正常、脑输出基线／断开、DN断开或置乱、AN切断／延迟／回放和恢复连接分别预登记，不能强求DN断开后身体必须立刻倒下——低层站立可能保留。证据应是任务相关脑贡献确经颈部传递及AN反馈被利用。未来电极信号必须区分刺激伪迹、群体混合与真实神经响应，不能直接用模拟port值冒称实际记录。

## 当前可执行的标准库检查

`python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T214803Z/interface_contract/test_validator.py`

玩具fixture仅用于拒绝旁路／错误方向／重复ID／缺源SHA／未来反馈等软件检查，临时文件自动清理；正例也绝不返回生物准入。

`python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T214803Z/interface_contract/validator.py /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T214803Z/interface_contract/MODEL_CONTRACT.json`

预期退出码2，输出valid_contract=true、admitted=false、NOT_ADMITTED_NECK_SOURCE_PENDING。这是按设计拒绝未绑定真实数据的训练准入，不是已训练或结构通过。代码／契约测试、源结构真实性、动力学有效性、行为、意识必须分层，不用电活动或运动成功证明意识保留。
