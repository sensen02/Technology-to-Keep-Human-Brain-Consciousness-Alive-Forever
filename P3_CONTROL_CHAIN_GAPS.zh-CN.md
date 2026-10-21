# P3 脑—VNC—肌肉控制链缺口：有界只读审计

**手工研究原型，不是成品。对应 P3/S3，兼顾 S4。没有运行新神经网络或原生物理模拟；没有改资产、连接、符号、参数、历史或追踪表。** 这份交付完成的是有界缺口归并，不是控制链补齐或稳定站立/行走验收。P2和追踪表由父代理处理。

## 1. 当前包与可信范围

当前P1：`/run/media/sensen/Data2/cell_wound_prototype/outputs/experiment_platform/20261013_233237_7be93a1d/result.json`；逐通道真实电流：`/run/media/sensen/Data2/cell_wound_prototype/outputs/experiment_platform/20261013_233237_7be93a1d/traces.npz`。包内结论仍为 `autonomous_goal_status=NOT_ACHIEVED`、`biological_mapping_verified=false`。50ms短时工程追溯，不是稳定行为。

本轮用标准库解析JSON、gzip CSV；仅使用NumPy只读打开归档NPZ电流，不导入项目或仿真模块。按照 `Root ID` 与P1的12000个根唯一连接原始注释：`/run/media/sensen/Data2/cell_wound_prototype/data/flywire/banc_neurons.csv.gz`。每根恰好一条，未缺失。历史感觉审计3788个 `(leg,channel,root)` 三元组与P1精确一致；历史12000根顺序与P1一致；历史路径诊断202个运动目标根集合与P1一致。它们让历史结构结论可作同身份参照，**不让历史动态协议变成当前P1实验**。

机器清单：`/home/sensen/Desktop/engine修复验收/P3-control-chain-gaps.json`。包含输入哈希、逐腿计数、部分池缺根、未解析执行器、共享组、缺口分类、来源和置信范围。高置信仅指本地身份/计数；生物映射未经验证，整体连接体完整性未评估。

**无图不能推断全部缺失路径**：本轮没有构建原始/严格模型邻接图，没有重新搜索下降→VNC局部→运动的路径。`missing_path_count=null`，不写0；不宣称找全所有缺路，也不把缺证据写成动物无通路。

## 2. 不可把所有问题归为“缺VNC”

对当前P1真实tier的原始CSV `Super Class` 计数如下（保留数据字段语义，不以细胞名推断解剖）：

| 类别 | 根数 |
|---|---:|
| ventral_nerve_cord_intrinsic | 5927 |
| sensory | 4149 |
| ascending / sensory_ascending | 774 / 34 |
| central_brain_intrinsic | 453 |
| descending / sensory_descending | 194 / 2 |
| motor | 393 |
| visual_centrifugal / visual_projection | 9 / 2 |
| glia / visceral_circulatory / 空标签 | 6 / 4 / 53 |
| 合计 | 12000 |

已有脑、VNC、颈部跨越类与运动根；不是FlyWire纯脑直接接腿肌，也不是完全无VNC。**存在这些类别不证明具体脑→下降→VNC局部→运动的全链路已验证。** 393个motor标签不等于393个可用输出；本包映射执行器只使用202个唯一在tier根。4149 sensory标签也不等于3788个实际分配腿感觉根。

选择与裁剪规则来源：`/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/adapters.py`；严格递质过滤来源：`/run/media/sensen/Data2/cell_wound_prototype/engine/neck_cut_data.py`。不能根据类别计数决定盲目扩大网络。

## 3. 逐腿感觉：群存在、实际注流、调谐验证三者分开

| 腿 | claw位置 | hook方向 | hairplate | campaniform | club | bristle | multidendritic | 总根 | P1正常臂曾非零电流根 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LF |25|24|28|28|53|450|7|615|49|
| RF |24|25|36|24|40|442|4|595|49|
| LM |33|20|31|2|52|466|6|610|53|
| RM |12|22|40|2|56|481|5|618|34|
| LH |33|22|33|2|51|489|4|634|55|
| RH |22|23|39|2|71|551|8|716|45|
| 总计 |149|136|207|60|323|2879|34|3788|285|

42个身份群均非空；**正常臂实际只有12群（每腿claw/hook）曾注入非零电流**。按归档电流向量逐元素 `!=0` 检查，不用可能正负抵消的电流和：285个唯一根均至少一个区间非零；每个活跃群在10区间中的9区间非零，初始区间零。其余30群在正常臂全部零。`channel_record` 的身体代理值或modelled标志不能冒充过滤后的真实注流；诊断触觉臂不能冒充正常臂感觉。

来源：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/channel_attribute_audit/channel_attribute_audit.json` 和当前P1电流数组。历史逐根审计保留 `confirmed_root_specific_joint=null`、`confirmed_tuning_polarity=null`；没有按SNpp编号发明屈伸极性。30根注释复核队列仍是局部证据问题：1根campaniform/HP矛盾、2根HP/CS名称歧义、15根外周目标空/orphan、12根bristle功能混合。详见 `/run/media/sensen/Data2/cell_wound_prototype/BANC_RUNTIME_STRUCTURE_AUDIT.zh-CN.md`。

## 4. 逐腿运动池、可用执行器与身份裁剪

| 腿 | 模型执行器 | usable | complete | partial | unresolved | 非空不同池根集合 | 在tier/完整声明唯一运动根 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LF |15|14|12|2|1|12|38/39|
| RF |15|14|12|2|1|12|38/39|
| LM |15|11|11|0|4|10|32/32|
| RM |15|11|11|0|4|10|32/32|
| LH |15|12|9|3|3|11|30/33|
| RH |15|12|10|2|3|11|32/33|
| 总计 |90|74|65|9|16|66|202/208|

usable仅表示有至少一个映射tier根，不是有脉冲、独立可控、足够力矩或解剖已验证。complete也仅是当前声明池的身份全覆盖。9条partial实际只缺**6个唯一根**；不能把共享束重复算成缺9个神经元：

| 受影响执行器 | 在tier/声明池 | 缺失根 |
|---|---|---|
| LFF_trochanter_flexor_a、LFF_trochanter_flexor_b |各7/8|720575941511883985|
| RFF_trochanter_flexor_a、RFF_trochanter_flexor_b |各7/8|720575941391252886|
| LHF_sterno-tergo-trochanter_extensor_b |2/3|720575941500775137|
| LHF_accesory_trochanter_flexor |2/3|720575941497037218|
| LHF_trochanter_extensor |1/2|720575941434056663|
| RHF_trochanter_flexor_a、RHF_trochanter_flexor_b |各6/7|720575941559306355|

原样保留 `accesory` 拼写保证可追溯。池按PRESENT细胞取均值；裁剪不仅影响根覆盖，还可能改变归一化，不能无声明地补根后当相同模型。

8个共享池组驱动16束：六腿各自的trochanter_flexor_a/b；另LF、RF各自的tergopleural_promotor_a/b。它们不是16个独立运动控制通道；同一池给多束相同命令是未验证的招募假设。源码：`/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/motor_output.py`，其5ms滤波、100Hz满控制量不是实测生理标定。

### 16个未解析执行器的分解

- 六腿各自 `C_pleural_promotor`：6条。
- LM/RM/LH/RH各 `C_tergopleural_promotor_a/b`：8条。
- `LMC_sternal_adductor`、`RMC_pleural_remotor_and_abductor`：2条。

全部精确名字列在机器清单。这是**14条promotor模型/注释对应问题 + 2条侧别目标注释问题**，不是16条都可简单索取新VNC补齐。历史 `/run/media/sensen/Data2/cell_wound_prototype/LEG_MOTOR_COVERAGE.zh-CN.md` 曾偏向“注释缺口”；后续 `/run/media/sensen/Data2/cell_wound_prototype/MODEL_PROVENANCE_STRUCTURE.zh-CN.md` 指出源OpenSim仅15块LF肌肉复制到六腿，修正为需要先核对身体肌组的归属。后者的全上游/论文检索本轮没有重做，不把旧推论升级为独立新验证。源副本：`/run/media/sensen/Data2/cell_wound_prototype/outputs/upstream_muscle_round11/best_combined.osim`。

LF/RF复合tergopleural/pleural-promotor名称可作为共享候选，但当前未接线，不能算已解决。中后腿无对应标签不是动物无同源神经元；MANC男性个体根也不能无证据搬入BANC。模型不含的tarsus/femur等肌组属于反向模型边界，不在90执行器分母中；本轮不新增身体自由度。

## 5. NT与路径：按证据层分开

### 5.1 当前逐腿verified字段缺失

从原始CSV连接当前感觉/运动根，感觉verified字段：

| 腿 | 感觉空verified | 感觉ACh verified | 在tier映射运动根空verified |
|---|---:|---:|---:|
| LF |514|101|38|
| RF |495|100|38|
| LM |503|107|32|
| RM |508|110|32|
| LH |526|108|30|
| RH |586|130|32|

这是已审计字段统计，不是整个文献无证据。**202个运动根自身NT空不等于它们不能接收突触或不能被解码成肌肉命令**；本模型依据突触前标签决定中心网络输出符号，而脉冲→肌肉译码不检查其NT。不能用运动根预测GABA解释其输入膜响应。

### 5.2 历史34个实际发放却verified缺失的源

`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_loop/missing_nt_evidence_audit.json` 记录34个prediction_only：27 sensory、6 VNC intrinsic、1 ascending。34根来自**历史LF感觉20ms脉冲/100ms观察**，不是当前P1正常臂活动集合；72个已发放源中31 ACh、7 GABA、34空verified。

| 互斥原始输出类别 | 行 | 原始突触数 |
|---|---:|---:|
| tier外 |133|670|
| tier内非motor，被严格NT规则排除 |1412|15897|
| tier内motor，被严格NT规则排除 |114|1194|
| 总计 |1659|17761|

tier内合计1526行/17091突触（96.23%原始输出突触），说明这个34根集合的损失主要不在tier外。26源直接指向28个唯一运动根，本轮按当前逐腿池根交叉全部属于LF；其他五腿在这份114边清单中为0，**只代表该协议的已审计目标为零，绝不代表其他腿无NT缺口/无路径**。所有114边仍sign=null、modelled=false；未应用每边上限和电导缩放，突触数不是电导或强度。

原始表与缓存/v888一致只证明摄取一致；不使预测变实测。即使verified已给ACh/GABA/Glut，统一ACh→exc、GABA/Glut→inh也缺逐靶受体验证。已有后续来源核查 `/run/media/sensen/Data2/cell_wound_prototype/IN21A022_MAPPING_FOLLOWUP.zh-CN.md` 与 `/run/media/sensen/Data2/cell_wound_prototype/NT_SCOPE_FOLLOWUP.zh-CN.md` 尚未解决IN21A022具体出生群/实例或成年BANC运动目标受体；不能把幼虫NMJ或lineage群证据直接赋给成年单根。

### 5.3 已有路径证据不是“无路径”

历史 `/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_loop/flybody_connectome_path_audit.json` 使用实际模型矩阵 `W[target,source]` 合并E/I搜索：LF hairplate位置代理群、LF hook速度群各自都能无符号到达202/202运动根。按腿目标计数即LF38、RF38、LM32、RM32、LH30、RH32，**源仍只有LF两个诊断群，不是每腿自身42群路径普查**。

该历史实验运动池仍0脉冲；最接近阈值差2.088/1.040mV，结构可达不等于事件成功传播，更不等于肌肉可控或站立。本次只核查归档，不重跑。零输入臂 `shortest_unsigned_hops=null` 是空源条件，不是证实结构断路。

因此当前可支持的是“部分输入结构可达，但生物符号/传播/招募与完整脑下降链路未验证”；**不支持“完全无路径”“只要加预测符号/增益即可走路”或“缺了整个VNC”**。要推断真正缺哪条路径，需要原图/严格资格图和外部边界，不能由计数、细胞名称、缺字段或0脉冲反推。

## 6. 缺口—来源—修复类别—下一最小动作

分类不是互斥：`public-source`=公开原文/版本化交叉表可查；`available-data`=现有本地数据仍可只读量化；`needs-experiment`=最终生物验收需要新测量，非本轮授权去做动物实验。

1. **G1 感觉群与调谐**（available-data/public-source/needs-experiment）：来源 `/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/channel_attribute_audit/channel_attribute_audit.json`、`/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/sensory_channels.py`、当前P1电流NPZ。先逐根复核30条异常注释和有出处的关节/类型证据，未知极性保留null；最终用刺激—发放数据标定，不为站住启用未知群。
2. **G2 已知6根裁剪**（available-data）：来源当前P1 manifest、`/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/motor_output.py`、`/run/media/sensen/Data2/cell_wound_prototype/data/flywire/banc_connectome.npz`。下一步先只读列6根原始出入边及严格NT资格，量化裁剪影响；不能由6根身份直接声称缺6条通路，不盲目增大tier。
3. **G3 未解析16执行器**（public-source/available-data）：来源当前manifest、`/run/media/sensen/Data2/cell_wound_prototype/MODEL_PROVENANCE_STRUCTURE.zh-CN.md`、`/run/media/sensen/Data2/cell_wound_prototype/LEG_MOTOR_COVERAGE.zh-CN.md`。先核肌组复制/逐腿解剖归属；再查两条侧别目标的新版BANC注释。禁止把MANC同名当逐根证据；前腿共享只是候选。
4. **G4 共享招募/译码**（public-source/needs-experiment）：来源当前manifest及 `/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/motor_output.py`。先逐组记录原肌肉目标出处；最终需运动根发放—肌肉活化/力矩同步测量验证共享比例、增益、时间常数。工程重放不能代替它。
5. **G5 NT输出损失**（available-data/public-source/needs-experiment）：来源 `/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_loop/missing_nt_evidence_audit.json`、`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_loop/active_output_coverage_audit.json`、`/run/media/sensen/Data2/cell_wound_prototype/engine/neck_cut_data.py`。下一步把34源及28目标整理成根→类型/版本→实证NT→受体的证据行；优先处理IN21A022身份歧义，不把预测、群lineage或NMJ证据升级。
6. **G6 脑/下降/局部VNC全链路与真实缺路**（available-data/public-source）：来源 `/run/media/sensen/Data2/cell_wound_prototype/data/flywire/banc_connectome.npz`、`/run/media/sensen/Data2/cell_wound_prototype/data/flywire/banc_neurons.csv.gz`、`/run/media/sensen/Data2/cell_wound_prototype/tools_connectome_path_diagnostics.py`、历史path JSON。下一次独立有界任务才构建raw与strict两图，按实际NT资格、方向、每腿目标输出路径见证和tier外边界；先诊断已有VNC裁剪，再决定是否扩展。当前缺路总数未知。
7. **G7 成年目标受体/延迟/肌肉身体响应**（public-source/needs-experiment）：来源 `/run/media/sensen/Data2/cell_wound_prototype/NT_SCOPE_FOLLOWUP.zh-CN.md`、`/run/media/sensen/Data2/cell_wound_prototype/engine/neck_cut_data.py`、`/run/media/sensen/Data2/cell_wound_prototype/GOAL_ROADMAP.zh-CN.md`。先区分成年中枢和幼虫NMJ证据范围；最终需感觉—神经—肌肉—运动同步数据与留出验证，不做无来源增益/符号搜索。

## 7. 验证与未执行事项

- 当前JSON/CSV逐根连接、归档NPZ实际非零向量、完整/部分/未解析执行器逐项核计；历史身份顺序、感觉三元组、运动目标集合精确核对。
- 6唯一缺根而非9根；8共享组而非16独立池；74 usable不等于74完整或生物已验证；本轮没有新增路径搜索。
- 实际调用图像视图打开 `/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/channel_attribute_audit/runtime_sensory_coverage.png`，42格计数与表一致、轴/限制说明可读。这是历史图视觉核查，没有新图或模拟。
- 没有公共搜索/网络请求；精确外部来源未成为本次计数所必需，因此不引入新的、未经核查的文献断言。历史文献作用范围按旧报告限定。
- `python`不可用退出127后改用`python3`；初始错误假设unresolved行也有leg导致KeyError，已改按执行器前缀和source_annotation处理；误找不存在的motor_manifest文件后定位实际motor_output源码。均未造成任何源文件改变。成功审计命令退出0。
- 输出以项目报告、桌面中文副本、机器JSON为限。未改追踪表、历史包或资产；没有站立/行走达成声明。

**交付边界：P3有界缺口清单完成；S3控制链补齐和生物验收未完成。下一最小动作建议G6只读图边界审计与G2六根裁剪清单，不先购GPU、补预测符号或做表现搜索。**

## 8. 2026-10-14 P3有界图审计后续修正（保留上文历史结论）

新只读工具 `/run/media/sensen/Data2/cell_wound_prototype/tools_p3_path_eligibility.py` 与测试 `/run/media/sensen/Data2/cell_wound_prototype/test_p3_path_eligibility.py` 已手动实施运行，不是成品。证据 `/run/media/sensen/Data2/cell_wound_prototype/outputs/p3_path_eligibility/20261014_001205/result.json`、报告 `/home/sensen/Desktop/engine修复验收/P3-路径资格审计报告.md`、JSON `/home/sensen/Desktop/engine修复验收/P3-path-eligibility.json`。实际构建严格We/Wi非零有向无符号支撑291986边；NT过滤前缓存tier图442843边，包含生物学未知NT边，不是允许未知符号的修复模型。原始CSV单次完整流式扫描3990039行，相关正边561506行，唯一支撑与缓存差0。未运行神经/身体动态、未改模型或P1；16个导入源码哈希与P1一致。

42感觉群：12个P1实际注流群均严格可达202目标；其余30个零注流群中14个严格可达202、16个严格可达0（六腿bristle/multidendritic共12群，加中后腿campaniform四群）。原始图42群全部可达202。195实际下降readout集合及196全tier下降集合分别在两图均可达202。逐腿目标LF38/RF38/LM32/RM32/LH30/RH32；JSON逐目标最短路径见证带根和class。结构可达不等于激活，也不证明每个下降根均可达或完整生物控制链。

**修正上文“六根被裁剪”的笼统表述**：四根在缓存节点全集但第二轮候选rank超过3394席位：720575941391252886(rank7925)、720575941497037218(rank3706)、720575941511883985(rank9342)、720575941559306355(rank7026)。四根verified NT均空；原始tier输入分别8/41、23/177、10/36、10/42（行/突触），其中突触前NT资格行7/17/10/8；原始输出至tier均0行。另720575941434056663与720575941500775137不在缓存节点全集，原始注释CSV也未找到，完整原始连接CSV中出入行均0。后二者属于该快照上游身份/摄取边界，不是tier rank裁剪，不能据此断言动物无细胞或无连接；精确NT/class保留null。

最终CLI managed job退出0，12.88秒，峰值RSS1008.57MB，guard need1600MB/cap4608MB。4项方向/见证/断连/截断单元测试通过；输入哈希、源码ZIP、逐腿计数与见证长度独立检查退出0。实际read_image打开 `/run/media/sensen/Data2/cell_wound_prototype/outputs/p3_path_eligibility/20261014_001205/reachability.png`，44行/6腿数值及12个注流星号一致。初始两CLI退出1（guard函数名误用、误以为六根均在缓存），已修正；首次独立检查误索引缺失原注释字段退出1后改为显式null并复验通过。未做全脑/跨tier外中间节点闭包、边替换或预测符号修复；下一步优先查两缺失原身份及真实NT/受体证据，不扩大网络制造表现。
