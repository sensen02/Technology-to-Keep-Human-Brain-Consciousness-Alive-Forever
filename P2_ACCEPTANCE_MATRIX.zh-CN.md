# P2 固定输入数值诊断验收矩阵

**手动研究实施，不是成品。此轮仅读取冻结数据、复算验收和哈希；没有新模拟、模型变更、测试再生成或生产方法切换。** 对应P2/S1。本报告区分“可适用合同失败”“无适用规范的诊断敏感性”和“未执行”；不通过放宽旧容差解决失败。

## 1. 结论与词义

- **P2诊断交付完成；整体收敛未证明；S1阶段尚未通过。** 重复/身份/时序/事件到命令的工程复现不等于整体收敛、动物有效性或稳定站立。
- 两种电压方法相邻神经时钟的命令最大差均为1.0，但检索到的原合同没有对此50ms固定输入逐池命令定义数值通过阈值。该结果是**diagnostic-not-applicable：强步长敏感性**，不能伪造为“超过某旧命令容差”的FAIL，也不能判PASS。
- 历史100ms耦合闭环的正常联合四分步对联合半步首次跌破高度时间差2.15ms，确实超出原预声明1ms，是**applicable / FAIL**。历史基本比较还存在正常身体半步高度0.054318模型mm、神经半步高度0.056727模型mm/门槛1.8ms、联合半步门槛2.4ms的FAIL。此轮从保存记录复算，不是当前源码重跑或新的历史数值复现。
- 当前固定正常命令50ms身体诊断三档步长，高度和门槛均在旧描述性数值参考内；它不满足旧合同100ms/三干预臂/耦合闭环的完整范围。只报告**参考内**，不能升级为原合同整体PASS。

`applicable`表示合同范围与当前检查匹配；`diagnostic-not-applicable`表示已分析但无对应通过规范或原规范范围不匹配；`not-executed`表示此轮没有运行所要求的实验。状态与PASS/FAIL/REFERENCE_WITHIN/NOT_ESTABLISHED分别保存，不能用一个总PASS覆盖不同层级。

## 2. 原合同及精确范围

|原始路径与行号|原预声明内容|适用范围与排除|
|---|---|---|
|`/run/media/sensen/Data2/cell_wound_prototype/AUTONOMOUS_NEXT_PROTOCOL.zh-CN.md`:11–13|5ms命令；身体/神经基础0.1/0.5ms、身体半步0.05/0.5ms、神经半步0.1/0.25ms、联合半步0.05/0.25ms、联合四分步0.025/0.125ms；各正常/感觉切断/15ms运动切断100ms。胸高最大差≤0.05模型mm，首次跌破90%初高时间差≤1ms；四分步须比较半步；失败不改容差。|历史100ms耦合派生初态分支适用。不是50ms固定电流神经实验的命令收敛合同；不是固定命令身体实验完整验收。|
|`/run/media/sensen/Data2/cell_wound_prototype/CLOCK_REFINEMENT.zh-CN.md`:5–9|神经400步/命令40周期固定、200ms闭环、12.5/6.25μs身体细化；COM最大误差0.01mm仅报告参考；[0,T]身体/[0,T)命令/(0,T]事件窗口。|旧200ms对应分支参考，不能对本轮胸高或50ms固定输入强套COM规范。|
|`/run/media/sensen/Data2/cell_wound_prototype/FIXED_REPLAY_REFINEMENT.zh-CN.md`:5–9|200ms既有40×90命令；12.5/6.25μs身体回放；最细须精确复现源轨迹。|旧200ms固定命令分支；未声明新误差通过阈值，不能升级所有输入机械收敛。|
|`/run/media/sensen/Data2/cell_wound_prototype/P2_STATE_TRACE_PROTOCOL.zh-CN.md`:9–10、20–24；`/run/media/sensen/Data2/cell_wound_prototype/run_p2_state_trace.py`:26|50ms三神经时钟、5个事后诊断根；旧事件/计数/命令精确一致，来源不变；独立状态绝对残差1e-12（各字段原单位）。|状态重算工程正确性适用，不是整体数值解与动物的误差容差，5根不是留出验证集。|
|`/run/media/sensen/Data2/cell_wound_prototype/P2_VOLTAGE_METHOD_PROTOCOL.zh-CN.md`:7–17、21–25|12000细胞/权重0.2/零背景/50ms固定输入、BE与frozen_exact三钟各基准重复；事件独立重建计数、解码及一窗口延迟。明确不新建数值通过阈值，收敛另需预声明标准。|身份、BE精确参照、重复和因果时序适用；端点时间不同本身不定义收敛FAIL；frozen_exact不是连续全网络精确解。|
|`/run/media/sensen/Data2/cell_wound_prototype/tools_p2_event_sampling.py`:59–80、83–96、166、174|固定保存时间戳，右闭事件箱；离散滤波与连续脉冲解析；重建速率误差≤1e-10Hz、命令atol1e-12/rtol0。|代码中的工程重建容差，不是生物标定或跨步长命令收敛容差。|

哈希固定的是当前读到的合同/源码字节；旧100ms结果没有归档该协议的历史哈希，本报告不宣称证明了协议文件当年所有字节的先后性。旧结果本身存有`predeclared_descriptive_tolerance`，与原文数值一致；历史失败保留。

## 3. 分层矩阵

|层级|范围与证据|适用性 / 判定|不能推导|
|---|---|---|---|
|重复与参照|固定输入基准重复；方法比较两方法重复；BE三钟精确匹配原事件/时间、计数、率、命令|applicable / PASS（冻结结果；本轮只读数组再次核对）|跨钟收敛/动物重复|
|归档身份|五目录NPZ/ZIP/identity/图与声明SHA256；ZIP成员对照来源摘要；所有原文件读前后未变|applicable / PASS或实际失败记录；详见JSON逐项结果|执行归档代码、完整资产重建或历史数值重跑|
|时间链|50ms/十个5ms输入窗口；神经0.5/0.25/0.125ms；命令来源区间-1,0…8，下一窗口才应用|applicable / PASS（工程）|神经延迟等于动物延迟|
|固定电流|10×12000实际nA电流和tier顺序在固定输入/状态/方法包中精确相同，零输入三钟均无事件|applicable / PASS（身份与模型内零输入控制）|入口幅度/极性/募集已生物标定|
|事件计数|BE总7074/7072/7083、唯一运动392/397/405；frozen总7702/7375/7213、运动459/436/420；各cell×窗口重建|重建applicable / PASS；跨钟diagnostic-not-applicable / NOT_ESTABLISHED|总数近似相同代替逐池收敛；共享池累计事件等于唯一细胞事件|
|解码分区不变量|同0.5ms事件网格改变处理分区，速率/命令精确一致|applicable / PASS（工程）|改变采样dt也不影响滤波|
|固定时间戳采样|已保存全部事件，三网格对连续解析命令最大差0.04837418/0.02458849/0.01239648；同base事件base-half/quarter采样差0.02394149/0.03621336|diagnostic-not-applicable / 误差下降观察，未设收敛阈值|新实验必需、全网络连续解、生物滤波真值|
|不同事件同网格|原base与half/quarter事件在统一网格下命令差1.0；两电压方法各相邻钟也1.0|diagnostic-not-applicable / SENSITIVE，未证明收敛|没有适用容差时虚构FAIL；唯一可加因果分解|
|状态/传播次序|到达→active→BE端点阈值/重置→衰减→延迟环入队；所选根各字段残差0，原2ms环到达一致|applicable / PASS（冻结状态重算）|整体积分准确、生物延迟正确、一个根独自引发全网差异|
|固定命令身体高度|50ms正常命令回放：base-half/half-quarter/base-quarter高度差0.00713749/0.00112813/0.00709631模型mm，门槛差0.55/0.125/0.675ms|diagnostic-not-applicable / REFERENCE_WITHIN；基准精确P1为applicable / PASS|100ms三臂耦合合同完成、全状态机械收敛|
|历史耦合高度/时间|15个100ms冻结episode，联合quarter-half正常高度0.018864模型mm、门槛2.15ms；其他基本比较如第1节|applicable / 高度PASS而门槛FAIL；不得放宽|这次当前版本复现了100ms；把两个指标合为选择性PASS|
|当前版本完整耦合合同|本轮无新100ms五钟×三臂|not-executed / 未执行|旧失败已修复|
|生物有效性|感觉实际有符号电流向量/输出映射、极性、时间常数/增益等仍为未标定接口|not-executed / NOT_VALIDATED|数值容差通过证明动物接口正确|
|稳定站立/行走|固定命令三钟首次90%门槛34.4/33.85/33.725ms；P1正常34.4ms下沉；本轮无新行为实验|历史证据未站稳；本轮not-executed / NOT_ACHIEVED|保持50ms可计算就是站立；扰动响应就是恢复|

原规范正下限与零下限派生分支须分开，参考`/run/media/sensen/Data2/cell_wound_prototype/STRICT_ACCEPTANCE_BOUNDARY.zh-CN.md`:9–24；本报告不把旧原规范零ctrl正act失败归给当前零下限派生分支。

## 4. 冻结证据路径

1. 固定输入：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/p1_fixed_input_clock_replay/20261013_234617_bd6f9aa2`。
2. 固定命令身体：`/run/media/sensen/Data2/cell_wound_prototype/outputs/p2_body_clock/20261013_234422_9814ef63`。
3. 采样分离：`/run/media/sensen/Data2/cell_wound_prototype/outputs/p2_event_sampling/20261014_000748_5a68d98c`。
4. 状态trace：`/run/media/sensen/Data2/cell_wound_prototype/outputs/p2_state_trace/20261014_002347_4efdb3df`。
5. 方法比较：`/run/media/sensen/Data2/cell_wound_prototype/outputs/p2_voltage_method_comparison/20261014_003713_eba8a3fc`。
6. 历史耦合：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/projected_pose_convergence/20261013_072227/convergence_results.json` 与 `/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/projected_pose_convergence/20261013_072227/convergence_traces.npz`。

JSON包含以上原始结果、源码/合同、报告哈希及只读核验，来自既有冻结包的PASS与本轮新核验分别标记。实际打开既有 `/run/media/sensen/Data2/cell_wound_prototype/outputs/p2_event_sampling/20261014_000748_5a68d98c/sampling.png`：左侧同事件误差随细化减小，右侧不同事件误差大；图标题明确NOT convergence。不生成新图替代旧证据。

## 5. 最小下一动作，不继续无穷审计

1. **无需再跑同一固定事件解码实验**：已有`sampling.npz`存全部base/half/quarter事件流×三网格×连续解析的逐池速率/命令。优先复用这些数组列出最大差对应池、时间戳、未裁剪率和饱和状态，用既有连续公式做针对性端点核验；不要重复聚合图或再改电压方法名。此处的1e-10Hz/1e-12仅验证重建，不是新收敛验收。
2. 随即转为**接口标定输入清单**：冻结当前有符号逐群电流向量、单位、方向来源/假设、饱和范围、输出池分母和别名；列出需要的动物感觉响应/运动募集/延迟数据与留出验证条件。不能凭软件找到“正确符号/增益”。这是S4前置，不称生物PASS。
3. 如果仍要新的数值实验，仅允许先另写有界协议、选一个已定位运动池/事件的时间延迟反事实（其它条件固定，结果只作定位）。`/run/media/sensen/Data2/cell_wound_prototype/P2_STATE_TRACE_PROTOCOL.zh-CN.md`:28–30已说明需要此额外声明；本轮未执行。不重新跑整网三钟、搜索参数、延长倒下轨迹或宣称该反事实为原模型行为。

S1保留“部分工程已验证、阶段未通过”：当前固定输入整体收敛无通过证据，适用历史合同存在真实FAIL，当前完整耦合合同未重跑。此轮的成果是明确验收边界及停止重复诊断，不是修复成品。

## 6. 本轮交付与验证记录

- 项目报告：`/run/media/sensen/Data2/cell_wound_prototype/P2_ACCEPTANCE_MATRIX.zh-CN.md`。
- 桌面数据：`/home/sensen/Desktop/engine修复验收/P2-验收矩阵.json`。
- 桌面报告：`/home/sensen/Desktop/engine修复验收/P2-验收矩阵报告.md`。
- 有界只读核验/生成工具：`/run/media/sensen/Data2/cell_wound_prototype/tools_p2_acceptance_matrix.py`。
- 工具命令：`/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python /run/media/sensen/Data2/cell_wound_prototype/tools_p2_acceptance_matrix.py`。实际退出码与逐项结果写入JSON；检查失败即保留FAIL，不修改原始输入或容差。
- 初版核验退出1：工具误把ZIP的`sources/000000`匿名成员当路径，造成假阴性；实际检查`manifest.json`后改为声明path→id→SHA256映射，未改原包。初版报告及JSON保存为 `/home/sensen/Desktop/engine修复验收/P2-验收矩阵报告-初版ZIP路径核验失败.md` 与 `/home/sensen/Desktop/engine修复验收/P2-验收矩阵-初版ZIP路径核验失败.json`，不删除或覆盖历史。
- 不运行单元测试/全量回归，不改变其最新85/97历史统计；父代理负责tracker与goal更新。本工具不导入或执行归档代码、不启动模拟。
