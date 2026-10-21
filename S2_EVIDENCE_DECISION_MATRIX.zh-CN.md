# S2/P2 证据结论矩阵与下一步决策

手动研究原型，不是成品。本轮纯文档归并与只读证据核查，无模型加载/forward/step、新轨迹、参数搜索或生产改动。表内数字是历史已交付实验，不是本轮重跑。工程验证、动态因果、生物标定、稳定行为分别列，不用通过测试数代替科学验收。

## 一、总体判定

S2已交付一组有来源、可复核的身体独立诊断；**阶段尚未整体通过**。已解释Euler目标活化超调、全局状态标签误筛选与质量存储误用；已观测固定刺激响应、接触时钟差以及初端点stationarity。尚未证明全程身体收敛、接触峰差的物理原因、绝对肌力生物标定或稳定支撑。

v2初端点g小不等于10ms动态端点g小，更不等于静止；零控制/零活化不等于零肌力。人工双球只作接口覆盖，不算果蝇行为。

## 二、证据矩阵

| ID | 问题/已取得证据 | 适用范围与当前结论 | 不能推出 | 来源编号 |
|---|---|---|---|---|
| E01 | 184固定刺激案例；90目标ctrl .1，双floor各92，10ms | 历史投影就座初态；moment转置乘force重建actuator广义力一致；native ctrl0被原生.0001夹紧；zero floor只改编译下限 | 原keyframe有效、零肌力、生物力矩或神经控制 | R01 |
| E02 | Euler100us命令.1首步act .2 | 激活动力学与dt/tau尺度可解释超调；不是生物招募；保留原参数与XML | 完整身体异常均由活化积分导致、生产应直接切RK4 | R01,R02,R10,R11 |
| E03 | 实际RK4 12例，10/12满足人工标量maxerror<=.001；pulse100us两floor失败 | target0独立标量活化参考；25/12.5us接触8案例在共同端点均满足该筛查 | 全90执行器已生物校准、接触/姿态/系统收敛 | R02,R03 |
| E04 | 25vs12.5us四分支pairmax差约.00116.. .00120模型力；typed姿态差已记录 | 10ms/target0/native-zero pulse-decay局部时钟敏感性；12.5us非真值，无物理通过阈值 | 充分收敛、足够支撑、接触力误差已小于生物容许量 | R03 |
| E05 | 早峰1.9ms为floor0/LFTarsus5 geom26；晚峰8.2..8.225ms pair26或25 | 离线定位到几何pair和共同时间窗口；sum-vector跨pair相消，不是系统净支撑力 | 全接触点身份稳定、峰由某一个接触切换导致 | R04 |
| E06 | native pulse raw接触峰窗口pair26三点，晚pair25一点；距离/局部力随时间变化 | 原始接触观测与pair合力可复算；状态直方图不是跟踪物理点，更不是最终solver分支真值 | 点数/全局state不变意味着接触稳定，相关性就是物理原因 | R05,R06 |
| E07 | 全局state1筛选得到旧e≈.812754；每case424行r>=0/f0；分段force差约1.22e-15/1.05e-15 | 两条native pulse动态档案：旧大e来自分支筛选误用，并非最终活跃行本构违例证据；848标签差异完整保留 | 动态档案直接保存过iefc_state；完整KKT已验证；.813是接触峰原因 | R06 |
| E08 | 实际3.9.0初forward row55 global1→observed island final0；人工实际2岛stats地址0/200 | 当前安装初端点直接观测global标签不等于最终state；非identity映射仅合成验证 | 历史所有端点已直接复现、通用稀疏/线程/monolithic适配全面通过 | R07 |
| E09 | v1误把CSR d.M给树序fullM导致非SPD而停止 | 采集代码错误；canonical1forward/artificial0，无数值现场保存，保留失败记录 | 真实惯性非SPD/物理模型或MuJoCo失效 | R08 |
| E10 | v2 qM/tree、M/CSR、fullM(qM)、mulM全基向量矩阵逐项一致，SPD通过 | 新canonical与人工各1forward、time0不积分；g_rows canonical平移1.42e-14/旋转3.55e-15模型单位，人工0/1.22e-16，REPORT_ONLY | 10ms全过程stationarity/静态站立/时钟收敛/完整KKT | R09 |
| E11 | v2 relative1出现于约1e-16量级项，scalar/BLAS相消结果略异 | 必须同时看绝对项与分母，零分母undefined；独立复核认为符合舍入范围 | 100%物理误差或solver失败率 | R09 |
| E12 | 肌力/力矩、控制下限、来源/单位分支可追踪 | 仍是生成规范standing派生模型、模型力/力矩；活化/衰减时间常数的生物依据及绝对单位标定未知 | 生物标定、完整严格神经控制链或稳定站立/自主行走 | R01,R03,R09 |

## 三、动态M采集是否必要：本轮决策

**不立即追加全10ms逐端点质量矩阵采集，也不继续重复初态forward。** 理由：大e疑点已由分支语义解释，v2质量接口已在限定初端点交叉验证；再次证明这些不直接回答早/晚接触差峰原因。现无预声明用途相关支撑阈值，用更多小残差不能制造收敛结论。

就本支线，下一次动态M优先限于以下明确问题（并非否认M的其他用途，也不以生物支撑阈值未知阻止数值异常调查）：验证既定早/晚峰窗口中，联合stationarity是否存在超过浮点累积/既定工程解释范围的异常，从而排查求解终止/记录混时。它不能单独解释接触几何、积分截断、肌力路径或峰差因果。必须在新运行同端点保存，禁止用初态M拼历史force。

### 竞争解释与最小可区分证据

| 候选解释 | 现有支持/限制 | 最小下一证据 | 何时停止 |
|---|---|---|---|
| 接触力解码/坐标或全局state误用 | force重建及分段式吻合，state问题已解释 | 复用现有档案，不新增重复实验 | 不再把旧.813作为失败论据 |
| 动态求解stationarity异常/混时 | 动态缺M；初端点小残差不能外推 | 仅在已声明峰邻域保存同forward M/smooth/final state与全部项，保留solver终止统计/舍入比较 | 若窗口g也在舍入解释内，将该假说降级，不继续靠加迭代/改容差制造表现 |
| 时钟累计状态/接触几何差异 | qvel/typed pose/dist/force已有同步差异，未因果分离 | 先离线列出控制切换/接触存在/几何与力变化的事件时间表，最终分支只能由r/正确映射识别，不能用global histogram | 若只有相关，仍写unknown；不硬命名根因 |
| 生物参数/支撑指标不适用 | 未有绝对力标定和用途阈值 | 明确单位与来源，选有生物依据的负载/接触测量及留出验收 | 缺数据时列未知，不从“站得住”搜索参数反推标定 |

## 四、下一最小动作（不在本轮执行）

**停止S2接触峰描述支线，转向S4来源/单位标定缺口。** 独立审查进一步读取 `/home/sensen/Desktop/engine修复验收/S2-raw-contact-20261014T204339Z-7ba64625/report.json`，发现其机器档案已经包含早[1.5,2.3]ms、晚[7.7,8.7]ms两时钟完整逐端点pair时间表（力差/点数/dist/type/global histogram等），并非只报告展示的±2样本。R06另有全400端点r分段和848差异CSV，因此再造完整窗口表实质重复，当前未找到值得补联表的具体分支/力异常。既有global histogram须按R06理解为非最终标签；1ms控制切换在窗口前，不能靠追加该背景区分原因。现有描述不能区分几何/积分/求解因果，这不是“没有因果问题”的证明。

下一有界动作：读取已存在S4接口标定/真实calcium/留出观测证据，选一个阻止生物解释的来源或单位缺口，明确可取得数据与不能映射的量（calcium不能直接当Hz/nA或肌肉活化）；先交付来源证据表，不做参数搜索、数据虚配或新身体轨迹。本轮不执行S4新工作。

若上述表提出具体动态联合残差问题，再另声明两时钟固定协议，并只加必要窗口的M观测；本矩阵不自动授权新轨迹。若未发现可区分问题，停止此S2诊断支线，回到S4来源标定或S3链路缺口，而非无限细化。

## 五、证据索引（完整路径）

- R01 `/home/sensen/Desktop/engine修复验收/S2-固定刺激诊断报告.md`
- R02 `/home/sensen/Desktop/engine修复验收/S2-mujoco-rk4-final/实际RK4诊断报告.md`
- R03 `/home/sensen/Desktop/engine修复验收/S2-contact-clock-protocol-complete-20261014-v1/协议补验中文报告.md`
- R04 `/home/sensen/Desktop/engine修复验收/S2-contact-peak-audit-20261014T191106Z-a50cc4fb/接触差峰离线审计报告.zh-CN.md`
- R05 `/home/sensen/Desktop/engine修复验收/S2-raw-contact-20261014T204339Z-7ba64625/原始接触观测报告.zh-CN.md`
- R06 `/home/sensen/Desktop/engine修复验收/S2-contact-state-audit-20261014-v2/状态标签语义更正报告.zh-CN.md`
- R07 `/home/sensen/Desktop/engine修复验收/S2-forward-adapter-20261014T215241Z/真实接口无积分核查报告.zh-CN.md`
- R08 `/home/sensen/Desktop/engine修复验收/S2-stationarity-real-20261014T222312Z/采集失败与布局核查报告.zh-CN.md`
- R09 `/home/sensen/Desktop/engine修复验收/S2-stationarity-v2-20261014T230905Z/真实同端点力平衡报告.zh-CN.md`
- R10 `/home/sensen/Desktop/engine修复验收/S2-activation-reference/独立活化参考报告.md`
- R11 `/home/sensen/Desktop/engine修复验收/S2-activation-clock/活化时钟诊断报告.md`（初导数2000/s，100us Euler更新.2，局部有效上升尺度50us；非恒定tau）
- R12 `/home/sensen/Desktop/engine修复验收/S2-dynamic-residual-20261014T211552Z-7fbf1130/同端点残差报告.zh-CN.md`（global-state筛选解释已由R06更正，原始动态字段仍冻结保留）

## 六、本轮验收边界

只核对报告与来源路径、当前冻结哈希和矩阵表述，不重跑历史实验或全量测试。旧报告中的“未测stationarity”按当时动态档案范围保留，新增初端点证据不改写旧报告。最新全量回归仍是历史85/97（8失败+4NO_TESTS），本轮不以此前45个聚焦测试冒充新执行。此文档及桌面副本、核查回执为新增成果；无新仿真图，不以旧图冒充新运行。
