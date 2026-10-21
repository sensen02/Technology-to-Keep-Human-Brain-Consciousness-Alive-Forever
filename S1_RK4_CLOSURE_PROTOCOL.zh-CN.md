# 显式RK4短时闭环：执行前协议

手动研究非成品，P1/S1/S2。当前仅契约代码/单元测试/协议，**尚未运行新闭环**。不是站立验收，不重复原生四case。

## 身份与时钟

独立runner，显式zero-floor+native RK4+25us分支，原canonical不写、不更改旧DT_BODY全局变量。复用原projected_seated_qpos初态（工程搜索派生，非测量站姿）、dynprm原值、actlimited false/actearly false及原生重放预检。freshdata零qvel/act，一次初始化，不每步修正。神经TierConfig(background_mean_nA=0,background_sd_nA=0)，原manifest/感觉组/NT假设和We/Wi*.2等沿用并明记，非生物映射确认，不加gain/连接/电流制造动作。

body25us/neural.5ms/command5ms，整数比20/200与10；50ms每臂2000身体步/100神经步/10decoder consume。RK4 callback不设神经/译码，可无callback沿用已验证原生行为；任何非None全局control/act回调拒绝，不覆盖。新增RK4ClosureClock只算时间/来源，不推进模拟。

每区间k：从身体t_k观察、编码真实current并保持；神经10步得到本区间spikes；身体只施加pending（源k-1，valid_from=t_k），200步；区间末decoder.consume(spikes,start=t_k,end=t_{k+1})一次，生成下一区间pending，不能提前施加。虽顺序先计算神经再body，pending不得读取当前spikes。源-1只初始零命令，正常后续必须k-1。k4在5ms边界仍旧命令，下一body step才换。body/net实际时钟每区间断言1e-10，无重置。

## 六臂与匹配条件

正常→精确重复→匹配命令重放→感觉切断→运动切断→输出静默，各50ms。全部无外推/人为touch/背景电流，统一初态/配置；不能重复旧normal无push而cut有push的混杂。正常产生新分支的命令重放，禁止拿旧Euler命令当新normal。每臂fresh网络/decoder/data。

感觉只复用P1已声明claw/hook通道，未知禁用；normal有身体输入，sensorycut清实际input并记录；motorcut从t>=15ms起将applied直接0，区间末connected=False清decoder记忆但不清物理act或神经；silence只将输出给decoder的事件清0，不冒称根神经元silence。三个cut必须和同无push normal比较，保存raw及emitted事件区别。重放是明确控制臂，命令来自本轮正常而非新神经因果。

## 记录与工程门

每区间保存实际sensory local index及nA、raw/emitted motor probe .5ms事件、全部神经count（聚合不可重建精细filter时须保留完整spike位压缩）、pool counts/filter rates/decoded_next/pending source/validfrom/applied命令；每body步保存qpos/qvel/act/ctrl/force、零外载、warning与时间，moment若资源不足应在运行前明确只命令边界采，不能伪造全端点力解释。至少保存每5ms完整moment和总广义力分项。原生preflight/依赖hash/71mesh/tier身份/实际source_snapshot。

工程必须全有限/warning0、inputhash不变、clock/source/command保持成立、act[-1e-12,1+1e-12]、repeat状态/命令/事件精确、normal与replay身体精确。重放精确仅在确定同输入/force操作时宣称。cut零命令与解码记忆规则精确验证；禁止要求cut清活化。每条新command都须可由已保存细事件独立重建，时序/归一化/共享pool不变。

神经命令是否非零、切断是否改变动作是待测指标，不是通过要求，允许零效应；不加触觉/噪声或调译码gain补救。若normal无事件就报告本输入下未招募，不称失败软件或擅自扩脑。act范围通过不自动继承旧输入下max误差.01证明；对本轮实际normal命令可离线分段参考核activation精度，若新输入超原范围验证需拒绝数值资格并保留。

## 预算/停止/边界

独立进程guarded_run沿用现有memory_guard与峰值上限，不绕过；六臂50ms最多12000body step/600neural step，外部120s硬上限，磁盘<=500MB。若预算不能完成，先预声明减少臂的分阶段执行，不能运行中偷偷删对照。首次normal出现非有限/warning/时钟/act越域即存FAIL停止，不继续五臂；repeat/replay不精确亦停止，不调阈值。旧分支数据不改。

结果只有短时闭环工程因果链，非神经/身体各自收敛、生物有效、恢复稳定或站立；当前A门静态支撑/初态来源问题仍在，禁止用非稳定背景下的差异称平衡恢复。交付raw/report/图（实际查看）/独立核验和分支声明。

下一“实施/继续”可实现此独立runner，但本轮未启动新神经或身体仿真。若需更长时长/触觉电流/支架或身体参数更改须另确认，不纳入这份协议。
