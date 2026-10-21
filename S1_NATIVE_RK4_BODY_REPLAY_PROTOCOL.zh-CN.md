# 显式原生RK4身体重放资格协议（尚未执行）

手动研究原型非成品，S1/S2。仅固定命令body-only资格，不站立、不跑新神经/感觉闭环、不变canonical文件或主runner。

## 源码核对结论

MuJoCo3.9.0上游engine_forward.c的mj_step先mj_forward，随后mj_RungeKutta4使用已算F0；stage1/2/3设置qpos/qvel/act/time并mj_forwardSkip(NONE,1)，故通常四次动力学/控制回调，时刻t,t+h/2,t+h/2,t+h。skipsensor=1跳传感器，不跳控制回调、actuation、acceleration或constraint；能量路径不由skipsensor统一跳过（源码内注释概括不能替代实际条件分支）。积分末尾恢复原state/time，用B=(1/6,1/3,1/3,1/6)聚合导数交mj_advance。

engine_support.c的mj_nextActivation对muscle用act+h*act_dot；这里传入的是RK加权导数，不能看到注释otherwise Euler就误认RK活化只做Euler。actlimited才最终限幅；stage直写act本身不自动clamp。actearly=true会用预测act求force，故必须断言本模型actearly全false。mj_step1/2路径RK4降为Euler，禁止使用。mj_step后力/派生字段仍可能来自最后stage，不等于新积分端点；必须补显式mj_forward再采端点并分清阶段。

来源及SHA在桌面sources.json，版本匹配上游不是已安装so的构建证明；未来运行callback日志/act对独立参考验证才是本机实证。

## 初态/分支/身份

只复用已归档B-local-replay baseline的初始projected-seated qpos来源（原initial_pose.npz），初qvel/act0。freshmodel canonical hash核对，显式派生改变仅ctrl lower0、opt.integrator=RK4、opt.timestep=25us或12.5us；保留其他参数逐数组对照，原XML不写。不能把该工程初态当动物站姿。无外推/锁定/adhesion/CPG/PD/逐步姿态覆盖。

预检nu/na90、nq49/nv48，dy=muscle、每肌一act且地址有序、dynprm原值、actlimited false、actearly false、无sleep/disabled actuator/history delay/plugin/user actuator回调。若不满足保存FAIL停止，不自动禁用原功能迎合模型。

## 四案例及stage观察

执行顺序严格为25us baseline→25us精确重复→25us callback关闭等价性案例（不新cut）→12.5us细化；必须先确认观察hook无效应再运行细化。均50ms同10x90冻结applied命令；最多10000原生mj_step（2000+2000+4000+2000），预检只load/forward，不增加积分案例。

control callback限定只读记录：ctrl/time/act/qpos/qvel，不写model/data，不运行encoder/decoder/net，不推进时间。若发现预先已有callback则拒绝，而不是覆盖他人回调。仅本独立进程注册观察hook，finally恢复原callback（应None），无跨服务副作用。

每step前按整数step index锁定u_row=floor(step/steps_per_command)，ctrl整个step含k4不变。特别在5ms边界最后k4时刻虽等于下个命令边界，**仍保持本步旧命令**；下一step才切换。不按callback data.time重新索引，避免右端stage提前换命令。外载qfrc/xfrc每step清0并保存，神经计数恒0。

每step四stage日志应为t,t+h/2,t+h/2,t+h，容差1e-12，ctrl完全等本步row。补mj_forward的回调另标endpoint，不算第五积分stage；初始forward也单列。不得通过sensor callback代替stage观察，因为skip sensors。force/moment不能在control callback采成“当前stage力”（此时actuation尚未执行，字段可能上一stage）；本协议只保存callback输入stage状态，力只补forward后记录。若要力stage需另协议，不塞假同步数据。

## 时钟与采集

25us每5ms命令200步、.5ms神经标尺20步；12.5us400/40整数。这里只神经标尺断言不运行net。保存整数索引/真实time/command source及所有stage act/ctrl；每step补forward保存qpos/qvel/act/force/moment/qfrc各项、warning、qfrc_applied/xfrc_applied、contact count。原step次数与forward次数明确区分；四案例成功应有10000step、50004动力学forward evaluations（40000积分stage+10000endpoint+4initial，非全是Python mj_forward显式调用），hook-on三案例callback总40003，hook-off为0；预检forward另计。数据<=500MB，外部120s硬上限，超过停止，不增时长/删失败。

## 验收标准（执行前冻结）

1. 身份输入hash前后同，callback finally清理、stage时刻及ctrl保持通过；全有限、无native warning/auto reset，零外载有raw证据。
2. native RK4 act在对应经典独立trace max差<=1e-10；对DOP853 all-step max<=.01/RMS<=.001，全端点及所有stage act在[-1e-12,1+1e-12]。若不同必须停解释，不修改参/阈值。25us和12.5us均需通过。
3. 25us重复所有采集状态逐位相同；观察hook on/off端点qpos/qvel/act逐位相同，否则observer侵入FAIL，不能把新轨迹当合格。force moment恒等式rtol/atol1e-10：按moment_rownnz/rowadr/colind恢复90xnv稀疏moment矩阵，禁止reshape稀疏values。预检并归档无actuator-level gravcomp（jnt_actgravcomp与body_gravcomp组合）和无joint-level actuator force limit；若存在则保存FAIL并停止，不能忽略M.T*force后附加/钳位项。actuator自身force limit已在保存force中体现，不与joint-level混淆。
4. 只作25与12.5us身体差报告：在25us公共端点按root平移模型长度、root旋转用mj_differentiatePos/quaternion差、标量关节rad、速度线性/角度分别统计，contact模式差另列，不混单位max。不以两档差小就宣布整体收敛，本协议没有身体收敛资格阈值。
5. 不要求匹配旧100us Euler身体轨迹；改integrator/dt正是显式新数值分支，不能为匹配旧轨迹调参。不是B门新单池干预，没有新因果/稳定站立资格。

## 停止线/交付

本轮仅源码/协议，没有执行四案例。下一次实施需独立runner，不碰prepare_body/DT全局常量绕过旧断言。预检/首25us失败即停，重复或hook失败停，不再跑细化。归档源/依赖/71mesh/XML/初态/commands/native so hash、NPZ/JSON及图（实际查看）、独立离线核验；报告手动研究非成品。四case成功仅原生activation和stage保持资格，不自动启动神经闭环或姿态搜索。
