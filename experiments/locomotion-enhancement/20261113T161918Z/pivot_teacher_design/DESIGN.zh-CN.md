# 停步／原地转向教师最小代码设计：待父审阅

本目录 ONLYCODE；未导入数值库、未读NPZ数组、未创建身体、未运行网络或物理。父已报告旧789f实际ABI检查通过，本设计不修改789f，也不把算法教师称为新脑原地转向。

当前新增API是 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/pivot_teacher_design/teacher.py` 的 `make_teacher(c, condition, *, order_indices)`、`body_length_from_compiled_mesh(c)`；条件固定stop/left_pivot/right_pivot，参数只有一个配置。尚无完整runner，父审阅后再准备最小实际身体诊断，不应直接启动训练。

## 源码依据与坐标修正

`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/static_teacher.py:44-49,75-84,125-163` 的StaticTeacher默认私有FK根固定在构造初姿，XY和Jacobian均用initial_rot/initial_origin。直接把旧sample改成切向足目标会在身体实际转动后继续用初始姿态求解。新局部子类每个action仅将私有q_des非执行器位置同步实际qpos、保持42个关节q_des连续，并更新初始变换为本次实际Thorax变换；真实c.d不写入。继承原IK、共同支撑分配、有限PD输出逻辑；每次前后精确检查真实qpos/qvel/xfrc/qfrc未变。

同源 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/static_teacher.py:165-250` 用实际接触点雅可比共同解六足竖直载荷分配，再按参考净空抑制摆动腿；support fractions只是求解变量，不直接施力。252-301的action把私有IK目标和tau_ff转为42归一化增量率。本候选完整继承这些代码，不复制物理循环、不改Core。该求解只含竖直准静态载荷，未建模水平摩擦扭矩；五腿计划支撑、COM在支撑多边形内不能证明实际pivot可行。

## 目标构造

连续旧流程1s settle+4.8s旧walkingteacher，到5.8s才用实际完整状态构造新teacher（mode=hold，避免另读测量NPZ）。步序从旧warmteacher.reference.order_indices复制，固定六腿顺序，不引入新动物时序声明。新教师没有真实源果蝇pivot数据，全部标工程目标。

新teacher捕获实际Thorax原点O、初始方向R0与六足世界锚点。请求目标yaw为 ±.15rad/s，以1s quintic平滑启动；期望方向 `R_des(t)=Rz(omega*integral_ramp(t))*R0`。原点目标保持O；它是教师目标，不写rootqpos/root速度/根外力。

支撑足worldanchors保持不动；足目标表达为 `R_des(t).T*(worldanchor-O)`。私有FK使用actualThorax方向，方向误差因而只通过关节动作与真实足地作用产生反力矩：当actual方向等于desired方向且原点无漂移时支撑目标回到固定worldanchor；不能用actual方向直接转换同一锚点后就声称有转向驱动。小角度下足端身体目标导数为 `-omega*J*p_i`，对应主动身体yaw；重新落脚放置在未来旋转方向（`+omega*J*p_i`），符号两者不同。

每.8s只摆一腿，XY用quintic插值，竖向 `.12*16u²(1-u)²`；落脚点是期望landing方向再向前预测2s（其余五个.8s支撑的半段），相当于固定工程的旋转落脚前置角。落脚时锚点提交为计划endpoint，绝不移动真实足；实际未接触必须在hypothetical support/actualcontact记录中显式失败。没有按实际接触重试的状态机、没有额外参数搜索。这个前置角可能过大/超IK范围，首批用于发现可行性与饱和，不能假设会成功。

stop条件固定六足锚点和零期望yaw、不摆腿。真实支撑是否保持仍需完整身体验证；它不是把所有动作乘0。pivot参考不提供虚假的零速度字段，原StaticTeacher以实际连续q_des有限差分生成qdot_des。

## 体长与测量

实际XML `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-triad-pad/20261110T010000Z/MODEL_TRIAD_x200.xml:25-31` 给出Thorax、A1A2/A3/A4/A5/A6、Head各自mesh与scale1000；115-137有腹部与头的层级/位移。不能拿global statistic extent或腿翼外框当体长，也不能仅用Head/尾部body origin距离代替表面长度。

新增body_length_from_compiled_mesh只选这些身体的准确命名geom，从编译mesh_vert经实际geom_xmat/geom_xpos投到共同初始站姿Thorax x轴，取max-min；排除腿、翅、眼、触角。记录每个geom/meshID、顶点数和投影区间。父在共同settle完成、旧warm前调用一次冻结值；目前没有创建模型，因此长度数值未知。必须实际检查compiledmesh对应正确type与变换，禁止把函数存在当尺寸已核验。原生单位，SI NOT_ESTABLISHED。

yaw从实际Thorax旋转矩阵每ms取atan2并unwrap，与目标yaw分开。漂移给实际COM相对候选交接时COM的最大XY偏移/体长，并另存最终偏移、累计路径、身体前侧速度。支持门按原.90、COM≥1.20、nativeNF=0和外力=0；pivot目标建议9.6s有符号yaw≥45°、最大漂移≤.5体长；stop不要求换腿周期。stop尚未冻结瞬时速度阈值，首轮仅报告，不能单凭漂移宣布站停通过。原forward≥.05与净位移滑移按旧定义计算并保留失败，单独解释原地不适用；不得借新任务追改旧五门或linewalking回归。

pivot换足门应检查实际卸载→抬起→重新承载，不能沿用要求向前advance的旧行走CycleTracker直接判零速转向失败；同时保存旧周期结果。逐腿接触切向速度/体长每秒、载荷分母单列，不用趋零COM位移作pivot滑移分母。

## 父审阅后的最小实际下一步

固定3条件/一个配置，共最大3,696,000native步，全部保留。先在小runner里执行相同身份/初态/步序的旧预热，5.8s连续切新算法教师9.6s，保存完整1ms/native峰值与5ms参考/IK/支撑/饱和诊断；无NN/无optimizer。出现数值异常立即停止，连续100ms COM<.8或NF>.5W按既有保护停止并保留失败。完成独立保存审核后才能判断教师是否够格给ANN学习，不靠增加epoch掩盖身体教师失败。

父代码审阅应首先核对desiredframe/actualshadow的旋转符号、worldanchor提交与实际接触分离、头胸腹compiledmesh体长、私有根同步没有实际状态写入。需要改动在本目录新版本记录，不改旧科学源。本目录没有部署CPG，教师时钟/目标yaw/worldanchor仅供训练数据和可行性诊断；未来ANN部署输入仍是实际感觉+经来源DN2的独立两任务量，不接本教师的时钟/世界锚点/目标根轨迹。
