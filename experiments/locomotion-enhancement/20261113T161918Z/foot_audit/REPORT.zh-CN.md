# 当前完整身体足端机械接触、爪钩与黏附只读审计

审计范围：P1/P2/S2/S5 运动增强前置证据；对象为 NECK_CURRENT_STATE 指定的 20261113T064313Z 身体实现。仅静态源码、XML、既有 JSON、资源路径／大小／SHA256 与 STL 头部元数据；未导入项目模块、Torch、MuJoCo、NumPy，未打开 NPZ 数组、未构造模型、未物理积分、未安装。只新建本目录的报告和来源清单，旧科学源与全局 tracker 均未修改（交由父代理统一记账）。

## 核心结论

1. **当前足端是每脚三颗球，共18颗球的工程接触垫。没有实现可挂住粗糙突起的机械爪钩，也没有实现真实 pulvilli 的毛状接触／分泌液／附着与剥离机制。** 名字含 pad 不构成生物足垫验证。
2. 六个 Tarsus5 外观 STL 对应的 mesh geom 都是 `contype="0" conaffinity="0"`，当前不参与自动碰撞；模型无显式 contact pair 绕过此设定。即使外观有弯钩也不会提供挂钩力。未读取 STL 顶点，不能把其原始形态记作已证明的双爪或凹钩。
3. **凸包没有可挂住突起的内凹钩口。** 对整个弯爪取单一凸包会填平内凹；单纯打开一个爪形 mesh 的碰撞不能据此承诺 hook。当前足端球本身也是凸形。此几何事实不需要物理运行；本次没有计算原始 mesh 的凸包、凹度或尺寸。
4. 当前实际14条件均关闭旧工程 adhesion。已有 `adhesion_on` 实现是接触触发、按体重比例的世界向下外加力，不是机械钩，也不是已标定的果蝇黏附。提高摩擦仅改变接触摩擦阻力，不能产生几何钩口或分离时的真实附着。
5. **粗糙地形前有测量口径阻塞**：足载、非足触地和滑移只计与 `plane` 的配对，且用世界 z 力、世界 xy 速度／世界 z 高度。新增 box、mesh、heightfield 地形可能物理上碰撞却在验收中漏计；必须先新建并声明地形集合和局部法向／切向指标，保留旧平面门。

## 入口与实际模型身份

|证据|绝对路径与行号|含义|
|---|---|---|
|当前实现选择|`/run/media/sensen/Data2/cell_wound_prototype/NECK_CURRENT_STATE.json:6`|指向 20261113T064313Z；不能凭目录名猜最新身体。|
|064313 runtime|`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T064313Z/sensory_body_repair/runtime.py:4`、`:5`、`:7`|继承045302 runtime，只重绑新感觉／融合权重和元数据。|
|064313 runner继承|`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T064313Z/sensory_body_repair/runner.py:2`；`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T064313Z/sensory_body_repair/adapter.py:4`、`:8`、`:11`|执行旧045302 runner的相同AST并重绑runtime命名空间。|
|045302→033622|`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T045302Z/sensory_body/runtime.py:4`、`:7`、`:12`|BASE由033622传递。|
|实际BASE与pin|`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T033622Z/visual_body/runtime.py:6`、`:52`|BASE固定为152714；pin验证 student_wave 实际路径。|
|sw.Core身份|`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/student_wave.py:21`|Core从同目录physics导入。|
|Core继承|`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/physics.py:24`、`:30`、`:43`、`:49`|加载快照repair_core；包装器默认adhesion=True，但当前调用者覆盖为False。|
|实际XML|`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/sources/repair_core.py:35`、`:87`|加载 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-triad-pad/20261110T010000Z/MODEL_TRIAD_x200.xml`。|
|模型编译及外观|`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/sources/incremental_actuator.py:51`、`:60`、`:64`、`:67`|FlyGym解析资源并编译；把71个mesh alpha设1只改变可见性，不开启碰撞。|

静态XML核对：71个mesh asset、71个mesh geom、18个sphere、1个plane，共90个world geom；0个显式contact pair，0个adhesion actuator。有限执行器验证契约在 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/sources/incremental_actuator.py:34` 要求49/48/42/71、无equality；本次未编译重验。XML timestep为12.5微秒、gravity为`0 0 -9801`（XML第5行）；绝对SI仍NOT_ESTABLISHED。

注意命名冲突：repair_core.py第34行的另一个 `BASE` 是历史 `MODEL_MEASURED_STAND.npz` 路径，并非当前runtime的Python BASE或实际PAD_XML。本次没有读取该数组。

## 足端geom、外观资源与实际碰撞

以下行号均在 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-triad-pad/20261110T010000Z/MODEL_TRIAD_x200.xml`。

|足|外观mesh geom关闭碰撞|三球pad开启碰撞|
|---|---|---|
|LF|248|249、250、251|
|LH|285|286、287、288|
|LM|324|325、326、327|
|RF|373|374、375、376|
|RH|409|410、411、412|
|RM|448|449、450、451|

每个pad均为sphere、radius=0.008、`contype=2 conaffinity=1 solref=".002 1"`，没有独立关节或附着控制。LF的`m_tarsus5_marker`在第252行，仅为局部原点site；site没有接触体积。其他跗节mesh仍可能碰地，例如LF Tarsus1—4在第232、236、240、244行均开启碰撞，Tibia在第226行开启；但这些geom不在pad前缀表内，会被现有平面审计记为nonfoot，不能说整条跗节都属于受载脚垫。

实际资源路径由 `/run/media/sensen/Data2/cell_wound_prototype/venv_body/lib/python3.12/site-packages/flygym/compose/fly/musculoskeletal.py:142`—`:164` 决定：优先XML目录相对文件，不存在时使用缓存。六个足端相对文件均不存在，本轮确认缓存文件存在，路径如下（只读二进制STL前84字节与大小，未读取顶点）：

|绝对路径|字节|STL头声明三角形数|
|---|---:|---:|
|`/home/sensen/.cache/flygym_assets/neuromechfly_musculoskeletal_meshes_20260623a/LFTarsus5.stl`|18084|360|
|`/home/sensen/.cache/flygym_assets/neuromechfly_musculoskeletal_meshes_20260623a/LHTarsus5.stl`|147084|2940|
|`/home/sensen/.cache/flygym_assets/neuromechfly_musculoskeletal_meshes_20260623a/LMTarsus5.stl`|49384|986|
|`/home/sensen/.cache/flygym_assets/neuromechfly_musculoskeletal_meshes_20260623a/RFTarsus5.stl`|18084|360|
|`/home/sensen/.cache/flygym_assets/neuromechfly_musculoskeletal_meshes_20260623a/RHTarsus5.stl`|147084|2940|
|`/home/sensen/.cache/flygym_assets/neuromechfly_musculoskeletal_meshes_20260623a/RMTarsus5.stl`|49384|986|

每个文件均符合84+50×三角形数的binary STL长度。完整71资源均已读取文件字节计算SHA作为元数据核查，未作几何运算；本目录清单重点固定六个末端STL及审计源码，既有完整资源绑定见 `/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T064313Z/sensory_body_repair/trials/V_empty/IDENTITY.json:914` 等。

三球的工程构造来源是 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-triad-pad/20261110T010000Z/build_triad.py:17`—`:33`：取历史Tarsus5 mesh最低顶点，把三球底面放到该高度，默认spread=0.006、radius=0.008。它不是根据双爪尖和两片pulvilli分别分割／测量出来的构造。当前XML的三球坐标符合此默认spread。**不能误用同目录TRIAD_LEDGER作为当前XML精确台账**：`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-triad-pad/20261110T010000Z/TRIAD_LEDGER.json:4` 实际指向另一个 `MODEL_TRIAD_s012_x200.xml`，spread=0.012（第7行），不是runtime加载的文件；旧记录保留。

项目另有明确命名爪形视觉资产，例如 `/run/media/sensen/Data2/cell_wound_prototype/upstream/flybody/tarsal_claw_T1_left_brown.obj`。其XML `/run/media/sensen/Data2/cell_wound_prototype/upstream/flybody/fruitfly.xml:14` 默认外观mesh关闭碰撞；第494行是爪形外观，第495行真正碰撞是capsule，第496行touch site也是capsule；另外五脚同样结构（第539、584、629、674、719行附近）。第892—897行另列adhesion actuators。因此这个备用模型也不能仅凭“有爪mesh”证明机械凹钩；它不是当前身体模型，不能把其尺寸／单位／附着器默默移植到当前模型。

## condim、摩擦、convex hull的准确含义

当前XML地面第100行明确condim=3；默认geom第13行没有friction覆盖；repair_core.py第88行把所有geom的滑动摩擦设为mu，默认MU=1.5（第42行）。

本轮利用既有V_empty身份JSON里的**数组摘要**核对元数据：用stdlib struct构造90个int32值3的字节流、90组float64 `[1.5,0.005,0.0001]` 的字节流并SHA256，两者与归档摘要逐字一致；没有读取任何实际数组或NPZ。证据为 `/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T064313Z/sensory_body_repair/trials/V_empty/IDENTITY.json:15017`、`:15046`。其SHA分别为 `f491c7dee1fff79e2b972ef747bdf6ad46cf6904cb8e5dae670cf10e0f5c9b6d`、`9e4772c767a531c27d15ca4193cfb803eb895562521a315ed9c92237336398bf`。这给出该归档实际全部90 geom的condim=3、摩擦=(slide1.5,spin0.005,roll0.0001)的摘要一致证据，不是新编译验收。spin/roll的参数存在，不代表condim=3接触启用了独立自旋／滚动摩擦约束；condim=3是法向加两切向。

本地MuJoCo头文件 `/run/media/sensen/Data2/cell_wound_prototype/venv_body/lib/python3.12/site-packages/mujoco/include/mujoco/mjmodel.h:456`、`:472` 分别说明contact dimensionality及slide/spin/roll含义，第634行说明mesh_graph为convex graph data。普通mesh凸碰撞不能保留整体内凹钩口；把钩拆成多个凸子体，组合边界才可能保留开口与内凹。当前足端没有这种子体链、双钩或钩尖碰撞命名，也没有SDF插件或凹钩接触实现。原始STL是否存在肉眼可见爪形／内凹仍未由本轮顶点／影像验证；“碰撞未实现”不依赖这个未知项。

## foot测量不是实际单一脚尖接触点

`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/sources/repair_core.py:100`—`:107` 只把名字以 `{leg}Tarsus5_pad` 开头的geom映射为足；第146—168行遍历真实MuJoCo contacts并累计每腿足载和载荷加权接触速度；第175—179行以三球最低底面作padz、以**三球中心算术平均**作foot，再转换到胸部坐标系。

因此foot是简化的足端参考标记，不是某一爪尖、实际接触点或压力中心；padz是世界水平面间隙近似，不能直接当粗糙地形净空。接触本身不是人为固定单一点：MuJoCo产生的contact.pos被真实使用，多个球可以有多个接触。`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/physics.py:227`—`:252` 还记录geom pair、contact.pos、frame、力和接触点速度，说明不能把“参考点简化”误说成“引擎无真实接触”。

主要地形盲点：repair_core.py第149行与physics.py第235行只认PLANE，repair_core.py第154行取世界z载荷、第162行取xy速度；physics.py第244行同取world-z。有侧壁／斜面／钩住突起时，水平反力和局部切向滑移恰恰关键；沿用旧门会漏掉力和接触，不具备可直接验收粗糙地形的语义。

## 旧adhesion_on：明确外力，不是钩／生物足垫

实际调用 `/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T045302Z/sensory_body/runner.py:15` 为 `sw.Core(adhesion=False,recorder=recorder)`，064313 adapter保留此调用。已只读解析14个trial的IDENTITY.json，全部adhesion.enabled=false；V_empty证据在 `/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T064313Z/sensory_body_repair/trials/V_empty/IDENTITY.json:26802`，RESULT第334行extra_force_max=0、第408行extra_forces=false。本轮不重新计算轨迹来替代原审核。

若开启旧选项，`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/sources/repair_core.py:206`—`:236` 的行为是：先清空xfrc_applied；仅对载荷>0.1×WEIGHT且没有pad正z上升速度的腿施力；每条合格腿总幅度0.25×WEIGHT，平均分给与plane接触点，在世界-z施加，加入该body外力和力矩。`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/physics.py:117`—`:137` 修正力矩到body COM并记录，`:146`—`:148` 每40原生步刷新一次，即0.5ms。

这会增大地面压载及可得摩擦，但没有钩口几何、粗糙突起身份、毛束接触面积、分泌液桥、方向依赖贴合／剥离或墙面局部法向。它不是力场自动随表面法向变化的生物黏附。提高mu或condim、启用该选项都必须作为不同物理模型明确登记，不能作为“补上爪钩”的暗改。

## 对果蝇脚钩问题与最小后续验证建议

生物问题应区分硬质pretarsal claws的几何挂接与pulvilli足垫的附着机制；两者不能互相代称。本轮受工具范围限制没有做网络文献检索，项目sources目录的Markdown搜索也未找到pulvilli／pretarsal机制证据，故**不把通用昆虫结构印象或FlyBody视觉资产当果蝇定量形态／材料／受力来源**。由父代理的生物来源审计补充果蝇物种、足位、成像尺度、爪尖／钩口曲率、pulvilli分区以及粗糙度／附着实验文献后，才能标“有来源”；目前这些参数是UNKNOWN。

建议下一最小实验按以下顺序预登记，全部是待实施建议，本轮0次物理：

1. **先把机械接触测量做对**：在新目录派生当前完整身体，保留71mesh、有限PD／42神经输出、重力与完整旧模型；明确terrain geom集合，记录所有足与非足接触，增加局部法向／切向力、实际contact.pos、接触点相对速度及最近地形间隙；旧平面工程门和新地形门分开。不能靠geom名字把胫节接触改名为足钩通过。
2. **先机械单元验证，后整身行为**：从可追溯形态得到一个足端双爪的坐标、开口和尖端尺寸；用少量capsule／凸多面体的**组合**表达带内凹的弯钩，固定在原Tarsus5局部坐标。不得用整个单一convex hull堵住钩口。保留旁边pulvilli占位为未实现；先做无黏附的硬钩，不借用旧世界-z附着外力。几何与材料缺口必须列为工程假设。
3. **同一微地形的四项反事实**：三球原型／有凹钩组合／把钩口填平的凸对照／反转钩朝向；摩擦、加载／牵拉条件相同，分别在光滑平面和尺寸已登记的突起／棱边上验证。首要证据是突起进入钩口后的接触对、侧向反力／负载路径与可逆释放；仅滑移降低不证明挂钩。低摩擦附加对照可分离摩擦与几何阻挡，但有限摩擦值和结果都要如实报告。
4. **显式禁止隐藏吸附**：全程adhesion=False、xfrc_applied/qfrc_applied=0、无weld／equality绑定、无脚点位置写入、无接触后开关“锁脚”。机械测试夹具／牵拉源若需要外力，须显式列为台架输入并单独记账，不能混入自主行走；完整身体运动阶段保持0外部辅助。机械反力来自碰撞约束和声明的足端几何。
5. **验证单元后再验证运动增强**：原地转向单列yaw变化、COM平移、全足切向滑移、完整周期和非足接触；粗糙地形单列跨越／跌落／夹卡与释放，不把现有转弯当原地旋转。脑控制仍经真实颈部接口，不能因新接触模块绕过。未完成物理反事实前不能宣称机械钩必能解决当前RH周期或转弯滑移失败。

本报告支持“当前足端机械实现有明确缺口、粗糙地形审计须先扩展”的结论；不证明身体极限、不承诺爪钩一定改善行为、不继承旧平面行走或视觉嗅觉通过为新运动能力。

## 本轮验证与产物

元数据检查命令均为系统python3的stdlib独立脚本，退出码均0；未执行任何项目科学源码。已核对XML计数、71资源路径／文件SHA、六足端STL头长度、14个adhesion关闭标记，以及V_empty两项compiled-array元数据摘要；报告与清单完成后再独立JSON解析、来源SHA和行号存在性核验。

- 报告：`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/foot_audit/REPORT.zh-CN.md`
- 来源SHA清单：`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/foot_audit/source_hashes.json`
