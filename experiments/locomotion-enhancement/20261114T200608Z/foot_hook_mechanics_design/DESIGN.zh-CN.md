# 果蝇足端爪钩：来源边界与独立机械台架设计

本轮只做来源与设计备忘录，**没有可执行物理脚本，也没有构造足端／全身模型**。硬爪尺寸、尖端半径、材料和动态姿态尚不足以建立物种标定模型；此时写入“真实果蝇爪”参数会超过证据。没有读取几何顶点、PT、NPY/NPZ数值payload，没有导入NumPy／Torch／MuJoCo、调用Core、渲染、积分或触及生物硬件。仅新增本目录的本文件与SOURCE_LEDGER.json，旧source／weights／tracker及父144训练不改。

## 当前证据怎样约束本设计

旧足端审计：`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/foot_audit/REPORT.zh-CN.md:7`—第11行；长腱补充：`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/foot_audit/BIOLOGICAL_SOURCE_ADDENDUM.md:5`—第13行。本轮仅增量读取其结论。

当前完整身体为71mesh、42有限PD、18颗工程足端球、滑动μ=1.5、adhesion关闭；原生长度与绝对SI换算未标定。原参数见 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/sources/repair_core.py:35`、第38—43行；LF球及外观geom见 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-triad-pad/20261110T010000Z/MODEL_TRIAD_x200.xml:248`—第251行。Tarsus5外观mesh关闭碰撞；**它是否实际含有可分辨的双爪／凹钩几何仍UNKNOWN**，本轮未读顶点，不能用文件名、STL头、材质或外观印象替代几何验证。单一凸包会填平凹钩口，开启一个mesh碰撞开关不等于实现几何互锁。

实际读取 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/CURRENT_STATE.json:17`、第21、24、26行：旧789f／完整336／来源Neck的匹配平地38.4s持续走，最后窗口六腿各2周期、action RMS约.08149，原slip约.4853；rough后窗动作约.000214、后续周期0。只删除六个boxes的比较同时改变光学与机械场景，**不能拆分两者因果，更不能据此断言缺爪导致静止**。新600组件四身体任务全部失败且右转翻倒也保留；本设计不修饰这些结果。父144同架构600→900训练是当前主线，本台架只作独立机制验证准备。

## 四类结构身份，不能互换

|结构|直接果蝇依据与具体支持|可宣称／不能宣称|
|---|---|---|
|terminal claws／ungues|[FlyBase FBbt:00004660](https://flybase.org/cgi-bin/cvreport.pl?id=FBbt:00004660)定义为末端成对、中空的几丁质钩，是硬化的表皮／角质层延伸；[Kimura 2020 Fig.1a](https://pmc.ncbi.nlm.nih.gov/articles/PMC7260203/)明确D. melanogaster pretarsus背侧成对爪|支持结构与材料类别；没有给出本模型所需尖端半径、三维曲线、杨氏模量或承载上限。|
|pulvilli／footpads|[Kimura 2020 Results“Footpads in Drosophila”](https://pmc.ncbi.nlm.nih.gov/articles/PMC7260203/)与[FlyBase FBbt:00004655](https://flybase.org/cgi-bin/cvreport.pl?id=FBbt:00004655)：每足两片腹侧足垫，每片约30根毛、6—7排，毛尖匙状；液体介导接触|这是附着垫，不能用硬爪或三球代称。细胞／毛结构缺陷会影响光滑面攀爬；不能直接推出具体黏附力定律。|
|empodium|[FlyBase FBbt:00004659](https://flybase.org/cgi-bin/cvreport.pl?id=FBbt:00004659)定义为两爪之间、足底中央的刺状突起；Kimura2020 Fig.1a用星号标出|独立的中央结构，不是第三个硬爪，也不是把两侧pulvilli合为一片的中央吸盘。其承载／感觉作用与几何尺寸在本轮未定量验证。|
|sex comb／性梳|[Kopp 2011的D. melanogaster专节与Fig.1](https://pmc.ncbi.nlm.nih.gov/articles/PMC3462374/)：雄性T1前足第一跗节ta1上由横排刚毛改造的结构；[Hurtado-Gonzales等2014](https://doi.org/10.1111/eth.12316)为交配抓握的一手研究线索，但全文获取被阻|它位于近端跗节，性别／足位特异，不能当所有六足末端爪，不能把交配抓握结果转为步行钩爪承载参数。这里只把可读综述当身份来源，一手功能全文尚未核验。|

[FlyBase图像报告FBim0000855](https://flybase.org/reports/FBim0000855)明确物种D. melanogaster，图为Bryant1978 p252 Fig16b的跖面结构示意，分别注释empodium／pulvillus／unguis等。它证明同一足端的身份区分，**示意图不提供本轮认可的标尺测量**。

长腱拓扑另有本轮实际读取的[FBbt:00058136](https://flybase.org/cgi-bin/cvreport.pl?id=FBbt:00058136)：由近端股节延至pretarsus，股节／胫节pretarsal depressor通过它回缩claws。它支持连接身份，不给出关节转角、弹簧、行程、时间常数或力。FeCO文献中的“claw／hook”神经元亚型只是形态命名，不能当末端爪钩机械来源；蛹期唾液胶也不是成年足垫分泌液。

## 数值与机制证据：已读到什么、还缺什么

|参数／证据|本轮可用值或状态|来源和限制|
|---|---|---|
|D. melanogaster毛尖宽|约1—2 µm|Kimura2020上述Results正文；是**软足垫setal spatula的宽**，不是硬爪尖半径／直径。|
|每片足垫毛数／排数|约30根／6—7排|同文正文；约数，未分六足、性别、个体变异，不固定成精确30。|
|单根fly footpad hair附着面积|3±0.1 µm²，mean±SE，n=5|同文“Biomimetic adhesive device”比较段；不是整片pad面积，更不是硬爪接触面积；没有把它乘毛数推算真实有效接触面积。|
|带标尺影像可得性|Kimura2020 Fig1a 20µm、1b 2µm；Tanaka2023 Fig1 E—N图注10µm|只读取文字图注，未对图片或mesh量坐标。可作为未来正式分割／测量候选，当前不从图片估爪长。固定／脱水SEM的姿态也不等于在体受载姿态。|
|硬爪长、开口、尖端半径、曲率、两爪间距、壁厚|UNKNOWN|尚无本轮核实的定量果蝇来源。不可用足垫毛尖宽替代。|
|硬爪长／体长比、微米到当前native长度映射|UNKNOWN|没有同个体／同标定体长与爪测量，也没有模型SI标定；不能由通常果蝇体长猜。|
|硬爪／足垫材料参数|几丁质硬化类别有依据；E、ν、硬度、断裂、黏弹性、局部摩擦均UNKNOWN|FlyBase给类别，不给可直接仿真的本构。μ=1.5是项目工程设置，不是此处实测果蝇材料常数。|
|收爪姿态／运动、长腱行程和松弛刚度|只有来源拓扑；定量UNKNOWN|FlyBase长腱定义不等于动态控制模型。|
|直接果蝇粗／滑面功能分离|Tanaka2023 nw突变体毛尖损伤、光滑瓶壁攀爬下降；filter paper上仍可爬，作者以爪解释|[原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10119690/)Description与Methods；有对照行为，未分离单爪力、未测filter paper粗糙度谱、不能提供asperity阈值或负载上限。|
|足垫湿附着／分泌力|D. melanogaster毛尖形态和湿接触机制有论文支持；液量、黏度、表面张力、接触角、单足力速曲线UNKNOWN|[Hüsken2015摘要](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:25714570&resultType=core&format=json)报告Su(z)2异常毛尖无法密接光滑面、只用爪、失去光滑面攀爬。摘要不够标定流体模型。|
|文中779±108 mN等大力值|**不准作为果蝇力**|Kimura2020这些是nylon／alginate人工大尺度纤维的力，52.8g负载同属人工装置；其杨氏模量／几何也不能替代果蝇爪。|

重要文献辨识：Hüsken2015原论文 DOI为**10.1242/jeb.108332**，218:1159—1165；搜索易命中**10.1242/jeb.125120**，218:1613，这是Erratum。EuropePMC实际元数据已区分两者。出版社正文与PDF本轮403，故只采用实际返回的一手摘要和后续可读论文，不声称原全文已核验。

跨物种资料仅供方法：Zhou等2014 [PMC4233698](https://pmc.ncbi.nlm.nih.gov/articles/PMC4233698/)使用Nauphoeta cinerea蟑螂和Gastrophysa viridula叶甲，标准微柱与同步力／接触成像可借鉴；其中1.4µm柱、0.4mN预载、270kPa估计等**均不进入果蝇参数表**。论文引言“粗糙突起大于约爪尖直径易互锁”为一般假说／引用背景，不能填补Drosophila硬爪尖径缺口。检索还把虱蝇96µm pulvilli、瓢虫模量与果蝇混排，这些未采纳。

## 最小独立台架，验证几何互锁而非“黏住”

**对象只含一个足端工程样件、基座和表面样件，无flyroot、无ANN、无71mesh全身。** 可固定基座、施加外部有限拉拔力／力控预载，因为它们是显式器械输入；基座反力、导轨和夹具约束均记账。禁止把这种支持迁移到全蝇根部冒充自主行走。

先登记`NOT_SPECIES_CALIBRATED`的无量纲几何：工程钩跨距L、尖端半径r、钩口g、突起高度h／棱边半径ρ，全部以L归一，双爪拓扑来源与工程尺寸身份分开。钩由少量凸子体组合形成真实开口；整个凹钩单一凸包不合格。不为贴合既有mesh而猜坐标。等质量／同材质／同外包络对照不完全可同时满足时，逐项报告差异，不宣称完全匹配。有限小集合的尺寸敏感性必须预先固定，不逐失败调尖端到通过。

建议最小控制矩阵如下；先干接触、液体模块关闭：

|条件|要隔离的解释|
|---|---|
|无爪、同材质与相同μ／预载，对同一棱边|突起或普通摩擦本来就可阻挡，不能把任意侧向反力称hook。|
|带凹爪，对光滑平面|无几何挂点时的基线；单纯较大μ可有切向阻力，不应有跨间隙吸力。|
|带凹爪，对固定圆角棱边／单突起|真实接触进入开口后，法向反力能否形成抵抗拉拔的分量；必须保留接触位置与几何配置。|
|把同一钩口填平的凸对照，对同一棱边|检测开口几何作用；不能仅比较“有爪外观”和“无爪外观”。|
|同凹爪反向牵拉／预登记小角度松钩|方向选择性与可逆释放；全过程连续，不能contact后锁死。|
|上述关键带钩／无钩条件在固定低摩擦与μ=1.5各一次|区分摩擦放大与几何阻挡；低摩擦数值、接触求解器最小摩擦必须预登记，不能把引擎最小μ写作精确0。|

液体附着是**后续独立试验臂**：只有在分泌液和材料参数可追溯后才加明确模型，配同几何dry/wet、已知液量／拉速／接触角对照。当前先不实现它，更不调用旧`adhesion_on`或MuJoCo adhesion执行器作为“爪钩替代”。

器械时序拟为：初始有正间隙→低速接近→固定正常预载→有限切向力斜坡／保持→撤去拉力和预载、仅自由响应观察→实施预登记反向／旋转释放→拉开并确认无接触。**撤力不保证自动脱钩**，几何可仍套在棱边上；必须报告“无载仍几何套合”与“仍有非零吸力”区别，不能通过脚本删接触。加载时样件平移／转动自由度要有可释放方向，不能用全刚性锁死所有运动再把夹具反力记为钩承载。

负载限额可用工程参考重量`W_ref=m_ref·g_ref`登记0.05、0.1、0.25 W_ref三个固定载荷级，各保持0.1s，达到上限即停止，不搜索破坏载荷。**这不是活体果蝇每足承载实测，也不沿用旧0.25W附着力**。若未来用项目W=24.446354755878005 native归一，须另标`W_project_native_reference`，不转换为N／μN；fixture自身惯性、重力及夹具负载分别保留。具体m_ref、几何、弹性、速度和求解器数值未定，故这些只是后续预登记模板，不是已准入仿真的参数。

## 必须保存和预先判定的机械证据

1. 原生时间步保存所有样件—表面contacts：geom IDs、位置、局部法向／切向、局部force／世界wrench、相对点速度、接触间隙与穿透；用surface registry，不能只认plane或只看world-z。钩侧壁的水平法向同样重要。
2. 同步记录独立几何钩口间隙／与棱边相对配置、样件6D位姿、夹具反力、外力和施力点、驱动功。MuJoCo margin内可存在零力contact，不能把接触计数等同承载。
3. 拉拔阻力用外力方向投影和法向／切向分解，分别报告峰值、保持区间位移／速度／蠕变、力—位移曲线。能量账为外部功、动能／重力势能、弹性储能、摩擦／阻尼耗散及数值残差；刚体近似没有材料应变能就明确写未建模。
4. 只有带钩粗面出现可视／几何可证的开口进入与真实受力，且与同μ无钩／填平／光滑面对照有预登记差异、方向释放可复现，才称**工程hook接触验证**；仅slip降低或静态高反力不足。低μ下几何反力仍可阻挡有助于支持互锁，但不据此声称完全无摩擦。
5. 空间分离到所有相关gap大于margin并留裕量后，dry trial不得存在持续拉力。任何contact触发的外加吸力、weld／equality切换、位置夹紧、锁脚／无限PD、改root状态都令该试验不具备无隐藏附着资格。显式试验台预载／拉拔可非零，必须与接触反力分栏。
6. 先用无接触自由响应和简单平面接触核对外力／反力符号与能量账，再以两档时间步／合理接触刚度做有限数值收敛对照；穿透不得接近钩口尺寸导致“穿过去再卡住”。误差阈值按工程L、W_ref和动力学时标在跑前冻结，当前不编造物种阈值或预报PASS。

本轮不写ONLYCODE fixture脚本：硬钩几何／材料准入仍不足，且数值稳定性阈值尚未登记。下一最小动作是取得带标尺、注明物种／足位／性别／制备条件的硬爪定量资料，并核对源mesh分辨率是否足以表达该结构；或者由父明确另立“工程形状敏感性”小任务，接受NOT_SPECIES_CALIBRATED身份后再写单台架。任何台架成功也不能直接外推全蝇38.4s行走、当前新NN停止／转向或rough静止原因。
