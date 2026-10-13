# 腿部触地／负载／地面反作用力感觉：有界一手文献研究

## 状态与边界

**手工研究，不是成品。** 本文件是人工检索与阅读一手文献后的研究指引，**不是**已验证的编码器、**不是**连接组映射、**不是**仿真结果、**不是**经过验证的模型结论。

- 物种：*Drosophila melanogaster*（另有 1 篇不同物种文献，已明确标注）。
- 只新增两个文件：本报告，以及 `outputs/neural_muscle_closure/research_load_contact_sources.json`。**未修改任何既有项目文件**，未运行仿真，未做重计算，未批量下载。
- 只读检查（不写入）：`outputs/leg_sensory_channels/leg_sensory_channels.json`、`engine/embodied/sensory_channels.py`、`data/flywire/banc_neurons.csv.gz`、`data/flywire/fafb_consolidated_cell_types.csv.gz`（流式读取，未写回）。
- 检索日期：2026-10-12（取自本机 `date -u`）。
- 所有英文引文都是**逐字摘录**；凡是转引自别的论文的引文，都单独标为"二手"。

### 本地前提（本报告必须与之相容，未重复审计）

1. 本地腿部 campaniform（钟形感器，CS）通道**已观测到对运动输出与身体动力学有影响**，因此"应变型通道影响行为"在本地部分成立。
2. 本地审计发现 450 个大场 tactile 毛细胞在本模型 BANC 数据中**没有 Verified NT 标签**，经"已验证递质"过滤后其出边被清空，**触觉输入当前在该模型中不可用**。
3. 不得编造连接、不得填补未验证的递质标签。

### 先说结论（一句话）

**在读到的一手来源里，没有任何一篇在真实行走中记录过有身份标注的果蝇腿部钟形感器传入，并同时测力／负载。** "果蝇腿部 CS 编码负载／触地力"目前是**由干预＋运动学推出的结论**，加上**从竹节虫／蟑螂借来的电生理**；而唯一一篇真正用有限元算过"行走地面反作用力是否作用于腿部 CS 场"的果蝇研究，结论是**在那个位置不够**。因此本地模型把 campaniform 直接等同于"该腿地面接触法向力／load proxy"，**是未经验证的假设，不是文献事实**。

---

## 一、六篇一手来源一览

| 编号 | 文献 | 类型／状态 | 与本地问题的关系 | 能否映射到 BANC/FlyWire |
|---|---|---|---|---|
| S1 | Custódio RD, Gorostiza EA, Pierzchlińska A, Haustein M, Godesberg V, Dübbert M, Bockemühl T, Büschges A (2026) *Role of leg campaniform sensilla sensory feedback in Drosophila melanogaster adaptive walking*. bioRxiv，doi 10.64898/2026.07.22.740025 | 一手研究，**预印本，未经同行评审** | **最直接的果蝇行走证据**：批量抑制／激活全部 CS，测行走速度、步态运动学、腿间协调、体高 | **否**。eyg-Gal4 是类别级标记，无逐感器身份，无 root ID |
| S2 | Pratt BG 等 (2026) *Proprioceptive limit detectors contribute to sensorimotor control of the Drosophila leg*. Nature Communications 17:2664，doi 10.1038/s41467-026-69333-z | 一手研究，已评审 | **不是负载细胞**，但提供了"命名细胞类型→钙成像→FANC 重建→光遗传验证"的完整可复制范式；并给出果蝇腿部各感觉类别的数量与 CS 分布位置的原话 | **部分**：仅 hair plate（CxHP8 等）在 FANC 有重建；对 CS 无 |
| S3 | Goldsmith CA 等 (2025) *A parametric finite element model of leg campaniform sensilla in Drosophila…*. J R Soc Interface 22(226):20240559，doi 10.1098/rsif.2024.0559 | 一手研究，已评审，**计算研究** | **唯一一篇算过"行走 GRF 是否足以激活腿部 CS 场"的果蝇研究，结论是否定的** | **否**。只有角质层感器编号 CS1–CS11，无神经元身份 |
| S4 | Kuan AT 等 (2025) *Divergent neural circuits for proprioceptive and exteroceptive sensing of the Drosophila leg*. Nature Communications 16:4491，doi 10.1038/s41467-025-59302-3 | 一手研究，已评审 | 说明果蝇 VNC 里 CS 轴突向下游如何并入本体感觉流；并明确说明腿部感觉轴突重建困难 | **否**（CS 只作聚合类处理）；但方法学警示直接适用 |
| S5 | Zhang L 等 (2025) *The TRP-channel painless mediates substrate stiffness sensing in the legs during Drosophila oviposition*. PLoS Genetics 21(12):e1011980，doi 10.1371/journal.pgen.1011980 | 一手研究，已评审 | **跗节 CS 的最强果蝇功能证据**；给出跗节 CS 命名（Ta1S/Ta1G/Ta3G/Ta5G）与其**背面、关节旁**位置 | **否**。感器组名在本地 BANC 元数据中不存在 |
| S6 | Zill SN, Dallmann CJ 等 (2024) *Mechanosensory encoding of forces in walking uphill and downhill…* J Neurophysiol 131:290-307，doi 10.1152/jn.00414.2023 | 一手研究，已评审，**物种为竹节虫 Carausius morosus，不是果蝇** | 提供定量响应性质锚点（发放率、适应、方向分组、阈值）——这是果蝇负载通道**将来**必须对标的参照 | **否**（跨物种、无连接组）。**绝不能当作果蝇证据引用** |

未取得全文的两篇（已如实记录，不用其内容）：Dinges 等 2021（J Comp Neurol 529:905-925，**订阅墙**，仅摘要）；Chockley 等 2022（JEB 225:jeb244245，**非 OA**，仅摘要）。另有一篇 Phelps 等 2021（Cell 184:759-774）本次未取全文，凡涉及它的内容均在文中标为**二手转引**。

---

## 二、逐篇证据

### S1（bioRxiv 2026 预印本）——果蝇行走的唯一直接证据

**获取**：[全文页](https://www.biorxiv.org/content/10.64898/2026.07.22.740025v1.full)／[PDF](https://www.biorxiv.org/content/biorxiv/early/2026/07/27/2026.07.22.740025.full.pdf)；[EuropePMC 记录](https://europepmc.org/article/PPR/PPR1286636)。
**状态：加载成功**（HTML 全文页只返回摘要＋导航，改用 PDF 下载后 `pdftotext -layout` 转文本，共 140932 字符）。过程中的失败：biorxiv 多次 HTTP 429（Cloudflare），加浏览器 UA＋referer 后成功；europepmc.org 文章页 HTTP 403；EuropePMC 的 PPR fullTextXML 接口 HTTP 500 空体（未使用该路径）。

**影像学／电生理学实际测了什么**

- CS 类别覆盖：`We verified three CS on the first tarsal segment (Ta1S) and two on the third tarsal segment (Ta3G) … On the fifth tarsal segment (Ta5G) we regularly located two cell bodies two less than expected.`
- 胫节分组：`We detected two somata forming the ventral group (TiGv…) and three forming the distal group (TiGd) (n=6 legs)`。
- 股节／转节：`we identified three eyg-positive cells comprising the trochanteral group (TrG), twelve comprising the trochanteral field (TrF), twelve comprising the femoral field (FeF) and one lone cell in the femur (FeS)` → TrG 3 + TrF 12 + FeF 12 + FeS 1 = 28。
- 全腿总数（作者原话，**计数出自 SEM 图谱即 Dinges 2021，本次未读原文**）：`which we found to label all CS, including all 42 in the legs`。
- 肌肉激活（单腿制备，仅左前腿）：`CS activation reliably elicited muscle activation in eyg>CsChrimsonmuscle femoral and tibial leg muscles. Contrarily, this protocol did not elicit muscle activation in emptySplit>CsChrimsonmuscle flies`；`CS activation reliably elicited muscle activation in all tarsal levator muscles… Interestingly no activity was observed in tarsal retro depressors`；刺激时长 2 s（Fig. 4B1）。
- 行为：攀爬性能 −40.9%（Fig. 5A 左；光对照 eyg>GFP +4.92%；头部／脑限制的 eyg>GtACR1brain 反而 +15.63%，说明表型不来自头／脑）。
- 行走：平均速度 **9 → 6.5 mm/s**（对照区间 4–30，峰值 9；抑制后 5–15，峰值 5.5）。
- 运动学（Fig. 7）：摆动时长 −14%~−16%，**支撑时长 +72%~+77%**，步周期 +41%~+46%，而**支撑幅度 −9%~−15%**（各腿对均如此，效应量大致不随腿变化）。
- 腿间协调（Fig. 8C1→8C2）：L3 之后走 L2 的概率 **97.6% → 86.6%**，L2→L1 90.5%，L1→L3 91.3%。
- 姿态（Fig. 9）：行走中体高（ground clearance）下降，取消抑制后回升；**但 `performing the same experiment on immobile, i.e. standing flies, showed no alterations in ground clearance`**，作者原话：`while CS load feedback has relevance for posture control in walking flies, its transient cessation is apparently no challenge for keeping body posture while standing`。
- 分子旁证：NompC RNAi（tubGal80ts，29 ℃ 允许温度 3 天）同样降低行走速度（**作者自己承认不同培育温度是混杂因素**）。
- 翅／平衡棒对照：截除翅和／或平衡棒后，CS 抑制的行走表型不变 → 表型归因于**腿** CS 而非翅／平衡棒 CS。
- 作者自陈的局限：`To differentiate excitatory from inhibitory influences the field will require a detailed mapping between specific CS and muscles, along with similar investigations using sparser labeling lines.`

**（a）直接测量**：eyg-Gal4 表达谱与 SEM 描述的类别级比对；单腿制备中肌肉 GCaMP6f 对批量 CS 激活的响应；抑制下的攀爬、行走、运动学、腿间相位、体高；行走 vs 站立的体高对照。
**（b）推断**：把"支撑时长变长＋支撑幅度变小"读作"身体负载再分配未被感知"；把负载／力反馈的机制归给 CS；把表型归给腿 CS（依据是翅／平衡棒＋头部对照，但 eyg 阳性的 VNC 群体本身并未被排除）。
**（c）对本地模型未知**：任何 spike rate vs 力／应变的曲线；哪个节段（转节／胫节／跗节）的 CS 贡献哪部分表型；到具体运动神经元的符号；任何 root ID 对应；**站立条件下未观测到表型**（这恰恰削弱"站立需要负载通道"的动机）。

> **对本地模型最要紧的一条**：本地 `leg_sensory_channels.json` 把 campaniform 定义为"该腿地面接触法向力（cuticular strain / load proxy）"。S1 是支持这一定义**最接近**的文献，但它**没有测力、没有记录 CS 发放**，且它自己的站立实验是**阴性**。

### S2（Nat Commun 2026）——命名细胞类型的范式，但对象是限位感受器

**获取**：[PMC13009157](https://pmc.ncbi.nlm.nih.gov/articles/PMC13009157/)。**状态：加载成功**（PMC 全文＋EuropePMC JATS）。

- `The six Drosophila legs have 214 hair plate mechanosensory neurons that are clustered into 42 hair plates`（引文，源自既有解剖学工作）。
- `The fruit fly's front leg contains over 200 proprioceptive sensory neurons, including hair plates, campaniform sensilla, and chordotonal neurons. Chordotonal neurons are primarily concentrated in the FeCO, which senses the femur/tibia joint. The more proximal coxa and trochanter joints contain similar numbers of campaniform sensilla and hair plates.` → **CS 集中分布在近端关节（基节／转节）**，这是本报告关于"CS 在哪些节段、可能编码什么"的关键位置线索。
- `Connectomic and electrophysiological evidence suggest that signals from these different proprioceptor classes are highly integrated in downstream circuits. For instance, work in the stick insect has shown that load and movement signals encoded by campaniform sensilla and femoral chordotonal organs converge onto leg motor neurons.` → **负载编码被归给竹节虫工作**，本文不为果蝇 CS 主张负载功能。
- 命名细胞类型验证：CxHP8 = 前腿基节前向极限的限位检测器；8 根轴突在 FANC 重建；33 根 hair plate 轴突做下游分析；每条 CxHP8 输出突触 `2279 +/- 482`；75% 输出到前运动／运动神经元。
- 另有它自己先前的工作：慢性沉默包含 CS 与 hair plate 的一小群神经元（含 CxHP8）会损害快速行走时的步态运动学与腿间协调（**混合群体，不是纯 CS**）。

**（a）直接测量**：CxHP8 钙成像调谐、光遗传激活／沉默的行为效应、FANC 重建与连接。**（b）推断**：CxHP3/4、TrHP5/6/7 的功能由连接预测。**（c）未知**：任何 CS 生理；任何 CS 的 BANC 对应。**注意**：本文明确说 hair plate 轴突在 MANC 中因重建质量差而无法辨识——同样的问题适用于 CS。

### S3（J R Soc Interface 2025）——唯一算过"行走 GRF 是否够用"的果蝇研究，结论是否定的

**获取**：[PMC12056673](https://pmc.ncbi.nlm.nih.gov/articles/PMC12056673/)。**状态：加载成功**（PMC HTML 转文本 87452 字符；EuropePMC 的 JATS 接口对该 PMCID 返回 HTTP 500 空体，未使用）。

- 摘要：`Our results indicate that ground reaction forces alone are unlikely to serve CS field activation during forward walking.`
- 结果 3.5：`Overall, the displacements at the receptor level are below 7 Angstrom, which means that the displacement cannot be differentiated from random thermal noise.`
- 结果 3.4–3.5：`For the physiological CS field location, forces transmitted from a step of the leg via the FeTi joint in straight forward walking were neither sufficient to open NompC channels nor carry directional information in strain responses of single CS and the whole CS field.`
- 讨论：`This work shows that in straight, forward walking the femoral CS are likely not directly activated by ground reaction forces… the physiological position of the femoral CS field on the leg is not optimal for resolving displacements induced by forward-walking ground reaction forces.`
- 定量：载荷取自行走雄蝇逆动力学的等效关节力**最大 46.019 µN**；判据是 NompC **7 Å** 热噪声位移下限（通道开放力约 20 pN）；生理 CS 场位置在股节长度 **2.5%** 处；支撑相股胫仰角 144.58 ± 9.66° → 65.57 ± 17.74°（n = 244 支撑相／12 只蝇），方位角 −19.66 ± 7.10° → 5.31 ± 5.00°。
- **两处不可调和**：本文说股节 CS 场 **11** 个，S1 数到 **12** 个（同一区域）。不要取平均。
- 作者给出的替代假说：`amplitude-/threshold-based coding is a potential mechanism of range fractionation in fruit fly CS`（按感器大小分级募集）——这是可检验的替代，而不是"负载幅度"。

**（a）直接测量**：后腿行走运动学、CS 场几何（TEM/SEM）、有限元模型。**（b）推断**：结论是**模型结果**，依赖材料参数与圆柱近似；作者报告填充材料属性贡献了场景间差异的 **>95%**。**（c）未知**：只覆盖**后腿股节场**；转节／跗节场未算；无力阈值、无发放率；不等于"CS 原则上不响应"。

### S4（Nat Commun 2025）——果蝇 VNC 里 CS 如何并入本体感觉流＋重建风险自陈

**获取**：[PMC12048489](https://pmc.ncbi.nlm.nih.gov/articles/PMC12048489/)。**状态：加载成功**。

- `Unfortunately, leg sensory axons are among the most difficult neurons to reconstruct in all available VNC connectome datasets, likely due to rapid degeneration that begins when the legs are dissected away from the VNC during sample preparation… As a result, we reconstructed roughly half of the FeCO axons from the front left leg (80 total axons).` 同一段还指出 MANC v1.2.1 该神经节只有 22 根 FeCO 轴突且多不完整。→ **这是"任何基于 VNC 连接组的腿部感觉身份/连接都不可靠"的原文依据。**
- Fig. 2E：`Neurons postsynaptic to claw and hook axons also integrate information from other leg proprioceptors including HP and CS axons.` → 果蝇 VNC 中 CS 轴突与运动本体感觉（claw/hook）汇聚到同一批局部伙伴，即 CS 被当作**本体感觉型输入**处理，而非独立的"触地／肢体状态"通道。
- 讨论中再次提到 `signals from load-sensing CS neurons can reduce the effect of FeCO activation on leg motor neurons`——同样是**转引其它昆虫**。
- 规模对比：FeCO 每腿约 150 个感觉神经元；CS 全果蝇约 42 个（S1/S6 口径）。这个对比在选择信任哪一路通道时很重要。

**（a）直接测量**：FANC 中重建的 FeCO 轴突突触连接、下游 hemalineage、上行投射在 FlyWire 中的匹配。**（b）推断**：本体／外感受功能由连接推断。**（c）未知**：CS 身份、CS root ID、任何负载信号。

### S5（PLoS Genet 2025）——跗节 CS 的最强果蝇功能证据（但测的是产卵基质硬度，不是行走负载）

**获取**：[PMC12768414](https://pmc.ncbi.nlm.nih.gov/articles/PMC12768414/)。**状态：加载成功**。

- 命名与位置：`Campaniform sensilla (CS) in the tarsi are organized either as singlets in tarsomere 1(Ta1S) or in clusters of 2-4 sensilla in tarsomeres 1, 3 and 5 (respectively named Ta1G, Ta3G and Ta5G). pain-Gal4 labelled neurons that innervate subsets of these CS, mostly positioned dorsally at the joints between tarsomeres or near the claws in the most distal tarsomere.` → **跗节 CS 在背面、跗分节关节旁、末端靠近爪**，**不在接触地面的腹面**。
- `We report that the stiffness-sensing role of tarsi is mediated through external sensory organs, namely ventral mechanosensory bristles and subsets of campaniform sensilla present primarily at the joints between tarsomeres.`
- `The involvement of ventral mechanosensory bristles appears straightforward, as they may directly contact the substrate.` → **"直接接触"被判给腹面毛细胞，而不是 CS。**
- `We propose that CS indirectly sense differences in substrate stiffness by sensing differential compression of the tarsal joints.` → **明确标为 proposal 的间接机制推断。**
- `Campaniform sensilla are known to be very sensitive organs and are seated in the cuticle at joints that experience maximum strain.`（引文，非本研究测量）
- 必要性／充分性：painless 缺失降低对较软基质的偏好；用 CS 与毛细胞驱动系表达 UAS-pain 可**部分到接近完全**拯救。
- 可复用试剂：`we used the generation-1 driver line R34E04-GAL4 (CS1), as well as two split-GAL4 lines, SS69574-Gal4 (CS2), and SS69602-Gal4 (CS3)`——三个 CS 子集驱动系（本文未声称任一为"仅腿部"）。
- 定量：偏好实验用 0.5% vs 1.25% 琼脂糖。

**（a）直接测量**：跗节切除、递质阻断（Kir2.1、shits）、pain RNAi 与拯救对产卵偏好选择的影响；pain-Gal4 外周支配的共聚焦解剖。**（b）推断**：CS 经关节压缩间接感知硬度。**（c）未知**：无力、无应变、无发放率；**完全没有站立／行走负载的证据**；无 BANC 映射。

### S6（J Neurophysiol 2024，竹节虫）——定量响应性质参照系（**不是果蝇证据**）

**获取**：[PMC11286306](https://pmc.ncbi.nlm.nih.gov/articles/PMC11286306/)。**状态：加载成功**。

- `discharges of sensilla of large amplitude during the ramp increases in force in the direction of joint extension that adapted completely (or fired at a low level, <10 Hz) during the hold phase. Receptors with smaller spike amplitude fired tonically throughout the hold phases. Ablation of cuticular caps confirmed that these discharges… were derived from 6B sensilla.` → **大振幅群：力上升期相位性放电，保持期完全适应或 <10 Hz；小振幅群：保持期持续放电**（N = 12）。
- `All campaniform sensilla at the FT joint are, therefore, generating a fine-grained reflection of the rate of force (torque) increase even in sloped walking.` → **编码 dF/dt，不是绝对负载水平。**
- 方向分组与阈值不对称：`discharges of sensilla of the opposite subgroup (for example, 6A firing in level and downhill walking) generally did not occur to small force decreases but were at higher threshold`。
- 符号结构（个人感器水平）：`punctate stimulation of cuticular caps of individual 6B sensilla excited the slow tibial extensor motor neuron, whereas stimulation of 6A receptors inhibited`。
- 关键限制（作者自陈）：`Although recordings of campaniform sensilla have not been obtained from free walking stick insects (due to technical problems of cross talk and the narrow diameter of leg segments), discharges of tibial sensilla have been extensively recorded in free-moving cockroaches.` → **连竹节虫都没有自由行走时的 CS 记录**；果蝇腿更细，只会更难。

**（a）直接测量**：竹节虫受控力斜坡下的胞外 CS 发放；单感器帽刺激对运动神经元的作用。**（b）推断**：负载稳定行走的一般化。**（c）对本地模型未知**：一切物种特异量（果蝇的力阈值、亚群结构、感器—肌肉符号）。

---

## 三、综合：证据强度分级

### 支持"果蝇腿部 CS 承载 load/force 信息"

1. **S1**：CS 抑制产生**以支撑相为主**的签名——支撑时长 +72%~+77%、支撑幅度 −9%~−15%、速度 9→6.5 mm/s、腿间摆动次序概率 97.6%→86.6%、行走体高下降；有 NompC-RNAi 旁证；有翅／平衡棒与头部对照。这是**目前最强的果蝇行为层面证据**，但它是**预印本**，且没有力测量。
2. **S2**：一个独立的含 CS 腿群体（CS＋hair plate，混合）被沉默后损害步态运动学与腿间协调。
3. **S4**：果蝇 VNC 中 CS 轴突与运动本体感觉汇聚，说明 CS 属于本体感觉流。

### 反对"CS 就是行走 GRF 负载通道"

1. **S3**：唯一果蝇专用力学计算 → 股节 CS 场在生理位置**低于 7 Å 热噪声下限**，且方位信息很弱。
2. **S1 vs S3**：同一股节场的计数不一致（12 vs 11）。
3. **S5**：跗节 CS 在背面关节旁、末端近爪，**不在接触面**；其果蝇功能证据是产卵基质硬度偏好，**与行走负载无关**。
4. **S1 自身**：**站立时未观测到姿态表型**——"站立需要负载通道"的动机缺乏该实验支持。
5. **S6**：即使竹节虫也无自由行走 CS 记录；负载编码量来自受控力斜坡与蟑螂自由行走记录。

### 身份信息可映射性（逐条，不含任何未支持的主张）

| 来源 | 身份粒度 | 是否声称连接组映射 | 能否落到 BANC root ID |
|---|---|---|---|
| S1 | **类别级**（eyg-Gal4 = 全部 CS）＋外周体群名（TrG/TrF/FeF/FeS/TiGv/TiGd/Ta1S/Ta1G/Ta3G/Ta5G） | 无连接组分析 | **不能**。类别级标记＋本地无体群名字段 |
| S2 | **命名细胞类型** CxHP8、CxHP3/4、TrHP5/6/7（hair plate，非 CS） | 是（FANC＋CAVE 注释） | **仅方法学**；CS 无，且本文未细分本地 `*_hair_plate_neuron` 行 |
| S3 | 感器级编号 CS1–CS11（角质层表面方案） | 无 | **不能**。与神经元身份无关 |
| S4 | FeCO 亚型级；CS 仅作聚合"CS axons" | 是（仅 FeCO） | **不能**（对 CS）；提供重建风险警示 |
| S5 | 感器组级 Ta1S/Ta1G/Ta3G/Ta5G ＋ 3 个 CS 子集驱动系 | 无 | **不能**。本地元数据无这些字段；驱动系是未来实验工具 |
| S6 | 竹节虫胫节 CS 亚群 6A/6B | 无 | **不能**（跨物种） |

### 本地只读观测（**不是文献结论**，也不是验证）

在 `data/flywire/banc_neurons.csv.gz` 中流式只读统计：

- `Class = campaniform_sensillum_neuron` 共 **576** 行；其中 Sub Class 含"leg"的 **62** 行：前腿 50、前腿 bilateral 4、中腿 3、中腿 bilateral 1、后腿 bilateral 4。
- 这 62 行的 `Verified NT type`：**acetylcholine 12 行，空 50 行**（注意其中若干行的 `Predicted NT type` 是 GABA 且置信度偏低——这种冲突**不得被静默抹平**）。
- `Primary Cell Type`：SNpp53(11)、SNxxxx(9)、SNcs01f(8)、SNpp45(5)、DNx01(3)、CS(3)、SNppxx(3)、SNpp43(3)、SNpp52(2)、SNxx30(2)、SNpp39(1)、IN13B054(1)、SNpp22(1)、SNta21(1)。
- **元数据中没有节段信息**（转节／胫节／跗节），因此**任何文献感器名都无法挂到具体行上**。`SNcs01f` 里虽有 "cs" 字样，但本次读的文献**从未定义**它，**不得靠猜解码**。
- 另查 `fafb_consolidated_cell_types.csv.gz`：**没有任何 campaniform 类型**（该表只有 root_id/primary_type/additional_type(s)，campaniform 计数为 0），所以这条映射链在 FAFB 侧也是断的。

---

## 四、什么**没有**被确立（必须写进任何后续分支的前提里）

1. **NE1**：没有任何果蝇腿部 CS 的"力／应变 → 发放率"曲线被测量。任何声称有该曲线的编码器，都在从竹节虫／蟑螂或从有限元模型外推。
2. **NE2**：本地 BANC 的腿部 CS 行**没有逐感器身份**（节段、感器名都缺）。
3. **NE3**：没有任何来源指出某个果蝇跗节／足部细胞类**可验证地**编码触地或地面反作用力。最接近的候选是腹面机械感毛（被判为"可能直接接触"）与跗节 CS（被判为"可能间接感知关节压缩"），两者在本地都缺已验证递质（毛细胞是已知阻塞点）。
4. **NE4**：**站立条件特别弱**——唯一测过站立的 S1 报告阴性。
5. **NE5**：转节 CS→胫节屈肌运动神经元的连接在本报告中是**二手**（转引自 S1 对其参考文献 Phelps 等 2021 的引用），未直接读原文。
6. **NE6**：42 个腿 CS 总数与全部感器组名，都依赖**订阅墙**的 SEM 图谱（Dinges 等 2021），本次未读原文。
7. **NE7**：没有任何来源为本地腿部 CS 行（或 tactile 毛细胞行）填补 Verified NT 提供依据。**不得填补。**

---

## 五、一个可辩护的下一步建模分支需要什么

以下仅为**前置条件与验收口径**，本报告**没有实施**，也不构成授权。

**R1 定位**：现有 campaniform 通道继续作为**显式标注的假说**保留，**不得**被描述为已验证的负载编码器，更不得从"未测量它的文献"反推参数。

**R2 分支必须自陈四件事**：(a) 它作用于哪个本地 BANC 细胞集合（root ID 清单＋release/hash）；(b) 这些细胞的**节段身份未知**；(c) 响应形式及其来源；(d) 该来源是**跨物种**还是**模型计算**。

**R3 至少并列两个竞争响应形式，不得只留一个**：
- (i) **率编码负载幅度**（须同时带上 S3 的反证：股节场在生理位置可能低于阈值）；
- (ii) **dF/dt 的速率／适应编码＋方向分组**（S6 的形态）。
两者分别报告，不取平均、不择优后隐藏另一个。

**R4 站立与行走分开报告**：按 S1，通道若真实应以支撑相为主；**不得**用"让站立更稳"反向拟合参数，因为站立侧本来就缺证据。

**R5 不要靠改名解决触觉通道的递质问题**：触地信号若要独立存在，需要独立的递质来源，或另建**显式的 class-prior 分支**并严格与 Verified NT 分表；绝不能写回 Verified NT。

**R6 任何"文献感器名 → root ID"的映射，必须先获得本次六篇都**没有**提供的逐细胞身份来源**（例如带注释的腿部感觉重建，或 S2 那样的驱动系—连接组对应关系）。在拿到之前，映射表留空。

**R7 S1 按预印本对待**：以它为依据的分支须在同行评审完成后复核；其效应量**不得**被转移到单条腿或单个感器上（它自己的数据表明各腿效应量大致一致，这恰恰说明它测的是群体效应）。

**R8 最小可判别实验（若将来有实验通道）**：用 S5 已发表的三个 CS 子集驱动系（R34E04-GAL4／SS69574-Gal4／SS69602-Gal4）做**分段**采样，配合受控力或受控基质条件，才能把"跗节 CS"与"转节 CS"分开；在此之前，任何分段归因都是猜测。

---

## 六、获取与复核

| 来源 | 路径 | 状态 |
|---|---|---|
| S1 | [bioRxiv 全文页](https://www.biorxiv.org/content/10.64898/2026.07.22.740025v1.full)／[PDF](https://www.biorxiv.org/content/biorxiv/early/2026/07/27/2026.07.22.740025.full.pdf)／[EuropePMC PPR1286636](https://europepmc.org/article/PPR/PPR1286636) | **成功**（PDF→pdftotext）。失败记录：biorxiv HTTP 429（多次）；europepmc.org 文章页 403；EuropePMC PPR fullTextXML 500 空体 |
| S2 | [PMC13009157](https://pmc.ncbi.nlm.nih.gov/articles/PMC13009157/) | **成功** |
| S3 | [PMC12056673](https://pmc.ncbi.nlm.nih.gov/articles/PMC12056673/) | **成功**（PMC HTML）。EuropePMC JATS 对该 PMCID 返回 500 空体 |
| S4 | [PMC12048489](https://pmc.ncbi.nlm.nih.gov/articles/PMC12048489/) | **成功** |
| S5 | [PMC12768414](https://pmc.ncbi.nlm.nih.gov/articles/PMC12768414/) | **成功** |
| S6 | [PMC11286306](https://pmc.ncbi.nlm.nih.gov/articles/PMC11286306/) | **成功**（注意：物种为竹节虫） |
| Dinges 等 2021 | [doi:10.1002/cne.24987](https://doi.org/10.1002/cne.24987)／[PubMed 32678470](https://pubmed.ncbi.nlm.nih.gov/32678470/) | **失败**：订阅墙（EuropePMC 标 "Subscription required"，isOpenAccess=N，inPMC=N）。**仅得摘要**，其数字在本报告中只作为 S1/S5 的转引出现 |
| Chockley 等 2022 | [doi:10.1242/jeb.244245](https://doi.org/10.1242/jeb.244245) | **失败**：非 OA（EuropePMC 36268799：isOpenAccess=N，inPMC=N）。**仅得摘要**，且研究对象是 FeCO 不是 CS，未作为 CS 证据使用 |
| Phelps 等 2021 | doi:10.1016/j.cell.2021.01.009 | **未取全文**（超出本次有界范围）。涉及它的转节 CS→屈肌运动神经元内容一律标为**二手** |

机器可读版本（含每篇的逐字摘录、章节/图号、检索状态、量化值与映射限制）见：
`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/research_load_contact_sources.json`

**再次声明：手工研究，不是成品。** 本文件是研究指引，不是已验证的编码器，也不是连接组映射；未修改任何既有项目文件，未运行仿真，未填补任何递质标签。
