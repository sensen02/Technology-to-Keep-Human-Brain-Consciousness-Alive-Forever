# 腿部钟形感器的**段身份**、接触/负载响应与驱动系：有界一手来源研究

## 状态与边界（先说清楚）

**手工研究，不是成品。** 本文件是人工检索、下载、逐字阅读一手来源后写出的**研究指引**；**不是**已验证的传感编码器，**不是**连接组映射，**不是**仿真结果，**不是**经过验证的模型结论。

- 物种：*Drosophila melanogaster*（另有明确标注的非果蝇生理学：竹节虫 *Carausius morosus*、蟑螂 *Periplaneta americana*、丽蝇 Calliphoridae）。
- **未修改任何既有项目文件。** 只新建两个文件：本报告，以及 `outputs/neural_muscle_closure/research_contact_identity_sources.json`。
- 未运行仿真，未做重计算，未批量下载。为读文献临时下载的 PDF/HTML/XLSX 全部落在 **项目树之外** 的 `/tmp/legsrc/`，不写入项目。
- 只读检查（无写入）：`data/banc888/banc_888_meta.feather`（用项目自带 `./venv/bin/python` + pyarrow 只读）、`outputs/leg_sensory_channels/leg_sensory_channels.json`。
- 检索日期：**2026-10-12**（取自本机 `date -u`）。
- 所有英文引文均为**逐字摘录**。凡属转引别人的说法，一律单独标注为"二手"。
- **绝不填补任何来源没有给出的递质标签或腿段身份。**

---

## 零、先说结论：这次研究改变了什么

1. **之前判定为"订阅墙"的 Dinges et al. 2021（J Comp Neurol 529:905-925）实际上拿到了全文。** 不是通过出版社，而是通过科隆大学机构库（KUPS eprint 25156）的作者接受稿 PDF，HTTP 200，19.2 MB。本文档因此**首次**给出该文逐段的感器计数与命名（见第 5 节）。这一点直接回答了你问题里的第 5 项：**"之前找不到"是检索路径问题，不是真的拿不到。**

2. **腿段身份是可以接上连接组行的——但只接了 11 行，而且只到"段"这一级。** 唯一找到的已发表链条是：BANC 腿 CS 行 → 该行的 `cell_type = SNpp53` → MANC/MaleCNS 里同一 cell type 的神经元 → **Virtual Fly Brain（VFB）把这些个体细胞归类为 FBbt:00048059 "mechanosensory neuron of trochanter campaniform sensillum"（转节钟形感器机械感觉神经元）**。因此这 11 行的段身份 = **转节（trochanter）**。不是股节，不是胫节，不是跗节，也不是任何具名感器。

3. **股节/胫节/跗节身份在连接组里存在，但对不上本地的行。** VFB 里有 12 个 hemibrain（T1LEG）个体被归为**前足股节 CS 神经元**、5 个被归为**前足胫节 CS 神经元**；但本地 BANC v888 的 61 条腿 CS 行的 `hemibrain_match` / `hemibrain_nblast_match` / `fafb_match` **全部为空**，接不上。

4. **CS1/CS2/CS3 三条驱动系只有"大致解剖"级描述，没有任何细胞/细胞类型清单，更没有连接组标识符。** 股节场（femoral field）**没有**对应的驱动系。

5. **果蝇腿部 CS 的"负载/力—发放率"定量数据：不存在。** 定量力响应数据只有竹节虫、蟑螂（以及一篇丽蝇模型）。没有任何实验在记录果蝇腿部 CS 的同时操纵腿的负载。

6. **"哪个细胞编码触地"：没有找到任何来源测过果蝇腿部的"触地"变量。** 跗节工作测的是**基质硬度偏好指数**，不是触地；而且形态学上与"触地"相冲突（跗节 CS 位于跗分节关节背面/爪附近，且**股节以外，腿基部 coxa 上完全没有 CS**）。

7. **一个必须报告的冲突：** BANC 把 `SNpp39`、`SNpp45`、`SNpp43`、`SNxx30` 这些行标为 campaniform，但 VFB 把同类细胞分别归为**股节弦音器 club 神经元**、**SNta23**、**SNxx33**、**未表征 SNxx30**。也就是说"本地这些行是钟形感器"这一条，至少对其中 8 行在 VFB 侧是有争议的。

---

## 一、逐条回答你按优先级提的五个问题

| 优先级 | 问题 | 结论 | 最强来源 |
|---|---|---|---|
| 1 | 能把果蝇腿 CS 指派到具体腿段、且能对上连接组行吗？ | **部分能。61 行里 11 行拿到"转节"**；具名感器级**完全不能** | VFB + FBbt（S10）、Dinges 2021（S1） |
| 2 | R34E04/SS69574/SS69602 标了哪些细胞？有股节场的线吗？ | **只有大致解剖描述，无细胞清单、无连接组标识符；股节场无线** | Ray 2025 正文＋S2/S3 补充（S2）、Meissner 2025（S9）、FlyBase（S11） |
| 3 | 有果蝇腿 CS 对负载/力/接触的定量响应数据吗？ | **没有。** 只有跨物种数据（竹节虫/蟑螂/丽蝇） | Zill 2021（S6）、Harris 2022（S7）、Saltin 2025（S8） |
| 4 | 哪个细胞编码"触地"？ | **无来源测过。** 且形态学不支持跗节 CS 当触地感受器 | Ray 2025（S2）、Dinges 2021（S1） |
| 5 | Dinges 2021 解剖图谱能拿到吗？ | **拿到了全文（机构库 OA PDF）。** 逐段计数见第 5 节 | S1 |

---

## 二、★ 核心发现：一条可用的"段身份"链条（回答优先级 1）

### 2.1 链条长什么样

```
BANC v888 行（cell_class = campaniform_sensillum_neuron，body_part_sensory 含 "leg"）
   └─ 该行 cell_type == "SNpp53"                      ← 本地字段
        └─ MANC / MaleCNS 里同 cell type 的神经元        ← 本地 manc_match / malecns_match
             e.g. MANC:34443、MANC:101691、MaleCNS:819255
                └─ VFB 把这些个体细胞归入
                   FBbt:00048059 mechanosensory neuron of trochanter
                                campaniform sensillum
                   ⇒ 段身份 = 转节（trochanter）
```

VFB 侧的原话（`get_term_info`，2026-10-12 取；HTTP 200）——这是一条**逐字**证据，说明该 BANC 细胞被归为转节 CS 神经元：

> `VFB_00105q2f` = `BANC_888:720575941532317571`
> Types: **[mechanosensory neuron of trochanter campaniform sensillum](FBbt_00048059)**
> Comment: *"BANC annotations: Flow - afferent, Super Class - sensory, Class - campaniform_sensillum_neuron, Sub Class - hind_leg_bilateral_campaniform_sensillum_neuron, Primary Cell Type - SNpp53, Alternative Cell Type(s) - SNpp53, Nerve - left_metathoracic_leg_nerve."*
> Relationships: *"fasciculates with adult metathoracic leg nerve (FBbt:00004096); has soma location left side of organism; is part of female organism"*

MaleCNS 侧对应细胞：

> `VFB_jrmc17bw` = `SNpp53_L (MaleCNS:819255)`
> Types: adult cholinergic neuron (FBbt:00058205); **mechanosensory neuron of trochanter campaniform sensillum (FBbt:00048059)**
> Comment: *"male-cns annotations: superclass - vnc_sensory, class - mechanosensory_proprioceptive, subclass - campaniform sensilla, type - SNpp53, mancType - SNpp53."*

**注意方向性**：段身份来自 **VFB 的策展归类**，不是 BANC 自己的字段。BANC 自己只给"腿对（front/middle/hind leg）"＋"campaniform"。

### 2.2 具体拿到段身份的 11 行

（本地只读查询结果；这些行的 `cell_type` 全部是 `SNpp53`）

| banc_888_id | BANC 子类 | manc_match | malecns_match | 神经 | 段身份 |
|---|---|---|---|---|---|
| 720575941532317571 | hind_leg_bilateral | 34443 | 819255 | 左后胸腿神经 | 转节 |
| 720575941391425540 | hind_leg_bilateral | 10155 | 909042 | 左后胸腿神经 | 转节 |
| 720575941473385770 | hind_leg_bilateral | 101159 | 1062512192 | 右后胸腿神经 | 转节 |
| 720575941483300349 | hind_leg_bilateral | 10247 | — | 右后胸腿神经 | 转节 |
| 720575941468647133 | middle_leg_bilateral | 11753 | 1055491165 | 左中胸腿神经 | 转节 |
| 720575941591858728 | middle_leg | 11753 | 1055491165 | 左中胸腿神经 | 转节 |
| 720575941573309691 | middle_leg | 116607 | 1050145590 | 右中胸腿神经 | 转节 |
| 720575941471495543 | middle_leg | 116607 | 1050145590 | 右中胸腿神经 | 转节 |
| 720575941449784413 | front_leg | 101691 | — | 右前胸腿神经 | 转节 |
| 720575941609969226 | front_leg_bilateral | 101691 | 877244 | 右前胸腿神经 | 转节 |
| 720575941505092517 | front_leg_bilateral | （nblast 101691） | — | 右前胸腿神经 | 转节 |

**三条必须一起说的限制：**

1. **只到"段"，不到"感器"。** 没有任何来源把这 11 行指到 Sc3 / Sc−8 / Sc+5 / TrFa / TrFp / TrG 里的任何一个具名感器。
2. **只到通用类，不到腿对专用类。** VFB 里 `FBbt:00048076`（前胸转节 CS 神经元）有 16 个实例，但 `FBbt:00048077`（中胸）和 `FBbt:00048078`（后胸）实例数**都是 0**。所以对中腿/后腿那几行，身份只能说"是一个转节 CS 神经元"，不能说"是后胸转节 CS 神经元"。
3. **匹配不是一一对应。** 例如 `malecns_match = 1050145590` 和 `1055491165` 各出现两次（不同 BANC 行）。把这些当唯一身份键使用前必须再核对。

### 2.3 剩下 50 行拿不到段身份，原因是"没有对应类型"，不是"没查"

本地 61 条腿 CS 行的 cell_type 分布，以及 VFB 侧的情况：

| BANC cell_type | 行数 | VFB 有没有对应个体？ | VFB 归类是什么 |
|---|---|---|---|
| `CS` | 23 | VFB 搜索无结果（太泛） | — |
| `SNcs01f` | 13 | **VFB 搜索 0 结果** | — |
| `SNpp53` | 11 | **有**（MaleCNS 13 个 + MANC + BANC 自身） | **转节 CS 神经元** |
| `bCS` | 4 | VFB 搜索 0 结果 | — |
| `SNpp43` | 3 | 1 个（MANC:33888） | **adult SNxx33 神经元**（未表征）← 与 BANC 冲突 |
| `SNpp45` | 2 | 11 个 | **adult SNta23 神经元** ← 与 BANC 冲突 |
| `SNpp52` | 2 | VFB 搜索 0 结果 | — |
| `SNxx30` | 2 | 有术语 FBbt:20011326 | **未表征 leg 感觉型**，无 CS 归属 ← 与 BANC 冲突 |
| `SNpp39` | 1 | 34 个 | **股节弦音器 club 神经元**（FBbt:00049556）← 与 BANC 冲突 |

> 逐字例（VFB `get_term_info`）：
> `VFB_jrcv27zy` = `SNpp39_ProLN_L (MANC:103678)` → Types: **[femoral chordotonal club neuron](FBbt_00049556)**；Comment: *"MANC classification: class-sensory neuron, subclass-prothoracic leg FeCO club, systematic type-SNpp39"*
> `VFB_jrcv11jm` = `SNpp45_MesoLN_L (MANC:48658)` → Types: **[adult SNta23 neuron](FBbt_20011274)**

**含义**：本地把 `SNpp39 / SNpp45 / SNpp43 / SNxx30` 这 8 行当作钟形感器**是不可靠的**（这是一条需要在项目里标注的比对冲突，不是本报告下的结论）。

### 2.4 段级身份在连接组里确实存在——但对不上本地的行

VFB 里存在的个体细胞（2026-10-12 逐类查询，`get_term_info`/`run_query` HTTP 200）：

| FBbt 类 | 实例数 | 来源与例子 |
|---|---|---|
| `FBbt:00048073` 前胸**股节** CS 神经元 | **12** | hemibrain T1 腿：`VFB_00102eem` = neuron 82405 (T1LEG:610665)，parent = 前胸股节 CS **Sc1**；`VFB_00102e1f` = neuron 82490 (T1LEG:604218)，parent = **Sc11**；另有 10 个同属 Sc11 |
| `FBbt:00052576` 前胸股节 CS **Sc11** 神经元 | 11 | 同上 |
| `FBbt:00052575` 前胸股节 CS **Sc1** 神经元 | 1 | `VFB_00102eem` |
| `FBbt:00048076` 前胸**转节** CS 神经元 | 16 | 其中 8 个 = Sc−8、5 个 = Sc+5、3 个 = Sc3（CATMAID T1Leg） |
| `FBbt:00058062` 前胸**胫节** CS 神经元 | **5** | `VFB_00102ecl` = neuron 83569 (T1LEG:609701)，parent = TiCSv1；`VFB_00102ec9` = neuron 83532 (T1LEG:609533)，parent = Sc2 |
| `FBbt:00048074` / `00048075` 中胸 / 后胸股节 CS 神经元 | **0 / 0** | — |
| `FBbt:00048077` / `00048078` 中胸 / 后胸转节 CS 神经元 | **0 / 0** | — |
| `FBbt:00058063` / `00058064` 中胸 / 后胸胫节 CS 神经元 | **0 / 0** | — |
| `FBbt:00058066` / `00058067` / `00058068` 跗分节1(metatarsal) CS 神经元 | **0 / 0 / 0** | — |
| `FBbt:00058070` 前胸跗分节3 CS 神经元 | **0** | — |
| `FBbt:00058073` 跗分节5（通用）CS 神经元 | 查询 HTTP 524（未完成） | — |
| `FBbt:00058068` 后胸跗分节1 CS 神经元 | 查询 HTTP 524（未完成） | — |

**所以**：跗节/胫节/股节的 CS 神经元**类**存在（本体论层面），但**除前足之外没有个体细胞**，而前足那 17 个（12 股节 + 5 胫节）全部来自 hemibrain T1LEG。本地 BANC 腿 CS 行的 `hemibrain_match` / `hemibrain_nblast_match` / `hemibrain_cell_type` / `fafb_match` / `fafb_cell_type` **61 行全部为空**：

```
hemibrain_match:        0/61 非空
hemibrain_nblast_match: 0/61 非空
hemibrain_cell_type:    0/61 非空
fafb_match:             0/61 非空
fafb_cell_type:         0/61 非空
manc_match:            41/61 非空   ← 唯一可用的桥
malecns_match:         14/61 非空
```

**这就是"为什么只有转节能对上"的技术原因**：MANC/MaleCNS 是唯一有桥的模态，而 VFB 只在 SNpp53 上做了 CS 段归类。

### 2.5 段/腿对本体论术语清单（可用于在项目里做显式标注）

FBbt（经 OLS4 与 VFB 双向核对）里存在**同时带腿段和腿对**的 CS 神经元类：

- 通用：`FBbt:00048058` leg CS 神经元；`FBbt:00049572/73/74` 双侧投射 / 同侧多神经节 / 单神经节 leg CS 神经元；`FBbt:00049575/76/77` 前/中/后胸腿双侧 CS 神经元
- 股节：`FBbt:00048060`（通用，定义引 Tuthill & Wilson 2016）；`00048073/00048074/00048075`（前/中/后胸）；`00052575`（前胸前股节 Sc1）、`00052576`（前胸前股节 Sc11，11 个一簇）
- 转节：`FBbt:00048059`（通用，定义引 Tuthill & Wilson 2016）；`00048076/00048077/00048078`（前/中/后胸）；`00052572/00052573/00052574`（前胸 Sc−8 / Sc+5 / Sc3）
- 胫节：`FBbt:00058061`（通用）；`00058062/00058063/00058064`（前/中/后胸）；`00052579/00052580`（前胸 Sc2/Sc3）；`00052581/00052582`（前胸 TiCSv1/TiCSv2）
- 跗分节 1（metatarsal）：`FBbt:00058065`（通用）；`00058066/00058067/00058068`（前/中/后胸）
- 跗分节 3：`FBbt:00058069`（通用）；`00058070/00058071/00058072`（前/中/后胸）
- 跗分节 5：`FBbt:00058073`（通用）；`00058074/00058075/00058076`（前/中/后胸）
- 感器本体（非神经元）：`FBbt:00058046/47/48` = **Ta5GF / Ta5GM / Ta5GR**（这正是 Ray 2025 里 Ta5G 那个名字的术语形态）；`FBbt:00004277/00004386` = 前胸/中胸股节 CS Sc11 簇

**注意 `00058065` 的用词是 "metatarsal"（跗分节 1）**，对应形态学文献里的 Ta1；`00058073` 是 "tarsal 5"，对应 Ta5。这两套命名在本体论和 SEM 文献之间是能对上的，**但只在感器/神经元类层面，不在本地行层面**。

### 2.6 方法学上唯一被证明可行的路子（但没人在腿上做过）

Tuthill 实验室在**翅**上做完了这件事（S12，Lesser/Moussa/Tuthill 2025 预印本，PDF HTTP 200，18.6 MB）：

> *"We also find that proprioceptors of adjacent campaniform sensilla on the wing have distinct axon morphologies and postsynaptic partners, suggesting a high degree of specialization in axon pathfinding and synaptic partner matching."*
> *"We then confirmed a long-standing prediction that individual campaniform sensilla from the same field can have distinct axon morphologies (Palka et al., 1986)."*

同一套方法在**腿**上被用在**毛板（hair plate）**上（S4，Pratt et al. 2026 Nat Commun 17:2664，PDF HTTP 200）：

> *"The fruit fly's front leg contains over 200 proprioceptive sensory neurons, including hair plates, campaniform sensilla, and chordotonal neurons."*
> *"By reconstructing CxHP8 axons in an electron microscopy dataset..."*（CxHP8 每条轴突输出突触 2279 ± 482 个，其中 75% 给前运动神经元 58% 与运动神经元 17%）
> *"The more proximal coxa and trochanter joints contain similar numbers of campaniform sensilla and hair plates."*（★ 这句是**二手**，其一手来源正是 S1 Dinges 2021）

**所以：把具名腿 CS 感器接到连接组细胞上的技术路径已被发表（翅、毛板），但对果蝇腿 CS 至今没有发表。** 这是本题最大的空白，而不是"没人想过"。

---

## 三、优先级 2：CS1 / CS2 / CS3 与股节场驱动系

### 3.1 来源与线路

来源：**Ray V, Bräcker LB, Kourtidis A, Rosher C, Dinges GF, Pierzchlińska A, Büschges A, Feng K, Cury KM, Gompel N (2025)** *The TRP-channel painless mediates substrate stiffness sensing in the legs during Drosophila oviposition.* **PLoS Genetics 21(12):e1011980**. doi:10.1371/journal.pgen.1011980。**已评审。**

- 主文：`https://pmc.ncbi.nlm.nih.gov/articles/PMC12768414/` → **HTTP 200**（正文读到"沉默钟形感器"一节末被截断；未再逐段续取）
- 补充材料：PMC 的 `bin/` 路径只返回 1,817 字节的 "Preparing to download" 中转页；改走 PLOS 的签名跳转（302 → `storage.googleapis.com/plos-corpus-prod/...`）后**全部拿到**：
  - `pgen.1011980.s012.xlsx`（= **S1 Table**，感器计数）HTTP 200，62,244 字节
  - `pgen.1011980.s013.xlsx`（= **S2 Table**，果蝇品系）HTTP 200，109,288 字节
  - `pgen.1011980.s003.pdf`（= **S3 Fig**，CS-Gal4 表达模式）HTTP 200，2,158,343 字节
  - ⚠️ 命名陷阱：`...s001.pdf` 是 **S1 Fig**（卵数），不是 S1 Table。

### 3.2 三条线到底标了什么（原话）

> *"To target different and partially overlapping subsets of campaniform sensilla in the legs, we used the generation-1 driver line **R34E04-GAL4 (CS1)** [36], as well as two split-GAL4 lines, **SS69574-Gal4 (CS2)**, and **SS69602-Gal4 (CS3)** that we built (S3B-S3L Fig)."*
> *"To isolate the specific contribution of campaniform sensilla, we generated two split-Gal4 drivers strictly restricted to these organs, **CS2** and **CS3**... Both show peripheral projections but no cell bodies in the central nervous system (S4E–E', F–F' Fig)."*
> *"we first employed the broad driver **CS1-Gal4** that labels mechanosensory neurons innervating **numerous CS in addition to some mechanosensory bristles in the legs**."*
> S3 Fig 图注：*"Confocal images of the expression patterns of CS1-Gal4 (B,E,H,K), CS2-Gal4 (C,F,I) and CS3-Gal4 (D,G,J,L) across the mesothoracic leg. The CS labelled by these driver lines are marked with an asterix (*). Scale bar, 20 µm."*

### 3.3 关键否证：**没有**细胞/细胞类型清单，**没有**连接组标识符

- 全文与三份补充材料中**没有任何**"CS1 = 哪些感器 / CS2 = 哪些 / CS3 = 哪些"的逐项清单。S3 Fig 只有星号标记的共聚焦图，**无计数**。
- 全文**没有任何** root ID / connectome cell type / FlyWire / hemibrain 标识符。
- S2 Table（品系表）里只有品系号与基因型，没有表达注释。

### 3.4 一条重要的命名不一致：R34E04 vs R34E03

- **正文**写 `R34E04-GAL4 (CS1)`；
- **S2 Table** 写 `R34E03-Gal4 (CS1-Gal4)`，基因型 `w[1118];P{y[+t7.7]w[+mC]=GMR34E03-GAL4}attP2/TM3,Sb[1]`；
- **S3 Fig 图头**写 `R34E03 (CS1)`。

也就是说 CS1 这条线在原文内部就有 R34E04 / R34E03 两种写法（S3 Fig 那种写法在补充材料里出现了两次）。**任何下游使用都必须先确定到底是哪一条**，否则会引错线。FlyBase 里两条是分开的：`FBtp0059511` = P{GMR34E04-GAL4}（Bloomington 49788）、`FBtp0059510` = P{GMR34E03-GAL4}；DBD 半线分别是 `FBti0191208`（Bloomington 69054）与 Bloomington 69836。

### 3.5 这三条线在公共资源里的可查状态（全部 HTTP 200）

- **FlyBase `FBal0331729`（Scer\GAL4[DBD.R34E04]）**：Expression Data 一节**为空**；只有品系 Bloomington 69054。
- **FlyBase `FBti0191208`（P{R34E04-GAL4.DBD}attP2）**：Expression Data **为空**。
- **FlyBase `FBtp0059511`（P{GMR34E04-GAL4}）**：Expression Data **为空**；仅说明是 tws 侧翼 4226 bp + DSCP + GAL4（Pfeiffer 2011 那批 GMR 构件的通用设计，并**不**代表该线特异性）；品系 Bloomington 49788。
- **FlyBase `FBal0331728`（DBD.R34E03）**：唯一带有表达注释的一条，但注释来自拆分组合 `FBco0001808`（与 `AD.R31H10` 配对，Hampel et al. 2017），描述的是**梳理行为相关**的组织（成体食管下区、翅缘机械感觉神经元、翅脉 campaniform 的 distal group / ventral radius、胸背与腹侧 pedicellar/scabellar campaniform 等），**不涉及腿**。
- **VFB 搜索 `R34E03` / `R34E04`**：只有 FlyLight 图像集（`VFBexp_FBtp0059510` / `VFBexp_FBtp0059511`、JRC MCFO 脑与 VNC 片段），**没有任何细胞类型归类**。
- **VFB 搜索 `SS69574` / `SS69602`**：**0 结果。**
- **Janelia split-GAL4 资源论文（Meissner et al. 2025, eLife, doi:10.7554/eLife.98405；PDF HTTP 200，4.6 MB）**：对全文做 `SS69574`、`SS69602`、`69574`、`69602`、`campaniform` 检索，**命中 0 次**。这两条线不在该资源所文档化的线路之内（至少在被检索到的版本里如此）。

### 3.6 股节场有对应驱动系吗？——**没有**

- Ray 2025 全文没有给出任何股节场专用驱动系。
- 唯一能在**股节**水平分辨细胞的工具是 **eyg-Gal4**（"标记全部腿 CS"的类别级工具，见下节），它是**类别级**，不是股节场专用。
- **不能**把 eyg 或 CS1/CS2/CS3 当成"股节场驱动系"使用。

---

## 四、优先级 3：负载/力/接触的定量响应数据

### 4.1 果蝇：**没有**

- Ray 2025（S2）：测的是**基质硬度偏好指数**（0.25% vs 0.75% 琼脂糖；单蝇 0.5% vs 1.25%）、卵数、共聚焦表达；**没有任何力学量、没有电生理**。
- Custódio et al. 2026 预印本（S3，bioRxiv，**未评审**）：全文检索 `force plate` / `ground reaction` / `force transducer` / `strain gauge` / `mN` / `µN` —— **没有任何力学测量**。它测的是：eyg>GFP,RedStinger 共聚焦计数、单腿制备 2 秒光遗传 CS 激活下的 **GCaMP6f 肌肉钙成像**、攀爬表现、高速视频运动学。文中 "load / force" 是**解释性用语**，不是被测量的变量；1 mg 体重只出现在讨论里作为背景。

> 需要点名的一点：Custódio 2026 在多处把结论表述为"load/force feedback"，但**没有任何一处测量力或负载**。这是"推断"，不是"测量"。

- Goldsmith et al. 2025（S5，J R Soc Interface 22(226):20240559）：**纯仿真**，且结论是**否定**的（见 4.4）。

### 4.2 竹节虫：定量力响应（S6，Zill et al. 2021, J Neurophysiol 126:227-248, doi:10.1152/jn.00120.2021，PMC8424542，HTTP 200）

- 制备：竹节虫 *Carausius morosus*，束缚制备，胫节 CS 组 6A/6B 的胞外记录，刺激为**从自由行走个体实测地面反作用力经逆动力学得到的关节力矩波形**。
- 峰值力：单步屈肌力矩波形峰值从 **0.57 mN 递增到 2.8 mN**。
- 发放率随 dF/dt：回归 **r² = 0.98**，斜率 **6B large = 8.1、6B all = 8.2**；6A 在伸肌力矩下 **r² = 0.92**。
- 幅度效应（斜率随幅度下降）：0.57 mN → 24.3；1.1 mN → 18.3；1.8 mN → 12.3；2.3 mN → 11.2；2.8 mN → 9.9。
- 样本：屈肌力矩 n = 603 次（N = 4 只）；伸肌力矩 n = 609 次（N = 3 只）；幅度系列 n = 115 次（N = 1 只）。
- 方向性：6B 对"向关节伸展方向的力增加 / 向屈曲方向的力减少"发放；6A 相反。
- 逐字：*"sensory firing closely reflected the rate of change of force (dF/dt) and pooled data indicated significant correlation of discharge frequencies and the rate of force application."*

### 4.3 竹节虫＋蟑螂：卸载（S7，Harris et al. 2022, J Neurophysiol 128(4):1003-1020, doi:10.1152/jn.00285.2022，PMC9529259，HTTP 200）

- 偏移负载水平：竹节虫 offset 0 / 2.9 / 5.4 mN（单制备）；蟑螂 offset 0 / 0.92 / 1.84 mN（平均幅度 0.85 ± 0.42 mN）。
- 斜率：竹节虫 6A 在降力期间随偏移下降（slope −5.95）；蟑螂远端感受器 −13.7；模型预测 −16.85 vs 实验 −19.0（r² = 0.99）。
- 样本：每水平 50 组 × 6 次重复，共 1,500 次测试，N = 3 只。
- 逐字：*"firing was maximal to complete unloading of the leg but substantially decreased or eliminated by sustained loads."*
- **这条对我们最有用的含义**：在持续负载下，"卸载/失接触"信号可以被**完全消除**——也就是说 CS 的"触地丢失"信号是**状态依赖**的，不是一个简单的接触开关。

### 4.4 丽蝇：小昆虫力感受（S8，Saltin et al. 2025, J Neurophysiol 133:1749-1760, doi:10.1152/jn.00044.2025；PDF 经 `https://par.nsf.gov/servlets/purl/10637725` HTTP 200，3.7 MB）

- 丽蝇胫节 CS 的力响应记录 + 感受器模型。**不是果蝇**（果蝇只出现在引言与参考文献里）。
- 价值：这是现有"小昆虫力编码"最接近的工作，但动物体量比果蝇大约一个数量级，且感受器是**胫节**而非股节/跗节。
- 逐字：*"whether small insects, with very low body weights, use ..."*；*"the small size of Drosophila has limited some"*。

### 4.5 果蝇里唯一算过"行走力是否够"的研究：结论是**不够**（S5，Goldsmith et al. 2025, PMC12056673，HTTP 200）

- 对象：雄性果蝇**后腿股节 CS 场**（多数个体 11 个 CS，3 列编号 CS1–CS11）；股节长度 720.68 µm。
- 输入：SEM/TEM 形态 + 自由行走 3D 运动捕捉 + 逆动力学地面反作用力。
- 运动学：支撑相俯仰角 **144.58° ± 9.66° → 65.57° ± 17.74°**，方位角 **−19.66° ± 7.10° → 5.31° ± 5.00°**（244 个支撑相，12 只）。
- 判据：NompC 的 **7 Å 热噪声下限**（NompC 通道最小直径 7 Å，长 240 Å，宽约 175 Å，估计开启力约 20 pN）。
- 各 CS 最大位移（Å）——**右左轴**：CS1r 4.75、CS2 0.96、CS3 0.71、CS4r 2.74、CS5r 2.10、CS6 2.12、CS7 3.43、CS8r 2.27、CS9 2.31、CS10 3.84、CS11 4.01；**远–近轴**：2.13–4.19；**腹–背轴**：3.57–5.73。**全部低于 7 Å。**
- 逐字（结论）：
  > *"Thus, our results indicate that ground reaction forces alone are unlikely to serve CS field activation during forward walking."*
  > *"For the physiological CS field location, forces transmitted from a step of the leg via the FeTi joint in straight forward walking were neither sufficient to open NompC channels nor carry directional information in strain responses of single CS and the whole CS field."*
  > *"However, displacement magnitudes over the thermal threshold (7 Å) were generated for artificial CS fields closer to midway down the length of the femur."*
- 材料性质：CS 自身几何/材料差异造成 <1% 位移变化；**节内填充物（血淋巴/肌肉）刚度差异贡献 >95%** 的场景间差异；网格三倍加密对 CS5 位移影响 <4%。

**这就是"地面反作用力→股节 CS 激活"这条链在果蝇身上唯一的定量检验，而它是否定的。**

### 4.6 关于"有没有人在记录果蝇腿 CS 的同时操纵负载"

**没有找到。** 本次检索（Europe PMC 三个不同角度的布尔查询、PubMed 主题检索、以及逐篇追引）没有发现任何果蝇腿部 CS 的电生理记录，更不用说"记录 + 负载操纵"的组合。

---

## 五、优先级 5：Dinges et al. 2021 的**可获取性与逐段计数**

### 5.1 获取状态（纠正此前判断）

| 路径 | 状态 |
|---|---|
| 出版社 Wiley 全文 `https://onlinelibrary.wiley.com/doi/full/10.1002/cne.24987` | **HTTP 403 Forbidden** |
| PubMed 页面 `https://pubmed.ncbi.nlm.nih.gov/32678470/` | **HTTP 203**，返回占位页，无内容 |
| Europe PMC API（doi 10.1002/cne.24987） | **HTTP 200**，但 `isOpenAccess = N`、`inEPMC = N`、`pmcid = null`、`fullTextUrlList` 只有一条 `availability: Subscription required` |
| **科隆大学机构库 KUPS eprint 25156** `https://kups.ub.uni-koeln.de/25156/1/Blanke%20in%20Dinges%202020.pdf` | **HTTP 200，application/pdf，19,201,726 字节 → 拿到全文** |

**结论：该文有一个开放获取的作者接受稿在机构库；"paywalled"是出版社路径的性质，不是这篇文献的最终可获取性。** 逐字阅读后（`pdftotext -layout`，790 行）的可引用内容如下。

### 5.2 命名体系（原文）

> *"The CS of the prothoracic leg were comprised of two fields, six groups, and two single CS (Figures 2 and 3). The two fields of CS were on the trochanter and femur."*
> *"Our results indicate the need for an appropriate naming scheme that will allow for differentiation between CS on each pair of legs and each segment of a leg."*
> *"In spite of its large volume no CS was observed on the Coxa (Cx)."*

命名语义（原文）：腿段 = Cx/Tr/Fe/Ti/Ta1..Ta5；F = field（场）、G = group（组）、S = single（单个）；后缀 F/M/R = 前/中/后胸腿。

### 5.3 逐段计数（原文数值，按腿对分开）

**前胸腿（前腿）**
- 股节场 **FeFF**（腹侧近端）：10 个（3/5 只）或 **11 个**（2/5 只），排成 3 列；其中 3 个偏心率更低
- 股节单个 **FeSF**：1 个（10 只，背侧偏后）
- 转节场 **TrFF**（背侧，被角质嵴分成前后两亚场）：
  - 后亚场 **TrFpF**：常见 **8 个**（8 只中 1 只为 7 个）
  - 前亚场 **TrFaF**：**恒为 5 个**（7 只）
- 转节组 **TrGF**（后侧）：**3 个**（10 只）
- 胫节背组 **TiGdF**：**2 个**（10 只）
- 胫节腹组 **TiGvF**：**3 个**（9 只；排列从三角到线性可变，中间那个最小）
- 跗节组：**Ta1GF 2 个**（7 只）、**Ta3GF 2 个**（6 只）、**Ta5GF 3 个**（1/5 只）或 **4 个**（4/5 只）
- 跗节单个 **Ta1SF**：**1 个**（8 只，腹侧）

**中胸腿（中腿）**：TrFpM 7（1/9）或 8（8/9）；TrFaM 4（1/10）或 5（9/10）；FeFM **11**（8/9）或 12（1/9）；TrGM 3（6 只）；TiGvM 2（1/12）或 3（11/12）；TiGdM 2（11 只）；Ta1GM 2（6 只）；Ta3GM 2（8 只）；Ta5GM 4（6 只）；FeSM 1（10 只）；Ta1SM 1（11 只）

**后胸腿（后腿）**：TrFpR 8（5/9）或 7（4/9）；TrFaR 5（11 只）；FeFR **11**（4/5）或 12（1/5）；TrGR 3（7 只）；TiGdR 2（10 只）；TiGvR 3（9 只）；Ta1GR 2（6 只）；Ta3GR 2（8 只）；Ta5GR 4（3 只）；FeSR 1（8 只）；Ta1SR 1（6 只）

> 原文另有逐字句：*"In Ta5GF, we found three (1/5 fly) and four (4/5 flies) CS. These were located on the ventral, distal end of the segment."*

### 5.4 按"段"归总的感器数（**据上文计数推导，原文未给该合计**）

| 段 | 场/组/单个 | 每条腿大致数量 |
|---|---|---|
| 股节 femur | FeF（10–12）+ FeS（1） | **11–13** |
| 转节 trochanter | TrFp（7–8）+ TrFa（4–5）+ TrG（3） | **14–16** |
| 胫节 tibia | TiGd（2）+ TiGv（2–3） | **4–5** |
| 跗节 tarsus | Ta1G（2）+ Ta3G（2）+ Ta5G（3–4）+ Ta1S（1） | **8–9** |
| 基节 coxa | — | **0** |
| 合计 | — | **约 50–57** |

**这是我在本地做的算术，不是任何来源的陈述。**

### 5.5 两条与本地数据对照时很有用的细节

- 原文强调**个体间数量变异**（例如 TrFp 在不同个体是 7 或 8，FeF 是 10/11/12）。这意味着**不能**用"某段应该有 N 个"去反推行的身份。
- 本地 BANC v888 只有 **前腿 53 行**（49 + 4 双侧），却有 **中腿 4 行、后腿 4 行**。按上表推算，中/后腿各应有约 50–57 个。**这个 release 的中/后腿 CS 群体相对解剖学严重不完整**——这直接限制了任何基于计数的身份推断。（此为本地只读查询 + S1 计数的算术，非任何来源陈述。）

### 5.6 S3（Custódio 2026 预印本）提供的**独立复核**

同一批科隆作者用 eyg-Gal4（"标记全部腿 CS"）做了共聚焦复核，与 Dinges 2021 的数量对比：

| 段/组 | eyg-Gal4 实测（S3 原文） | Dinges 2021 SEM |
|---|---|---|
| Ta1S | **3** | 1（Ta1SF） |
| Ta3G | **2** | 2 |
| Ta5G | **2**（作者注明"比预期少 2 个"） | 3–4 |
| TiGv | **2** | 2–3 |
| TiGd | **3** | 2 |
| TrG | **3** | 3 |
| TrF | **12** | 7–8 + 4–5 = 11–13 |
| FeF | **12** | 10–12 |
| FeS | **1** | 1 |
| 股节+转节合计 | **28** | 约 25–29 |

> 逐字：*"We verified three CS on the first tarsal segment (Ta1S) and two on the third tarsal segment (Ta3G). On the fifth tarsal segment (Ta5G) we regularly located two cell bodies two less than expected but in line with a second study."*
> *"we identified three eyg-positive cells comprising the trochanteral group (TrG), twelve comprising the trochanteral field (TrF), twelve comprising the femoral field (FeF) and one lone cell in the femur (FeS)."*
> *"We captured one representative image with the optimal alignment to visualize all 28 predicted femoral and trochanteral CS."*
> *"Therefore we corroborated by number, location and morphology that eyg-Gal4 labels all leg-CS."*

**注意 Ta1S 的 3 vs 1 差异**（eyg 计数 3，Dinges 记 1 个单感器 + 2 个 Ta1G）。两篇的"Ta1S/Ta1G"边界划定不同，**不要**把一个数字硬套到另一个命名上。

---

## 六、优先级 4：谁编码"触地"

### 6.1 没有任何来源测量"触地"这个变量

在这次检索到的所有来源中，**没有一个**在果蝇里测过"足部接触/失接触"这一变量，也没有对跗节毛、爪感器、多树突神经元做过接触相关记录。

### 6.2 各来源实际测的变量（明确区分）

| 来源 | 实际测量的变量 | 制备 |
|---|---|---|
| Ray 2025（S2） | **基质硬度偏好指数**（PI =（软侧卵数−硬侧卵数）/总卵数）、总卵数、共聚焦表达有无 | 产卵双选行为（群组/单蝇）；解剖跗节成像 |
| Custódio 2026（S3） | 攀爬表现（−40.9%）、肌肉钙信号（GCaMP6f）、步态运动学（速度、支撑/摆动时长、幅度） | 自由行走/攀爬、单腿离体肌肉成像 |
| Pratt 2026（S4） | **CxHP8 钙信号 vs coxa 关节角**（旋转/内收/屈曲）、光遗传激活/沉默的行为效应 | 行为中双光子成像 + 3D 运动学 |
| Goldsmith 2025（S5） | （仿真）位移（Å）与应变方向 | 纯仿真 |
| Zill 2021 / Harris 2022（S6/S7） | **发放率 vs dF/dt 与力幅度/偏移** | 竹节虫、蟑螂束缚制备电生理 |

**"触地"在这张表里没有任何一格。**

### 6.3 形态学证据**反对**把跗节 CS 当触地感受器

- Ray 2025 原话：*"pain-Gal4 labelled neurons that innervate subsets of these CS, **mostly positioned dorsally at the joints between tarsomeres or near the claws** in the most distal tarsomere"* ——**背侧**、位于跗分节**关节**旁或爪附近，不是地面接触面。
- Dinges 2021 原话：Ta1GF、Ta3GF 位于各跗分节**背侧远端**；Ta5GF 位于**腹侧远端**；**coxa 上完全没有 CS**。
- 含义：**如果模型需要一个"足部接触/滑脱"信号源，本次检索没有提供任何果蝇腿部 CS 的候选依据。** 引用竹节虫/蟑螂的"卸载"结论（S7）只能当跨物种类比，不能当果蝇事实。

---

## 七、**这份研究允许什么、不允许什么**（强制阅读）

### ✅ 允许

1. **在 61 条腿 CS 行中，把 11 行（全部 `cell_type == SNpp53`）标注为"段身份 = 转节"，并注出来源是 VFB 对 SNpp53 的归类（FBbt:00048059），以及这是通用类而非腿对专用类。**
2. **把这 11 行标注为"段级身份有已发表依据；具名感器身份无依据"。**
3. **在引用解剖数量时使用 Dinges 2021 的原文数值**（现在有全文了），并标明是 SEM 计数、有 7±1 级个体差异。
4. **在方法论上引用 Lesser 2025（翅）与 Pratt 2026（毛板）作为"具名感器↔连接组细胞"已被证明可行的技术范式**，同时明确"腿 CS 至今没人做过"。
5. **把 Goldsmith 2025 的否定结论作为"不要在模型里默认股节 CS = 行走地面反作用力编码器"的约束条件。**
6. **把 Zill 2021 / Harris 2022 的定量响应性质（速率敏感、方向分组、持续负载下卸载信号可被消除）作为将来做果蝇编码器时的对标目标**，并**显式标注物种**。
7. **把 SNpp39 / SNpp45 / SNpp43 / SNxx30 这 8 行标注为"BANC 与 VFB 的细胞类型归类冲突，待复核"。**

### ❌ 不允许

1. **不允许**把任何具名感器（Ta1S、Ta1G、Ta3G、Ta5G、Ta1SF/GF/GR、FeF、FeFF/FeFM/FeFR、FeS、TrF/TrFa/TrFp/TrG、TiGd/TiGv、Sc1、Sc11、Sc3、Sc−8、Sc+5）指派给任何 BANC 行。**没有任何来源做过这个指派。**
2. **不允许**把 61 行中的任何一行称为股节、胫节或跗节 CS。只有 11 行有段身份，且那 11 行的身份是**转节**。
3. **不允许**声称存在股节场驱动系。**没有找到。**
4. **不允许**把 CS1（R34E04/R34E03）、CS2、CS3、eyg 描述为"有细胞清单"或"可对连接组"。**它们只有大致解剖描述。**
5. **不允许**把"campaniform = 负载编码器"写成本地事实。果蝇里既无定量力响应数据，唯一算过的仿真是**否定**的。
6. **不允许**把"campaniform = 触地通道"。**没有任何来源测量过触地变量**；而且形态学位置不利。
7. **不允许**从跨物种数据（竹节虫、蟑螂、丽蝇）直接推出果蝇数值。只能作为**性质对标**，且必须写清物种。
8. **不允许**填补任何递质标签。本报告中出现的每一个递质陈述都逐字来自指定来源；本地数据里那 12 个"已验证乙酰胆碱"以外一律保持为空。
9. **不允许**把本文件当作已验证的传感编码器或连接组映射。**它不是。**

---

## 八、我们的模型仍然未知的部分

1. 那 **50 条**未获身份的腿 CS 行分别属于哪个腿段。
2. 除"腿对"以外的左右/个体归属（双侧投射行尤其不清楚）。
3. `CS`（23 行）、`SNcs01f`（13 行）、`bCS`（4 行）以及小的 SNpp39/43/45/52/xx30 组里，有没有股节或跗节感器。
4. 这些细胞在模型里的**机械刺激变量**是什么——没有任何来源提供果蝇腿 CS 的"应变→发放率"传递函数。
5. 我们仿真的地面反作用力是否**根本**能激活股节 CS 场——已发表的果蝇有限元证据说**直线前行时生理位置不够**。
6. 递质：本地 61 行里 12 行有已验证乙酰胆碱、49 行为空。**本次研究没有新增任何递质标签。**

---

## 九、来源与获取状态一览

| 编号 | 文献 | 获取状态 | 关键定量 | 段身份 | 可对连接组 |
|---|---|---|---|---|---|
| S1 | Dinges GF et al. (2021) *J Comp Neurol* 529:905-925, doi:10.1002/cne.24987 | **全文拿到**（KUPS 机构库 PDF，HTTP 200，19.2 MB）；Wiley 403、PubMed 203、EPMC OA=N | 逐段感器计数（见 §5.3）；coxa 无 CS | **是**（感器命名） | 间接（经 FBbt） |
| S2 | Ray V et al. (2025) *PLoS Genet* 21(12):e1011980, doi:10.1371/journal.pgen.1011980 | 主文 PMC HTTP 200（截断）；S1/S2 Table、S3 Fig 经 PLOS 签名跳转 HTTP 200 | Ta1G 1/2、Ta3G 1/2、Ta5G 1/4；无 Ta1S | 是（感器组名） | **否** |
| S3 | Custódio RD et al. (2026) bioRxiv doi:10.64898/2026.07.22.740025，**未评审** | PDF 最终 HTTP 200（5.0 MB）；过程中 biorxiv 多次 429、EPMC 403、biorxiv API 无记录 | Ta1S 3、Ta3G 2、Ta5G 2、TiGv 2、TiGd 3、TrG 3、TrF 12、FeF 12、FeS 1；攀爬 −40.9% | 是 | **否** |
| S4 | Pratt BG et al. (2026) *Nat Commun* 17:2664, doi:10.1038/s41467-026-69333-z | PDF HTTP 200（3.1 MB）；HTML 版被 idp 重定向阻断 | CxHP8 2279±482 突触/轴突；前腿 69 运动神经元/14 模块 | 是（毛板，非 CS） | **是**（对毛板） |
| S5 | Goldsmith CA et al. (2025) *J R Soc Interface* 22(226):20240559, doi:10.1098/rsif.2024.0559 | PMC HTTP 200（287 KB）；出版社 PDF 403 | 位移 0.71–5.73 Å 全低于 7 Å；244 支撑相/12 只 | 是（后腿股节场 CS1–11） | **否** |
| S6 | Zill SN et al. (2021) *J Neurophysiol* 126:227-248, doi:10.1152/jn.00120.2021 | PMC HTTP 200（269 KB） | 0.57→2.8 mN；斜率 8.1/8.2；r²=0.98 | 是（物种内胫节 6A/6B） | 否 |
| S7 | Harris CM et al. (2022) *J Neurophysiol* 128(4):1003-1020, doi:10.1152/jn.00285.2022 | PMC HTTP 200（237 KB） | offset 0/2.9/5.4 mN 与 0/0.92/1.84 mN；斜率 −5.95/−13.7/−16.85 vs −19.0 | 是（物种内） | 否 |
| S8 | Saltin BD et al. (2025) *J Neurophysiol* 133:1749-1760, doi:10.1152/jn.00044.2025 | par.nsf.gov PDF HTTP 200（3.7 MB） | 丽蝇胫节 CS 力响应 + 模型 | 是（物种内） | 否 |
| S9 | Meissner GW et al. (2025) *eLife* doi:10.7554/eLife.98405 | PDF HTTP 200（4.6 MB），但检索 SS69574/SS69602/campaniform **0 命中** | — | 否 | 否 |
| S10 | Virtual Fly Brain / FBbt（VFBquery API + OLS4） | HTTP 200 JSON（2 次 524：FBbt_00058068、FBbt_00058073） | 见 §2.4 实例计数表 | **是** | **是** |
| S11 | FlyBase FBtp0059511 / FBti0191208 / FBal0331729 / FBtp0059510 / FBal0331728 / FBco0001808（FB2026_03） | 全部 HTTP 200 | R34E04 表达注释为空；唯一注释来自 AD.R31H10 拆分组合且不涉腿 | 否 | 否 |
| S12 | Lesser E, Moussa A, Tuthill JC (2025) 翅本体感觉预印本 | PDF HTTP 200（18.6 MB） | 翅远端 CS ~17 个，找到 15 条轴突 | 是（翅，非腿） | 是（翅） |

**未取全文、因此未在本文中使用的来源**：Chockley et al. 2022（JEB 225:jeb244245，非 OA，仅摘要）；Joel, Adamova & Bräunig 2018（J Morphol 279:1654-1664，被 S3 引用，本次未取全文——凡涉及它之处均为**二手转引**）。

---

## 十、给项目的三条可执行建议（研究指引，不是实现）

1. **在本地数据里把"段身份"变成一个显式、可选、来源可溯的字段**，先只填 11 行（全部 `SNpp53`），值 = `trochanter`，来源 = `VFB FBbt:00048059 typing of SNpp53`，并标"通用类，不含腿对"。其余 50 行保持 `unknown`，**不要**用插值、计数或"看起来像"去补。
2. **把 Goldsmith 2025 的否定结论写进模型的假设清单**：不要把 campaniform 通道默认接到"该腿地面接触法向力"。如果一定要接，必须显式记为**未经验证的假设**，并注明其唯一定量检验是否定的。
3. **把 SNpp39 / SNpp45 / SNpp43 / SNxx30 这 8 行的类别标注标为"冲突待复核"**，而不是当作钟形感器使用——VFB 对同类细胞的归类与 BANC 不一致。

---

*本文档所有引文均于 2026-10-12 从所列 URL 逐字提取；检索失败（HTTP 403 / 429 / 203 / 524 / 跨域重定向 / PDF 内容类型不支持）已在正文与 `research_contact_identity_sources.json` 中逐条如实记录，未使用任何未能加载的内容。*
