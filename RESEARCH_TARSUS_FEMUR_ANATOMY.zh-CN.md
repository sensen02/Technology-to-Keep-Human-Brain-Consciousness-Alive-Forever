# 果蝇腿跗节（tarsus）与股节（femur）肌肉解剖学原始文献调研

> **【重要声明 — 必读】**
>
> 1. **本文是手工调研（手工研究，不是成品）。** 由子代理通过网页检索与抓取逐条核实完成，未经同行评审，也不是经过验证的数据集或代码。其中所有"数值"均为从原始文件/网页**逐字复制**并标注来源；除此之外**没有任何数值是本代理推算或补全的**。
> 2. **research guidance only —— 仅供研究指引用。** 本文不是模型的输入数据。
> 3. **禁止把任何附着坐标写入模型，除非某个来源明确写明了它。** 本文件明确区分"来自文献的数值"与"缺失、必须继续声明为假设（declared）的量"。凡本文标注 `【缺】` 或 `【仅定性】` 的项，**不得**被当作测量值填入 MuJoCo 模型。
> 4. 本项目已有的一次失败尝试（用**自造**附着几何 + 自造关节轴加一个跗节自由度，10% 胸高逃逸时间由 23.15 ms 退化到 8.75 ms）正是本条规则的反面教材。本次调研的首要结论恰恰是：**那一步很可能根本不需要自造**，见 §5。
>
> 检索完成时间：本次会话。所有 HTTP 状态码为该次实际抓取的真实返回码。

---

## 0. 一句话结论（先给可操作结论）

| 问题 | 结论 |
|---|---|
| 有没有现成的、开源的、已发表的果蝇模型**已经含跗节+股节自由度**？ | **有。flybody（TuragaLab，Apache-2.0）已经含每腿 9 个关节，包含 tibia–tarsus 及 4 个跗分节间关节，并给出真实关节轴、角度范围与关节中心坐标。** |
| 有没有现成模型含**肌肉附着点坐标**？ | **有，但只到胫节为止。FlyMimic（Özdil et al., ICLR 2026，Apache-2.0）给出 15 条肌肉的真实附着点三维坐标；但它的跗节是 WeldJoint（焊死），且明确说明"胫节内肌肉被排除"。所以跗节压肌/提肌的附着坐标依然【缺】。** |
| 跗节压肌/提肌的**定性**解剖可靠吗？ | **可靠**，有 FlyBase/果蝇解剖本体（FBbt）直接引自 Soler et al. 2004 的定义；另有 Azevedo et al. 2024 Nature 补充材料给出**逐肌肉的动作定义与图号**（见 §2.7）。 |
| 跗节压肌/提肌的**定量**附着坐标/力臂/肌腱长度？ | **【缺】—— 在任何公开来源中都没有找到。** 原因已查明且是硬性的：**唯一可能含这些坐标的 X 射线纳米断层（XNH）数据集"数据止于胫节约 3/4 处，胫–跗关节不可见"**（Azevedo et al. 2024 补充材料逐字，见 §3.6）。Soler et al. 2004 全文亦关闭获取。 |
| **有没有"站着时胫–跗关节"的定量数据？** | **有，而且是本次最有价值的新发现。** Oeftger et al. 2026（iScience，CC BY，可完整读取）给出**胫–跗角 ε 随载荷的斜率**（wild-type **−0.31 °μN⁻¹**）、坐姿单腿载荷 **1.7 μN**、肌腱直径 **3.25 ± 0.15 μm**、腿刚度 **13.1 ± 7.97 μN mm⁻¹**。见 §2.8。 |
| 股节收肌（femur reductor）、股节压肌、附属肌/长腱肌的**定性**解剖？ | **可靠**（FBbt 逐条，见 §1.5–1.7）：起点/止点/纤维数齐全，例如 **femoral reductor = 转节基部 → 近端股节，4–5 纤维**；**femoral depressor = 转节内 → 近端股节，7–9 纤维**；**accessory tibial flexor (tirm/tibial reductor) = 远端股节 → 近端胫节，29 纤维**。 |
| 股节肌肉的**定量**附着坐标/力臂？ | **【缺】**。FlyMimic 有**股–胫**两条肌肉（`LFTibia_flex_93434`、`LFTibia_extensor_93932`）的真实附着坐标（见 §3.2），**是唯一可用的股节相关附着几何**；但**股节收肌/股节压肌本身的附着坐标无任何来源**，力臂全部【缺】。 |
| 转节–股节关节（trochanter–femur）的轴与范围？ | **有现成数值可抄**：flybody 给出 `femur_twist`（轴 `0 1 0`，`-1..1`）与 `femur`（轴 `1 0 0`，`-0.15..2`），并按腿给出 `springref`（见 §2.1）。**但那是模型约定，不是解剖测量。** 另需注意：**该关节是否可动在解剖学上仍有争议** —— Haustein et al. 2024 逐字："its functional role in Drosophila is debated, with opinions ranging from it being **fused and immobile** ... to **at least limited mobility**"。 |
| **关节轴的真实朝向（不是模型约定）？** | **有，且是最有价值的补充发现。** Haustein et al. 2024（§3.7）用**真实标本 μCT 的关节髁位置**导出关节轴，证明果蝇腿关节轴是**斜的（oblique）**而非正交，并**开放了髁位置数据**（Harvard Dataverse DOI 10.7910/DVN/XWREDA，含 `j_tib_tar_1-1.obj` 等）。**本项目可自行从该数据导出真实轴朝向。** |
| 所以现在能做什么？ | 见 §5：**跗节自由度、关节轴数值、角度范围、以及转节–股节/股–胫关节可以直接抄 flybody 的已发表数值（不是自造）；关节轴的真实朝向可从 Haustein 的开放 μCT 数据自行导出；肌肉驱动只能沿用"单自由度等效作动器"或 flybody 的固定肌腱耦合，力臂/附着点必须继续声明为假设。** |

---

## 1. 从 FlyBase / 果蝇解剖本体（FBbt）取得的肌肉解剖（定性，权威）

FlyBase 的 FBbt 本体条目**直接引用 Soler et al. 2004** 撰写定义，是目前可公开获取的最权威的定性描述。以下 Definition 均为逐字摘录（英文原文，避免翻译引入歧义）。

### 1.1 跗节压肌（tarsal depressor muscle, TaD / tadm）

- 本体 ID：**FBbt:00058846**
- URL：<https://flybase.org/cgi-bin/cvreport.pl?id=FBbt:00058846>（HTTP 200）
- 检索状态：**公开可读（HTTP 200）**
- 逐字 Definition：

  > "Adult leg muscle that extends along the tibia and attaches to the proximal tarsus, via the tarsal depressor tendon (Soler et al., 2004). It is composed of 20 to 33 fibers, with a markedly lower number of cells in the prothoracic leg than in the mesothoracic and metathoracic legs (Soler et al., 2004). It extends the tibia-tarsus joint, which raises the fly's body when standing (Azevedo et al., 2024)."

- 别名：`TaD muscle` / `tadm` / `tarsus depressor muscle`
- 来源文献：FlyBase:FBrf0180184（Soler 2004）、FBrf0186027、FBrf0218373、FBrf0259940（Azevedo 2024）

**可以确定的（定性）：**
- **起点（origin）：沿胫节（tibia）内延伸** —— 即肌纤维位于胫节内部。
- **止点（insertion）：近端跗节（proximal tarsus）**，通过 **tarsal depressor tendon（跗节压肌腱）** 附着。
- **作用（action）：伸展 tibia–tarsus 关节**；站姿时该动作**抬高**身体。
- 纤维数：20–33（**这是跨三条腿合并的范围，不是单条肌肉的数**）。

**⚠️ 纤维数需按腿分段（逐条核实自 FBbt 子条目）：**

| 本体 ID | 名称 | 逐字 Definition（摘录） | 纤维数 |
|---|---|---|---|
| FBbt:00058872 | prothoracic tarsal depressor muscle **43** | "extends along the tibia, **medial to** the tarsal levator muscle **42**." | **20–24** |
| FBbt:00058930 | mesothoracic tarsal depressor muscle **74** | "...**medial to** the tarsal levator muscle **73**." | **28–33** |
| FBbt:00058959 | metathoracic tarsal depressor muscle **95** | "...**medial to** the tarsal levator muscle **94**." | **28–33** |

**⇒ 压肌位于提肌的「内侧（medial）」。** 配合同表所列提肌的"远端胫节、背侧"，可得二者在胫节内的**相对空间关系**：**TaD 偏内侧、TaL 偏外侧（lateral）且偏背侧，二者均在胫节（提肌在远端）。**

- **TaD 与 TaL 的动作方向互为正反**（Azevedo 2024）：TaD **伸展** tibia–tarsus 关节（站姿时抬身体），TaL **屈曲** 该关节（站姿时降身体）。详见 §2.7 的重要警告 —— **这个名字逻辑与动作不直观对应，建模时极易搞反。**

**补充（来自 Azevedo et al. 2024 补充材料，逐字）：**
- "**Tarsus depressor muscle** — Extend the tibia-tarsus joint. **The joint is flexed when the fly is standing.** Extension moves the fly's body away from the substrate."（别名 `tadm`）
- "**Tarsus retro depressor muscles** — Muscle fibers originate on tibia cuticle that is **distal to their insertion sites on the tarsus depressor tendon**."（别名 `tarm 1 and 2`）→ **即：retro depressor 分为 anterior/posterior 两条（FBbt:00059106 后侧 4–5 纤维、FBbt:00059107 前侧 3 纤维，合计 7–8），且纤维起点在其止点的远端（retrograde 排列）。**
- **重要更正史（FBbt:00059105 的策展注记，逐字）**："Soler et al. (2004) say that fibers of these muscles do not attach to a tendon, this is corrected in Azevedo et al. (2024), who show that they insert on the tarsus depressor tendon." → **即"tarm 是否止于腱"在 2004 年曾被误判，2024 年更正。二手转述时务必区分年份。**

> ⚠️ **"proximal tarsus" 的精确指认（到底是不是 tarsomere 1？）本文未能取得一手原文确认。** FlyBase 措辞为 "proximal tarsus"，未编号。**不得**据此断定是 tarsomere 1 还是 tarsomere 1 的近端某点。

### 1.2 跗节提肌（tarsal levator muscle, TaL / talm）

- 本体 ID：**FBbt:00058853**
- URL：<https://flybase.org/cgi-bin/cvreport.pl?id=FBbt:00058853>（HTTP 200）
- 逐字 Definition：

  > "Adult leg muscle located dorsally within the distal tibia, attaching to the proximal tarsus via the tarsal levator tendon (Soler et al., 2004). It flexes the tibia-tarsus joint, which lowers the fly's body when standing (Azevedo et al., 2024). It is composed of 8 to 10 fibers (Soler et al., 2004)."

- 别名：`TaL muscle` / `talm` / `tarsus levator muscle`

**可以确定的（定性）：**
- **起点：胫节远端内部、背侧（dorsally within the distal tibia）** —— 比压肌更靠远端、更靠背侧。
- **止点：近端跗节**，通过 **tarsal levator tendon**。
- **作用：屈曲 tibia–tarsus 关节**；站姿时**降低**身体。（注意：与压肌互为拮抗，且此处的"屈/伸"命名与压肌相反。）
- 纤维数：8–10。

### 1.3 跗节后压肌 / 跗节收肌（tarsal retro depressor muscle, tarm）

- 本体 ID：**FBbt:00059105**
- URL：<https://flybase.org/cgi-bin/cvreport.pl?id=FBbt:00059105>（HTTP 200）
- 逐字 Definition：

  > "Adult leg muscle with fibers that originate at the distal end of the tibial cuticle and attach to the tarsal depressor tendon in a position proximal their origin sites (Soler et al., 2004; Azevedo et al., 2024)."

- 别名：`tarm` / `tarsal reductor muscle`
- 意义：这是**第三条**作用于 tibia–tarsus 关节的肌肉。它**不独立止于跗节**，而是**并入同一条 tarsal depressor tendon**，且止点相对起点更靠**近端（proximal）**。对建模的含义：tibia–tarsus 关节由**至少 3 条肌肉**（TaD、tarm、TaL）驱动，其中两条共用一条腱 —— **单自由度双拮抗作动器在解剖上是不完整的**（但不完整的近似仍是合法选择）。

### 1.4 跗节前压肌 / "长腱肌"（pretarsal depressor muscle / long tendon muscle）

- 本体 ID：**FBbt:00058847**
- URL：<https://flybase.org/cgi-bin/cvreport.pl?id=FBbt:00058847>（HTTP 200）
- 逐字 Definition：

  > "Adult leg muscle attached via the long tendon to the unguitractor plate of the pretarsus. There are two of these muscles, one located in the femur and one in the tibia (Soler et al., 2004)."

- 别名：`long tendon muscle`

**这条最关键：它是一条"跨关节"肌肉** —— 一条肌肉在**股节**、一条在**胫节**，都通过**同一条长腱（long tendon）**连到前跗节的 **unguitractor plate**。

**细分（逐字来自 FBbt 子条目）：**
- **FBbt:00059117 femoral pretarsal depressor muscle** —— "composed of around **6 or 7 fibers** and is attached to the **posterior part of the femur** (Soler et al., 2004)."
- **FBbt:00059118 tibial pretarsal depressor muscle** —— "composed of around **3 fibers** and is attached to the **posterior part of the tibia** (Soler et al., 2004)."

**⚠️ 一段必须知道的历史误判（FBbt:00003324 策展注记，逐字）：**

> "This muscle was originally described as located in the **proximal tarsal segments** (where it was numbered **44, 75, and 96** in the prothoracic, mesothoracic, and metathoracic leg respectively) with a tendon extending to the pretarsus. It was later found that there are **no muscle fibers in any of the tarsal segments** and that the tendon originates much more proximally in the femur, with muscle fibers located in both the femur and the tibia (Chapman, 1998; Soler et al., 2004)."

**⇒ 这一条对本项目直接有用：文献上曾明确记载"**跗分节内没有任何肌纤维**"。这从解剖学上支持"跗分节间不需独立肌肉驱动"的建模选择。**

FBbt:00004658 **unguitractor plate** 逐字："A small ventral, median sclerite of the pretarsus that is partially invaginated into the end of the **5th tarsal segment** and is the attachment site of ... a long tendon connected to the pretarsal depressor muscle." → **长腱的远端附着点在「第 5 跗分节末端」的腹侧、正中硬化片。**

相关本体：**thoracic long tendon cell（FBbt:00058136）**，逐字 Definition：

  > "Cell that forms a tendon that originates in the proximal femur and extends to the pretarsus. The pretarsal depressor muscles in the femur and the tibia act on this tendon to retract the claws (Chapman, 1998; Soler et al., 2004)."

- **这对本任务极重要**：用户问的 "unguitractor / claw depressor" 与 "tarsus depressor" **在本体中是两个不同实体**。若模型里要把"爪的收放"与"跗节的弯曲"分开建，必须区分：
  - **TaD/tarm/TaL → 作用于 tibia–tarsus 关节**（改变跗节相对胫节角度）
  - **pretarsal depressor + long tendon → 作用于前跗节爪**（retract the claws），肌腱**起于近端股节、贯穿整条腿到前跗节**

### 1.5 股节收肌（femoral reductor muscle, ferm）

- 本体 ID：**FBbt:00058855**
- URL：<https://flybase.org/cgi-bin/cvreport.pl?id=FBbt:00058855>（HTTP 200）
- 逐字 Definition：

  > "Reductor muscle of the adult leg that originates at the base of the trochanter and attaches to the proximal femur. It is composed of 4 or 5 fibers (Soler et al., 2004)."

- 别名：`femur reductor muscle` / `ferm`
- **起点：转节基部（base of the trochanter）**；**止点：近端股节（proximal femur）**；纤维数 4–5。

### 1.6 股节压肌（femoral depressor muscle, fedm）

- 本体 ID：**FBbt:00059100**
- URL：<https://flybase.org/cgi-bin/cvreport.pl?id=FBbt:00059100>（HTTP 200）
- 逐字 Definition：

  > "Depressor muscle that originates in the trochanter and attaches to the proximal femur. It is composed of 7 to 9 fibers (Soler et al., 2004)."

- **起点：转节内（in the trochanter）**；**止点：近端股节**；纤维数 7–9。

### 1.7 "附属/长腱"类肌肉（用户特别问到的 accessory / long tendon）

| 本体 ID | 名称 | 逐字 Definition（摘录） | 起点 → 止点 |
|---|---|---|---|
| FBbt:00059104 | accessory tibial flexor muscle（别名 tibial reductor / TiR / tirm） | > "Muscle that originates in the distal femur and attaches to the proximal tibia (Soler et al., 2004; Azevedo et al., 2024). It is the shorter of two muscles that flex the femur-tibia joint ... It is composed of 29 fibers (Kuan et al., 2020)." | **远端股节 → 近端胫节**（股节–胫节关节的两条屈肌中较短者） |
| FBbt:00058845 | tibial depressor muscle（别名 tibial flexor / TiD / tidm） | > "Muscle that extends along the femur and attaches to the proximal tibia. It is composed of 26 fibers (Kuan et al., 2020). It flexes the femur-tibia joint (Azevedo et al., 2024)." | **沿股节 → 近端胫节**（屈股–胫关节） |
| FBbt:00058852 | tibial levator muscle（别名 tibial extensor / TiL / tilm） | > "Muscle that extends along the femur and attaches to the proximal tibia. It is composed of 36 fibers (Kuan et al., 2020). It extends the femur-tibia joint (Azevedo et al., 2024)." | **沿股节 → 近端胫节**（伸股–胫关节） |
| FBbt:00059096 | accessory trochanter levator muscle（别名 trochanter reductor / trrm） | > "Adult leg muscle that originates at the distal end of the coxa, on the opposite surface to the trochanter levator muscle, and attaches to the trochanter, in a similar position to the trochanter levator tendon (Soler et al., 2004; Azevedo et al., 2024). It flexes the coxa-trochanter joint (Azevedo et al., 2024). It is composed of 3 fibers (Soler et al., 2004)." | **远端基节 → 转节** |
| FBbt:00053098 | tarsal depressor tendon cell（别名 tadt） | > "Tendon cell of the tarsal depressor muscles (Soler et al., 2004; Azevedo et al., 2024)." | 腱细胞（肌腱本身） |
| FBbt:00053099 | tarsal levator tendon cell（别名 talt） | > "Tendon cell of the tarsal levator muscle (Soler et al., 2004)." | 腱细胞 |
| FBbt:00053090 | prothoracic trochanter extensor/depressor tendon cell（trdt） | > "Tendon cell that fibers of muscles that extend the prothoracic trochanter attach to." | 腱细胞 |
| FBbt:00053089 | prothoracic trochanter levator/flexor tendon cell（trlt） | > "Tendon cell of the prothoracic trochanter levator muscle (Soler et al., 2004)." | 腱细胞 |

**说明**：FBbt 命名里出现 `reductor / retro depressor / accessory` 三套词，指的是**不同实体**，不要混用：
- `reductor` = 收肌（使节段"收回"），如 femoral reductor、tibial reductor(=accessory tibial flexor)、trochanter reductor(=accessory trochanter levator)、tarsal reductor(=tarsal retro depressor)。
- `retro depressor` = 起点在远端胫节、止点在腱上更近端的压肌。

**从 FBbt 能得到的全部定量信息只有"纤维数"。没有任何一个 FBbt 条目给出附着点坐标、肌腱长度、力臂或关节轴。** 所有 FBbt 条目的定量来源都指向 Soler et al. 2004，而该文全文关闭获取（见 §4.1）。

---

## 2. 关节轴与活动范围（已有可用的真实数值）

### 2.1 flybody（含 tibia–tarsus 与跗分节间关节）— **推荐直接采用**

**这是本次调研最重要的发现。** 项目的 MuJoCo 模型若基于 flybody / flygym 血统，则**跗节自由度不需要自造**。

- 仓库：<https://github.com/TuragaLab/flybody>
- 许可：**Apache-2.0**（GitHub API `license.spdx_id = Apache-2.0`，HTTP 200 核实）
- 论文：Vaxenburg R, Siwanowicz I, Merel J, Robie AA, Morrow C, Novati G, Stefanidi Z, Both G-J, Card GM, Reiser MB, Botvinick MM, Branson KM, Tassa Y, Turaga SC. *Whole-body physics simulation of fruit fly locomotion.* **Nature 643:1312–1320 (2025).** DOI [10.1038/s41586-025-09029-4](https://www.nature.com/articles/s41586-025-09029-4)
- 我实际抓取的文件：`flybody/fruitfly/assets/fruitfly.xml`，**65,787 bytes**，HTTP 200（`https://raw.githubusercontent.com/TuragaLab/flybody/main/flybody/fruitfly/assets/fruitfly.xml`）
- 单位为 **CGS**（论文补充材料 §1.1 第 3 步明确："Model units were chosen to be CGS, for better numerical precision."）

**每腿关节链（6 条腿 × 9 关节 = 54 个腿关节；全模型 102 个关节）：**

```
coxa_abduct → coxa_twist → coxa → femur_twist → femur → tibia
  → tarsus → tarsus2 → tarsus3 → tarsus4 → tarsus5
```

论文补充材料 §1.1 第 1 步逐字给出的每腿 link 列表（即为权威结构定义）：

> `links = ['coxa', 'femur', 'tibia', 'tarsus', 'tarsus2', 'tarsus3', 'tarsus4']`

**关节轴与范围（逐字复制自 `fruitfly.xml` 的 default class；角度单位 radian）：**

| 关节 | class | axis | range | springref | 备注 |
|---|---|---|---|---|---|
| `femur_twist`（转节–股节扭转） | `twist_femur` | `0 1 0` | `-1 1` | — | T2 用 `twist_femur_T2` springref `0.608`；T3 用 `twist_femur_T3` springref `0.558` |
| `femur`（**股节伸展 DoF**） | `extend_femur_T1/2/3` | `1 0 0` | `-0.15 2` | T1 `-0.142` | 基类 `extend_femur` |
| `tibia`（**股–胫关节**） | `extend_tibia_T1/2/3` | `1 0 0` | `-1.35 1.3` | T1 `-1.29`、T2 `-1.34`、T3 `-0.292` | 基类 `extend_tibia` |
| **`tarsus`（胫–跗关节）** | `extend_tarsus_T1/T2/T3` | **`1 0 0`** | T1 `-0.7 1.2`、T2 `-1 1.8`、T3 `-0.8 1.2` | T1 `-0.242`、T2 `0.608`、T3 `0.258` | 基类 `extend_tarsus`，父类 `tarsus` 声明 `joint axis="1 0 0"` |
| `tarsus2..tarsus5`（**跗分节间**） | `extend_tarsus` | `1 0 0` | **`-0.36 0.36`** | — | `group=3`, `stiffness=0.1`, `damping=0.002` |
| 全部腿关节（父类） | `leg` | — | — | — | `stiffness=0.01`, `damping=0.01`, `geom density=1.18` |

**关节中心（body 锚点）坐标 —— 逐字复制自 `fruitfly.xml`，单位 CGS（cm）。左前腿 T1：**

| body | `pos`（相对父体原点） | `quat` |
|---|---|---|
| `coxa_T1_left` | `0.0317 0.0209 -0.0272` | `-0.532 0.787 -0.311 -0.0229` |
| `femur_T1_left` | `0 0.0437 0` | `0 0 0.252 0.968` |
| `tibia_T1_left` | `0 0.0697 0` | `0.186 0.162 0.677 -0.694` |
| `tarsus_T1_left` | `0 -0.051 0.00175` | `0.039 0.998 0.00674 0.0474` |
| `tarsus2_T1_left` | `-0.000165 0.0235 0.00174` | （无） |
| `tarsus3_T1_left` | `0.000113 0.0136 0.000556` | （无） |
| `tarsus4_T1_left` | `-0.000255 0.00909 0.0002` | （无） |
| `claw_T1_left` | `-7.12e-05 0.00906 -0.00081` | （无） |

其余 5 条腿（T1_right / T2_left/right / T3_left/right）的锚点已同样逐条核实存在且数值不同（论文补充材料 §1.1 第 5–6 步保证"左右经矢状面镜像、语义一致：正向=伸展/外展，负向=屈曲/内收"）。完整数值见配套 JSON 台账。

**由上述锚点直接得到的节段长度（本项目自行做的一次算术，非文献数值，仅供核对量级）：**

| 节段 | 长度 |
|---|---|
| coxa → femur | 43.700 mm（CGS 单位） |
| femur → tibia | 69.700 |
| tibia → tarsus | 51.030 |
| tarsus1 → tarsus2 | 23.565 |
| tarsus2 → tarsus3 | 13.612 |
| tarsus3 → tarsus4 | 9.096 |
| tarsus4 → claw | 9.096 |
| **合计 coxa → claw** | **219.799** |

> ⚠️ 上表是"由已发表锚点做的算术"，**不是文献直接给出的长度**；且单位是 flybody 的 CGS 单位体系。请勿与本项目模型的单位直接混用（见 §6 单位警告）。

### 2.2 跗节如何被驱动：flybody 的固定肌腱耦合（解释了"为什么加一个自由跗节 DoF 会变差"）

flybody 的 `fruitfly.xml` 中 `<tendon>` 段落逐字内容（6 条腿各一条）：

```xml
<fixed name="tarsus2_T1_left" class="extend_tarsus">
  <joint joint="tarsus2_T1_left" coef="1"/>
  <joint joint="tarsus3_T1_left" coef="0.5"/>
  <joint joint="tarsus4_T1_left" coef="0.5"/>
  <joint joint="tarsus5_T1_left" coef="0.5"/>
</fixed>
```

即：**tarsus2 与其后 3 个跗分节的关节被一条固定肌腱按 1 : 0.5 : 0.5 : 0.5 耦合，由单个作动器驱动。** 生成脚本 `make_fruitfly.py` 的注释逐字为：

> `# Tendon couples tarsi starting from tarsus2.`

论文补充材料 §1.1 第 10 步逐字解释：

> "Added tendons to the abdomen and the tarsi. These 'fixed' tendons are a simplification of spatially routed tendons, whose length corresponds to a linear combinations of joint angles, allowing a single actuator to act on multiple joints e.g., to flex the multiple links of the tarsus or abduct the entire abdomen."

**建模含义（高价值）**：真实的跗节不是"一个自由关节"，而是**多节耦合的链**。如果用单个自由 DoF 替换一个原本被动耦合的链，就等于改变了接触/被动动力学 —— 这很可能就是 23.15 ms → 8.75 ms 退化的机制来源之一（**这是本代理的推理，不是文献结论，需实验验证**）。

### 2.3 flybody 作动器配置（权威数字，来自 Nature 补充材料 Supplementary Table 2）

逐字摘录：

> "Added a total of 78 actuators:
> • 8 actuators in each leg (coxa: 3, femur: 2, tibia: 1, tarsus: 2), using desired angle (position) semantics, with gains chosen so that a force of approximately one body weight can be applied at the end-effector at the base pose."

> **Supplementary Table 2**：`position` — legs (excluding tarsi) **42**；`position, coupled by tendon (several DoFs per actuator)` — tarsi **6**；`adhesion` — leg tips 6；`labrum` 2；`torque` — wings 6；`position` — neck 3、proboscis 5、antennae 6；`abdomen` 2；**total 78**。

> 另：每个 actuator 默认驱动一个 DoF；tarsi 是"多 DoF 耦合到单作动器"。

### 2.4 肌肉质量/几何对照（补充材料 Supplementary Table 1，逐字）

| Body part | Mass (mg) |
|---|---|
| head | 0.15 |
| thorax | 0.34 |
| abdomen | 0.38 |
| **leg (each)** | **0.0162** |
| wing (each) | 0.008 |
| fly total | 0.983 |

> "Averaged over 52 female flies."

### 2.5 其他来源给出的关节轴（FlyMimic，仅到胫节）

详见 §3.2。其 `joint_LFTibia_pitch` 轴为 `0 1 0`（**与 flybody 的 `1 0 0` 不同** —— 说明两套模型的局部坐标系不同，**不可直接互换轴向量**）。

### 2.6 关于"关节轴相对腿平面的朝向"与实测 ROM —— **【缺】**

- **没有任何来源**给出"胫–跗关节轴相对腿平面的夹角"这类**几何朝向**的定量值。
- flybody / FlyMimic 给出的是**各自模型局部坐标系下的轴向量**（`1 0 0` / `0 1 0`），这**是模型约定，不是解剖测量**。把 `1 0 0` 当作"解剖学上关节轴指向 X"是错误解读。
- 实测 ROM：**只找到 femur–tibia 关节的实验范式描述，没有找到系统的 tibia–tarsus / 跗分节间 ROM 测量**。见 §4.2（Mamiya 2022）。
- flybody 论文自己**也不声称**其关节轴是解剖测量值。逐字（Vaxenburg et al. 2025, Nature，PMC12310536）：

  > "Biological joints were modelled as single hinge joints (1 DoF) or as combinations of two or three hinge joints (2 and 3 DoFs, respectively). These joint models are **simplified approximations**, particularly for complex articulations such as the neck joint, wing hinge and thorax-coxa articulation."

  > "We manually segmented 67 body components using Fiji, then simplified their meshes in Blender ... and defined the kinematic tree by linking them at **66 identified joint locations** (Fig. 1b,d and Extended Data Fig. 1), **yielding 102 degrees of freedom (DoFs)**."

  > "Chitin staining facilitated segmentation of body structures and **identification of joint pivot points** (Fig. 1b)."

  > "in multi-segmented structures such as the tarsi and abdomen, **multiple DoFs were coupled through a MuJoCo tendon and actuated together** for coordinated bending (Fig. 1k)."

  **⇒ 关节"枢轴点（pivot point）"是通过几丁质染色定位的**，这是比"随意指定"强得多的依据；但**关节轴方向本身仍是建模简化**。

### 2.7 ⭐ Azevedo et al. 2024（Nature）补充材料 —— **站着时胫–跗关节的动作方向是反的**

- 完整引用：**Azevedo A, Lesser E, Phelps JS, ... Dickinson M, Pacureanu A, Seung HS, Macrina T, Lee WCA, Tuthill JC.** *Connectomic reconstruction of a female Drosophila ventral nerve cord.* **Nature 631(8020):360–368 (2024).** DOI [10.1038/s41586-024-07389-x](https://doi.org/10.1038/s41586-024-07389-x). **PMID 38926570, PMCID PMC11348827**（开放获取）。
- 我实际抓取：`https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07389-x/MediaObjects/41586_2024_7389_MOESM1_ESM.pdf` → **HTTP 200，9,029,439 bytes**，已用 `pdftotext` 提取（78,540 字符）。MOESM2（2,435,616 B）、MOESM3（859,803 B）亦均 HTTP 200。
- 补充材料中**逐条给出跗节三条肌肉的动作定义**（逐字摘录，来自 Supplementary Methods 的肌肉表，Figure A17）：

  > **Tarsus depressor muscle** — "Extend the tibia-tarsus joint. **The joint is flexed when the fly is standing. Extension moves the fly's body away from the substrate.**" （别称 `tadm`）

  > **Tarsus retro depressor muscles** — "Muscle fibers originate on tibia cuticle that is distal to their insertion sites on the tarsus depressor tendon." （别称 `tarm 1 and 2`）

  > **Tarsus levator muscle** — "Flex the tibia-tarsus joint. **The joint is flexed when supporting the fly's weight. Flexion brings the fly closer to the substrate.**" （别称 `talm`）

  > **Long tendon muscle 2** — "**Located in femur.** Pull on the long tendon."（Figure A15-16）
  > **Long tendon muscle 1** — "**Located in the tibia.** Pull on the long tendon."（Figure A15-16）

- 补充材料另有关于命名与动作**不一一对应**的明确警告（逐字）：

  > "Second, the action that levator (or depressor) muscles have on a joint is not always the same: **the trochanter levator flexes the coxa-trochanter/femur joint, whereas the tibia levator extends the femur-tibia joint.**"
  > "We retain the terms 'levator' vs. 'depressor' for the muscles that 'lift' or 'push down' the tarsus. **When the fly is standing, the tarsus depressor muscle appears to ...** the levator muscle appears to flex the tibia-tarsus joint further."

- **【对本项目极高价值的建模含义】**：
  1. **站姿下胫–跗关节是"屈曲"状态**，压肌（TaD）做的是**伸展**（把身体推离地面）。**如果模型里把跗节轴的正方向或默认弹位设反了，站姿就会立刻被破坏** —— 而 flybody 的 `springref`（T1 `-0.242`）正是决定站姿平衡位的量。
  2. **命名不可信**：`depressor`/`levator` 在果蝇腿上是**按对跗节的作用命名**的，跨关节不一致。**绝不能按名字反推关节力矩符号。**
  3. **"站立稳定性"与胫–跗关节的耦合是真实的、有文献支持的** —— 用户的 10% 胸高逃逸时间指标恰恰是站姿稳定性指标，与该关节强相关。

### 2.8 ⭐ Oeftger et al. 2026（iScience）—— **唯一给出胫–跗关节定量力学的一手来源**

- 完整引用：**Oeftger S, Moussian B, Lehmann FO.** *Levels of elastic resilin modulate leg stiffness but not elasticity in Drosophila.* **iScience 29(7):116404 (2026).** DOI [10.1016/j.isci.2026.116404](https://doi.org/10.1016/j.isci.2026.116404). **PMID 42375520, PMCID PMC13310938**（**开放获取 CC BY**）。
- 我实际抓取：`https://pmc.ncbi.nlm.nih.gov/articles/PMC13310938/` → **HTTP 200，247,933 bytes**。（注：`cell.com` / `sciencedirect.com` 的对应页面均为 **HTTP 403**；PMC 可用。）
- 逐字摘录 —— **长腱路径与跗节无肌肉**（直接回答用户第 3 项关于跗分节间关节的问题）：

  > "A major component that is responsible for leg function is the **long leg tendon (pretarsal apodeme, retractor unguis apodeme)**. The tendon is controlled by long tendon muscles in the **femur (retractor unguis muscle 1)** and **tibia (retractor unguis muscle 2)** and **runs ventral through the femoro-tibial joint and up to the end of the fifth tarsal segment where it connects to the unguitractor plate.** It directs and helps to stabilize all joints distal to the femur, **including the tarsal segments that lack muscles**, during jumping and grasping."

  > "The pretarsus only has a depressor muscle, and levitation is thought to result from elasticity of its basal parts as shown in beetles and stick insects."

- 逐字摘录 —— **跗分节间关节在加载中可视为刚性**（**这是本项目最需要的一条**）：

  > "**We did not attempt to estimate angular changes between tarsal segments because these changes were negligible.**"

- 逐字摘录 —— **四个腿角的定义与载荷斜率（定量）**：

  > "The coxo-trochanteral angle γ slightly increases with increasing loading with values ranging from **1.81° μN⁻¹** ... to **5.99° μN⁻¹** in wild-type flies. The femoro-tibial angle δ slightly decreases by minimum **−3.32° μN⁻¹** ... to maximum **−5.33° μN⁻¹** ..., and the rate of change of the **tibio-tarsal angle ε ranges from −0.31° μN⁻¹ in wild type to −7.63° μN⁻¹ in act5C-cas9**. By contrast, the angle between tarsus and substrate η is largely independent of loading and ranges from −1.62° μN⁻¹ to 1.61° μN⁻¹."

  （角度定义：γ = coxo-trochanteral，δ = femoro-tibial，ε = **tibio-tarsal（胫–跗）**，η = tarsus-to-substrate。腿被建模为"连接 coxa 与前跗节的**可压缩悬挂支柱**"。）

- 逐字摘录 —— **肌腱直径（唯一的肌腱尺寸定量）**：

  > "The mean tendon diameter in wild-type flies is **3.25 ± 0.15 μm** (n = 10 flies), at an average tendon variance of 1.88 ± 0.05 μm (width of Gaussian fit curve to all tendon diameters; R², 0.94 ± 0.01)." （`res+-GFP` 株为 3.37 ± 0.13 μm。）

- 逐字摘录 —— **腿刚度（定量）**：

  > "We found minimum stiffness (maximum compliance) in the knockout strain res− (**7.75 ± 3.78 μN mm⁻¹**) that is ∼3.9-fold less than maximum leg stiffness in res+-Venus (**30.1 ± 23.2 μN mm⁻¹**) ... **Wild-type flies (13.1 ± 7.97 μN mm⁻¹)** and res+-GFP strains (14.3 ± 7.94 μN mm⁻¹) lie between these two extremes."

- 逐字摘录 —— **载荷量级（可与 flybody 的质量表交叉核对）**：

  > "...it flies experiences up to **47 μN** load, and the **estimated leg loading in a sitting fly is 1.7 μN at 10 μN body weight**."
  > 另：加载到约 1.5 倍坐姿力（**2.5 μN**）时做姿态测量。

- 逐字摘录 —— **胫–跗关节处的肌腱形态**：

  > "In Drosophila, both structures are visible at the **joint between tibia and first tarsus** (Figure 3C) and also to some degree at the femoro-tibial joint (Figure 3D). However, **we did not find local thinning of the tendon at the leg joints** as reported for beetles."
  > "...at the joint, tendon diameter **expands** and the tendon appears to be **fibrous**."
  > "Muscle attachment sites of long tendon muscles in the tibia and femur **appear as local swellings** near single muscle fibers (Figures 3 and 4)."
  > Figure 3 图注：`ti-ta1, transition between first tarsus and tibia (tibio-tarsal joint); fe-ti, ...`

- **【对本项目极高价值的建模含义】**：
  1. **跗分节间关节（tarsus2..5）在站姿/加载下角度变化可忽略 → 近似刚性是**有实验支持的**，因而 flybody 把它按近零范围（`-0.36..0.36` rad）用固定肌腱耦合驱动，方向上是合理的。
  2. **真正承载站姿的胫–跗关节自由度只有一个**（tibia ↔ tarsus1），这与"给 tarsus 一个 DoF"的思路一致 —— **但**恰是这一步做得不对（见 §2.7 的符号/弹位问题）。
  3. **长腱从股节近端贯穿到第 5 跗分节的 unguitractor plate**，是一条**跨 3 个关节的腱**。它在物理上是**跗节的被动稳定元件**。**若在模型中加了跗节 DoF 却没有表示这条贯穿腱，就等于抽掉了原有的被动稳定来源** —— 这为 23.15 → 8.75 ms 的退化提供了一个有文献支持的机制假设（**仍属推理，需实验验证**）。
  4. 肌腱直径 **3.25 μm** 与腿刚度 **13.1 μN/mm** 是**可直接用于参数标定**的真实测量值。

---

## 3. 已发表的果蝇肌骨模型（用户第 4 项，逐条核实）

### 3.1 flybody（TuragaLab + Google DeepMind / Janelia）

| 核查项 | 结果 |
|---|---|
| 是否存在 | **是** |
| 是否公开可下载 | **是**（`raw.githubusercontent.com` HTTP 200；GitHub API HTTP 200） |
| 许可 | **Apache-2.0** |
| 单位 | **CGS**（补充材料 §1.1 第 3 步） |
| 是否含**股节**关节 | **是**：`femur_twist*` + `femur_*`（每腿 2 个） |
| 是否含**跗节**关节 | **是**：`tarsus_*` + `tarsus2..tarsus5_*`（每腿 5 个） |
| 是否定义**肌肉附着点** | **否。** 对 `fruitfly.xml` 全文检索 `muscle / apodeme / unguitractor / depressor / levator / flexor / extensor / attach` 返回**零命中**；全文仅有 10 处 `tendon`，全部是 §2.2 的固定肌腱耦合。`make_fruitfly.py`（52,899 bytes，已抓取）同样**不含任何肌肉定义**，只有 abdomen/tarsus 肌腱耦合与 site 生成代码。 |
| 是否含关节轴数值 | **是**（见 §2.1） |
| 是否含附着几何数值 | **否**（有 body 锚点/关节中心，但**没有肌肉起止点**） |

> **重要澄清（针对任务描述中的"90 actuators / 15 肌肉 × 6 腿"）**：flybody **当前 main 分支**的 MJCF 并非如此。它的腿作动器是每腿 8 个 `extend_*`（共 48）+ 8 个 adhesion，总 78。**"15 肌肉 × 6 腿 = 90"精确对应的是 FlyMimic 的 15 条肌肉（见 §3.2）被复制到 6 条腿**。请核对本项目实际加载的构件到底是哪一个（详见 §6 待核对项）。

### 3.2 FlyMimic（Özdil et al., ICLR 2026）— 唯一含真实肌肉附着点坐标的公开模型

- 论文：Özdil G, Ning ..., Phelps JS, Wang-Chen S, Elisha ..., Ijspeert AJ, Ramdya P. *Musculoskeletal simulation of limb movement biomechanics in Drosophila melanogaster.* **ICLR 2026**；arXiv:2509.06426（<https://arxiv.org/abs/2509.06426>）
- 论文摘要逐字（节选）：

  > "Here, we introduce the first 3D, data-driven musculoskeletal model of Drosophila legs, implemented in both OpenSim and MuJoCo simulation environments. Our model incorporates a Hill-type muscle representation based on high-resolution X-ray scans from multiple fixed specimens."

- 仓库：<https://github.com/gizemozd/FlyMimic>（GitHub API HTTP 200，**Apache-2.0**，未归档）
- README 指向的**构建管线仓库 `gizemozd/neuromechfly-muscles` 返回 HTTP 404**（main 与 master 均试）→ **模型构建/参数优化管线未公开，只有成品模型文件公开。**

**我实际核查的成品文件：**

| 文件 | URL | 状态 | 大小 |
|---|---|---|---|
| OpenSim 模型 | `.../flymimic/assets/models/opensim/best_combined.osim` | HTTP 200 | 483,413 B |
| MuJoCo 模型 | `.../flymimic/assets/models/best_combined_arm_damping_stiff_cvt3.xml` | HTTP 200 | 49,670 B |

**单位（关键，OpenSim 文件内逐字）：**
```
<length_units>m</length_units>
<force_units>N</force_units>
<gravity>0 -9806.6499999999996 0</gravity>
```
MuJoCo 转换件逐字：`<compiler angle="radian" autolimits="true"/>`、`<option timestep="1e-04" gravity="0 0 -9801"/>`、mesh `scale="1000 1000 1000"`。

**⇒ 单位说明（已核实，勿误读）**：mesh 的 `scale="1000 1000 1000"` 只是把以**米**为单位的 STL 资产换算到以**毫米**为单位的模型，**不代表模型被放大**。FlyMimic 的数值**就是真实果蝇的毫米尺度**（整条左前腿链合计 2.183，股节 0.57，胫节 0.51，跗分节 0.087–0.51 —— 全是合理毫米值）。
⚠️ **唯一真实的坑**：OpenSim 文件声明 `<length_units>m</length_units>`，但其自身的 `<gravity>0 -9806.6499999999996 0</gravity>` 是 **mm/s²**，且整腿链长 2.183（真实果蝇全腿约 2–2.5 mm）→ **数值实为毫米，`length_units=m` 的声明自相矛盾（差 1000×）**。任何按 OpenSim 元数据自动导入的流程都会得到错 1000 倍的模型。详细单位对照见 §6.2。

**FlyMimic 的关节（逐字）：**
- 仅左前腿（LFC/LFF/LFTibia 前缀）可动：`joint_LFCoxa_yaw|pitch|roll`、`joint_LFTrochanter_yaw|pitch|roll`、`joint_LFTibia_pitch` → **7 DoF，3 个关节**。右前腿同名关节存在但被 `*_locked` 等式的 `polycoef="0 0 0 0 0"` 锁死。
- `joint_LFTibia_pitch` 逐字：`pos="0 0 0" axis="0 1 0" range="0.4789 2.502" springref="2"`
- `joint_LFCoxa_roll` / `yaw` / `pitch` 轴分别为 `0 0 1` / `1 0 0` / `0 1 0`，range 分别 `0.1436 0.6236` / `-0.597 0.2745` / `-0.5783 0.7375`
- `joint_LFTrochanter_*` 轴 `1 0 0` / `0 1 0` / `0 0 1`，range `-1.196 0.2745` / `-3.242 -1.217` / `-0.2745 1.384`

**FlyMimic 的跗节 —— 【关键否定】：** 在 OpenSim `best_combined.osim` 中 `joint_LFTarsus1..5` **全部是 `WeldJoint`（焊死）**；MuJoCo 转换件中**根本没有 tarsus 关节**。可动关节只有 coxa/trochanter/tibia 三个。**跗节是刚体。**

**FlyMimic 的肌肉（逐字，MJCF）：** 15 条肌肉，均为 `<spatial>` 空间肌腱 + `<site>` 附着点：
```
LFC_tergopleural_promotor_a, LFC_tergopleural_promotor_b,
LFC_pleural_remotor_and_abductor, LFC_pleural_promotor,
LFC_sternal_anterior_rotator, LFC_sternal_posterior_rotator, LFC_sternal_adductor,
LFF_trochanter_flexor_b, LFF_sterno-tergo-trochanter_extensor_a,
LFF_sterno-tergo-trochanter_extensor_b, LFF_accesory_trochanter_flexor,
LFF_trochanter_extensor, LFF_trochanter_flexor_a,
LFTibia_flex_93434, LFTibia_extensor_93932
```
共 **15 条 spatial tendon / 84 个 site 定义（含 12 个 `m_*_marker` 零点标记）/ 0 个 fixed tendon / 17 个 `<general>`**。

**其中作用于股节–胫节关节的两条肌肉，附着点坐标逐字摘录（父体局部坐标，模型单位 = m，注意 1000× 比例）：**

| 肌肉 | 附着点 site | 所属 body | `pos` |
|---|---|---|---|
| `LFTibia_flex_93434`（股–胫屈肌） | `..._bifemlh_r-P1` | `LFFemur` | `-0.0415201 0.0475376 -0.0877767` |
| 同上 | `..._bifemlh_r-P2` | `LFFemur` | `-0.0151968 0.110902 -0.546627` |
| 同上 | `..._pp21995` | `LFTibia` | `-0.0108694 0.0114727 -0.0107331` |
| `LFTibia_extensor_93932`（股–胫伸肌） | `..._bifemlh_r-P1` | `LFFemur` | `0.0276828 0.0384591 -0.0895311` |
| 同上 | `..._bifemlh_r-P2` | `LFFemur` | `0.0267536 0.105256 -0.597078` |
| 同上 | `..._pp21995` | `LFTibia` | `0.035 0.002 0.000927515` |

> 肌肉命名中的 `93434` / `93932` 疑为 FlyWire/MANC 神经元或解剖编号；`Miller28a/29/30/31/32/33` 明显指向 **Miller (1950) 的肌肉编号体系**（见 §4.3）—— 这是本模型与经典解剖学的连接点，但**本文未能取得 Miller 原文以核实编号对应**。

**FlyMimic 的致命缺口（逐字引自论文，由子代理核实）：**

> "Muscles housed in the tibia were excluded because the tibia segment was partially captured in the X-ray data."

胫节内肌肉 = **跗节压肌（TaD）与跗节提肌（TaL）就住在胫节里（§1.1、§1.2）**。因此：**跗节压肌/提肌的附着坐标在 FlyMimic 中根本不存在，且原因是成像时胫节未完整采集。** 这是一个**原因明确的结构性缺口**，不是遗漏。

### 3.3 NeuroMechFly v2 / flygym（NeLy-EPFL）

- 仓库：<https://github.com/NeLy-EPFL/flygym>（GitHub API HTTP 200，**Apache-2.0**）
- 论文：NeuroMechFly v2，Nature Methods 2024。
- flygym **内嵌了 flybody**：`src/flygym/assets/model/flybody/fruitfly.xml`（**67,648 bytes**，比 flybody 上游的 65,787 B 更大 → 经过修改/增强，**不是同一份文件**）。
- flygym 另有肌骨模型：`src/flygym/assets/model/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml`（49,670 bytes）。
  - 我逐条核实其关节：**只有** `joint_LF/RF Coxa_pitch|roll|yaw`、`Trochanter_pitch|roll|yaw`、`Tibia_pitch` → **没有任何 tarsus 关节，止于胫节**。
  - 17 个 `<general>` 作动器，**0 个 `<fixed>`**。
- ⚠️ **注意**：flygym 中这份肌骨模型文件名与 **FlyMimic 的 MuJoCo 成品文件同名且同样 49,670 bytes** → 极可能是**同一份 FlyMimic 模型的再分发**。**因此 flygym 的"肌骨模型"也就是"没有跗节"的那一个。**

**结论：目前唯一公开的"果蝇腿肌骨模型"恰好是唯一没有跗节的那一个。**

### 3.4 NeuroMechFly v1（NeLy-EPFL/NeuroMechFly）

- 仓库：<https://github.com/NeLy-EPFL/NeuroMechFly>（GitHub API HTTP 200，**Apache-2.0**，**已归档 archived: true**）
- 基于 PyBullet。其网格资源含**每腿独立的跗分节网格**（`lf_tarsus1..5.stl`、`lh_tarsus1..5.stl`、`lm_tarsus1..5.stl`）与 `*_trochanterfemur.stl`，即**几何上确实把 trochanter-femur 合并为一个节段、把跗节分成 5 段**。
- 但其 `rigging.yaml`（6,425 bytes）为关节绑定文件，**未发现肌肉附着点定义**。
- 是否含肌肉附着：**【否】**（未找到任何肌肉定义）。

### 3.5 其他候选的核查结论

| 候选 | 结论 |
|---|---|
| **FlyMimic** | **存在**（非虚构），见 §3.2。**是唯一含真实肌肉附着坐标的公开果蝇模型**，但**无跗节肌肉**。 |
| **Haustein et al. 2024** | **存在，且极为有用**，见 §3.7。**唯一给出关节轴真实朝向（μCT 关节髁导出）的公开来源**，并开放了髁位置数据。**但无肌肉、无 OpenSim 模型。** |
| **NeuroMechFly v3** | **不存在。** flygym 最新为 **v2.1.0**（release `published_at` 2026-06-24；PyPI latest 2.1.0）；分支列表含 `dev-v2.1.1` 但无 v3；`neuromechfly.org/v3/` 等路径均 **HTTP 404**；README 中 `v3` 出现 0 次。 |
| **MuJoCo Menagerie** | **含 flybody。** `google-deepmind/mujoco_menagerie` tree HTTP 200，69 个模型目录中含 `flybody/`（93 个文件，`flybody/fruitfly.xml` 67,727 B，Apache-2.0），README 列为 "Biomechanical"，**102 DoF**。**但它就是 §3.1 的 flybody，同样无肌肉附着几何**（`dyntype="muscle"` 0 次，15 个具名 site 全非肌肉 site）。**⇒ 这是一条独立渠道，可用来核对 flybody 的文件是否被改动过。** |
| **OpenSim / SimTK 上的果蝇腿模型** | ❌ **未找到独立的。** 见下方逐条核实。 |
| **MyoSuite** | ❌ **仅人体**（human arm/hand/leg/torso），882 个路径中无果蝇/昆虫。 |
| **DeepFly3D** | ❌ **无肌肉附着几何**（显式）。输出仅为 2D/3D 姿态关键点（`(7, 15, 38, 2)`），路径中 `muscle\|sdf\|urdf\|mjcf\|tendon\|attachment` 匹配数均为 **0**。 |
| **FlyMimic 的构建管线** | `gizemozd/neuromechfly-muscles` **HTTP 404**，未公开。 |

**OpenSim / SimTK 的逐条核实（真实返回码）：**

| 检查 | 结果 |
|---|---|
| `api.github.com/repos/opensim-org/opensim-models/git/trees/master?recursive=1` | **HTTP 200，1,168 个路径；匹配 `fly\|drosophil\|insect` 的路径 = 0 个** |
| 同上 `main` 分支 | **HTTP 404**（默认分支是 `master`） |
| `api.github.com/orgs/opensim-org/repos?per_page=100` | HTTP 200，29 个仓库，**无任何昆虫相关** |
| GitHub 仓库搜索 `drosophila+opensim` | **HTTP 200，`total_count = 0`** |
| `opensim.stanford.edu/` | HTTP 200，`Drosophila`/`fruit fly`/`insect` 出现 **0 次**；`/models/` → **HTTP 404** |
| **simtk.org** | **⚠️ 未能核实（部分可达，但检索索引不可用）** —— 详见下方专门说明。**这是本报告唯一"未决"而非"否定"的项目，不得当作"不存在"。** |
| MyoSuite（`MyoHub/myosuite`） | HTTP 200，882 路径，**仅人体模型** |

> **⇒ 结论：公开渠道上不存在独立的 OpenSim/SimTK 果蝇腿模型。唯一含附着坐标的 OpenSim 果蝇模型（FlyMimic）托管在 GitHub 而非 SimTK。simtk.org 本身未能核实，须换网络环境复查。**

**⚠️ simtk.org 专门说明（诚实记录，含我自己的更正）：**

| 检查 | 结果 |
|---|---|
| `https://www.simtk.org/`（带 `www.`） | **HTTP 200**，41,695 B，`<title>SimTK: Welcome</title>`，`Server: Apache/2.4.62 (Debian)`，自述 "2048 projects"。**首页中 `drosophila` / `fruit fly` / `insect` 出现 0 次。** |
| `https://simtk.org/`（不带 `www.`） | **连接被拒 `Errno 111`** —— 这是本报告最初"完全不可达"结论的来源，**该结论已更正**：带 `www.` 的主机是可用的。 |
| `/search?q=Drosophila`、`/search?SearchableText=Drosophila` | HTTP 200，但为 **JS-only 外壳**（body 内含 `<style> body { display:none !important; } </style>` + frame-buster），**搜索结果不在返回的 HTML 中** |
| `/projects/` | **HTTP 403 Forbidden** |
| 首页表单揭示的真实检索端点 `/search/search.php?srch=Drosophila&search=search&type_of_search=soft` | **`Errno 111 Connection refused`**（无 HTTP 码）；`/export/rss20_projects.php`、`/export/search_plugin.php` 同样被拒。约 6 次请求后所有 simtk.org 地址持续拒连约 10 分钟 → **疑似 IP 限流** |
| 我本人（本报告作者）独立重试 `www.simtk.org` 首页与检索端点 | **全部 `Errno 111 Connection refused`**（与上述限流一致）→ **确认本环境无法核实 SimTK** |

**⇒ 因此："SimTK 上没有果蝇项目"这一说法【未被证明】。它必须从不同 IP 或用具备 JS 能力的浏览器，以 `srch=Drosophila`、`"fruit fly"`、`"insect leg"`、`"fly leg"`（`type_of_search=soft`）重试。本报告把 SimTK 列为待决问题，而不是否定结果。**

**旁证（弱，但方向一致）**：web 检索只能浮出一个 SimTK 昆虫相关项目 "Messor barbarus ant biomechanics"（<https://simtk.org/frs/?group_id=2191>）—— **是蚂蚁，不是果蝇**；且该页本环境打不开。另外我尝试用 Wayback Machine 侧路核实：`http://archive.org/wayback/available?url=simtk.org/projects/opensim` → **HTTP 200**，确实存在 `simtk.org/projects/opensim` 的快照（2026-10-08），说明 SimTK 上**存在** OpenSim 项目页；但**这不能替代对果蝇项目的检索**。

### 3.6 ❌❌ **为什么跗节肌肉的定量坐标在公开来源中根本不存在 —— 已查明根本原因**

**这是本次调研最重要的"否定性发现"，它把"我没找到"升级为"已确认不存在且原因明确"。**

Azevedo et al. 2024 的 Supplementary Methods（MOESM1，我完整取得并读取）逐字说明了它所依赖的 X 射线全息纳米断层（X-ray holographic nano-tomography, XNH）数据集（源自 Kuan et al. 2020）的覆盖范围：

> "An X-ray holographic nano-tomographic (XNH) dataset of the fly's front leg (Kuan et al., 2020) shows each of the prothoracic nerves and their branches into the musculature ... We also used the XNH dataset to **determine leg muscle fiber origins, insertions and numbers**, as well as how the tendons move the leg joints (Figure A1). At the proximal end of the volume, the insertions of the thoracic muscles onto tendons and apodemes that contact the coxa are visible. **In most cases, the origins of the thoracic muscle fibers on the thoracic cuticle are not visible.** Distally, **the dataset ends ~¾ the length of the tibia. Many of the tibia muscle fibers are visible except for a few notable fibers. The tibia-tarsus joint is not visible.**"

**⇒ 三重锁死（每一条都是逐字原文）：**

1. **"the dataset ends ~¾ the length of the tibia"** —— 数据只到胫节的约 3/4 处。
2. **"The tibia-tarsus joint is not visible."** —— **胫–跗关节根本不在成像体积内。**
3. **"Many of the tibia muscle fibers are visible except for a few notable fibers."** —— 位于**胫节远端**的肌肉纤维恰好落在被截断的区域。

而按 §1.1 与 §1.2：**跗节压肌（TaD）"extends along the tibia"、跗节提肌（TaL）"located dorsally within the distal tibia"** —— 二者都是**胫节内**肌肉，且提肌明确在**远端**。
补充材料还把 "Tarsus depressor muscle | Figure A17 | 6 | **Uncertain, cut off**" 明确标注为"不确定，被截断"；并注明：

> "...muscles ... fibers that originate on the distal tibia cuticle."

**⇒ 结论（可以放心下断言）：**
**跗节压肌与提肌的附着点三维坐标，在目前最先进、最完整的果蝇腿 X 射线断层数据集里根本不存在 —— 因为成像体积在到达胫–跗关节之前就结束了。这不是任何人的疏漏，是成像视野的物理限制。**

这同时**独立地解释了 FlyMimic 的同一条限制**（§3.2 逐字："Muscles housed in the tibia were excluded because the tibia segment was partially captured in the X-ray data."），因为 FlyMimic 用的是**同一类 XNH 数据**。

**⇒ 并且它给出了本项目那次失败实验的最可辩护的机制解释**：项目想在模型里加一个跗节 DoF，而**世界上现有的果蝇腿成像数据恰恰在最需要的地方是空的**。所以那一步不是"查文献就能拿到坐标"，而是**必须先有新的成像**，或者接受等效作动器简化。

**Azevedo 2024 仍是有价值的下游线索**（用于**其他**肌肉）：正文逐字：

> "We also used the XNH dataset to determine the number of muscle fibres in each leg muscle (Fig. 4a), along with the **fibre origins and cuticle attachment points**."

但需注意：这些点**仅在 XNH 体积内有效**（即**不含**胫–跗关节附近）。其 Figure 4a **我未能提取**：`pmc.ncbi.nlm.nih.gov/articles/PMC11348827/figure/F4/` 返回 HTTP 200 但为 reCAPTCHA 拦截页；Nature 正文重定向至 `idp.nature.com`（付费墙）。**该图中是否有可机读的坐标仍未核实。**

### 3.7 ⭐⭐ Haustein et al. 2024 —— **唯一给出关节轴朝向与各关节 ROM 的公开来源**

**这条填补了本报告先前标注为【缺】的第 3 项需求（关节轴朝向 + 实测活动范围）。**

- 完整引用：**Haustein M, Blanke A, Bockemühl T, Büschges A.** *A leg model based on anatomical landmarks to study 3D joint kinematics of walking in Drosophila melanogaster.* **Front. Bioeng. Biotechnol. 12:1357598 (2024).** DOI [10.3389/fbioe.2024.1357598](https://doi.org/10.3389/fbioe.2024.1357598)
- 取用状态：`https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2024.1357598/full` → **HTTP 200，1,320,875 bytes**（我实际抓取并提取正文）。
- **这是什么**：一个**运动学**腿模型，**关节旋转轴由 μCT 扫描的关节髁（condyle）位置导出**，因此得到的是**"斜的"（oblique）主轴**，而不是正交分解。

**逐字要点：**

> "For instance, **the natural alignment of joint rotational axes was largely neglected in previous kinematic analyses**. In this study, we therefore present a detailed kinematic leg model in which not only the segment lengths but also **the main rotational axes of the joints were derived from anatomical landmarks, namely, the joint condyles**."

> "Our model with natural oblique joint axes is able to adapt to the 3D leg postures of straight and forward walking fruit flies with high accuracy. When we compared our model to an orthogonalized version, we observed that our model showed a smaller error as well as **differences in the used range of motion (ROM)**, highlighting the advantages of modeling natural rotational axes alignment for the study of joint kinematics."

> 扫描参数逐字："joint condyles positions and leg segment lengths were extracted from a **high-resolution synchrotron radiation μCT scan** carried out at the Paul-Scherrer Institute (PSI, Villigen, Switzerland). A single adult female wild-type Drosophila melanogaster specimen was scanned at the TOMCAT beamline of the PSI in absorption contrast at 10 keV with ×20 magnification."

> 关节集合逐字："Each kinematic chain comprised the same joints which we tracked with our motion capture system, i.e., **ThCx, CxTr, TrFe, FeTi, TiTar, and Tar**"，追踪用 DeepLabCut，"we tracked six keypoints: the thorax-coxa joint (ThCx), the coxa-trochanter joint (CxTr), **the trochanter-femur joint (TrFe)**, the femur-tibia joint (FeTi), **the tibia-tarsus joint (TiTar)**, and the tip of the tarsus (Tar)"。

- **⚠️ 关于 ROM 数值的重要限定（逐字）：**

  > "**Since the actual ROMs of joint DOFs in Drosophila are not known**, angle constraints were empirically adjusted to ensure an appropriated model fit but minimizing solutions that reached the range limits."

  **⇒ 该文的 Table 1（"Joint DOF angle constraints of the kinematic leg model"，单位**度**）中的数值是"经验调整的求解约束"，作者明确说明真实 ROM 未知。因此 Table 1 的数值【不是实测 ROM】，只能作为"合理范围"的参考，使用时必须声明。** 这正是本报告 §5.3 第 6 条【缺】项依然成立的原因。

- **关于跗节的处理（逐字）：**

  > "the bending movements of the five individual tarsal segments, could not be resolved accurately enough in the camera views and were thus **simplified to a variable-length monolithic tarsal segment**."

- **⭐ 一条极有价值的、直接支持本项目建模方向的独立论断（逐字）：**

  > "**tarsal segments do not have muscles that allow independent control by the nervous system, but are moved together by muscle tension on the long tendon (Soler et al., 2004).** Consequently, our model with a simplified tarsus **incorporates already the majority of joints which are under active control by the nervous system**. ... This suggests that a more detailed kinematic model of the tarsus is needed, for instance for dynamic simulations."

  **⇒ 双重结论：(i) 跗分节由**同一条长腱**一起驱动、无独立肌肉 → 支持"单个跗节等效自由度"；(ii) 但该文同时明确指出"若要**动力学**仿真，则需要更精细的跗节模型" → 也支持本项目当初想做的那一步的方向是对的。**

- **关于 TrFe（转节–股节）关节（逐字）：** "Although the TrFe is moveable in many insects (Frantsevich and Wang, 2009), **its functional role in Drosophila is debated**, with opinions ranging from it being **fused and immobile** (Sink, 2006; Lobato-Rios et al., 2022) to **at least limited mobility** (Feng et al., 2020). We found that a roll DOF in the TrFe drastic[ally/...]..." → **即：flybody 给股节两个 DoF、而 NeuroMechFly v1 与 FlyMimic 把股节与转节合并/焊死，这个分歧在解剖学上尚未定论。**

- **开放数据（可直接下载，用于自行导出关节轴）：**
  > "...publicly available at Harvard Dataverse, V1. **The μCT scan dataset** of the female adult fly used to extract joint condyle locations is available at https://doi.org/10.7910/DVN/XWREDA , and **the tracking dataset** including 3D-reconstructed leg postures is available at https://doi.org/10.7910/DVN/BBNPYX ."

  经核实：两个 DOI 均 **HTTP 202**，Dataverse API（`.../api/datasets/:persistentId?persistentId=doi:...`）均 **HTTP 200**。XWREDA（"micro-CT scan data of single adult female wild-type Drosophila melanogaster"）文件含 **`j_bod_cox_1-1.obj`、`j_cox_tro_1-1.obj`、`j_tro_fem_1-1.obj`、`j_fem_tib_1-1.obj`、`j_tib_tar_1-1.obj`、`j_leg_1-1.obj`** 与 `KinematicModel_InitalPosture_Coords3D-2.csv` —— **即每个关节的髁位置对象都在里面**（注意 `j_tib_tar_1-1.obj` 仅 174 B，是关节髁标记点，不是网格）。BBNPYX 含 `Coords2D.csv` / `Coords3D.csv` / `modelAngles.csv` 与视频。

  **⇒ 这是本项目若要沿用"真实关节轴"路线时，唯一可自行获取一手关节轴几何的公开数据集** —— 而且是**真实标本的 μCT**，不是模型约定值。

- **不包含的**：`OpenSim` 在全文出现 **0 次**；**无肌肉、无肌肉附着几何**。该文是纯运动学模型。

---

## 4. 经典解剖学文献（用户第 5 项）—— 取用状态如实报告

### 4.1 Soler et al. 2004 —— **决定性缺口：关闭获取（closed access）**

- 完整引用：**Soler C, Daczewska M, Da Ponte JP, Dastugue B, Jagla K.** *Coordinated development of muscles and tendons of the Drosophila leg.* **Development 131(24):6041–6051 (2004).** DOI [10.1242/dev.01527](https://doi.org/10.1242/dev.01527). **PMID 15537687**。
  > ⚠️ **更正**：任务中未给 PMID；核查中发现一个易错点 —— PMID **15563524 是错误的**（那是 Wingert et al. 的斑马鱼 chianti 论文，Development 131(24):6225-35, DOI 10.1242/dev.01540）。**正确的 Soler 2004 PMID 是 15537687。**
- **取用状态（全部实测）：**

  | 路径 | 状态 |
  |---|---|
  | `journals.biologists.com/dev/article/131/24/6041/43068/` | **HTTP 403**（Cloudflare "Just a moment..."） |
  | `journals.biologists.com/dev/article-split/131/24/6041/42694/` | **HTTP 403** |
  | `journals.biologists.com/dev/article-pdf/131/24/6041/1148341/6041.pdf` | **HTTP 403** |
  | `paperity.org/p/45054938/...` | **HTTP 403** |
  | Europe PMC REST API（DOI 10.1242/dev.01527） | HTTP 200，但记录字段为 `"isOpenAccess":"N"`, `"inPMC":"N"`, `"inEPMC":"N"`, `"hasPDF":"N"`，`fullTextUrlList` = "Subscription required" |
  | Semantic Scholar API | HTTP 200，`isOpenAccess: false`，`openAccessPdf.status = "CLOSED"` |
  | OpenAlex API | HTTP 200，`oa_status = "closed"`，`any_repository_has_fulltext: false` |
  | Crossref API | HTTP 200，**仅**返回元数据 + 约 1200 字符摘要 + 一个 403 的 PDF 链接 |

- **结论：** 该文**没有任何绿色/金色开放获取副本被索引到**。**我未能取得该文 Results 的任何一句话，也无法读到任何图（图号亦无法核实）。** 所有 FBbt 中关于跗节肌肉的解剖学表述，其一手来源（含全部纤维数与附着关系）**都在这道付费墙之后**。
- **因此：本文 §1 全部为"经 FlyBase 本体转述的二手信息"。** 虽然 FBbt 由专业策展人从 Soler 2004 直接摘录，可信度高，但**本文不声称曾阅读 Soler 2004 原文**。

### 4.2 Azevedo et al. 2024 / 2020、**Agrawal et al. 2020**、Mamiya et al. 2022

> ⚠️ **两处引用更正（重要）**
>
> **(1) "Kuan et al. 2020 eLife 60299" 引用有误。** eLife 2020;9:**e60299**（*Central processing of leg proprioception in Drosophila*）的作者是 **Agrawal S, Dickinson ES, Sustar A, Gurung P, Shepherd D, Truman JW, Tuthill JC**（DOI [10.7554/eLife.60299](https://doi.org/10.7554/eLife.60299)，PMID 33263281，PMCID PMC7752136，**开放获取 CC BY**）。"Kuan et al., 2020" 是**另一篇更早的论文**，是被 e60299 引用的；FBbt 中"accessory tibial flexor 29 fibers (Kuan et al., 2020)"即引自它，**但本代理未独立核实 Kuan 的完整引用信息**。
>
> **(2) Azevedo et al. 2024 不是一篇"腿肌肉"论文**，而是 **Nature 631(8020):360–368 (2024)** 的**雌性果蝇腹神经索连接组重建**（FANC）。它之所以关键，是因为其补充材料含逐肌肉表与 XNH 附着点分析（见 §2.7、§3.6）。**开放获取（PMCID PMC11348827），我成功取得其三个补充 PDF。**

- **Azevedo AW et al. (2020)** *A size principle for recruitment of Drosophila leg motor neurons*, **eLife 9:e56754**, DOI [10.7554/eLife.56754](https://doi.org/10.7554/eLife.56754)，PMID 32490810，PMCID PMC7347388（**开放获取 CC BY**）。经核实的关键定量信息：
  - 逐字："The leg of the fruit fly, Drosophila melanogaster, contains **14 muscles** which are innervated by just **53 motor neurons**."
  - 逐字力值："a single fast motor neuron spike produced **~10 μN of force**, resulting in a 50 μm movement of the force probe. An intermediate neuron spike produced **~1 μN** that moved the tibia 5 μm."（慢神经元 <0.1 µN/spike。）
  - 逐字："during take-off, the peak force production of the fly leg is **~100 μN**"；"the femur-tibia joint can produce enough force to lift approximately **ten times the fly's body weight**."
  - 测力探头弹簧常数 k = 0.2234 N/m；蝇质量 ~1 mg，体重 ~10 μN。
  - 关节角区间（用于分类）：伸展时 "femur-tibia joint **>120°**"，屈曲时 "**<30°**"。
  - 逐字（与 §1.7 呼应）："The muscle fibers innervated by these two neurons connect to the **same tibia flexor tendon** (Soler et al., 2004)."
  - **判定：(a) 无附着几何数值；(b) 无关节轴；(c) 定性解剖 + 定量力学/电生理。**

- **Agrawal et al. 2020（eLife 60299）** —— **开放获取 CC BY**；PMC 全文 HTTP 200。
  - **它确实同时监测三个关节**，逐字："we measured how optogenetic activation altered movements of the **three major leg joints (coxa-femur, femur-tibia, and tibia-tarsus)**" —— **即 tibia–tarsus 被当作一个独立可测关节**，这对本项目是好消息。但报告形式是"平均关节角变化"（Fig. 3C/D），**没有绝对 ROM 表**。
  - 逐字：FeCO 是最大本体感受器官，"its **152 neurons** can be divided into at least three anatomically distinct subtypes: the claw, hook, and club neurons."
  - 关节角阈值：13Bα 活动仅在 "the tibia was **extended past ~90°**" 时增加。
  - 摆动速度刺激：fast **720 °s⁻¹**、slow **240 °s⁻¹**。
  - 迟滞："the steady-state membrane potential at a given tibia position was greater when the tibia was extended to reach that position than when it was flexed."
  - 振动刺激：1600 Hz @ 0.9 µm 振幅。
  - **判定：(a) 无附着几何数值；(b) 无关节轴向量（仅有 ~90° 阈值与 240/720 °s⁻¹ 这类标量）；(c) 主要为神经生理，解剖为定性。**

- **Mamiya A, Gurung P, Tuthill JC (2022)** *Biomechanical origins of proprioceptive maps in the Drosophila leg*, bioRxiv 2022.08.08.503192v1（<https://www.biorxiv.org/content/10.1101/2022.08.08.503192v1>）—— **成功取得全文 PDF（14,804,556 bytes，HTTP 200，经 python urllib 抓取；普通 web_fetch 通道返回 429）**，CC-BY-NC-ND 4.0。**这是本次唯一完整读到的"关节力学实验范式"一手文献。**
  - 关于 **femur–tibia** 关节的实测/实验定量信息（逐字摘录）：
    > "we placed the camera to the side and used a prism (Edmund Optics) to capture the view from below"
    > "adjusted its position so that the magnet moved in a circular trajectory centered at the femur-tibia joint, ... mm from the center of the femur-tibia joint and the distance between the pin and the magnet was ~ 300 µm"
    > "Because it is difficult to fully flex the femur-tibia joint without the magnet colliding with the abdomen or other legs, we ..."
    > 关节角定义（Fig. S4 图注逐字）："...the difference from the arculum angle **when tibia angle is at 90˚**, and **clockwise rotation is defined as positive** change in the angle."
    > 方法：磁力控制系统在 ~300 µm 处以圆周轨迹驱动胫节，加速度设定 `72000 deg/s2`；用 200 fps 高速相机 + DeepLabCut 追踪。
  - **可得：** femur–tibia 关节确实被当作**绕关节中心的近平面旋转**（圆周轨迹驱动 + 平面视角成像）→ 支持**单铰链（1 DoF）近似**；给出了 90° 参考位与正方向约定。
  - **不可得：** **没有给出关节轴的解剖学朝向向量，没有给出 ROM 上下限数值，没有 tibia–tarsus 或跗分节间的任何测量。** 该文研究硬骨感受器（FeCO/club/claw 神经元）与 femur–tibia 关节，**不涉及跗节肌肉**。

### 4.3 Miller (1950) *The morphology of the Diptera* 及同类经典图谱 —— **未取得，取用状态如实报告**

- 任务提示该文可能收费。**本次调研没有尝试绕过任何付费墙**，也**没有**取得 Miller 1950 的任何正文或图。
- **但有一个有价值的间接发现**：FlyMimic 的肌肉命名中直接出现 `Miller28a`、`Miller29`、`Miller30`、`Miller31`、`Miller32`、`Miller33` 这样的编号（见 §3.2 逐字 site 名）。这说明 **FlyMimic 的肌肉身份是挂接到 Miller 编年体系的**。
- **因此：Miller (1950) 是最有希望的"跗节肌肉编号 + 附着关系"经典来源，本文未取得其正文，故不引用其任何内容。** 若需要，这是一条明确的后续获取方向（图书馆/馆际互借），而不是可以靠网页抓取解决的。
- 同理，任务提到的其他老图谱（如 Snodgrass）**本次均未取得**，不作任何转述。

### 4.4 iScience 2026 韧弹性（resilin）论文中的长腱结构

- 检索中出现：*Levels of elastic resilin modulate leg stiffness but not elasticity in Drosophila*, iScience（PII S2589-0042(26)01779-7），其 **Figure 3** 图题为 "Structure of the leg's long tendon (pretarsal apodeme, retractor unguis apodemе) in wild-type Drosophila"。
- 取用状态：`cell.com/.../fulltext/S2589-0042(26)01779-7` → **HTTP 403（Cloudflare）**；`sciencedirect.com` → **403**。**正文与图均未取得。**
- 但图题本身即是有用信息：**"leg's long tendon" = "pretarsal apodeme" = "retractor unguis apodeme" 是同一结构的三个名字**，与 FBbt:00058847 / FBbt:00058136 的描述一致（§1.4）。**这确认了术语对应关系，但仍未取得该结构的任何坐标或长度。**

---

## 5. 【核心章节】哪些几何可以用来扩展模型，哪些必须继续声明为假设

> 本节回答任务要求的"explicit section"。**分三类：可直接采用（有文献数值）／可条件采用（需换算，有风险）／必须声明（无任何来源）。**

### 5.1 ✅ 可以直接采用 —— 有已发表、有许可、可下载的**真实数值**

**这些数值来自 flybody 的 MJCF 与 Nature 补充材料，是已发表的模型约定，不是自造。**

| 可用内容 | 来源（精确位置） | 备注 |
|---|---|---|
| tibia–tarsus 关节**存在**，且是单轴关节 | flybody `fruitfly.xml`，class `extend_tarsus_T1/T2/T3` | 用户原以为"该模型跗节无关节"——**与 flybody main 分支不符，务必先核对本项目实际构件** |
| tibia–tarsus 关节轴 `1 0 0` | 同上（含父类 `tarsus` 的 `joint axis="1 0 0"`） | **这是模型局部坐标约定，不是解剖测量** |
| tibia–tarsus ROM：T1 `-0.7..1.2`、T2 `-1..1.8`、T3 `-0.8..1.2` rad | 同上 | 可按腿分别采用 |
| springref：T1 `-0.242`、T2 `0.608`、T3 `0.258` rad | 同上 | 决定被动平衡位 |
| 股节（femur）2 个 DoF：`femur_twist`（轴 `0 1 0`, `-1..1`）与 `femur`（轴 `1 0 0`, `-0.15..2`） | 同上，class `twist_femur` / `extend_femur_T*` | **这是"转节–股节"关节的现成答案** |
| tibia 关节：轴 `1 0 0`, `-1.35..1.3` rad | 同上，class `extend_tibia_T*` | |
| 跗分节间关节：轴 `1 0 0`, `-0.36..0.36` rad，`stiffness 0.1`, `damping 0.002` | 同上，class `extend_tarsus` | |
| 关节中心坐标（每腿每节的 `pos`/`quat`） | 同上，body 定义 | **已逐条核实存在；6 条腿数值互不相同** |
| 跗节驱动应如何处理：1 : 0.5 : 0.5 : 0.5 固定肌腱耦合 | 同上 `<tendon>` 段 + `make_fruitfly.py` 注释 + 补充材料第 10 步 | **强烈建议照此建，而不是加一个自由 DoF** |
| 每腿作动器配置：coxa×3 + femur×2 + tibia×1 + tarsus×2 = 8，位置语义，增益标定为"基姿下末端可施加约 1 倍体重力" | 补充材料 §1.1 第 12 步 + Supplementary Table 2 | 权威 |
| 质量：每腿 0.0162 mg，全蝇 0.983 mg（52 只雌蝇平均） | 补充材料 Supplementary Table 1 | |
| 左右对称语义：正向=伸展/外展，负向=屈曲/内收；关节角参考值 0 对应 base pose | 补充材料 §1.1 第 6–7 步 | 定义符号约定 |

### 5.2 ⚠️ 可以条件采用 —— 有数值但必须处理血统与坐标系，风险较高

| 内容 | 来源 | 风险 |
|---|---|---|
| 股节–胫节**肌肉**附着点坐标（`LFTibia_flex_93434`、`LFTibia_extensor_93932` 各 3 个 site），**单位 = 毫米、真实尺度（无需缩放）** | FlyMimic `best_combined_arm_damping_stiff_cvt3.xml` / `best_combined.osim`（Apache-2.0） | **① 坐标系不同**：FlyMimic 的胫节轴是 `0 1 0`，flybody 是 `1 0 0` → **坐标不可跨模型直接搬运，必须做刚体变换**；**②** OpenSim 文件声明 `length_units=m` 却存毫米值，自动导入会差 1000×；**③** 肌肉身份名（`Miller28a` 等）未核实到 Miller 原文；**④** 这是 MyoConverter 转换件，site 语义需与原生 MJCF 对齐；**⑤ 仅左前腿**，右腿与中/后腿的附着重建提到过但**未随仓库发布** |
| FlyMimic 的关节 ROM（如 `joint_LFTibia_pitch` `0.4789..2.502`） | 同上 | 与 flybody 的 tibia 范围 `-1.35..1.3` **不一致** → 两套模型不能混用数值 |
| **关节轴的真实朝向 + 各关节范围（度）** | **Haustein et al. 2024**（§3.7），关节髁由 μCT 导出；开放数据 DOI 10.7910/DVN/XWREDA | **⚠️ 其 Table 1 的数值是"经验调整的求解约束"，作者逐字声明"actual ROMs of joint DOFs in Drosophila are not known" → 不是实测 ROM，使用必须声明。** 但**髁位置本身是真实标本的 μCT 测量**，可用于自行导出关节轴 |
| 股节–胫节**肌肉**参数（tendon_slack_length、max_isometric_force、optimal_fiber_length） | FlyMimic OpenSim 文件 | 这是**参数优化拟合结果**（论文标题即含 "optimizing unknown muscle parameters"），不是直接测量 |

### 5.3 ❌ 必须继续声明为假设（declared）—— **任何来源都没有数值**

**以下各项在任何已获取的来源中都不存在数值。若模型需要它们，必须显式标注为假设，并且（按任务要求）在有来源之前不得冒充测量值：**

1. **跗节压肌（TaD）的附着点坐标** —— **无，且原因已查明（§3.6）**：跗节压肌"沿胫节延伸"，而最强的 XNH 数据集**"ends ~¾ the length of the tibia. The tibia-tarsus joint is not visible."** 定性上只知道"纤维沿胫节延伸、经 tarsal depressor tendon 止于近端跗节"（FBbt，二手自 Soler 2004）。**起点在胫节内的具体三维位置、止点在跗节上的具体三维位置：全部【缺】。**
2. **跗节提肌（TaL）的附着点坐标** —— **无，同上原因。** 定性只知道"胫节远端、背侧、偏外侧（lateral）"（FBbt）。
3. **tarsal depressor tendon / tarsal levator tendon 的几何** —— **【部分】**：**直径有**（野生型 **3.25 ± 0.15 μm**，Oeftger 2026）；**长度、路径坐标、截面形状均无**。只知道止于近端跗节。
4. **所有跗节相关肌肉的力臂（moment arm）** —— **无。** 这是最关键的缺口：**没有附着点就无法算力臂；没有力臂就无法把生理肌肉力转成关节力矩。** 本文**不会**用 flybody 的 anchor 位置去"估算"力臂。
5. **tibia–tarsus 关节轴相对腿平面的解剖学朝向** —— **【部分可得，但需自行处理】**：flybody 的 `1 0 0` 是模型约定。**但 Haustein et al. 2024（§3.7）提供了由真实标本 μCT 关节髁导出的关节轴**，并开放了髁位置数据（DOI 10.7910/DVN/XWREDA，含 `j_tib_tar_1-1.obj` 等）。**⇒ 本项目可以自行从该数据导出"斜轴"朝向，但这需要自己做一次刚体对齐工作，本报告没有替你做，也没有给出任何轴向量。**
6. **tibia–tarsus 的实测 ROM** —— **仍然是【缺】**。**关键限定**：Haustein et al. 2024 的 Table 1 虽有各关节角度范围（度），但作者逐字声明 **"Since the actual ROMs of joint DOFs in Drosophila are not known, angle constraints were empirically adjusted to ensure an appropriated model fit"** → **那是经验约束，不是实测 ROM，不得当测量值引用。** 可用的定量只有载荷斜率：ε = **−0.31 °μN⁻¹（野生型）**（Oeftger 2026），以及"站立时该关节处于屈曲状态"（Azevedo 2024，§2.7）。
7. **跗分节间关节是否需要自由度** —— **【有证据支持"近刚性"】**：Oeftger 2026 逐字 "We did not attempt to estimate angular changes between tarsal segments because **these changes were negligible**"；且 FBbt 记载"跗分节内无肌纤维"。flybody 用 `-0.36..0.36` rad + 固定肌腱耦合来近似的做法，**方向上与实验证据一致**（但耦合系数 1:0.5:0.5:0.5 本身仍是 "simplification"）。
8. **股节收肌（ferm）、股节压肌（fedm）的附着点坐标** —— **无。** 只有"转节基部→近端股节"、"转节内→近端股节"的定性描述与纤维数（4–5、7–9）。
9. **转节–股节关节轴的解剖学朝向与实测 ROM** —— **无**（flybody 的 `twist_femur` + `extend_femur_*` 是模型约定）。Mamiya 2022 只覆盖 femur–tibia。
10. **long tendon（pretarsal apodeme / retractor unguis apodeme）的长度与路径** —— **【部分】**：**路径拓扑有**（逐字：从股节近端 ltm2 附着处起，"runs ventral through the femoro-tibial joint and up to the end of the fifth tarsal segment where it connects to the unguitractor plate"），**且有明确的电机学描述**（过胫–跗关节处直径**膨胀、呈纤维状**）；**但无长度、无力臂、无包裹/滑车几何坐标。**
11. **每个肌肉的生理横截面积、最优纤维长度、最大等长力（真实果蝇）** —— FlyMimic 给出其模型的 `max_isometric_force` / `optimal_fiber_length` / `tendon_slack_length` 参数值，但那是**模型优化拟合结果**（论文标题即含 "optimizing unknown muscle parameters"），**不是直接测量**，且**不含跗节肌肉**。
12. **整条腿的被动刚度** —— **【有】**：野生型 **13.1 ± 7.97 μN mm⁻¹**（Oeftger 2026），以及坐姿单腿载荷 **1.7 μN**、起飞峰值 **~100 μN**（Azevedo 2020）。这些**可直接用于标定**。

### 5.4 如果把最好的来源全部读完，仍然缺什么？

即使同时拿到 Soler et al. 2004 全文 **和** Azevedo et al. 2024 的 Figure 4a 与补充表，**仍然会缺**：

- **跗节压肌/提肌的坐标依然不会有** —— 这不是文献获取问题，而是**成像视野问题**（§3.6 逐字："the dataset ends ~¾ the length of the tibia. The tibia-tarsus joint is not visible."）。**Soler 2004 是发育细胞生物学论文**（GFP 标记、共聚焦、腱前体细胞内陷、duf-lacZ 成肌细胞、肌管沿腱导向），其图**大概率是低倍示意/拓扑图，不是带坐标的测量**。依据：FBbt 从该文提取的**全部定量信息只有纤维数**，**没有任何一条 FBbt 条目引用它给出坐标、长度或力臂**。
- **力臂（moment arm）**：即使有了附着点坐标，还需要**关节旋转中心的定义**（flybody 给了 pivot point，但那是模型约定），才能算力臂。**没有任何果蝇腿肌肉的力臂被发表过。**
- **长腱的包裹/滑车几何**：长腱**穿过**股–胫关节和胫–跗关节并改变方向（Oeftger 2026 明确描述其在胫–跗关节处"直径膨胀、呈纤维状"），**这种改变方向的几何（等效滑车半径）是力臂正确与否的决定因素，而它完全没有被发表。**
- **肌肉的生理参数**（PCSA、纤维长、羽状角、比张力）：无。
- **活体胫–跗关节 ROM 上下限**：无。只有载荷斜率（−0.31 °μN⁻¹）。
- **tarsomere 1 的精确止点位置**：所有来源只说 "proximal tarsus"。**未找到任何来源字面写出"tarsomere 1"。** **不得**把这一条作为引文断言。
- **跗分节间关节的被动 ROM**：Oeftger 2026 在其加载实验中说变化"可忽略"，但那是**特定加载条件下**的结论；被动 ROM 本身未测。

**⇒ 诚实的结论（最终）：**

**跗节肌肉的"可用附着几何"在公开来源中不存在，且现在已知其原因是数据采集视野的物理限制，不是检索不够努力。** 项目若需要它，只有两条路：
- **(a) 接受等效作动器简化**（用已发表的关节轴/范围 + 声明力臂为假设），或
- **(b) 自行做一轮针对胫节远端与跗节的成像测量**（在现有 XNH 视野之外）。

**不存在"再查一查文献就能拿到坐标"的第三条路。同样，"转节–股节"与"股–胫"的肌肉附着几何则有 FlyMimic 的实测坐标可用（但需处理 §5.2 的单位与坐标系问题），且这两个关节的肌肉附着也不含 mm 级精度保证。**

---

## 6. 对本项目的具体建议与待核对项

### 6.1 【最高优先级】先核对本项目实际加载的到底是哪一个模型

观察到两个互相矛盾的线索：

- 本项目称其模型"Femur 与 Tarsus1..5 **无关节**，90 个作动器 = 单条前腿 15 肌肉集复制到 6 条腿"。
- 我核实的事实：
  - **"15 肌肉 × 6 腿 = 90" 精确对应 FlyMimic 的 15 条肌肉。** 而 FlyMimic **跗节为 WeldJoint、且完全不含 tarsus 关节** → **与"跗节无关节"完全吻合。**
  - **但需注意一个限定：FlyMimic 发布的模型本身只有 15 个作动器、且全在左前腿**（中腿 7 MTU、后腿 8 MTU 提到过但**未随仓库发布**）。所以本项目的 90 个作动器 = **把 FlyMimic 的 15 条肌肉复制到 6 条腿的衍生版本**，而不是 FlyMimic 原样。**这个"复制到 6 腿"是本项目自己做的一步，需要核对它的附着点是否也做了对应的左右/腿间镜像变换。**
  - **flybody main 分支的跗节和股节都是有关节的**，且作动器是 78 个，不是 90。flygym/NMF v2 自身的腿关节集是**每腿 11 个命名 DoF**（ThC×3 + CTr×2 + FTi×1 + TiTa×1 + 4 个跗分节 pitch），**默认只驱动 7 个**，其余跗分节为 passive links。

**⇒ 本项目模型的血统很可能不是 flybody 的当前 MJCF，而是 FlyMimic（或 flygym 内那份同名的肌骨模型）骨骼 + 肌肉作动器的组合。** 若如此：

1. **本项目模型的跗节是"被焊死的刚体"**，所以"给它加一个跗节自由度"= 把一个 WeldJoint 换成一个铰链 —— **而 FlyMimic 的胫节网格在 X 射线数据里没采全，所以焊接本身可能就是成像缺陷的后果。** 这解释了为什么加 DoF 后变差：**不是"轴错了"这么简单，而是"原本靠焊死掩盖的缺失踝部动力学被解锁了"。**
2. **可用的替代路径**：改用 **flybody 的 MJCF 骨骼**（Apache-2.0，**跗节与股节关节齐备、含真实关节中心、含跗节耦合肌腱**），再把 FlyMimic 的 15 条肌肉附着点**变换**到 flybody 的骨骼上 —— 但注意 §5.2 的坐标系/单位风险，且 **FlyMimic 也没有跗节肌肉**，所以跗节驱动仍需等效作动器。

**建议动作（不修改现有文件的前提下）**：先确认本项目模型的 XML 来源文件名、`<compiler>` 单位、关节名清单、作动器清单；再决定是"给 FlyMimic 骨骼加跗节"还是"迁到 flybody 骨骼"。**在此之前不要再次改模型。**

### 6.2 ⚠️ 单位警告（已核实并修正；这是最容易出错的地方）

**核实结果（每条都有逐字依据）：**

| 模型 | 长度单位 | 依据（逐字） |
|---|---|---|
| **flybody 源 MJCF** | **厘米（CGS）** | `<option ... gravity="0 0 -981" density="0.00128" .../>`。−981 = 9.81 m/s² 的 cm/s²；0.00128 = 空气密度 **g/cm³**。交叉验证：`femur_T1_left pos="0 0.0437 0"` → 0.0437 cm = **0.437 mm**，与真实果蝇股节尺寸相符。 |
| **flygym 合成模型** | **毫米** | `src/flygym/assets/model/flybody/mujoco_globals.yaml` 逐字：`gravity: ['0','0','-9810']  # mm/s^2 (source XML is cm: -981 cm/s^2 x10 -> mm)`。即 **10×** 换算。NMF v2 论文逐字："We configured the meshes at 1000× scale in MuJoCo to obtain observation measurements in **mm and mN**." |
| **FlyMimic MuJoCo** | **毫米** | `<option timestep="1e-04" gravity="0 0 -9801"/>`（mm/s²）+ 所有 mesh `scale="1000 1000 1000"`（STL 以米为单位 → 模型用毫米）。 |
| **FlyMimic OpenSim 文件** | **文件声明 `m`，但数值实为毫米** | `<length_units>m</length_units>` 与 `<gravity>0 -9806.6499999999996 0</gravity>`（= 9806.65 **mm/s²**）自相矛盾；整条左前腿链长合计 **2.183 文件单位**，而真实果蝇全腿约 2–2.5 mm。**⇒ 数值是毫米，`length_units=m` 的声明是不一致的（差 1000 倍）。** |

**⚠️ 自我更正（重要）：** 本报告早前版本（§3.2 与 §5.2）曾推断 FlyMimic 是"被放大 ~1000× 的模型"，并据此把它列为"高风险、必须换算"。**该推断是错的，现予更正：** FlyMimic 的数值量级**就是真实果蝇的毫米尺度**（整腿 2.183、股节 0.57、胫节 0.51、跗分节 0.087–0.51 —— 全是合理的毫米值）。`scale="1000 1000 1000"` 只是把以**米**为单位的 STL 换算到以**毫米**为单位的模型，**不代表模型被放大**。因此 **FlyMimic 的附着坐标是按 1:1 真实尺度用的毫米值**，不需要 1000× 缩放。

**但仍然存在的真实风险（这些保留）：**
1. **flybody = cm，flygym/FlyMimic = mm → 相差 10×。** 跨这两套模型搬数字必须显式换算。交叉验证：同一锚点在 flybody 是 `0.0437`（cm），在 flygym 的 `rigging.yaml` 是 `'0.437'`（mm）—— 二者等价，证实 10× 关系。
2. **FlyMimic OpenSim 文件自身声明 `length_units=m` 却存毫米值。** 任何按 OpenSim 元数据自动导入的流程都会得到差 1000× 的模型。
3. **坐标系不同**（见 §5.2）：FlyMimic 的胫节轴是 `0 1 0`，flybody 的是 `1 0 0` —— 轴与坐标**不可跨模型直接搬运**，必须做刚体变换。
4. flybody 论文自己也警告跗节惯量极小（MKS 下 `1e-20 kg m²`，"差异对双精度浮点精度而言很显著"）。

**⇒ 请先确认本项目模型的单位制（cm 还是 mm），再做任何几何改动。好消息是：本项目报告的锚点量级（股节 0.437、胫节 0.697、跗节 0.51）与真实果蝇的毫米尺度完全一致，说明现有几何本身没有被单位错误支配 —— 那一次失败实验的原因更可能在符号/弹位/被动耦合上（见 §6.3），而不是单位。**

### 6.3 若确实要建模跗节驱动，最小可辩护方案（已按新证据更新）

按"证据等级"从高到低：

1. **照抄 flybody 的关节与耦合结构**：跗节 1 个铰链（轴 `1 0 0`）+ tarsus2..5 按 `1 : 0.5 : 0.5 : 0.5` 固定肌腱耦合到单作动器。**关节侧的数值全部有已发表来源，无需声明任何自造量。** 这一方案还得到两条**独立实验证据**支持：跗分节间角度变化"可忽略"（Oeftger 2026）、跗分节内无肌纤维（FBbt 引 Chapman 1998/Soler 2004）。
2. **必须做对的三件事（否则站姿一定会退化）**：
   - **(i) 站姿弹位/符号方向。** Azevedo 2024 逐字："The joint is **flexed when the fly is standing**." → 站姿下胫–跗关节处于**屈曲**状态。flybody 用 `springref` 编码这个平衡位（T1 `-0.242`、T2 `0.608`、T3 `0.258` rad，**按腿不同**）。**若设成 0 或符号相反，站姿立即被破坏。**
   - **(ii) 不要给跗分节独立的自由 DoF。** 实验证据（§5.3 第 7 条）支持"近刚性"。
   - **(iii) 保留长腱的被动稳定作用。** 按 §2.8，长腱从股节近端贯穿到第 5 跗分节，是"stabilize all joints distal to the femur"的元件。flybody 用固定肌腱耦合来部分体现这一点；**如果加 DoF 时把这条耦合去掉，等于抽掉了被动稳定源。** ← **这是对 23.15 → 8.75 ms 退化最可辩护的假设（仍需实验验证）。**
3. **若要区分压肌/提肌**：用**一个**等效双向作动器驱动 tibia–tarsus。形态学上至少 3 条肌肉（TaD 内侧、TaL 外侧背侧、tarm 分 anterior/posterior 两条并共用 TaD 的腱）→ **单自由度双向作动器是合法但已声明不完整的近似**。**力臂必须声明为假设**，因为没有任何来源给出附着点（§3.6 已证明这是成像视野限制）。
4. **不要**给每条"肌肉"建独立的 spatial tendon：那需要 3 条肌肉各自的起止点坐标，**而这些坐标不存在（§5.3、§3.6）**。
5. **可用的标定量（真实测量值，可直接用于参数校准）**：腿被动刚度 **13.1 ± 7.97 μN mm⁻¹**、坐姿单腿载荷 **1.7 μN**（10 μN 体重）、起飞峰值 **~100 μN**、胫–跗角载荷斜率 **−0.31 °μN⁻¹（野生型）**、长腱直径 **3.25 ± 0.15 μm**。

### 6.4 建议的后续获取方向（非网页可解决）

- **Soler et al. 2004（Development 131:6041）**：需图书馆/馆际互借或直接联系作者。**这是唯一可能含跗节肌肉附着图的一手来源。**已确认无任何开放获取副本。
- **Miller (1950) *The morphology of the Diptera***：同上。FlyMimic 的 `MillerNN` 命名说明其编号体系仍在使用，值得取得原文以对齐肌肉身份。
- **Azevedo et al. 2024**：需核实 DOI 与开放状态（本次未取得原文）。
- **若要做定量跗节 ROM**：需自行拍摄/追踪活体果蝇跗节运动 —— 本次调研**没有找到任何已发表的 tibia–tarsus ROM 测量**。

---

## 7. 检索失败与限制的诚实记录

| 目标 | 状态 |
|---|---|
| Soler et al. 2004 全文（Development） | **HTTP 403（多路径）／closed access（Europe PMC、Semantic Scholar、OpenAlex 一致判定）**。未取得任何正文与图号。 |
| Azevedo et al. 2024 正文 + **3 个补充 PDF** | ✅ **全部 HTTP 200**（MOESM1 9,029,439 B、MOESM2 2,435,616 B、MOESM3 859,803 B），已 `pdftotext` 提取。**但其 Figure 4a 本身未取得**（PMC 图页为 reCAPTCHA 拦截，Nature 正文重定向付费墙）。 |
| Azevedo et al. 2020（eLife e56754） | ✅ 成功（PMC 全文 HTTP 200）。 |
| Agrawal et al. 2020（eLife e60299） | ✅ 成功（PMC 全文 HTTP 200）。注意：**`elifesciences.org` 直连被反爬拦截（"Client Challenge"），必须走 PMC。** | 
| Oeftger et al. 2026（iScience） | ✅ **成功（PMC HTTP 200，247,933 B）**。注意：`cell.com` 与 `sciencedirect.com` 均为 **HTTP 403**。 |
| Mamiya et al. 2022 | ✅ **成功**（bioRxiv PDF，14.8 MB，经 python urllib；普通 web_fetch 通道返回 **HTTP 429**）。 |
| flybody `fruitfly.xml` | ✅ **成功**（raw.githubusercontent.com HTTP 200，65,787 B）。 |
| flybody Nature 正文（PMC12310536）与补充材料 PDF | ✅ **成功**（补充 PDF 759,993 B，已 `pdftotext` 提取）。 |
| FlyMimic OpenSim/MuJoCo 文件 | ✅ **成功**（HTTP 200）。 |
| `gizemozd/neuromechfly-muscles`（FlyMimic 构建管线） | ❌ **HTTP 404**（main 与 master 均 404）→ 未公开。 |
| 独立 OpenSim/SimTK 果蝇腿模型 | ❌ **未找到**（无结果的诚实报告，不排除因检索方式而未发现）。 |
| Miller (1950) 及同类经典图谱正文 | ❌ **未取得**；**未尝试绕过任何付费墙**；因此本文不引用其任何内容。 |
| Chapman (1998) *The Insects* ch. 8 | ❌ **未取得**（纸本教材，无 OA 文本）。仅通过 FBbt 转述获知其关于"跗分节内无肌纤维"的结论。 |
| FlyBase `fbcvq.html` 查询端点 | ❌ **HTTP 404**（该端点已失效；改用的 `cvreport.pl` 端点 ✅ 可用）。 |
| FlyBase 术语详情页（`cvreport.pl`） | ✅ 成功（HTTP 200，14 个术语逐条提取）。 |
| **Azevedo 2024 Figure 4a（含附着点）** | ❌ **未取得**。PMC 图页 HTTP 200 但为 reCAPTCHA 拦截页；Nature 印刷版重定向 `idp.nature.com`。**这是唯一"可能还有一点坐标"但未能读到的目标。** |
| Haustein et al. 2024（Frontiers） | ✅ **成功**（HTTP 200，1,320,875 B，全文提取）。 |
| Haustein 开放数据（Harvard Dataverse） | ✅ **成功**（两个 DOI 均 HTTP 202；API HTTP 200；文件清单已核实）。 |
| NMF v2 postprint（EPFL） | ✅ **成功**（HTTP 200，6,537,877 B PDF，`pdftotext` 提取 1,934 行）。注：普通 web_fetch 报 "unsupported content type: application/pdf"，那是抓取工具限制，**不是服务器错误**。 |
| simtk.org | ⚠️ **部分可达，检索索引未能核实**：`www.simtk.org` 首页 **HTTP 200**；但 `/projects/` **HTTP 403**，检索页为 JS-only 外壳，真实检索端点 `/search/search.php` **连接被拒 `Errno 111`**（疑似 IP 限流，我本人重试亦全部被拒）。**这是"未决"，不是"不存在"。** 详见 §3.5 专门说明。 |
| opensim-org/opensim-models | ✅ HTTP 200，1,168 路径，**0 个果蝇/昆虫匹配** → 明确的否定结果。 |
| MuJoCo Menagerie | ✅ HTTP 200（3,430 条目），含 `flybody/`，Apache-2.0 —— 可用于独立核对 flybody 文件。 |
| `gizemozd/neuromechfly-muscles` | ❌ **HTTP 404** |
| MyoSuite | ✅ HTTP 200，882 路径，**仅人体** |
| **Zenodo** 检索 `"OpenSim" AND "Drosophila"` | ✅ API 可达，**total = 0**（另 `neuromechfly` → 5 条，其中 NeuroMechFly v2 代码为 10.5281/zenodo.12973000，**均非 OpenSim 果蝇模型**） |
| **figshare** 检索 | ✅ **全部为空**：`'Drosophila OpenSim'` → empty；`'neuromechfly'` → empty；`'Drosophila musculoskeletal model'` → **仅返回 Haustein 的 Frontiers 补充文件**。⚠️ 该检索端点**必须用 POST**（GET 返回 HTTP 404，是端点用法问题，不是内容结果） |
| **Dryad** 检索 | ✅ `Drosophila OpenSim` → **count 0**；`'Drosophila leg model'` → count 4（**均为生理/行为数据，无模型**） |
| **GitHub 仓库检索** | ✅ 全部 **total 0**：`drosophila+osim`、`drosophila+opensim`、`fruit+fly+opensim`、`insect+osim`、`fly+leg+musculoskeletal`；`drosophila+musculoskeletal` → **total 1 = gizemozd/FlyMimic**（即唯一者） |
| **实验室级检索**（Büschges / Blanke 等） | ✅ 各 **total 0** → 这两个实验室公开的成果**只有**两个 CC0 的 Harvard Dataverse 数据集（见 §3.7），**无模型、无肌肉** |
| Haustein 补充数据 `DataSheet1`（10.3389/fbioe.2024.1357598.s001） | ✅ HTTP 200，3,409,084 B，解压后**仅含补充图片**（image1.tiff…image6.jpeg）：`OpenSim` 0、`muscle` 0、`attachment` 0、`.osim` 0、`.obj` 0 |
| `gizemozd/fly_mimic`（FlyMimic 的姊妹仓库） | ✅ HTTP 200，119 路径，**不含任何 `.osim`** → **不是附着几何的第二来源** |

**未做之事（明确声明）**：未克隆任何仓库；未批量下载任何数据；未修改本项目任何既有文件；未运行任何重型计算；未绕过任何付费墙。

---

## 8. 配套机读台账

本条调研的全部逐条记录（含 URL、实际 HTTP 状态、逐字摘录、数值、三个 yes/no 判定）见：

**`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/research_tarsus_femur_sources.json`**

---

**再次强调：本文为手工调研，不是成品；仅供研究指引；任何附着坐标在得到明确来源之前，不得写入模型。**
