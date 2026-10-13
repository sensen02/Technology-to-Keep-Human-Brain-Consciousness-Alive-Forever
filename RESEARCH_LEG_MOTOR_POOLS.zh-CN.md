# 腿部肌肉运动神经元池：一手来源研究（有界检索）

> **手工研究，不是成品。** 本文件是人工检索与逐条核对的结果，不是自动化流水线产物，也不是"已验证的映射表"。它可以用来指导下一个动作，但它本身不构成任何 actuator 的电机神经元池。**除非某个来源明确写出某身份，否则不得把任何 root ID 指派给任何池。** 本次工作只读：没有修改项目既有文件，只新增本报告与同目录账本 `outputs/neural_muscle_closure/research_leg_motor_pool_sources.json`；没有运行重型计算，没有镜像任何大数据集（本地 feather 只用已有副本读取，未下载任何数据集）。

配套机读账本：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/research_leg_motor_pool_sources.json`

---

## 0. 先复算基线（任务数字作为给定值，这里只做确认）

用项目自带 `vendor/pylibs`（pyarrow 25.0.1）+ `python3.12` 读取 `/run/media/sensen/Data2/cell_wound_prototype/data/banc888/banc_888_meta.feather`（188,508 行 × 81 列；官方数据字典记 188,162 行 × 79 列，本地副本多 `root_890`，非逐字节同版本，仅作本地口径基线）：

| 项 | 复算值 |
|---|---|
| `flow == "efferent"` | 1033 |
| 其中 `body_part_effector ∈ {front_leg, middle_leg, hind_leg}` | **396**（任务给定值确认） |
| 三条腿 efferent 分布 | front 141 / middle 127 / hind 128 |
| 四个目标类型的计数 | 与任务给定值**逐格一致**（见下表） |

| 肌肉类 | front L/R | middle L/R | hind L/R | 合计 |
|---|---|---|---|---|
| pleural_promotor_muscle | 0 / 0 | 0 / 0 | 0 / 0 | **0** |
| tergopleural_promotor_muscle | 4 / 4 | 0 / 0 | 0 / 0 | 8 |
| pleural_remotor_and_abductor_muscle | 2 / 2 | 1 / 0 | 1 / 1 | 7 |
| sternal_adductor_muscle | 1 / 1 | 0 / 1 | 1 / 1 | 5 |

另外核对了 ID 语义：这 5 个池内 ID 在 `banc_888_id / root_id / root_626 / root_850 / root_888 / root_890` 六列同一行给出相同字符串（例 `720575941644474312` 在第 21565 行），并且与 VFB 的 `BANC_888` 对齐页所印 `source_id` 完全一致。所以本地 `root_id` 可以直接和 VFB / Codex 的 BANC v888 对照，不存在版本错位。

**数字口径提醒（重要，不裁决）：** 按 `pleural_promotor_muscle` 缺 6 个 + `tergopleural_promotor_muscle` 缺 8 个 + `pleural_remotor_and_abductor_muscle` 缺 1 个 + `sternal_adductor_muscle` 缺 1 个，共 **17** 个 actuator 无池；任务描述写的是 16。我没有读模型 XML 的 actuator 清单，因此无法判断这 1 个差异来自哪一对 (肌肉, 腿) 的编码方式。更重要的是：下面 §3 的文献结论显示，任务描述中"tergopleural promotor 在 middle/hind leg 上未解决"这一对 (肌肉, 腿) 组合，**在解剖学文献里很可能根本不成立**。请务必回查模型里那 8 个 actuator 到底对应哪块肌肉。

---

## 1. 一句话结论

- **唯一给出"逐腿 + 逐肌肉 + 逐细胞"的已发布表格，是 Cheong et al. 2026（eLife 96084）的补充文件 3 与 6**；但它来自**雄性 MANC**，给的是 MANC bodyid，**不能**直接填 BANC 池。
- **BANC 自身**把腿身份放在 `body_part_effector`（front/middle/hind leg），把肌肉类放在 `peripheral_target_type`（**不含腿**），二者需要同行 join；**BANC 的 `peripheral_target_type` 受控词表里根本没有 `pleural_promotor_muscle` 这个值**。
- **Virtual Fly Brain** 已经把 FBbt 的逐腿 MN 类对齐到具体 BANC v888 root ID（例如 mesothoracic sternal adductor MN = `720575941644474312`），这是目前唯一"类名 + 腿 + BANC root ID"三者齐全的现成来源。
- **tergopleural promotor（以及 pleural promotor）在 T2/T3 上没有任何已发表的运动神经元**，而且本体与文献都倾向于这两块肌肉在中/后腿**不存在**。若模型的 middle/hind 腿上真有名为 "tergopleural promotor" 的 actuator，其解剖学依据本身需要重新核对。
- 没有比 **banc888** 更新的 BANC 物化版本。

---

## 2. 四类肌肉"在哪条腿存在"（决定"未解决"是注释缺失还是肌肉缺失）

FBbt（FlyBase 果蝇解剖本体，经 EBI OLS4 REST 读取，全部 HTTP 200）里这四类都是**跨体节的通用类**，其逐腿子类分布不均：

| BANC 字符串 | FBbt 通用类 | 有记录的腿 | 无记录的腿 |
|---|---|---|---|
| pleural promotor muscle | FBbt:00058832 | T1（muscle 30，FBbt:00058860）、T3（muscle 82，FBbt:00058945） | **T2：无该子类** |
| tergopleural promotor muscle | FBbt:00058830 | **仅 T1**（muscle 28，FBbt:00058858） | T2、T3 |
| pleural remotor (+abductor) | FBbt:00058836 | T1 = #29 名称含 "and abductor"（FBbt:00058859）；T2 = #63（FBbt:00058919）；T3 = #83（FBbt:00058947） | 无（三条腿都有，但只有 T1 名称带 abductor） |
| sternal adductor | **无通用类** | **仅 T1**（muscle 33，FBbt:00058864） | T2/T3 无该词条（T2/T3 是 sternal promotor 64/84 + sternal remotor 65/85） |

逐字定义（OLS4 API 原文）：
- `pleural promotor muscle` FBbt:00058832 = "Promotor muscle of the adult thorax that attaches to the inner surface of the pleura."；子类仅两个：`prothoracic pleural promotor muscle 30`（FBbt:00058860，"arises on the episternum and extends posteriorly to insert on the anterior coxa (Snodgrass, 1935; Miller, 1950)"）与 `metathoracic pleural promotor muscle 82`（FBbt:00058945，"Pleural promotor muscle of the adult metathoracic coxa that extends dorsoventrally…"）。
- `tergopleural promotor muscle` FBbt:00058830 = "Promotor muscle that arises from the tergum."；**唯一**子类 `prothoracic tergopleural promotor muscle 28`（FBbt:00058858）。
- `adductor muscle` FBbt:00058840 = "Muscle that draws some body part back to position."；**唯一**腿部子类是 `prothoracic sternal adductor muscle 33`（FBbt:00058864），其父类是 `prothoracic sternal muscle`（FBbt:00058861），不是"sternal adductor"类。
- 全 FBbt 里名称含 "promotor muscle motor neuron" 的类**只有 1 个**：FBbt:00053100 `motor neuron of adult dorsal prothoracic nerve` —— "There are four of these on each side and they collectively innervate the prothoracic pleural and tergopleural promotor muscles (Azevedo et al., 2024)." 并附注 "Unclear from Azevedo et al. (2024) whether neurons innervate one or both muscles."

策展层面证据（FlyBase 解剖本体 issue #1814 "adult muscle merges"，2024-04-11 开、2024-05-10 关）：`tergopleural promotor muscle → [prothoracic tergopleural promotor muscle 28]` 与 `adductor muscle → [prothoracic sternal adductor muscle 33]` 两条的 merge 判定都是 **"no"**，notes 为 **"this might be in other segments"**。即：本体维护者只把它们记录在前胸，**明确没有断言**其他体节存在。"might be in other segments" 是猜想，不是存在记录，本研究不据此推断任何腿。

**这不是"注释系统缺失"，而是"这些肌肉只在前胸被经典解剖学记录"。** 所以 `pleural_promotor_muscle` 在 BANC 中零细胞、`tergopleural_promotor_muscle` 只出现在 front_leg，不能简单当成注释遗漏。

---

## 3. 按任务指定顺序的来源清单

### 3.1（第 1 项）更新的 BANC 版本 / BANC 论文补充表 —— **表号已确定：Supplementary Data 2**

- 论文（正式版）：Bates AS, Phelps JS, Kim M, Yang HH, … Lee W-CA (2026) "Distributed control circuits across a brain-and-cord connectome." *Nature* 656: 957–970. DOI [10.1038/s41586-026-10735-w](https://doi.org/10.1038/s41586-026-10735-w)。预印本 [10.1101/2025.07.31.667571](https://doi.org/10.1101/2025.07.31.667571)（v1 2025-08-01 / v2 2025-08-02 / v3 2026-02-14）。
- **检索状态（逐条诚实记录）**：Europe PMC REST 全文 XML **loaded（HTTP 200，576,365 字节）**；Europe PMC 元数据 loaded（isOpenAccess=Y）；**nature.com 直连失败**："cross-origin redirect to https://idp.nature.com is not followed automatically"；**biorxiv.org 直连失败**：HTTP **429** + Cloudflare "Attention Required … We have received a high number of requests from this session"；Springer ESM PDF 直连失败（工具不接受 `application/pdf`），其中 MOESM1（Supplementary Notes）与 MOESM13（Peer Review File）经文本代理读到，MOESM3/4/12 返回 GCS 403 `AccessDenied`（该编号不存在）。
- **正文本身不含肌肉级证据**：通篇没有出现 promotor / remotor / adductor / pleural / tergopleural / sternal / `peripheral_target_type` 等字样；Fig. 2d/2e、ED Fig. 4f 只按 body part 汇总效应器。原文："We assigned effectors to 1 of 24 body part categories where possible (99.4%)."
- **该看哪张表、哪一列**：
  - **Supplementary Data 2 = `supplemental_data_2.txt`**（"BANC neuron metadata"，v888，187,590 个神经元）—— 唯一同时含 `peripheral_target_type` + `body_part_sensory` + `body_part_effector` + `nerve` 的官方逐神经元表。名义筛选：`flow == "efferent"` AND `peripheral_target_type %in% {四类之一}` AND `body_part_effector %in% {front_leg, middle_leg, hind_leg}`。
  - **Supplementary Data 1 = `supplemental_data_1.txt`**（受控词表，30,318 字节）—— 本次**完整读取**。`peripheral_target_type` 枚举**包含** `tergopleural_promotor_muscle`、`pleural_remotor_and_abductor_muscle`、`sternal_adductor_muscle`，**不包含** `pleural_promotor_muscle`（枚举字母序为 `photoreceptor → pleural_remotor_and_abductor_muscle → pleurosternal_muscle`，该值本应插在中间而缺席）。`body_part_effector` 枚举含并列的 `front_leg` / `middle_leg` / `hind_leg`。`nerve` 枚举含左/右 × pro/meso/metathoracic leg nerve，**但存在逗号连接多值项**（如 `left_mesothoracic_leg_nerve, right_mesothoracic_leg_nerve`），故 `nerve` 并非严格单值。
  - 官方数据字典（GCS `documentation/banc_888_meta.md`，HTTP 200）：`peripheral_target_type` = "The sensor or effector structure or organ targeted by an afferent / efferent neuron"；`body_part_effector` = "For efferent neurons, the body part targeted, taken at the site of action when this differs from the innervation site"。
  - **BANC 社区标注方案**（wiki "Annotations (cell-types, etc.)"）另有独立的 `muscle innervated`（含 `innervates tergopleural promotor` / `innervates pleural promotor` / `innervates pleural remotor and abductor` / `innervates sternal adductor`，名称归于 Azevedo 2024）与 `body part innervated`（`innervates T1/T2/T3 leg`）。**这两类标签不会整体并入 `banc_888_meta`**，只存在于 CAVE 的 `cell_info` / `codex_annotations`。
- **是否更新版本**：**没有**。GCS 桶 `compiled_data/` 只有 `banc_888/`，`neuron_annotations/` 只有 `v626/` 与 `v888/`；CHANGELOG 记 "Dataset: BANC v888"，Dataverse v3.0 (2026-07-01) 仅增加突触评估 CSV、参考骨架、示意图与 supplemental_notes.pdf，**没有新的物化版本、没有新注释列**。
- **腿分辨判定**：`peripheral_target_type` = **否（无腿分辨）**；`body_part_effector` = **是（逐细胞 front/middle/hind leg）**。因此对一个神经元"哪条腿的哪块肌肉"只能靠同一行两列 join —— **论文自己不做这个 join，也不为 tergopleural/pleural promotor 在 T2/T3 提供任何证据**。
- **能否填池**：**部分能**（在允许同行 join 的前提下，可给出 (肌肉类, 腿, 侧) → root ID）；**不能补 pleural promotor**（词表无此值）；**不能证明 T2/T3 的 tergopleural promotor 存在**。
- **未读项（诚实）**：Supplementary Data 2 本体（约 30 MB）与 Supp 7（`datafile/13917306`，两次 403）按"不下载批量数据"的约束未读；BANC CAVE `cell_info`/`codex_annotations` 需 CAVE token，未读；neuPrint API 需 Bearer key，未读。

### 3.2（第 2 项）FlyWire/FAFB 腿 MN 类型表 —— **对本问题无效，可排除**

- **FAFB 里没有腿运动神经元**：FAFB 是成蝇**脑**体积（Zheng et al. 2018；Dorkenwald et al. 2024 FlyWire），不含 VNC。FlyWire Codex FAQ 原文："FlyWire FAFB - Female Adult Fly Brain; default snapshot 783"。所以任何 "FAFB leg MN 类型表" 在概念上不存在。
- Codex 本身：静态 FAQ 页 HTTP 200，但 app 页与下载资源**需要 Google 登录 / JS / `api_token`**，本次无法取得公开非 JS 的类型表；试探的 `/api/cell_types` 等路径 404。
- **腿分辨**：不适用；**能否填池**：**否**。
- 真正含腿 MN 的是 **FANC**（Azevedo 2024，仅 T1）与 **BANC**（Bates 2026），已在 3.1 / 3.4 覆盖。

### 3.3（第 2 项）MANC 命名体系（Cheong et al. 2026 eLife 96084）—— **唯一"名称+腿"，但属雄性 MANC**

- 引用：Cheong HSJ, Eichler K, Stürner T, Asinof SK, Champion AS, Marin EC, Oram TB, Sumathipala M, Venkatasubramanian L, Namiki S, Siwanowicz I, Costa M, Berg S, Janelia FlyEM Project Team, Jefferis GSXE, Card GM (2026) "Organization of circuits linking descending input to motor output in the *Drosophila* Male Adult Nerve Cord connectome." *eLife* 13: RP96084. DOI [10.7554/eLife.96084](https://doi.org/10.7554/eLife.96084)；PMID 42474298；PMC13384506。
- **检索状态**：Europe PMC REST 全文 XML **loaded（HTTP 200，638,690 字节）**；`elifesciences.org/articles/96084` 返回 "Client Challenge - A required part of this site couldn't load"（**JS 墙**）；`api.elifesciences.org` 直连 **HTTP 406 not acceptable**；CDN 上的两个补充 CSV 经文本代理读取。
- **承载映射的表与列**（这是关键）：
  - **Supplementary file 3** "MN and efferent light-level matching in MANC"（csv），表头逐字：`class,bodyid,group,subclass,type,systematic_type,exit_nerve,target,synonyms,match_certainty(1-5),publication_match,line_match,match_notes`
  - **Supplementary file 6** "Serial Leg MN groups"（csv），表头逐字：`bodyid,group,serial,type,class,subclass,soma_neuromere,soma_side,plot_name,target`
- **逐字行**：`motor neuron,17091,17091,ml,Pleural remotor/abductor MN,MNml17,MesoLN_L,Pleural remotor/abductor`；`motor neuron,41482,41482,ml,…,MNml31,MesoLN_R,…`；`motor neuron,14168,14168,hl,…,MNhl17,MetaLN_L,…`；`motor neuron,14596,14596,hl,…,MNhl31,MetaLN_R,…`；`motor neuron,13817,13817,ml,Sternal adductor MN,MNml16,MesoLN_L,Sternal adductor`；`motor neuron,20667,20667,hl,Sternal adductor MN,MNhl16,MetaLN_R,Sternal adductor`；promotor 类只有 `motor neuron,12396,12396,fl,Tergopleural/Pleural promotor MN,MNfl56,DProN_L,Tergopleural/Pleural promotor`（另有 MNfl53/54/55，**全为 fl / dorsal prothoracic nerve**）。
- **决定性否定证据（逐字）**："We could not identify serially repeating homologs (in T2 and T3) for the Tarsus levators and depressor MNs as well as the Tergopleural/Pleural promotor MNs (Figure 7D)." 与 "For the Tergopleural/Pleural promotor MNs that innervate a muscle in the thorax, we could not find MNs in T2 and T3 that were similar in morphology and connectivity, suggesting that these MNs are either not present or have a different function and morphology"。Fig. 7D 图注："For three of the muscle targets, we were not able to find corresponding neurons in T2 and T3 leg neuropils. Thus, only the T1 leg neuropils have muscle target assignments."
- **腿分辨**：**YES，cell-type level**（`fl/ml/hl` 前缀 + exit nerve + `soma_neuromere` T1/T2/T3 + 逐细胞 bodyid）。**但 T2/T3 的肌肉靶标是串行同源推断**（Fig. 7E 图注："All MN targets in T1 were assigned by careful matching to the Female Adult Nerve Cord (FANC) dataset (Azevedo et al., 2024), while all MN targets in T2 and T3 were assigned by serial sets"）。
- **能否填池**：**否，不能直接填 BANC 池** —— 它给的是 MANC bodyid（如 `13817`），且物种/性别与 BANC（雌性）不同。它的价值是给出**覆盖预期**：pleural remotor/abductor 在 T1/T2/T3 各 **2 个/侧**；sternal adductor 在 T1/T2/T3 各 **1 个/侧**；tergopleural/pleural promotor 在 T2/T3 **无同源神经元**。

### 3.4（第 3 项）Virtual Fly Brain 查询 —— **目前唯一"类名 + 腿 + BANC root ID"齐备的现成来源**

- 接口：`https://v3-cached.virtualflybrain.org/`（VFBquery v1.22.57，Court et al. 2023 DOI 10.3389/fphys.2023.1076533），**无需 token / 无需 JS**，全部 HTTP 200。
- `list_connectome_datasets` 逐字给出：`{"label": "BANC web interface v888", "symbol": "BANC", "short_form": "BANC888"}`（同时有 flywire783、MANC 1.2.1、male-cns 1.0、hemibrain 1.2.1 等）。
- **可用的查询式**（VFB 的 "innervates <muscle>" 等价物）：
  - `run_query?id=<FBbt_id>&query_type=ListAllAvailableImages` —— 列出该类型下每个已登记个体及其**数据源 ID（BANC root ID）**；
  - `run_query?id=<FBbt_id>&query_type=NeuronsPartHere` —— 直接问"哪些神经元有部分位于这块肌肉"；
  - `search?query=...` / `get_term_info?id=...` —— 术语检索与报告；
  - `v3-cached.virtualflybrain.org/docs.json` —— 全部端点目录。
- **实测结果（逐字要点）**：
  - `FBbt_00053151`（metathoracic pleural remotor muscle motor neuron）→ BANC_888: `720575941537495489`、`720575941414478484`、`720575941562065920`（另有 MANC 15262/14596/14168/18286 与 male-cns 条目）。
  - `FBbt_00053567`（metathoracic sternal adductor muscle motor neuron，MNhl16）→ BANC_888: `720575941503422187`、`720575941611945389`。
  - `FBbt_00053566`（mesothoracic sternal adductor muscle motor neuron，MNml16）→ BANC_888: `720575941644474312`。
  - `FBbt_00053102`（prothoracic sternal adductor muscle motor neuron，Azevedo 的 "MN 7"）→ BANC_888: `720575941499964555`、`720575941527847608`。
  - `FBbt_00053104`（prothoracic pleural remotor and abductor muscle motor neuron，同义名 `pleural_remotor_and_abductor` / `_miller_29` / "MN 12-13"）→ BANC_888: `720575941494164864`、`720575941548546696`、`720575941593715569`、`720575941691007896`。
  - `run_query?id=FBbt_00058860&query_type=NeuronsPartHere`（prothoracic pleural promotor muscle 30）→ **rows = [] , count = 0**（没有任何已登记神经元部分位于该肌肉）。
  - `search?query=pleural promotor` → 8 个 male-cns 个体 "Tergopleural/Pleural promotor MN_L/R"（MaleCNS:800061/801173/807281/934091/801824/802072/805745/810535），**全部只出现在前腿（T1）语境**。
- **检索失败（诚实）**：`run_query?id=FBbt_00053150`（mesothoracic pleural remotor MN 的对齐列表）本次 **工具超时 30 秒**，未取到 BANC 条目；`https://www.virtualflybrain.org/reports/FBbt_*` 是 JS 单页、服务端不渲染术语内容（改用 `/blog/2022/01/01/<slug>-<fbbt_id>/` 可读全文）；MCP 服务 `vfb3-mcp.virtualflybrain.org` 未接入（需要 MCP 客户端）。
- **腿分辨**：**YES，cell-type level**（类名自带 pro/meso/metathoracic，且每类给出该腿的 BANC root ID）。**能否填池**：**部分能**（见 §4）。
- **必须记录的来源冲突**：VFB 把 `720575941414478484` 归入 FBbt_00053151（metathoracic pleural remotor MN），但本地 BANC v888 里该细胞 `cell_type = femur_reductor_dorsomedial_ribbon`、`peripheral_target_type = femur_reductor_muscle`（两者共用同一条 metathoracic leg nerve 束）。**任何池指派前必须先解决这个冲突，本报告不替它裁决。**

### 3.5（第 4 项）腿部肌肉解剖图谱 / 逐腿存在性

- 权威命名来源是 **Miller A. (1950) "The internal anatomy and histology of the imago of Drosophila melanogaster"**（Biology of Drosophila: 468–481；FlyBase **FBrf0186027**），它被上述每一个 FBbt 定义引用，其连续编号（muscle 28/29/30/33 在 T1，63/64/65/82/83/84/85 在 T2/T3）正是"同名肌肉在不同腿上有不同名称与编号"的机制。
- **检索状态（诚实）**：Miller (1950) 原文**本次未读到**，因此本报告**不直接引用 Miller 的任何原句**；我只通过 FBbt 定义与 FlyBase 参考号间接引用。Snodgrass 1935（FBrf0111704）同样未直接读取。Azevedo et al. 2024 的 Supplementary Methods 与 Supplementary Table 1（69 个 T1 MN → 18 块肌肉 的表）**也未读到**（Springer ESM PDF 工具不支持 `application/pdf`；PMC 图页 reCAPTCHA；nature.com 跨域重定向到 idp.nature.com）。
- 能读到的替代一手来源已足够回答"逐腿存在性"：FBbt 类层级（§2）+ FlyBase DAO issue #1814 + Cheong et al. 2026 的逐字否定句。
- **腿分辨**：**YES（对存在性），但只是"记录在哪些腿"而非"证明其他腿没有"** —— 必须把"无记录"写成"无记录"，不能写成"已证明不存在"。

### 3.6 未能读取的来源（失败清单，全部为工具/权限层面，不是"来源不存在"）

| 目标 | 实际错误 |
|---|---|
| `biorxiv.org` 正文/PDF（含 520299、543407、667571） | HTTP **429** + Cloudflare "Attention Required" |
| `elifesciences.org` HTML/PDF | "**Client Challenge** … site couldn't load"（JS 墙） |
| `api.elifesciences.org/articles/96084` | HTTP **406** not acceptable |
| `europepmc.org/article/...` | HTTP **403** Cloudflare |
| `europepmc.org/api/fulltextRepo?...` | HTTP **403** "PDF link has expired or is invalid" |
| `europepmc REST /PMC11356479/fullTextXML`、`/PMC11348827/supplementaryFiles` | HTTP **406** Not Acceptable |
| `pmc.ncbi.nlm.nih.gov` 图页 | reCAPTCHA |
| `static-content.springer.com` 所有 ESM PDF | 工具错误：unsupported content type **application/pdf**（MOESM1/MOESM13 仅经文本代理读到） |
| `rdcu.be/fncjS`（论文免费 PDF 直链） | "Human Verification … confirm you are human"；且 `Target URL returned error 405: Method Not Allowed` |
| `dataverse.harvard.edu/api/access/datafile/13917306`（Supp 7） | 文本代理两次 **403**；直连为跨域重定向到 S3，未跟随 |
| BANC CAVE（`cell_info` / `codex_annotations`） | 需要 CAVE token / `banc` Python 客户端，本次只有网络读取工具 |
| neuPrint API（BANC 自身 `peripheral_target_type` 词表） | 所有查询端点需 Bearer API key，无凭证 |
| FlyWire Codex app / 下载资源 | 需要 Google 登录 / JS / `api_token` |

---

## 4. 逐 actuator 判定：能填什么、不能填什么

下表以"每一 (肌肉类, 侧, 腿) 池"为单位。**只有当来源明确写出该池的细胞身份时才标"可补"。**

| 池 | BANC v888 细胞数 | 判定 | 依据 |
|---|---|---|---|
| pleural promotor · front L/R | 0 | **不可能** | BANC 词表无此值；VFB `NeuronsPartHere(FBbt_00058860)` = 0；前腿 4 个 dorsal prothoracic nerve MN 在 BANC 中整体标为 tergopleural promotor（FBbt:00053100 明说这群"collectively innervate"两块肌肉且"unclear … one or both"） |
| pleural promotor · middle L/R | 0 | **不可能** | FBbt 无中胸子类（第 3 方来源也不给中腿该肌肉 MN） |
| pleural promotor · hind L/R | 0 | **不可能** | FBbt:00058945 证明后腿**有这块肌肉**，但无 MN 类、BANC 无细胞、Cheong 找不到 T2/T3 promotor MN |
| tergopleural promotor · front L/R | 8（4/侧） | **已可用**（但 tergopleural 与 pleural promotor 在 BANC 未拆分） | 本地 `tergopleural_promotor_pleural_promotor(_miller_28_30)`，附着 dorsal prothoracic nerve；数量与 FBbt:00053100 的"每侧 4 个"及 Cheong MNfl53–56 一致 |
| tergopleural promotor · middle L/R | 0 | **不可能，且该肌肉很可能在 T2 不存在** | FBbt:00058830 唯一子类为前胸 28；curator 只写 "might be in other segments"；Cheong 明确找不到 T2/T3 同源 MN |
| tergopleural promotor · hind L/R | 0 | **不可能，且该肌肉很可能在 T3 不存在** | 同上 |
| pleural remotor and abductor · middle/left | 1 | **可补（1 个）** | `720575941553503460`，`pleural_remotor_and_abductor_MesoLN_dorsal_5`，left mesothoracic leg nerve；与 FBbt:00053150 及 Cheong MNml17/MNml31 一致。**但 Cheong 说每侧 2 个，BANC 只 1 个**，需复核是否遗漏 |
| pleural remotor and abductor · middle/right | 0 | **来源冲突，不可指派** | BANC 无右侧中腿该细胞；FBbt 断言该腿该肌肉有 MN，但本次未取得 FBbt_00053150 的 BANC 对齐列表（超时），**不得按对称性猜一个 root ID** |
| pleural remotor and abductor · hind L/R | 2（1/侧） | **可补（1/侧）** | `720575941537495489` / `720575941562065920`，`_dorsomedial_ribbon`，metathoracic leg nerve；VFB 归入 FBbt_00053151。Cheong 说每侧 2 个，BANC 只 1 个，需复核 |
| sternal adductor · middle/left | 0 | **不可能（无可引用身份）** | BANC 只有右侧一个；FBbt:00053566 (MNml16) 自身注记该肌肉身份 "unclear from references"；Cheong 给的是 MANC 雄性细胞 `13817`，**不是 BANC root ID，禁止转用** |
| sternal adductor · middle/right | 1 | **已可用（1 个）** | `720575941644474312`；VFB 明确把该 root ID 对齐到 FBbt_00053566 (MNml16) |
| sternal adductor · front L/R | 2（1/侧） | **已可用（1/侧）** | `720575941527847608`（left, leg nerve）/ `720575941499964555`（right, accessory nerve）；VFB 对齐到 FBbt_00053102（Azevedo "MN 7"） |
| sternal adductor · hind L/R | 2（1/侧） | **可补，但命名有冲突** | `720575941611945389` / `720575941503422187`，`sternal_adductor_MetaAN`，first abdominal nerve；VFB 对齐到 FBbt_00053567 (MNhl16)。**同一批细胞 `manc_cell_type` 却写 'Sternal anterior rotator MN'(L) 与 'MNhl62'(R)**，必须在文档里标注冲突、不能挑一个当事实 |

**净结果**：在"允许 BANC 同行 join + 采用 VFB 对齐"的既定口径下，**能补的只有 pleural remotor/abductor 与 sternal adductor 两类**（且中腿右侧、中腿左侧各留 1 个缺口）；**pleural promotor 六条腿全部无法填**；**tergopleural promotor 的 middle/hind 池无法填，且其解剖学前提可疑**。

---

## 5. 即使拿到最好的来源，仍然缺什么

1. **pleural promotor 四条腿（含前腿）的 BANC root ID**：BANC 把 tergopleural 与 pleural promotor 合并成一个细胞类型且只在 T1 出现；要拆分必须读 BANC 社区逐神经元 `muscle innervated` 标注（CAVE `cell_info` / `codex_annotations`），本次无 token 未读。
2. **tergopleural promotor 在 middle/hind leg 的运动神经元**：任何已发表来源都没有；FBbt 与 Cheong 都表明该肌肉在 T2/T3 很可能不存在。这 8 个 actuator **目前不能由任何来源填池**，且需要先确认模型里它们的解剖学定义。
3. **middle/left 的 sternal adductor**：BANC 无细胞；FBbt 对应类自身说"unclear which muscle"；Cheong 给的是 MANC 雄性细胞。三者都不能转成 BANC root ID。
4. **middle/right 与 hind 的 pleural remotor，以及每侧数量差异**（BANC 1 vs Cheong 2）：需要与 BANC 逐细胞标注或作者核对，判断是注释遗漏还是物种/性别差异。
5. **两处命名/身份冲突**：`720575941414478484`（VFB = metathoracic pleural remotor MN；BANC 本地 = femur_reductor_dorsomedial_ribbon / femur_reductor_muscle）以及 MetaAN sternal adductor 细胞的 `manc_cell_type`（Sternal anterior rotator MN / MNhl62）。
6. **没有读到的关键原文**：Miller (1950) 原文、Azevedo et al. 2024 的 Supplementary Methods 与 Supplementary Table 1（T1 的 69 MN → 18 肌肉）。因此对 T1 肌肉计数没有逐字一手引用。
7. **BANC CAVE / neuPrint 未访问**：所以"BANC 是否把 `tergopleural_promotor_pleural_promotor` 或 `sternal_adductor` 标到 T2/T3"只能以本地 v888 快照为限回答：快照答案是**否**。

---

## 6. 下一步的最小可执行查询（谁做都能得到确定答案）

1. **BANC CAVE（需 token）**：对 `cell_info` / `codex_annotations` 查 `annotation IN ('innervates pleural promotor','innervates tergopleural promotor','innervates pleural remotor and abductor','innervates sternal adductor')` 与 `('innervates T2 leg','innervates T3 leg')` 的交集，并按 `side` 分组 —— 这一步同时回答"pleural promotor 能否单独成池"和"tergopleural/pleural promotor 是否有 T2/T3 实例"。
2. **VFB**：重试 `run_query?id=FBbt_00053150&query_type=ListAllAvailableImages`（本次超时），以确定 middle/right 的 pleural remotor BANC root ID。
3. **本地**：核对 `720575941414478484` 与 `720575941611945389`/`720575941503422187` 两处命名冲突，以及在模型 XML 中确认那 8 个"tergopleural promotor" actuator 到底对应哪块肌肉、哪条腿。
4. **补充数据**：按官方列语义筛 `supplemental_data_2.txt`（`peripheral_target_type` × `body_part_effector` × `side`），与本地 feather 交叉验证一次，确认 396 与四个类型的计数在官方表上同样成立。

---

## 7. 使用这条研究的硬规则

1. **手工研究，不是成品**；本文件不构成任何 actuator → root_id 断言。
2. 任何 root ID 只有在被具名来源明确指向某 (肌肉类, 腿, 侧) 时才可入池；入池时必须同时记录来源 URL 与原文片段。
3. **禁止**按左右对称性补池；**禁止**用 MANC/FAFB 的 bodyid 或 Cheong 的 MN 名称（MNml17 等）直接生成 BANC root ID。
4. **禁止**把 tergopleural promotor 或 sternal adductor 推广到六条腿：文献只在前胸记录它们，且 Cheong 明确找不到 T2/T3 的 promotor 同源神经元。
5. **pleural promotor 目前不可能单独成池**：BANC 中它与 tergopleural promotor 共用同一细胞群，任何拆分都要先拿到逐细胞 `muscle innervated` 标注。
