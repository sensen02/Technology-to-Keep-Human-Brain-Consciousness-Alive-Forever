# 外部证据：Eon Systems 的虚拟果蝇、FlyWire、VNC 与"数字斯芬克斯"

本文件记录本轮实际下载并**逐条核对过的原文**，以及它们对我们这个"果蝇自身闭环"目标的直接含义。所有引文都来自下载到本地的原文，不是二手转述。

下载目录：`/run/media/sensen/Data2/cell_wound_prototype/outputs/external_literature_round16/`

## 一、Eon Systems 自己的原文（一手来源）

Eon Systems PBC 在 2026-03-10 发布的说明（已下载存档页 `eon_systems_blog_embodied_brain_emulation.html`，正文提取 `eon_systems_blog_text.txt`）中，**他们自己写明**：

- "The fly body is **not currently driven by the full downstream motor hierarchy** of the biological fly. Instead, we use a small number of descending outputs as a practical interface between the connectome model and the biomechanics."
- "**Fly walking was implemented using slight modifications to existing NeuroMechFly controllers, trained to imitate the walking behavior of the fly.**"
- "Activation of these specific descending neurons serve to **influence the controllers of the body, which have been trained by imitation learning** to mimic particular fly behaviors."
- 具体用了哪些神经元：转向用 **DNa01 / DNa02**，前进速度用 **oDN1**，梳理用触角下行神经元，取食用 proboscis MN9。
- 他们自己承认：**"this mapping can be somewhat arbitrarily chosen by hand (as is our case), or learned with reinforcement learning, or mediated by lower-level controllers, but in all cases it is still an approximation of the true motor hierarchy."**
- 他们指出的解决方向：**"more imaging or electrophysiology to understand the specific transformation between DN firing rate and specific behavior."**
- 脑-体同步周期 **15 ms**，并自承"may be too slow for some behaviors"（我们用的是 5 ms）。
- 他们结尾说：**"extending our model to include the VNC and other outputs is another useful direction."**

## 二、独立技术评述：VNC 与"什么才算 embodied"

`toward_fully_embodied_virtual_drosophila.pdf`（Zenodo 19511908，CC BY 4.0）逐条分析同一份公开材料后指出：

- Eon 的虚拟果蝇**用动画控制器而不是已发表的 VNC 连接组**与身体接口，因此"**decouples neural dynamics from motor output**"。
- 具体机制：下行神经元发放超过阈值→播放预定义行走动画→按触发神经元身份与频率施加身体旋转；梳理/取食同理。**"the behavior is produced by a learned motor controller, not by the emergent dynamics of the full motor hierarchy."**
- 它给出一条对我们**完全一致**的判据：真正 embodied 的路径必须是
  **Brain activity → Descending neurons → VNC input neurons → VNC circuitry → Motor neurons → Joint torques → Movement**。
- 也点出了原因：FlyWire 脑与 MANC 胸腹神经索来自**不同个体、不同性别**，神经元 ID 不对应，人工逐个配对约 1100 个下行神经元需数小时/个。
- 它自己的限制声明同样值得记录：LIF 权重可能不够准、神经递质推断可能有误、**MANC 连接组可能无法干净地映射到 87 关节的 NeuroMechFly 身体**、即使 pinout 正确，在缺少感觉反馈或神经调质时行走也可能不涌现。

## 三、"数字斯芬克斯"：行为逼真 ≠ 生物学意义

`digital_sphinx.pdf`（Brunton, Abe, Hu, Tuthill；eLife reviewed preprint 111516，12 页，已渲染核看首页）做了一个**反向实验**：把**线虫 C. elegans 的连接组**接上**果蝇身体**，用深度强化学习训练一个 ANN 把蠕虫运动神经元激活映射到果蝇腿部执行器。结果：

- "The resulting digital sphinx produces **highly realistic fly walking—yet it is biologically meaningless.**"
- 两条教训："DRL is a capable optimization algorithm... even when the imposed constraints are incomplete or outright wrong"；以及"**even when brain and body models are individually accurate, a biologically unrealistic interface between them renders the whole model meaningless.**"
- 更关键：蠕虫连接组在其中**只充当一个 RNN**，"Its role in the movement policy could be fulfilled equally well by a **randomly connected RNN**"，因为全部学习都发生在黑箱运动解码器里。

## 四、支撑性原文（已下载全文本）

| 论文 | 本地文件 | 与我们的关系 |
|---|---|---|
| Shiu et al. 2024, Nature（Eon 的脑模型基础） | `shiu2024_nature_brain_model.fulltext.xml/.txt` | 全脑 LIF；验证的是取食与触角梳理（MN9、aDN1/aDN2），**并明确说需要下行神经元的通路在 FlyWire 体积里不完整**；无肌肉、无身体 |
| Dorkenwald et al. 2024, Nature（FlyWire） | `dorkenwald2024_flywire_wiring_diagram.fulltext.xml/.txt` | 脑连接组本体 |
| Lappalainen et al. 2024, Nature | `lappalainen2024_connectome_constrained_networks.fulltext.xml/.txt` | Eon 的视觉通路来源 |
| Takemura et al. 2024, eLife（MANC VNC） | `takemura2024_manc_vnc_connectome.pdf/.txt` | 雄性 VNC 连接组 |
| eLife RP96084 (2026) 下行输入→VNC 运动输出 | `desc_to_motor_vnc_2026.fulltext.xml/.txt` | **与我们的接口问题最相关**，见下 |

**RP96084 的几条关键事实（对我们直接有用）**：

1. "**direct DN-MN connections are infrequent**"——下行神经元**很少直接**连到运动神经元，中间有 premotor 群落。这与我方"没有人工 DNa01 直连"的处理方向一致。
2. 前腿神经命名：**DProN / VProN / ProAN / ProLN**（背侧、腹侧、附属、腿神经）——**正是我们肌肉解剖解析器采用的四个前腿神经类**，这是外部独立印证。
3. MN→肌肉对应在原始数据里**看不到**："the muscles were outside our EM dataset, which included only the VNC, so we could not directly observe this correspondence"，他们只能靠**树突形态与文献已鉴定 MN 比对**来做映射。**这说明"MN→肌肉对应"是公开数据本身的结构性困难，不是我们独有的无能。**
4. "flies have been shown to walk **without proprioceptive feedback**"；同时指出被去头果蝇虽有运动模式但**不协调，可能与缺少下行输入有关**。
5. 生成行走节律的 CPG 神经元"**have not been identified**"。

## 五、对我们项目的直接含义

1. **我们退回人工 DNa01 桥接是正确的决定。** Eon 用的正是 DNa01/DNa02（转向）+ oDN1（前进），并且他们自己和独立评述都称其为"对真实运动层级的一种近似"，走路靠模仿学习控制器。
2. **用户的验收标准与文献判据一致**：要求 环境→感觉→神经→运动神经→肌肉→身体→环境，不允许用控制器/动画/阈值触发替代。这不是苛求，正是"fully embodied"的定义。
3. **"MN→肌肉对应"的困难有据可依**：连 MANC 原始论文都无法在数据内直接观测肌肉对应，只能靠形态+文献。这解释了我们的 46 个未解析输出为何不能靠猜补齐。
4. **我们的四个前腿神经类得到独立印证**（DProN/VProN/ProAN/ProLN）。
5. **两个反向警告**：行为逼真可以完全不代表生物真实（数字斯芬克斯）；以及把连接组当成黑箱 RNN + 学习型解码器，会让整个模型失去意义。

## 六、未取得的文件（如实记录）

- NeuroMechFly v2（Nature Methods 2024）正文付费；bioRxiv/eLife CDN 的预印本 PDF 本次请求返回失败，仅取得题录（`paper_metadata.json`）。FlyGym/FlyMimic 仓库即其代码与身体模型，本项目已经在用该身体。
- Eon Systems 官方站点原文通过 Wayback 存档取得（2026-04-10 快照），未取得其官方仓库中标注的 2026 基准论文 PDF。

以上均为外部资料，**其中任何一种做法都没有被本项目采纳为"完成"**；它们的作用是界定什么做法不算自主闭环。
