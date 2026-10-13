# FlyBody 躯干分支：换躯干会不会打断闭环？

**手工研究原型，不是成品。** 本文件里的每一个数字都来自本机实测产物，产物路径写在 §10。
没有任何一条结论被归功于神经回路之外的东西，也没有任何「会走路」的主张。

---

## 0. 先做备份（按你的要求）

| 备份方式 | 位置 | 规模 | 回滚方式 |
|---|---|---|---|
| git 快照 | 提交 `369a2bd`，标签 `pre-flybody-20261012_171343` | 源码 + 全部 .md 报告（5.2 MB 未跟踪文件一并提交） | `git -C /run/media/sensen/Data2/cell_wound_prototype reset --hard pre-flybody-20261012_171343` |
| 整盘文件级快照 | `/run/media/sensen/Data2/_snapshot_pre_flybody_20261012_171343/cell_wound_prototype/` | 23,345,713,320 字节，58,374 个文件（含 data/ outputs/ vendor/ 两个 venv） | `rsync -a --delete <快照>/ /run/media/sensen/Data2/cell_wound_prototype/` |

快照在本次实验**全部开始之前**完成（rsync exit 0）。`upstream/flybody/`（本次新下载的 151 MB 第三方模型）
是快照之后新增的输入，它可复现：来源与 commit 见 §2。

---

## 1. 直接回答你的问题

**切换 FlyBody 不会打断闭环，也不会破坏神经元回路。** 回路本体（BANC 分层、突触电导网络、
脉冲、0.5 ms 神经钟 / 5 ms 命令钟、感觉通道分组、运动池映射、解码器）在这条分支里**逐字节复用**，
一行都没有改。换掉的是两个**翻译层**：

| 层 | 规范体 | FlyBody | 本分支怎么处理 |
|---|---|---|---|
| 感觉提取 | 每腿 7 铰链：胫节偏差 + 多关节均值 + 足端接触力 | 每腿 **11** 铰链；没有肌肉、没有激活态 | 用翻译表指定的 6 个铰链 + 爪碰撞体，量纲按各自「单腿承重份额」归一（本项目已声明的可比化归一器） |
| 运动翻译 | 运动池 → **同名肌肉**（90 个肌肉执行器） | 每腿 **8 个关节执行器**（位置伺服）+ 1 个爪吸附 | 90 个肌肉激活**投影**到 36 个 FlyBody 关节执行器；两种声明变体：保留发布伺服 / 换成同权能力矩电机 |

**而且闭环仍然在因果链里**（这是可测的，不是声明）：把脉冲输出静音、或切断感觉输入、或在 15 ms 后
切断运动命令，**F lyBody 的轨迹都会改变**（最大状态差 22.4 个模型单位）；正常臂与静音臂不同 ⇒ 回路在环内。

真正变差的是**结果**，不是回路：

| 躯干 | 接上回路（normal） | 脉冲静音 | 读法 |
|---|---|---|---|
| 规范 FlyMimic/Miller 体 | 逃逸 **23.20 ms** | 14.90 ms | 回路把塌陷**推迟** 8.3 ms（有帮助） |
| FlyBody（保留发布伺服） | 逃逸 **119.10 ms**（末高 1.0752 mm） | **200 ms 内从不逃逸**（保持 1.0836 mm） | 回路让这个躯干**变差**了：它本来能自己站住 |
| FlyBody（伺服→力矩电机） | 逃逸 **101.40 ms**（末高 0.5262 mm） | 27.60 ms | 回路把塌陷**推迟** 74 ms，但两者都倒 |

一句话：**换躯干不打断回路，但把「谁在支撑身体」这件事暴露了。** FlyBody 之所以能站，是因为它**发布模型自带
的逐关节位置伺服**（`force = gain*(ctrl − qpos)`，`ctrl=0` 就是中性姿态），不是因为它被神经回路驱动。
我们的回路一旦接管这些伺服（把它当作偏置指令），反而破坏了它自己的姿态保持。这是**回路的输出层与
新躯干语义不匹配**，属于翻译层问题，不是回路被破坏。

---

## 2. 新增输入与来源（未做任何修改）

- 仓库 `https://github.com/TuragaLab/flybody`，commit `d015e9bfe441bd90ae431bac24c55cb74bdbce26`，Apache-2.0。
- 复制到 `/run/media/sensen/Data2/cell_wound_prototype/upstream/flybody/`（151 MB：`fruitfly.xml`、`floor.xml`、
  85 个 `.obj`、LICENSE）。
  - `fruitfly.xml` sha256 `d14946fd0311025ecca70c8eeb5de80e1fe18700d3072be37ecbb18d33d80fd8`
  - `floor.xml` sha256 `1490751ff170d089b8882aa4972778b64bcf56f6d39b0e87029156dcf0452019`
- 出处副档：`upstream/flybody/PROVENANCE.json`（含 commit、哈希、复制范围："python 包/任务套件/训练数据未复制"）。
- 本机另有 flygym 2.1.0 自带的同源副本（`venv_body/.../flygym/assets/model/flybody/`），但**它缺少 mesh**，
  无法独立加载；本分支用的是官方仓库版本。

**单位读法（实测证据链，非假定）**：FlyBody 是 **CGS（cm, g, s）**——重力 981 cm/s²、总质量 9.8462e-4 g
= **0.985 mg**、空气密度 1.28e-3 g/cm³，由此体重 = **9.659 µN**（真实果蝇约 10 µN ✓）。
规范 FlyMimic 体是 mm 尺度且质量单位未定（本项目早已记录），所以两个躯干的力**不能直接比数值**，
本分支一律按「各自单腿承重份额」归一后再进编码器。

---

## 3. 躯干体检（无神经参与）

FlyBody 结构：`nq=109, nv=108, nu=78, njnt=103, na=0`（**没有肌肉、没有附着点、没有激活态**）。
每腿 11 铰链（coxa_abduct/twist/coxa、femur_twist/femur、tibia、tarsus、tarsus2..5），
每腿 9 执行器（8 个关节位置伺服 + 1 个爪吸附），另有 6 个爪碰撞体 + 24 个跗节碰撞体、
8 个吸附执行器（6 爪 + 2 口器）、48 个腿部关节伺服、可动爪部与跗节链、`tarsus2` 经固定腱带动 2..5。

发布模型自带的姿态与地面（`ctrl=0` = 中性姿态）：

| 臂（实验者设定，均非神经） | 逃逸 | 末高（质心，mm） | 爪支撑 / 体重 | 躯干触地 |
|---|---|---|---|---|
| `published_command`（ctrl=0） | **从不** | 1.0836 | 1.149 | 无 |
| `published_plus_adhesion`（爪吸附=1） | 从不 | 1.1166 | 7.177 | 无 |
| `springs_only`（XML 层去执行器力） | **28.0 ms** | 0.797 | 0.631 | **腹部 `abdomen_7_collision` 触地** |

读法：**发布伺服是承重的**——去掉它，FlyBody 在 28 ms 内坐到自己肚子上；保留它，200 ms 内自己站得住。
这是「FlyBody 能站」的真正来源。

图：`outputs/model_branches/flybody_body/flybody_body.png`

---

## 4. 翻译表（全部可复现，规则声明、系数实测）

1. **不存在离散铰链配准**（实测，不是猜测）：用匈牙利算法在左右腿求和意义上找最佳一一对应，
   结果解剖上荒谬——中腿的 `Tibia_pitch` 被配到 `coxa_twist`、前/后腿的 trochanter 铰链被配到 tarsomere
   （T2 平均 |dot| 0.685）。**因此本分支不使用离散配对。**规范体在对称零位姿下六条腿的铰链轴完全相同
   （世界 XYZ），轴本身携带不了铰链身份信息——这与本项目此前记录的「无解剖配准」一致。
2. **段落身份声明 + 段内轴投影**：Coxa↔coxa、Trochanter↔femur、Tibia↔tibia（只用两个模型自己命名的
   段落连接关系）；每个肌肉先由**实测有限差分力矩臂**定出目标铰链，再把该铰链的力矩方向**投影**到
   同腿同段段的 FlyBody 铰链轴上（系数 = 轴点积 × 符号）。
   - 90 个肌肉 **0 个丢失**，最大 |系数| 分布：最小 0.303 / 中位 0.772 / 最大 0.958；
   - 驱动 **36 个** FlyBody 铰链（每腿 6 个），跗节铰链**不驱动**（规范体没有对应铰链和对应感觉量）。
3. **胫节符号端到端校验**（唯一有无歧义几何判据的铰链，判据：屈曲 = 爪靠近躯干）：
   **前腿与中腿 4/6 通过；左右后腿 2/6 失败**（规范屈肌投影到了伸展方向，幅度相等 ⇒ 纯符号反号）。
   规范体自身左右后腿的已知不对称是首要嫌疑。缺陷被**报告**而非隐藏。

系数的运行时形式：`outputs/model_branches/flybody_loop/flybody_translation.json`（含每个铰链的肌肉列表、系数、归一化范数）。
图：`outputs/model_branches/flybody_loop/flybody_translation.png`

---

## 5. 闭环结果（同一回路，三种躯干，200 ms，6 个臂）

时钟：神经 0.5 ms / 命令 5 ms / 躯干 0.1 ms，每个区间断言两者都精确落在命令网格上。
实验者刺激与规范分支**完全相同**（左前腿局部触觉 5–15 ms）。

| 躯干 | normal | no_external_touch | sensory_cut | motor_cut | 脉冲静音 | replay |
|---|---|---|---|---|---|---|
| 规范体 | 23.20 ms | 23.20 | 14.90 | 19.50 | 14.90 | 23.20（逐位一致） |
| FlyBody 保留伺服 | 119.10 ms | 119.10 | 从不逃逸 | 从不逃逸 | 从不逃逸 | 119.10（逐位一致） |
| FlyBody 力矩电机 | 101.40 ms | 101.40 | 27.60 | 27.60 | 27.60 | 101.40（逐位一致） |

**规范体这一列精确复现已发布基线 23.1999 ms**——这既是回归锚点，也证明本分支的回路搭建没有走样。

因果链判据（FlyBody 保留伺服）：`sensory_cut` / `motor_cut` / `spike_output_silence` 与 normal 的最大状态差
均为 22.4；`replay` 逐位一致（构造如此）。触觉刺激臂与 normal **完全相同**：左前腿 `bristle_touch`
在本分层中是 `NO_MODELED_MOTOR_PATH`（本项目早先已记录的证据覆盖结论），所以 0.3 nA 注入不产生任何可测差异。

图：`outputs/model_branches/flybody_loop/flybody_branch_summary.png`、`flybody_closure_200ms.png`

---

## 6. 变体：用实测胫节符号修正后腿（独立标注）

把两条后腿的胫节驱动符号按几何校验翻正后重跑：

- FlyBody 保留伺服：normal 逃逸 **129.10 ms**（末高 0.8608 mm，比声明的 119.10 / 1.0752 稍差）
- FlyBody 力矩电机：normal 逃逸 **96.20 ms**（末高 0.6354 mm，稍好）
- 规范体：**完全不变**（它没有覆盖项）

结论：符号缺陷是真实缺陷，但**不是**回路在 FlyBody 上表现变差的主因；两种方向各有小幅好坏，没有救场。
产物后缀 `_tibiasign_measured`，**永不与声明版本混用**。

---

## 7. 结论与建议（按证据排序）

1. **闭环没被打断，回路没被破坏**；换躯干只换两个已声明的翻译层，且实测回路仍在因果链内。
2. **FlyBody 的站立能力属于它自己发布的伺服**，不属于我们的回路。接上回路反而从「200 ms 站得住」
   变成「119 ms 塌陷」——这是当前回路的输出语义与新躯干不匹配的直接证据。
3. 想真正利用 FlyBody，必须解决的是**输出层**：让回路产出的是「比中性姿态多/少多少」，并且必须有来源地
   决定这个偏置（本分支的 `DRIVE_FRACTION=0.5 × 半量程` 只是声明，不是标定）。
4. 换躯干的**代价**已量化：肌肉级分辨率丢失（90 个肌肉压到 36 个关节铰链）、后腿胫节符号缺陷、
   力单位不可直接比（需自有归一器）、跗节自由度**未接入回路**。
5. 换躯干的**收益**也确认了：真实跗节 + 股节链、可动爪、爪吸附执行器（我们完全没有的能力）、
   内部自洽的 CGS 单位与真实质量（0.985 mg / 9.66 µN）。
6. **建议**：把 FlyBody 保持为**独立标注的第二谱系**（本分支即是），结论永不与 FlyMimic 谱系混用；
   若目标是行走（而非站立），下一步才是把跗节/股节与爪吸附接进回路，并先解决第 3 条的输出语义。

---

## 8. 不支持的主张

- 不支持「回路让 FlyBody 站住了」：保留伺服时，静音臂反而站得最好。
- 不支持「跗节自由度已接入闭环」：本分支只用 6 个铰链/腿，跗节铰链不驱动也不参与感觉。
- 不支持「符号/连接是解剖配准」：不存在离散配准（§4 已给退化证据）；只有胫节有无歧义几何校验，且 2/6 失败。
- 不支持任何行走、步态、逃逸行为主张；本项目自定的站立判据没有被套用到 FlyBody 上做结论。
- 不支持质量/力的 SI 标定：两个躯干各用自己的承重份额归一，跨躯干的力数值不可直接比较。
- 触觉刺激臂**不是**因果臂（它对输出无可测影响，因为该通道无已建模运动通路）。

---

## 9. 复现命令（全路径）

```bash
cd /run/media/sensen/Data2/cell_wound_prototype
./venv_body/bin/python run_flybody_body.py                       # 躯干体检
./venv_body/bin/python tools_flybody_translation.py              # 翻译表 + 力矩变体 XML
./venv_body/bin/python run_flybody_loop.py --duration 0.2        # 声明翻译的闭环
./venv_body/bin/python run_flybody_loop.py --duration 0.2 --tibia-sign measured
./venv_body/bin/python tools_flybody_branch_figure.py            # 汇总图
./venv_body/bin/python run_branch_regressions.py                 # 137 项回归（含本分支）
```

---

## 10. 产物清单（全路径）

**代码**
- `/run/media/sensen/Data2/cell_wound_prototype/run_flybody_body.py`
- `/run/media/sensen/Data2/cell_wound_prototype/tools_flybody_translation.py`
- `/run/media/sensen/Data2/cell_wound_prototype/run_flybody_loop.py`
- `/run/media/sensen/Data2/cell_wound_prototype/tools_flybody_branch_figure.py`
- `/run/media/sensen/Data2/cell_wound_prototype/test_flybody_branch.py`

**输入**
- `/run/media/sensen/Data2/cell_wound_prototype/upstream/flybody/`（+ `PROVENANCE.json`、`LICENSE`）

**结果**
- `/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_body/flybody_body.json`
- `.../flybody_body/flybody_body_traces.npz`、`.../flybody_body/flybody_body.png`
- `/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_loop/flybody_translation.json`
- `.../flybody_loop/flybody_translation.png`
- `.../flybody_loop/flybody_torque_body.xml`、`.../flybody_loop/flybody_torque_variant.xml`（声明变体，不写回发布资产目录）
- `.../flybody_loop/flybody_closure_200ms.json` / `.npz` / `.png`
- `.../flybody_loop/flybody_closure_200ms_tibiasign_measured.json` / `.npz` / `.png`
- `.../flybody_loop/flybody_branch_summary.png`
- `/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/branch_regressions.json`（137 项，0 失败）

**备份**
- `/run/media/sensen/Data2/_snapshot_pre_flybody_20261012_171343/cell_wound_prototype/`
- git 标签 `pre-flybody-20261012_171343`（提交 `369a2bd`）
