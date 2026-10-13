# 结构溯源：模型是**一条腿肌肉组的六重复制**

**手工研究原型，不是成品。** 只读比对上游OpenSim源模型、规范MJCF与BANC注释。**未做任何建模建议或改动。**

## 一、决定性发现

上游源文件 `/run/media/sensen/Data2/cell_wound_prototype/outputs/upstream_muscle_round11/best_combined.osim` 实测：

| 项目 | 实测值 |
|---|---|
| 身体段数 | 71，**六条腿的几何都在**（Coxa/Femur/Tarsus1–5/Tibia/Trochanter） |
| 定义的关节 | **仅6个：LF与RF各3个**（Coxa/Trochanter/Tibia） |
| 定义的肌肉 | **15块，全部属于左前腿** |

而规范MJCF：**六条腿全部有7个关节**（coxa yaw/pitch/roll + trochanter yaw/pitch/roll + tibia pitch），**每条腿都带同一套15块肌肉**（共90个actuator）。

**结论：整个肌肉系统是把"单条腿（左前腿）的15块肌肉"复制到六条腿**。这与之前发现的"每条腿7自由度、无股节/跗节关节"来自同一来源——**股节和跗节的几何体存在，但既无关节也无肌肉**。

## 二、于是"16个不可驱动actuator"得到正确分解

| 来源 | 数量 |
|---|---:|
| 模型复制到中/后腿、但BANC在该腿没有该肌肉（`tergopleural_promotor`，`pleural_promotor`） | **14** |
| 真正的单细胞缺失（中腿右pleural remotor、中腿左sternal adductor） | **2** |
| 合计 | **16** |

逐腿"模型有、该腿注释无"：LF 1项、RF 1项、LM/LH/RM/RH各2项（其中`tergopleural_promotor`每腿2束）。

**这意味着：向BANC索取这14个池，等于要求注释去匹配模型的复制假设，而不是补数据缺口。** 正确的处理是模型层面：要么声明这些actuator为"解剖未归属的模型构件"，要么按解剖来源逐腿修肌组。

## 三、独立复核的证据

**VFB查询（我亲自复现，无token）**：
- `FBbt_00053566`（中腿sternal adductor MN）→ 唯一BANC细胞 `720575941644474312`，正是池中已有的中腿右细胞 ✓
- `FBbt_00053150`（中腿pleural remotor MN）→ **BANC只有1个细胞** `720575941553503460`（中腿左，已在池中）；**右侧对应细胞只出现在雄性MANC**（902492/800776/…共14个条目）。**不同个体不能互相搬运**，所以中腿右仍无法补。
- `FBbt_00058860`（前腿pleural promotor muscle）`NeuronsPartHere` = **0**。

**解剖存在性**：BANC的`pleural_promotor_muscle`根本不在受控词表里；FBbt中pleural promotor只有T1与T3子类（**无T2**），tergopleural promotor**仅T1**；Cheong et al. 2026逐字：**"We could not identify serially repeating homologs (in T2 and T3) for … the Tergopleural/Pleural promotor MNs"**，并注明"only the T1 leg neuropils have muscle target assignments"。

**反向缺口依然存在**：BANC在六条腿上注释了模型完全没有的5类肌肉——`tarsus_depressor`(23)、`tarsus_levator`(7)、`femur_reductor`(34)、`long_tendon`(48)、`accessory_tibia_flexor`(71)。上游模型里**股节与跗节的几何体是存在的**，但既无关节也无肌肉。

## 四、对"距离行走"的直接含义

**脚部摆放的执行机构在几何上存在，但没有关节也没有肌肉**——这不是调参问题，也不是注释缺口，而是**身体模型本身缺少行走所需的执行自由度**。同时中/后腿的肌组是前腿的复制，与注释的实际肌组不一致。

## 五、产物与边界

- 溯源审计：`/run/media/sensen/Data2/cell_wound_prototype/run_model_provenance_structure_audit.py`、`test_model_provenance_structure_audit.py`（4项通过）、`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/model_provenance_structure/`
- 缺口判定：`/run/media/sensen/Data2/cell_wound_prototype/LEG_MOTOR_COVERAGE.zh-CN.md`
- 子代理检索：`/run/media/sensen/Data2/cell_wound_prototype/RESEARCH_LEG_MOTOR_POOLS.zh-CN.md`、`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/research_leg_motor_pool_sources.json`

**计数核对**：子代理按逐类相加减得17个，与我的16不一致。我直接按manifest逐条数：`tergopleural` 8 + `pleural_promotor` 6 + `sternal_adductor` 1 + `remotor` 1 = **16**（且与按原因分解的6+8+2一致）。差异来自逐类相加时的重复计数，以manifest逐条计数为准。

边界：单一上游版本与单一注释release；本地osim是项目副本，未取更新上游版本；"未注释"不等于"动物没有"。
