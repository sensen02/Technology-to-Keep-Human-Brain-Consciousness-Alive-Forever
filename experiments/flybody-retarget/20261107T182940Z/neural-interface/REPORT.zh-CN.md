# 脑—颈—VNC接口：本地源数据审计与候选清单

这是服务S3/S4/S5完整运动、摄食、选择平台的静态解剖数据审计，不是成品或行为验证。实际神经步、物理步、训练更新、创伤实验、外部检索均为0；未修改核心、AGENTS或tracker。原完整平台和术前冻结尚未完成，不执行病损。

主要结论：真实缓存支持保留神经身份和原始边的局部稀疏模块。旧版ID字典和严格递质筛选有实质缺口，不能直接把旧12k层改名为完整脑—颈—VNC。摄食有明确标签和连接，但当前严格符号规则使所检查的口部通路失去全部有效边；标签存在不等于摄食已实现。

完整机器清单：`/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface/NEURAL_INTERFACE_MANIFEST.json`

桌面副本：`/home/sensen/Desktop/engine修复验收/flybody-retarget/20261107T182940Z/neural-interface/NEURAL_INTERFACE_MANIFEST.json`

## 缓存与原表实核

本轮任务说明中的缩写路径`/data/flywire/banc_connectome.npz`作为绝对路径不存在；实际文件为`/run/media/sensen/Data2/cell_wound_prototype/data/flywire/banc_connectome.npz`。构建器`/run/media/sensen/Data2/cell_wound_prototype/build_banc_cns.py`也指向该项目内路径。

- 153,962个严格升序且唯一的int64 root ID；3,037,361个唯一有向pre/post聚合对；23,556,214个突触计数。
- 对`/run/media/sensen/Data2/cell_wound_prototype/data/flywire/banc_connections_princeton.csv.gz`全部3,990,039条神经毡行做150,000行分块，逐对重新累加，与缓存完全相等。
- 原始神经毡行最小计数是1，聚合对最小计数是3。构建器没有施加min3阈值；源导出的过滤历史未独立确证，不能把缺失对解释成生物学零连接。
- 原始连接表NT字段全部为空；缓存3,037,361个nt_pair全部为−1。缓存ann_nt_verified另有标签：89,075个为空，纯acetylcholine43,119、glutamate8,611、gaba6,820，其他混合/调质标签保留。
- 原始神经元表158,262行，153,962全部缓存ID都有唯一行，4,300个源ID不在缓存。8个构建器复制的注释字段逐元素一致。
- 不创建153962×153962稠密矩阵。最终运行峰值RAM见机器清单runtime；初次完成审计约1.44GiB，独立验证约590MiB。

## 实际可用的身份与分歧

以下数值都是缓存中实际匹配的ID，不是论文数字或未加载的表格。

| 身份规则 | 缓存细胞数 |
|---|---:|
| super_class ascending | 1,847 |
| sensory_ascending | 507 |
| ascending_visceral_circulatory | 5 |
| 上行合计 | 2,359 |
| descending | 1,315 |
| sensory_descending | 12 |
| 下行合计 | 1,327 |
| 上下行互斥并集 | 3,686 |
| VNC intrinsic | 12,827 |
| 明确六腿nerve且sensory/sensory_ascending/sensory_descending | 5,107 |
| class=leg_motor_neuron | 391 |
| motor且六腿nerve | 329 |

class字段给ascending_neuron1,850、descending_neuron1,317，与super_class不同。5条分歧保留，未强行归一：

| Root ID（十进制字符串） | super_class | class | primary_type |
|---|---|---|---|
| 720575941458374637 | sensory | ascending_neuron | SNxxx |
| 720575941507811422 | ventral_nerve_cord_intrinsic | ascending_neuron | IN10B032 |
| 720575941594292775 | ventral_nerve_cord_intrinsic | ascending_neuron | INXXX101 |
| 720575941566565174 | central_brain_intrinsic | descending_neuron | CB0191 |
| 720575941612401510 | central_brain_intrinsic | descending_neuron | CB0191 |

这只是源注释方向，不是独立形态学极性验证。绝不把3,686条全称为下行。

每组索引在`/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface/GROUP_CACHE_INDICES.npz`，组名、规则和来源在manifest。索引映射为原cache.root_ids[index]；完整153,962行身份和两套注释在`/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface/NODE_IDENTITIES.csv.gz`。所有文本ID均保留十进制字符串，不经过float。

## 版本和替代表的确切覆盖

核心新表`/run/media/sensen/Data2/cell_wound_prototype/data/banc888/banc_888_meta.feather`有188,508行。root_888唯一且覆盖全部缓存，但版本列不能混用：

| 来源ID列 | 唯一源ID | 缓存匹配 | 缓存缺失 | 重复ID行 |
|---|---:|---:|---:|---:|
| feather root_888 | 188,508 | 153,962 | 0 | 0 |
| feather banc_888_id | 188,508 | 153,962 | 0 | 0 |
| feather root_id | 188,340 | 139,342 | 14,620 | 168 |
| feather root_626 | 185,165 | 92,216 | 61,746 | 3,343 |
| feather root_850 | 188,196 | 142,215 | 11,747 | 312 |
| feather root_890 | 188,196 | 146,735 | 7,227 | 30 |
| supplemental_data_2.txt root_id | 155,927 | 152,806 | 1,156 | 0 |
| cell_info.parquet pt_root_id | 173,448 | 141,053 | 12,909 | 202,773 |
| codex_annotations.parquet pt_root_id | 158,227 | 153,928 | 34 | 1,682,851 |
| somas_v1.parquet pt_root_id | 153,633 | 122,369 | 31,593 | 132 |

cell_info/codex重复通常是同细胞多注释记录，不等于同数量的歧义细胞；soma多行需分别检验。每个mapping_*.csv.gz保留所有有效源ID、cache_index（未匹配−1）和source_row_count，不能把多行任取首条当唯一来源。

root_888与banc_888_id只有2条同列行分歧：720575941229247158→147423687484661800、720575940434812394→150730400321262552；这4个值都不在当前cache，且两行没有细胞注释。最初断言两列完全相等因此失败，随后改为显式报告这两条，未修改源表。精确覆盖支持本快照兼容性，不是对拆分/合并历史的独立生物身份认证。

即使按root_888正确连接，缓存与feather也有super_class109条、class113条、nerve24条、primary_type4,305条、verified NT1,654条差异。manifest分别给差异组；NODE_IDENTITIES同时保留双方字段，不用较新表悄悄覆盖缓存注释。

## 腿感觉与运动输出

旧感觉字典`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_rebuild/sensory_channel_lookup.json`的3,903项由原feather规则逐条重建完全一致。保留同一subclass+side选择后，旧root_id匹配3,847，新root_888匹配3,873，仍有30个源细胞不在缓存。每腿每channel（claw/hook/hairplate/campaniform/club/bristle/multidendritic）的完整源行、ID和−1缺失记录在`/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface/SENSORY_CHANNEL_CANDIDATES.csv`。这不是每细胞调谐或真实受体动力学标定。进一步实际检查2,945项bristle_touch规则：2,924项cache class为bristle_neuron，3项taste_bristle_neuron，1项cache为hair_plate_neuron但新subclass为bristle，17项不在cache。4项模态歧义的完整ID和索引在`/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface/SENSORY_CHANNEL_REVIEW.json`；后续未独立资格审查前不向这4项注入普通触毛驱动，也不把taste标签强行解释为纯味觉。

MN—肌肉目标并非完全没有来源。feather具有peripheral_target_type、body_part_effector、side等字段；旧90执行器表74个已解析pool、16个未解析。原208个唯一ID中206个在cache，按同源行root_888投影后208个全部在cache。候选逐行在`/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface/MOTOR_TARGET_CANDIDATES.csv`；16项缺口原样保存在同目录`MOTOR_UNRESOLVED_SOURCE_RECORDS.json`，6项整个表无该目标、10项目标不在指定腿/侧。

这些是旧90肌肉执行器池的标签，不是当前42增量关节的已验证解码，也没有提供力学/募集标定。当前391个leg_motor中，按cache nerve可唯一归到一条腿的有323个（LF43、LM60、LH59、RF42、RM60、RH59）；其余68个不能根据这个规则分腿。全部391个及明确eligible标记保存在`/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface/LOCAL_MOTOR_READOUT_PORTS.csv`。保留更丰富feather字段供后续单独资格审查，不擅自补腿或造肌肉边。

## 摄食与内部状态标签

| 明确源标签/规则 | 缓存数 |
|---|---:|
| proboscis_motor_neuron | 35 |
| pharynx_motor_neuron | 22 |
| crop_motor_neuron | 7 |
| taste_bristle_gustatory_neuron | 1,283 |
| 明确口器/咽部body_part且sensory类 | 606 |
| 消化道/crop body_part | 84 |
| hemolymph/hemolymph_sensory body_part | 8 |
| function精确token feeding / feeding_suppression | 4 / 3 |
| interoceptive / hemolymph_sugar | 4 / 1 |
| sugar / water | 534 / 25 |
| pharyngeal_pumping | 13 |
| pars_intercerebralis / pars_lateralis neurosecretory | 37 / 21 |
| SEZ / VNC neurosecretory | 11 / 146 |

这些组在GROUP_CACHE_INDICES有完整ID索引；NODE_IDENTITIES还保存原Verified Neuropeptide及feather neuropeptide_verified和功能字段。广泛的sugar标签不等于饥饿通路，神经分泌标签不等于已知能量状态动力学。没有创建虚构的饥饿/饱腹/腹部传入神经。摄食机械、摄入量、消化/能量平衡、反馈与行为竞争仍需要独立实施和验证。

## 原始有向边及边界

| 源组 → 目标组 | 聚合对 | 突触数 | 纯verified标签源边 |
|---|---:|---:|---:|
| brain intrinsic/visual → descending侧 | 69,628 | 793,633 | 4,276 |
| descending侧 → VNC intrinsic | 57,077 | 517,979 | 7,261 |
| VNC intrinsic → ascending侧 | 55,557 | 505,559 | 45,537 |
| ascending侧 → brain intrinsic/visual | 48,735 | 472,534 | 25,961 |
| 腿感觉 → VNC intrinsic | 55,171 | 328,078 | 13,467 |
| VNC intrinsic → leg_motor class | 43,897 | 777,882 | 36,087 |
| brain intrinsic/visual → VNC intrinsic | 0 | 0 | 0 |
| VNC intrinsic → brain intrinsic/visual | 9 | 52 | 0 |
| 口部感觉 → brain intrinsic/visual | 6,798 | 57,151 | 0 |
| brain intrinsic/visual → proboscis motor | 1,182 | 18,552 | 0 |
| brain intrinsic/visual → pharynx motor | 902 | 17,405 | 0 |

全部super_class×super_class互斥方向计数在`/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface/DIRECTED_SUPERCLASS_EDGE_COUNTS.csv`，和原总边/突触数完全相等。上表每条实际cache edge row索引另在同目录`DIRECTED_ROUTE_EDGE_ROWS.npz`。那9条VNC→brain直边保留而不解释成新解剖事实；可能涉及源标签问题，不能为满足架构悄悄删掉。

方向是真实pre→post，端点标签不是突触所在脑区或颈部轴突位置。缓存已经聚合神经毡，未含逐突触空间坐标。access_map的somas_v1是核/胞体点，只覆盖122,369个cache ID，不能替代颈纤维；其源注明各轴尺度为0.004/0.004/0.045µm，不能与其他脚本的0.016µm各向同性假设混用。本审计未运行电极或切断模型。soma来源署名：BANC (2026), Distributed control circuits across a brain-and-cord connectome, Harvard Dataverse doi:10.7910/DVN/7WTH1N, somas_v1.parquet, CC BY4.0；未导出坐标。

## 下一局部神经模块候选

`/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface/LOCAL_MODULE_CANDIDATE.npz`是静态身份/原边候选，不是训练模型。选择规则先保留全部颈跨越、腿感觉、leg motor（并集9,072）；再按与这些保留端点的入/出突触总量排序VNC intrinsic，同分按全局index，填到20,000上限。最终10,928个VNC intrinsic，705,068条原cache诱导边、6,703,494突触计数。每条边同时保存原edge row、全局节点、局部pre/post和syn，全部无伪造。

已有严格纯NT规则只允许435,126条边，269,942条不能默认为兴奋或抑制。独立检查发现：

| 端点组 | 入选细胞 | 有严格入边 | 有严格出边 |
|---|---:|---:|---:|
| ascending侧 | 2,359 | 2,231 | 1,316 |
| descending侧 | 1,327 | 1,224 | 133 |
| 腿感觉 | 5,107 | 4,419 | 842 |
| leg motor | 391 | 391 | 0 |
| VNC intrinsic | 10,928 | 10,855 | 8,796 |

MN无严格出边不阻止读取其状态；感觉/DN缺乏出边却会限制输入传播。严禁把全部1,327下行或5,107感觉端点称为已功能接通。口部相关表列路径的纯标签边为0，严格模式无法据此声称完成摄食。可先实现少量已支持端点的局部模块工程验证，再单列未知/混合递质的模型假设或预测资格审查；不从表现好坏倒推真实递质。

建议下一最小实施是：保持NPZ神经元身份和源稀疏拓扑，用有界点神经元状态和明确的感觉注入，先检查索引、稀疏散加、数值稳定和端点覆盖。可训练至多8组连接增益[0.5,2]、8组时间常数[5,100]ms、8组归一化阈值[−1,1]；这些是工程范围，不是生物测量。任何纯NT→兴奋/抑制规则仍是受体模型假设，不能宣称glutamate在全部位置都必定抑制。未知边保留于解剖产物但在严格动力学中关闭；若采用预测或学习符号，须另版本且标明假设，不能悄悄补齐。

读出先限定323个可明确分腿MN→每腿7个当前增量关节；最多2,261个权重+42偏置，权重/偏置[−1,1]、tanh输出[−1,1]，然后经过父任务已有角度、速度、力矩界。这是工程神经读出，不是MN—肌肉解剖映射。实际42关节坐标、次序和速率沿父任务重定向契约，不从此表猜测。不给VNC放普通MLP隐藏层再冒称神经身份；工程读出与真实稀疏局部网络分开命名。

脑模块未来只能经声明的descending/ascending身份接口收发；脑不得直接读全部身体状态或直接发42关节命令。此候选没有完整脑内部、视觉输入和摄食/内部状态动力学。当前源`/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/adapters.py`默认12k、腿播种、head/visual关闭，只是历史局部选择；本候选单独导出，未改旧运行层。

## 验证与重现

生成器：`/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface/audit_neural_interface.py`。

独立验证器：`/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface/verify_neural_interface.py`。它不导入生成器或engine，重新检查全部源/产物SHA256、ID唯一性/边界、组规则、上下行互斥并集、每条方向边、候选全部源边和局部索引、独立选择排序、153,962行身份及全部感觉/MN候选ID。结果写`/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface/VERIFICATION.json`；当前PASS仅指这些静态不变量，不是动力学、闭环或行为通过。

命令（没有模型加载/forward/step）：

```bash
/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python3 /run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface/audit_neural_interface.py
/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python3 /run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface/verify_neural_interface.py
```

正式导出另有镜像SHA256核对。仅使用授权的工作区和桌面neural-interface目录；无额外后台仿真或训练。后续平台未建成和冻结前仍不做创伤。
