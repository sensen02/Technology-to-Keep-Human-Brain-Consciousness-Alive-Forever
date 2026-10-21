# 运行时感觉根身份与结构证据审计

**手工研究原型，不是成品。只读数据审计，不是站立成功或生物学验证。**

## 精确连接及测试

从 current_run.json 验证 manifest 哈希，再验证 neural_identity.json 哈希。用运行时 sensory_groups 的 tier 索引取根，与 lookup 的 leg_channel_by_root 逐一核对；只用 v888 的 banc_888_id 唯一连接。拒绝重复根、缺失根、索引重叠、lookup 不一致及空审计，并记录输入 SHA256。

12,000 神经元 tier 中 **3,788/3,788 感觉根唯一匹配**；7 通道 × 6 腿均非空。未按数据集 subtype 精确等号重建运行时群，因此也覆盖 bilateral campaniform。

| 运行时通道 | 细胞数 |
|---|---:|
| bristle_touch | 2879 |
| campaniform_strain | 60 |
| claw_position | 149 |
| club_vibration | 323 |
| hairplate_angle | 207 |
| hook_direction | 136 |
| multidendritic_noxious | 34 |

命令：`venv/bin/python -m unittest test_banc_channel_attribute_audit test_sensory_tuning_annotation_audit`，本轮 **9 tests / OK**。这些测试验证身份与导出约束，不验证感觉增益、动力学或站立。

## 结构证据与界限

- hairplate 207 根中，body_part_sensory 有 **11 根 coxa、25 根 trochanter** 字面标注。peripheral_target_type 有 CoHP3 3 根、CoHP8 7 根、CoHP4 1 根；cell_type 的 CoHP8 是 9 根——字段计数不可混用。不能把全部 hairplate 视为同一 coxa DOF；也不能直接把片名等同于某个模型铰链。
- claw 149 根功能字段为 position，hook 136 根为 direction，club 323 根为 vibro_tactile。文献层面 FeCO 对应 femur–tibia，但这些字段没有给出每根的屈伸极性。club 字段名称也不能否认文献报告的双向运动响应。
- campaniform 60 根中，前/中/后腿为 52/4/4。9 根 subtype 含 bilateral；不能将其遗漏，也不能由计数推断哪条腿感觉更强。未找到可据此分配场方向和正负力响应的已验证根交叉表。
- 3,788 根在本次检查字段中均没有显式 flexion/extension 字面标签。这只是词项审计，不是断言所有潜在来源均不存在极性证据。
- cell_type、seed、nerve 等有组内差异；**存在类别分裂 ≠ 调谐分裂**。SNpp 编号不得转为屈伸符号。
- 所有 confirmed_root_specific_joint 和 confirmed_tuning_polarity 保留 null。没有改通道、符号、感觉电流、连接或身体动力学。

## 需要人工复核的注释

独立输出 30 根 review queue，不自动重分类；以下 flag 合计也是30，本次无重叠：

- 1 根 campaniform 运行时通道的 peripheral_target_type 为 hair_plate、功能为 joint_angle。
- 2 根 hairplate 通道的 cell_type 为 CS（只是标签歧义，不能仅据此判为 campaniform）。
- 15 根外周目标缺失或 orphan。
- 12 根 bristle 的功能字段非空：6 根 contact_pheromone、4 根 sugar/low_salt、2 根 joint_angle。把所有 bristle 简化为同一机械触觉量存在模态混合风险；这里先列证据，不擅自替换模型。

## 对先前解释的纠正

1. drive 与关节偏差正相关不能独立证明正反馈、缺少姿态参考或不稳定；drive 自身也会造成运动，需受控因果扰动。
2. 没有外周坐标来源证明时，不把 metadata position 当成外周胞体，不换算解剖单位，不用 PCA 猜关节附着。
3. 单个 CxHP8 研究不能支撑所有 hairplate→coxa；特定 femoral CS 场的受力结果不能推广到所有 CS。
4. 刺激范围的一半不是测得的感觉增益；6–30 ms 不是已建立的文献滤波时间常数范围。旧分支结果不得据此称为已校准生理参数。

## 交付与验证状态

主输出目录：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/channel_attribute_audit/`

- JSON：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/channel_attribute_audit/channel_attribute_audit.json`
- 逐根CSV：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/channel_attribute_audit/runtime_sensory_evidence.csv`
- 复核队列：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/channel_attribute_audit/annotation_review_queue.json`
- 覆盖图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/channel_attribute_audit/runtime_sensory_coverage.png`

用户修复读图接口后，已用 read_image 实际查看桌面副本 `/home/sensen/Desktop/cell_wound_prototype/channel_attribute_audit/runtime_sensory_coverage.png`：42格计数与审计表一致，标题、轴标签、色标和原型/局限说明可读，无明显裁切。**已完成该图的视觉核验**；这不验证生物学调谐或站立。

下一步应优先核查这30根的原始注释与外部类型交叉表，并验证身体分支实际读取了预期感觉角色；在此之前不依据编号或相关性分配极性，也不为站住而调符号。
