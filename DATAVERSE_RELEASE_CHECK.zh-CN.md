# 官方release核查：不需要token，缺口在源头就存在

**手工研究原型，不是成品。** 只读核查，未使用任何token。

## 一、结论先行

**不需要HuggingFace token，也不需要CAVE token。** 我直接从公开的Harvard Dataverse记录（`doi:10.7910/DVN/7WTH1N`，release 3）下载了与CAVE等价的三张表，并逐项核验：

| 文件 | 大小 | 内容 |
|---|---:|---|
| `supplemental_data_2.txt` | 31.7 MB | 逐神经元官方表（含peripheral_target_type、body_part_effector、nerve） |
| `codex_annotations.parquet` | 74.4 MB | **完整注释转储，184万行**（含classification_system与CAVE用户标注） |
| `cell_info.parquet` | 12.3 MB | CAVE校对/用户标注表 |

**核验结果：官方release与我们的本地副本在关键计数上完全一致**——efferent 1033、腿部efferent 396、四个缺口目标的逐腿逐侧覆盖**逐格相同**（`pleural_promotor` 0条；`tergopleural_promotor` 仅前腿8条；中腿remotor仅左侧1条；中腿adductor仅右侧1条）。

**因此缺口不是"我们本地数据旧了"，而是发布的注释里本来就没有。给token也换不来。**

## 二、CAVE用户标注不能用来填池

`cell_info.parquet`里确实有肌肉名标签，例如"Sternal adductor MN"(10个细胞)、"Pleural remotor/abductor MN"(20个)。但它们**混着多种肌肉**：`Pleural remotor/abductor MN`标签下的细胞，元数据里大多是`femur_reductor_muscle`；`Sternal adductor MN`标签下混着`trochanter_flexor`与`accessory_trochanter_flexor`。属近似人工标注，**不能据此分配池**。

完整注释转储里，所有promotor相关标注（33行、6种标签写法）对应的细胞**全部在前腿**。

## 三、真正的好消息：扩展在结构上可行，而且神经侧已就绪

1. **模型已含跗节与股节几何体**：编译后LF腿的body为Coxa/Trochanter/Femur/Tibia/Tarsus1–5，其中**只有Coxa、Trochanter、Tibia有joint**；Femur与Tarsus1–5**无joint**（被焊死）。
2. **这些肌肉的运动神经元已在当前tier里**：

| 注释肌肉 | tier内细胞数 | 覆盖 |
|---|---:|---|
| tarsus_depressor_muscle | 14 | 六腿双侧 |
| tarsus_levator_muscle | 7 | 六腿双侧 |
| femur_reductor_muscle | 26 | 六腿双侧 |
| long_tendon_muscle | 45 | 六腿双侧 |
| accessory_tibia_flexor_muscle | 57 | 六腿双侧 |

即**神经侧已就绪，缺的只是身体侧的关节与执行器**——这正是可以合理扩展的地方。

## 四、已启动的扩展

按你的授权，我已派出实现任务：**跗节自由度 + depressor/levator执行器**分支，要求：

- 只新增、不改动既有joint/actuator/几何/质量；
- 关节轴由既有腿的pitch轴约定推导，**附着点与力参数全部数值化声明**（力参数直接复制既有胫节肌肉，不引入新调参）；
- 运动池用`tarsus_depressor/levator`的真实注释root，逐个记录，不借不猜；
- 200ms、25μs、三条臂（正常/输出静默/精确回放），严格时钟不变；
- 编译数组对比：除新增DOF必然增长的数组外，其余必须逐位不变；
- **如结果更差（例如新增自由DOF带来新的塌陷模式），必须照实报告**。

实现里有一个已知约束：核心`prepare_body()`要求每条腿恰好7个关节，分支必须自带映射而不能改核心文件——我已明确交代。

## 五、产物

- 核查脚本：`/run/media/sensen/Data2/cell_wound_prototype/run_banc_release_check.py`
- 测试：`/run/media/sensen/Data2/cell_wound_prototype/test_banc_release_check.py`（6项通过）
- 结果：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/banc_release_check/banc_release_check.json`
- 下载的官方文件：`/run/media/sensen/Data2/cell_wound_prototype/data/banc_dataverse/`（含SHA-256）

边界：单一release；CAVE用户标签只用于说明"不能填池"；未做任何在线CAVE查询，未使用token。
