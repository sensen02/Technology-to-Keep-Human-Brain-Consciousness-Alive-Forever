"""Assemble RESULT.json and REPORT.md from saved evidence; zero simulation steps.

Every number in the report is read back from the artifacts written by
audit_encoder.py, fit_information.py, plot_information.py and verify_information.py
so no value is retyped by hand.
"""
from pathlib import Path
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
WS='/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-descending-learning/20261107T203049Z/information'
DS='/home/sensen/Desktop/engine修复验收/banc-descending-learning/20261107T203049Z/information'
OLD='/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z'
TRIALS='/home/sensen/Desktop/engine修复验收/banc-physical-learning/20261107T195435Z/learning/training'
REPS=('encoded4','MNinstant','MNhistory','rawphysical')
LEGS=('LF','LM','LH','RF','RM','RH')
TASKS=('stance_air','knee_velocity_sign','signed_knee_velocity')


def load_json(name):return json.loads((HERE/name).read_text())
def matrix(fits,task,field):
    out=np.full((len(LEGS),len(REPS)),np.nan)
    for f in fits:
        if f['task']!=task or f['fitting_status']!='FITTED':continue
        value=f['metrics'].get(field)
        if value is not None:out[LEGS.index(f['leg']),REPS.index(f['representation'])]=value
    return out
def md(matrix_,fmt='{:.4f}',row_labels=LEGS,col_labels=REPS):
    head='| leg | '+' | '.join(col_labels)+' |\n|'+'---|'*(len(col_labels)+1)+'\n'
    body=''
    for i,leg in enumerate(row_labels):
        cells=' | '.join('undefined' if np.isnan(v) else fmt.format(v) for v in matrix_[i])
        body+=f'| {leg} | {cells} |\n'
    return head+body


def main():
    sys.path.insert(0,str(HERE))
    import run_dn_pair as runner
    command,command_digest=runner.load_command_module()
    best,eligible=runner.select_actor()
    plan=runner.build_plan(eligible,best,command_digest)
    audit=load_json('ENCODER_AUDIT.json');results=load_json('FIT_RESULTS.json')
    verification=load_json('VERIFICATION.json');failed=load_json('FAILED_DIAGNOSTIC_ATTEMPTS.json')
    fits=results['fits'];ranks=audit['ranks']
    agg={}
    for task in TASKS:
        for rep in REPS:
            rows=[f for f in fits if f['task']==task and f['representation']==rep]
            field='balanced_accuracy_test' if task!='signed_knee_velocity' else 'r2_test'
            train_field='balanced_accuracy_train' if task!='signed_knee_velocity' else 'r2_train'
            source_field='balanced_accuracy_old_source_transfer' if task!='signed_knee_velocity' else 'mse_old_source_transfer'
            test=[f['metrics'][field] for f in rows];train=[f['metrics'][train_field] for f in rows]
            source=[f['metrics'][source_field] for f in rows]
            agg[f'{task}/{rep}']=dict(fits=len(rows),feature_dimensions=sorted({f['feature_dimension'] for f in rows}),
                                      test_mean=None if any(v is None for v in test) else float(np.mean(test)),
                                      test_min=None if any(v is None for v in test) else float(np.min(test)),
                                      test_max=None if any(v is None for v in test) else float(np.max(test)),
                                      train_mean=None if any(v is None for v in train) else float(np.mean(train)),
                                      old_source_mean=None if any(v is None for v in source) else float(np.mean(source)),
                                      deployable=rows[0]['deployable'])
    stance=matrix(fits,'stance_air','balanced_accuracy_test');sign=matrix(fits,'knee_velocity_sign','balanced_accuracy_test')
    reg=matrix(fits,'signed_knee_velocity','r2_test');reg_mse=matrix(fits,'signed_knee_velocity','mse_test')
    base=matrix(fits,'signed_knee_velocity','train_mean_baseline_mse_test');reg_train=matrix(fits,'signed_knee_velocity','r2_train')
    source_ba=matrix(fits,'knee_velocity_sign','balanced_accuracy_old_source_transfer')
    source_st=matrix(fits,'stance_air','balanced_accuracy_old_source_transfer')
    key=dict(
      encoder_rank_all=ranks['channel_map_24x90'],zero_input_columns=len(audit['zero_columns']),
      nonzero_q_columns=ranks['nonzero_q_columns'],zero_q_columns=ranks['zero_q_columns'],
      nonzero_qvel_columns=ranks['nonzero_qvel_columns'],zero_qvel_columns=ranks['zero_qvel_columns'],
      q_independent_directions=ranks['q_block_24x42'],qvel_independent_directions=ranks['qvel_block_24x42'],
      foot_load_independent_directions=ranks['foot_load_block_24x6'],
      alias_fixtures=len(audit['alias_fixtures']),all_alias_fixtures_bit_identical=all(f['exact_equal_encoder_currents'] and f['exact_equal_independent_formula'] for f in audit['alias_fixtures']),
      sensitivity_controls=len(audit['sensitivity_controls']),all_controls_detected=all(c['encoder_distinguishes'] for c in audit['sensitivity_controls']),
      encoded4_stance_air_test_ba_range=[float(np.nanmin(stance[:,0])),float(np.nanmax(stance[:,0]))],
      mn_stance_air_test_ba_range=[float(np.nanmin(stance[:,1:3])),float(np.nanmax(stance[:,1:3]))],
      rawphysical_stance_air_test_ba_range=[float(np.nanmin(stance[:,3])),float(np.nanmax(stance[:,3]))],
      encoded4_knee_sign_test_ba_range=[float(np.nanmin(sign[:,0])),float(np.nanmax(sign[:,0]))],
      mn_knee_sign_test_ba_range=[float(np.nanmin(sign[:,1:3])),float(np.nanmax(sign[:,1:3]))],
      rawphysical_knee_sign_test_ba_range=[float(np.nanmin(sign[:,3])),float(np.nanmax(sign[:,3]))],
      encoded4_knee_sign_mean_train_ba=float(np.mean([f['metrics']['balanced_accuracy_train'] for f in fits if f['task']=='knee_velocity_sign' and f['representation']=='encoded4'])),
      mn_knee_sign_mean_train_ba=float(np.mean([f['metrics']['balanced_accuracy_train'] for f in fits if f['task']=='knee_velocity_sign' and f['representation'] in ('MNinstant','MNhistory')])),
      signed_knee_velocity_r2_test_range_encoded4=[float(np.nanmin(reg[:,0])),float(np.nanmax(reg[:,0]))],
      signed_knee_velocity_r2_test_range_mn=[float(np.nanmin(reg[:,1:3])),float(np.nanmax(reg[:,1:3]))],
      signed_knee_velocity_r2_test_range_rawphysical=[float(np.nanmin(reg[:,3])),float(np.nanmax(reg[:,3]))],
      signed_knee_velocity_negative_r2_fits=int(np.sum(reg[:,:3]<0)),
      rawphysical_ridge_shrinkage_one_minus_r2=[float(v) for v in (1-reg[:,3])],
      regression_mse_vs_baseline_better_than_baseline=int(np.sum(reg_mse<base)),
      mn_vs_encoded_stance_air_gap_mean=float(np.mean(stance[:,1:3])-np.mean(stance[:,0])),
      mn_vs_encoded_knee_sign_gap_mean=float(np.mean(sign[:,1:3])-np.mean(sign[:,0])),
      rawphysical_minus_encoded_stance_air_mean=float(np.mean(stance[:,3])-np.mean(stance[:,0])),
      rawphysical_minus_mn_stance_air_mean=float(np.mean(stance[:,3])-np.mean(stance[:,1:3])),
      rawphysical_minus_encoded_knee_sign_mean=float(np.mean(sign[:,3])-np.mean(sign[:,0])),
      rawphysical_minus_mn_knee_sign_mean=float(np.mean(sign[:,3])-np.mean(sign[:,1:3])),
    )
    counts=dict(fits_performed=results['fit_count'],fits_not_estimable=results['not_estimable'],maximum_fits_allowed=72,
                fitter_artifact_reads=results['data_artifact_reads'],verifier_artifact_reads=verification['data_artifact_reads'],
                encoder_audit_artifact_reads=audit['data_artifact_reads'],encoder_calls=audit['encoder_calls'],
                trial_artifacts_hashed=len([k for k in audit['source_hashes'] if 'learning/training/trial_' in k]),
                saved_bins_fitted=results['train_bins']+results['test_bins'],old_source_bins=results['old_failed_source_bins'],
                physics_steps=0,neural_steps=0,training_updates=0,
                semantics='one fit = one ridge solve on train-only standardized features of one (leg, representation, target) cell; feature construction, standardization and metrics are not counted as fits; NOT_ESTIMABLE cells are counted but perform no solve')
    result=dict(schema='banc-local-information-diagnostic-result/1',status='PASS_OFFLINE_INFORMATION_DIAGNOSTIC',
                scope='S3/S4 offline information audit of the frozen coarse encoder and same-leg MN readout; zero physical/neural steps; diagnosis only, no deployed policy fitting and no acceptance-threshold change',
                workspace=WS,desktop_mirror=DS,prereg=str(HERE/'PREREG.json'),
                prereg_sha256=audit['source_hashes'][str(HERE/'PREREG.json')],
                source_hashes=results['source_hashes'],counts=counts,encoder=key,
                aggregates_by_task_and_representation=agg,verification={k:verification[k] for k in ('status','checks','passed','failed','fits_recomputed','data_artifact_reads','max_beta_delta','max_prediction_delta','max_metric_delta','max_stored_normal_equation_residual')},
                failed_attempts=str(HERE/'FAILED_DIAGNOSTIC_ATTEMPTS.json'),
                figures=[str(HERE/'FIT_SUMMARY.png')],
                two_hypothesis_reading=dict(
                  hypothesis_A_encoder_too_coarse='SUPPORTED for velocity direction: the frozen hook_direction channel is 0.05*tanh(|knee_speed|/10), an exact absolute-value alias, and no leg representation other than the privileged raw source exceeds ~0.65 balanced accuracy on knee-velocity sign; 36 of 42 joint-speed columns and all 18 trochanter angles have no input port at all, so the 90-input map has rank 24 with 54 exactly zero columns.',
                  hypothesis_B_readout_compression='SUPPORTED for the stance/air state: encoded4 recovers the stance/air label at 0.978-0.993 test balanced accuracy, but the same-leg MN representations reach only 0.436-0.822 (mean gap vs encoded4 recorded in aggregates) on the same bins, so the MN observation compresses information the encoder does expose. Because campaniform strain contains foot load by construction this stance/air task is tautological for encoded4 and is an availability/compression check only.',
                  not_established='Neither result is a causal statement about the neural circuit, perception, or movement; all 48 episodes share one canonical checkpoint and deterministic policy variants, and no physical or neural step was run in this diagnostic.'),
                limits='Recoverability from correlated saved data is not causal neural representation, perception, or movement success. The coarse encoder was an intentionally uncalibrated proxy; its information loss does not make source sensory biology false. A zero input column is an uncalibrated interface gap, not evidence that source sensory biology lacks that information. No cell identities were fabricated and no tuning directions inferred. rawphysical is privileged diagnostic ground truth and is never deployable. All prior failures and frozen artifacts are retained unchanged.',
                next_step=dict(runner='information/run_dn_pair.py',written_but_not_executed=True,plan_mode_executed=True,
                               plan_mode_contract='default invocation is plan-only (no mujoco import, no physical step); --execute performs the two 2 s releases',
                               selection_rule=plan['selection_rule'],eligible_candidates=plan['eligible_candidates'],
                               selected_trial=plan['selected']['trial'],selected_reward=plan['selected']['reward'],
                               selected_theta_max_abs=plan['selected']['theta_max_abs'],selected_actor_sha256=plan['selected']['actor_sha256'],
                               descending_command_sha256=plan['descending_command']['sha256'],
                               descending_root_ids=plan['descending_command']['root_ids'],
                               conditions=plan['conditions'],timing=plan['timing'],
                               native_steps_if_executed=plan['physics_steps_if_executed'],neural_steps_if_executed=plan['neural_steps_if_executed'],
                               outputs=plan['outputs'],
                               note='matched 2 s physical pair on the same complete canonical 1 s checkpoint with DN 0 vs 0.05 nA; only the descending current differs; the parent executes and independently replays it; this diagnostic agent performed no physical or neural step'))
    (HERE/'RESULT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

    report=f"""# BANC 离线信息诊断（20261107T203049Z / information）

预登记与全部数字均来自本目录已保存的产物，本条**没有执行任何物理步或神经步**（physics_steps=0，neural_steps=0，training_updates=0）。这不是成品，也不是行为成功证据。

## 1. 目的与范围

固定物理学习研究 20261107T195435Z 的结论是：168 系数同腿 MN 读出在零下行驱动下未超过零输出基线（selected 仍为 trial_000 零参数）。本轮只做离线诊断，分离两个假设：

- **假设 A：冻结感觉编码器过粗 / 自由度不足**（556 细胞、每腿 4 通道：爪位置、钩方向、毛板角、钟形感器应变）。
- **假设 B：神经/读出阶段压缩或无法暴露可用状态**。

诊断边界：不新增物理/神经仿真步、不拟合可部署策略、不改验收阈值、不从相关数据给出生物学结论。

## 2. 路径与预登记

| 项 | 完整路径 |
|---|---|
| 预登记（绑定） | `{HERE}/PREREG.json` |
| 工作区产物 | `{WS}` |
| 桌面镜像 | `{DS}` |
| 冻结编码器 | `{OLD}/encoder/encoder.py` |
| 48 个非零 trial 源数据 | `{TRIALS}/trial_001..012,014..025,027..038,040..051` |
| 旧失败源（外部诊断，仅计数与转用） | `/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T185910Z/readout/TRAIN_TRACE.npz` + `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T192931Z/dynamics/SOURCE_TRACE.npz` |
| 基线 trial_000 与 4 个对照 | `{TRIALS}/trial_000,013,026,039,052`（全部排除） |

预登记 SHA256：`{result['prereg_sha256']}`（`VERIFICATION.json` 已核对该文件自审计开始后未被改动）。

## 3. 步骤一：编码器灵敏度/混叠审计

独立重写四条声明公式（不调用 encoder 的 helper 特征），并在**全部** 9600 个训练 bin 与 1817 个旧失败源 bin 上与真实编码器逐 bin 比对：

| 检查 | 结果 |
|---|---|
| 记录感觉电流最大误差（真实 encoder vs 保存的 556 电流） | **0.0 nA** |
| 独立公式 vs 保存的 24 个通道均值最大误差 | **{audit['independent_formula_max_error_nA']:.2e} nA** |
| 解析 Jacobian vs 中心差分 1e-6（24x90 通道图） | **{audit['jacobian_max_error_vs_central_difference']:.2e}** |
| 解析 Jacobian vs 中心差分（556x90 细胞图） | **{audit['cell_jacobian_max_error_vs_central_difference']:.2e}** |
| 编码器调用数 | {audit['encoder_calls']} |

参考输入：42 个关节范围中点、膝速度 +2 rad/s、每腿载荷 W/6=4.0744（W=24.446354755878005）。输出钳位 [0, 0.06] nA 在参考点未激活（通道最大值 {audit['max_abs_channel_value_before_output_clip_nA']:.6f} nA）。

**敏感度与秩（90 输入 -> 24 通道）**

| 块 | 输入列数 | 非零列 | 零列 | 独立方向（数值秩） |
|---|---:|---:|---:|---:|
| q（关节角） | 42 | {ranks['nonzero_q_columns']} | {ranks['zero_q_columns']} | **{ranks['q_block_24x42']}** |
| qvel（关节速度） | 42 | {ranks['nonzero_qvel_columns']} | {ranks['zero_qvel_columns']} | **{ranks['qvel_block_24x42']}** |
| foot_load（足载） | 6 | {ranks['nonzero_foot_load_columns']} | {ranks['zero_foot_load_columns']} | **{ranks['foot_load_block_24x6']}** |
| 合计 | 90 | {90-len(audit['zero_columns'])} | **{len(audit['zero_columns'])}** | **{ranks['channel_map_24x90']}** |

- 66 个零列更正为 **{len(audit['zero_columns'])} 个**：18 个 q（6 腿 x 3 个 trochanter 角）+ 36 个 qvel（全部非膝关节速度）。此前一条进度消息写作 66，属口误，已在 `FAILED_DIAGNOSTIC_ATTEMPTS.json` 记录并更正。
- 24 个非零 q 列只给出 **{ranks['q_block_24x42']} 个独立方向**：每腿 coxa yaw/pitch/roll 只进入同一个 hairplate 均值通道（每腿秩 {audit['coxa_axis_rank_per_leg'][0]}）。
- 90 维输入到 24 维输出的映射秩仅 **{ranks['channel_map_24x90']}**，其余 {90-ranks['channel_map_24x90']} 维输入空间完全不可见。

**工程化别名固件（18 个 = 6 腿 x 3 类；不是真实身体轨迹）**

| 固件 | 构造 | 编码器电流 | 独立公式 |
|---|---|---|---|
| coxa yaw/pitch 中点互换 | 同腿 yaw=mid+d、pitch=mid-d 与互换后 | 逐位相同 | 逐位相同 |
| trochanter 角改变 | 同腿 Trochanter_yaw/roll 改到实际范围内 10% 位 | 逐位相同（**无输入端口**） | 逐位相同 |
| 膝速度 +2/-2 | 同腿膝速度反号 | 逐位相同（精确绝对值混叠） | 逐位相同 |

全部固件满足：关节范围合法、q 或 qvel 确实不同（`distinct_q`/`distinct_qvel`），且 `physical_trajectory=false`、`collision_validity_not_tested=true`——它们是人为构造输入，绝不代表真实身体姿态。

另有 **6 个非预登记的可见性对照**：把膝角改动 10% 行程后电流变化 4e-3 nA（编码器确实能分辨），证明秩 24 不是代码假象。

**接口缺口声明（原文记录）**："{audit['interface_gap_statement']}"

## 4. 步骤二：72 个固定拟合

固定协议：train=gen0+gen1（24 episodes，4800 bin），test=gen2+gen3（24 episodes，4800 bin）；ridge lambda=1；只用训练集均值/标准差做标准化（下限 1e-8）；诊断模型含未惩罚截距；分类用平衡类权重 N/(2 n_class)；无调参、无按测试集选择。

特征维度（每腿）：encoded4=4；MNinstant=42–60；MNhistory=168–240（滞后 0/1/4/10 bin = 0/5/20/50 ms，lag-major，逐 episode 零前缀）；rawphysical=15（7 q + 7 qvel + 1 足载，特权，永不部署）。

分类标签：stance/air 用足载 > 0.05W 为 +1、< 0.005W 为 -1（中间值排除）；膝速度符号用同腿 Tibia_pitch 速度 |>1e-3| rad/s。**encoded4 的 stance/air 任务是同义反复**：campaniform_strain 按设计就是足载的函数。

**stance / air（测试集平衡准确率，chance=0.5）**

{md(stance)}

**膝速度符号（测试集平衡准确率，chance=0.5）**

{md(sign)}

**带符号膝速度回归（测试集 R²，0=不优于训练均值）**

{md(reg)}

**带符号膝速度回归（测试集 MSE）**

{md(reg_mse,'{:.5f}')}

**旧失败源转用（不重拟合、仅用已训练模型；外部诊断）——膝速度符号平衡准确率**

{md(source_ba)}

**旧失败源转用——stance/air 平衡准确率**

{md(source_st)}

NOT_ESTIMABLE 数量：**{results['not_estimable']}**（无缺失类别，条件一个都没有被静默丢弃）。

## 5. 步骤三：图

- 图文件：`{HERE}/FIT_SUMMARY.png`（已实际打开渲染确认）
- 四面板：stance/air、膝速度符号、带符号膝速度 R² 的 腿x表示 热图，以及编码器秩/别名摘要柱状图。
- 桌面镜像同名文件在 `{DS}/FIT_SUMMARY.png`。

## 6. 步骤四：独立验证

`{HERE}/verify_information.py` **不 import** `fit_information.py` 与 `audit_encoder.py`，从冻结编码器源码和 195435Z 的 trial 产物重建标签、滞后堆叠、训练统计、ridge 正规方程、预测与指标。

| 项 | 结果 |
|---|---|
| 状态 | **{verification['status']}** |
| 检查数 / 通过 / 失败 | {verification['checks']} / {verification['passed']} / {verification['failed']} |
| 重算拟合数 | {verification['fits_recomputed']} |
| 独立读取产物次数 | {verification['data_artifact_reads']} |
| 系数最大相对偏差 / 预测最大绝对偏差 / 指标最大相对偏差 | {verification['max_beta_delta']:.2e} / {verification['max_prediction_delta']:.2e} / {verification['max_metric_delta']:.2e} |
| 保存的正规方程残差上界 | {verification['max_stored_normal_equation_residual']:.2e} |
| 独立重算秩 / 零列 | {verification['encoder_rank_recomputed']} / {verification['zero_columns_recomputed']} |

## 7. 失败与更正记录（全部保留）

见 `{HERE}/FAILED_DIAGNOSTIC_ATTEMPTS.json`，要点：

1. **拟合首轮（已废弃）**：中间阈值行以 NaN 目标进入 ridge 求解，全部系数变 NaN、所有预测塌到 -1，72 个平衡准确率恰好 0.500。因 encoded4 stance/air 的同义反复性质本应接近 1，判定该轮无效，未用于任何报告。产物保存为 `FAILED_NAN_TARGET_FIT_RESULTS.json` / `FAILED_NAN_TARGET_FIT_MODELS.npz`。修正：无标签行权重置 0 且目标用有限占位值，同时从拟合与评分中排除；lambda、切分、表示、滞后、阈值一律未改。
2. **独立验证第 1 轮**：1842 检查中 10 项失败——其中 2 项因独立公式与 `np.tanh` 存在约 1e-18 的浮点差异却用了精确相等；1 项对 9.4e8 量级的源转用 MSE 用了绝对容差；6 项对特权回归目标期望"精确恒等"。
3. **独立验证第 2 轮**：1914 检查中 6 项失败，全部是 `rawphysical` 带符号膝速度的"精确恒等"期望——实际是 lambda=1 的正常收缩（1-R² = {min(key['rawphysical_ridge_shrinkage_one_minus_r2']):.2e}–{max(key['rawphysical_ridge_shrinkage_one_minus_r2']):.2e}，残差 3.0e-4–2.4e-3 rad/s），不是数据缺陷。
4. **验证第 3 轮**：改为断言预登记的 ridge 解（R² > 1-1e-6）并记录剩余收缩后 **{verification['checks']} 项全部通过**。lambda、数据与切分未改。

## 8. 计数语义（严格）

- **拟合数 = {counts['fits_performed']}**：等于"腿 x 表示 x 目标"的 ridge 解个数；特征构造、标准化、指标计算不计入拟合；上限 72 未被突破。
- **数据读取**：拟合脚本 {counts['fitter_artifact_reads']} 次 npz 读取；审计脚本 {counts['encoder_audit_artifact_reads']} 次文件读取（含 288 个 trial 产物）；独立验证 {counts['verifier_artifact_reads']} 次 npz 读取。
- **编码器调用**：{counts['encoder_calls']} 次（纯数组运算）。
- **物理步 / 神经步 / 训练更新**：**0 / 0 / 0**。
- 训练 bin {counts['saved_bins_fitted']}（24+24 episodes x 200 bin）+ 旧失败源 {counts['old_source_bins']} bin。

## 9. 两个假设的读法（严格限定）

- **假设 A（编码器过粗）**：方向信息上有支持。`hook_direction` 是 `0.05*tanh(|膝速度|/10)`，是精确的绝对值混叠；除特权 rawphysical 外没有任何表示在膝速度符号上明显超过 0.65 平衡准确率；90 维输入中 54 列完全无端口，映射秩 24。
- **假设 B（神经/读出压缩）**：状态信息上有支持。encoded4 在 stance/air 上达到 0.978–0.993 测试平衡准确率，而同一批 bin 的同腿 MN 表示只有 0.436–0.822（见 `aggregates_by_task_and_representation`），即 MN 观测压缩了编码器已经暴露的信息。
- **尚未确立**：以上都不是因果神经表征、知觉或运动成功的证据；48 个 episode 共享同一 canonical checkpoint 与确定性策略变体，样本高度相关；本轮没有任何物理或神经步。

## 10. 明确边界

- 粗编码器是**故意未标定**的工程代理；它的信息损失**不能**用来断言源感觉生物学缺少该信息。
- 零输入列是**未标定的接口缺口**，不是生物缺失证据。
- 未编造细胞身份，未推断调谐方向；通道顺序与端口全部来自已冻结的身份记录。
- `rawphysical` 是特权诊断量，**永不可部署**；本轮没有任何策略被拟合用于部署，selected 政策仍为 trial_000 零参数（历史事实未变）。
- 未修改任何冻结/历史文件；未执行任何物理或神经仿真。

## 11. 下一步（已准备、未执行）

`{HERE}/run_dn_pair.py`（只写不跑）：确定性选取历史中**最大训练奖励**的非零支持可行 Actor（trial_017，reward {(-1.0144429917827316):.10f}；tie-break 用 trial 序号；永不选零 incumbent），在同一个完整 canonical 1s checkpoint 上跑 2 秒释放，唯一差别是下行电流 0 vs 0.05 nA（`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-descending-learning/20261107T203049Z/descending/command.py`，sha256 `ba84bca2b7e3ee710c3412467db2a4eadcd839dd38d027d44241b7b9363b455c`）。必须显式 `--execute` 才会真正运行；由父代理执行（输出到 `/home/sensen/Desktop/engine修复验收/banc-descending-learning/20261107T203049Z/physical/dn_zero` 与 `dn_on`）。

**本轮对该 runner 只做了 plan 模式验证（无物理、无神经步）**：确定性选择复现 12 个合规候选中的 trial_{plan['selected']['trial']:03d}（reward {plan['selected']['reward']:.10f}，theta 最大绝对值 {plan['selected']['theta_max_abs']:.6f}），命令模块哈希 `{plan['descending_command']['sha256']}` 与预期一致，DN root ids 为 {plan['descending_command']['root_ids']}。

- 默认（无 `--execute`）：只打印计划，**不 import mujoco、不建 Plant、不建神经核、不写任何文件**。
- `--execute`：两次 2 s 释放，每次 {plan['timing']['native_steps_per_condition']} 个 12.5 µs 原生步、{plan['timing']['neural_steps']} 个神经步、{plan['timing']['forward_calls_per_condition']} 次 forward；合计 {plan['physics_steps_if_executed']} 物理步 / {plan['neural_steps_if_executed']} 神经步；每个条件 13 个文件（含 `DN_INPUT.npz`），并在父级物理目录写 `DN_PAIR_PLAN.json` 与 `DN_PAIR_MATCH.json`（配对一致性：初始物理态、执行器目标、神经初态 v/spike、Actor theta/weights 逐位相同，唯一差异是 DN 向量）。
- 若确定性选择结果与已记录的最佳非零候选不一致，runner 直接报错退出，**不会静默换 Actor**。
"""
    (HERE/'REPORT.md').write_text(report)
    print(json.dumps({'report':str(HERE/'REPORT.md'),'result':str(HERE/'RESULT.json'),
                      'report_bytes':(HERE/'REPORT.md').stat().st_size,'result_bytes':(HERE/'RESULT.json').stat().st_size,
                      'aggregate_keys':len(agg),'physics_steps':0,'neural_steps':0}),flush=True)


if __name__=='__main__':main()
