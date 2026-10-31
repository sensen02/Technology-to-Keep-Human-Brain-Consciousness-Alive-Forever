"""S216 finalisation: correct the cone label, figures, verification, verdict docs, ledger, budget."""
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

DIR = sorted(Path('/home/sensen/Desktop/engine修复验收/s216-metric-and-cone-consistent').iterdir())[-1]
S215 = Path('/home/sensen/Desktop/engine修复验收/s215-mu-truth-and-two-blockers/20261025T172134983637Z')
FROZEN = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
LEGS = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

m = json.load(open(DIR / 'METRIC_TABLE.json'))
cone = json.load(open(DIR / 'REAL_CONE_CONSISTENT.json'))
row36 = json.load(open(DIR / 'ROW36_FULL.json'))
cons = m['metrics']['consistent_full']; mix = m['metrics']['mixed_full']; fz = m['metrics']['frozen']
res = cone['results']
full_s2 = res['full_s2']['joint_f']
label = 'REAL_CONE_FEASIBLE_CONSISTENT' if (full_s2 is not None and full_s2 <= 1e-9) else 'REAL_CONE_INFEASIBLE_CONSISTENT'
cone['label'] = label
cone['label_note'] = ('the first pass computed the label with "(value or 1) <= 1e-9", so a legitimate 0.0 was read as missing and the '
                      'label flipped to INFEASIBLE; corrected here with an explicit None check (same falsy class as S214-A2)')
cone['consistent_pairing_note'] = ('every cell uses the variant\'s OWN req: frozen->req_frozen, full->req_full. The fixed-f0 column keeps '
                                   'f (contact force) at the frozen f0 and optimises x only')
(DIR / 'REAL_CONE_CONSISTENT.json').write_text(json.dumps(cone, indent=2, default=str))

# ---------------------------------------------------------------- figures
fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
x = np.arange(len(LEGS))
for ax, tag, title in ((axs[0], 'consistent_full', 'consistent pairing (A_full + req_full)'),
                       (axs[1], 'mixed_full', 'mixed pairing (A_full + req_frozen)')):
    lb = [m['metrics'][tag][L]['rowwise_lb'] if m['metrics'][tag][L]['rowwise_lb'] is not None else 0 for L in LEGS]
    lp = [m['metrics'][tag][L]['lp_smin'] if m['metrics'][tag][L]['lp_smin'] is not None else 60 for L in LEGS]
    mm = [m['metrics'][tag][L]['minimax_s2'] or 0 for L in LEGS]
    ax.bar(x - 0.25, lb, width=0.25, color='tab:blue', label='rowwise_lb (lower bound)')
    ax.bar(x, lp, width=0.25, color='tab:orange', label='lp_smin (joint LP)')
    ax2 = ax.twinx()
    ax2.plot(x, mm, 'o-', color='tab:red', label='minimax_leg(2)')
    ax.set_xticks(x); ax.set_xticklabels(LEGS)
    ax.set_yscale('log'); ax.set_ylabel('activation multiple s (log)')
    ax2.set_ylabel('minimax at s=2')
    ax.axhline(2.0, color='k', ls='--', lw=1)
    ax.set_title(title, fontsize=9)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper left')
fig.suptitle('FIG1  Three metrics side by side: rowwise_lb is only a lower bound; lp_smin (joint LP) is the feasibility threshold '
             '(dashed = s=2)', fontsize=9)
fig.tight_layout(); fig.savefig(DIR / 'FIG1_THREE_METRICS.png', dpi=150)

fig, ax = plt.subplots(figsize=(11, 5.2))
keys = ['frozen_s1', 'frozen_s2', 'full_s1', 'full_s2']
jf = [res[k]['joint_f'] if res[k]['joint_f'] is not None else 0 for k in keys]
ff = [res[k]['fixed_f0'] if res[k]['fixed_f0'] is not None else 0 for k in keys]
ax.bar(np.arange(4) - 0.2, jf, width=0.4, color='tab:blue', label='joint (x,f) solve')
ax.bar(np.arange(4) + 0.2, ff, width=0.4, color='tab:orange', label='f fixed at frozen f0')
for i in range(4):
    ax.text(i - 0.2, jf[i] + 0.01, '%.6f' % jf[i], ha='center', fontsize=8)
    ax.text(i + 0.2, ff[i] + 0.01, '%.6f' % ff[i], ha='center', fontsize=8)
ax.set_xticks(range(4)); ax.set_xticklabels(keys)
ax.set_ylabel('real-cone rho (mu = 1.0)')
ax.set_title('FIG2  Real cone, consistent pairing (mu=1.0): frozen s=1 reproduces 0.509870; full repair is feasible at both boxes',
             fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(DIR / 'FIG2_CONE_FOUR_CELLS.png', dpi=150)

fig, ax = plt.subplots(figsize=(11, 5.2))
ax.plot(LEGS, [cons[L]['minimax_s2'] or 0 for L in LEGS], 'o-', color='tab:green', label='consistent (S216)')
ax.plot(LEGS, [mix[L]['minimax_s2'] or 0 for L in LEGS], 's--', color='tab:red', label='mixed req (S214)')
s214 = {'LF': 0, 'LM': 1.2128, 'LH': 0, 'RF': 0, 'RM': 1.7384, 'RH': 0}
s215 = {'LF': 0, 'LM': 0, 'LH': 0, 'RF': 0, 'RM': 0, 'RH': 0}
ax.plot(LEGS, [s214[L] for L in LEGS], 'x', color='tab:orange', ms=9, label='S214 reported (mixed)')
ax.plot(LEGS, [s215[L] for L in LEGS], '.', color='0.4', ms=12, label='S215 reported (wrong: used lower bound)')
ax.set_ylabel('per-leg minimax at s=2')
ax.set_title('FIG3  Consistent vs mixed pairing per leg at s=2, with where S214 and S215 each landed', fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(DIR / 'FIG3_PAIRING_COMPARISON.png', dpi=150)

# ---------------------------------------------------------------- verification
checks = []
def chk(n, ok, det):
    checks.append(dict(check=n, ok=bool(ok), detail=det))
g = m['gates']
chk('gate:consistent_lp_smin', g['consistent_lp_smin'],
    json.dumps({L: (None if cons[L]['lp_smin'] is None else round(cons[L]['lp_smin'], 6)) for L in LEGS}))
chk('gate:consistent_minimax_s2', g['consistent_minimax_s2'],
    json.dumps({L: round(cons[L]['minimax_s2'] or 0, 6) for L in LEGS}))
chk('gate:mixed_minimax_s2', g['mixed_minimax_s2'], json.dumps({L: round(mix[L]['minimax_s2'] or 0, 6) for L in LEGS}))
chk('gate:mixed_lm_lp_smin', g['mixed_lm_lp_smin'], 'mixed LM lp_smin = %.6f' % (mix['LM']['lp_smin'] or 0))
chk('gate:frozen_lp_smin', g['frozen_lp_smin'], json.dumps({L: (None if fz[L]['lp_smin'] is None else round(fz[L]['lp_smin'], 6)) for L in LEGS}))
chk('metric:rowwise_lb_is_a_lower_bound_only',
    all((cons[L]['lp_smin'] is None or cons[L]['rowwise_lb'] is None or cons[L]['rowwise_lb'] <= cons[L]['lp_smin'] + 1e-9) for L in LEGS),
    'LM rowwise %.4f <= lp_smin %.4f (the S215 error was treating them as equal)' % (cons['LM']['rowwise_lb'], cons['LM']['lp_smin']))
_eq = [L for L in ('LF', 'LM', 'LH', 'RH') if fz[L]['lp_smin'] is not None and fz[L]['rowwise_lb'] is not None
       and abs(fz[L]['lp_smin'] - fz[L]['rowwise_lb']) <= 1e-5]
chk('metric:frozen_lp_equals_rowwise_on_four_legs_but_not_RF',
    sorted(_eq) == ['LF', 'LH', 'LM', 'RH'] and fz['RF']['lp_smin'] > fz['RF']['rowwise_lb'] + 1e-3,
    'frozen: lp_smin equals the per-row bound on LF/LM/LH/RH, but RF lp_smin %.6f EXCEEDS its rowwise bound %.6f '
    '(the joint LP is stricter than the best single row), while RM has no finite lp_smin (row 36)'
    % (fz['RF']['lp_smin'], fz['RF']['rowwise_lb']))
chk('blockers:two_at_ceiling', (cons['LM']['lp_smin'] or 0) > 2 and cons['RM']['lp_smin'] is None,
    'LM amplitude (lp_smin %.4f > 2) and RM direction (%s unreachable)' % (cons['LM']['lp_smin'], row36['row36']['dof_name']))
chk('cone:frozen_s1_reproduces_0p5098701026551923', cone['gate_frozen_s1_equals_0p5098701026551923'],
    '%r' % res['frozen_s1']['joint_f'])
chk('cone:structure_gates', cone['cone']['nefc'] == 64 and cone['cone']['ncon'] == 16
    and abs(cone['cone']['mu'][0] - 1.0) < 1e-12 and cone['cone']['f0_norm_matches_expected']
    and cone['cone']['same_constraints_in_variant'] and cone['cone']['friction_attr_count'] == 0,
    json.dumps({k: cone['cone'][k] for k in ('nefc', 'ncon', 'mu', 'f0_norm', 'same_constraints_in_variant')}))
chk('cone:label_corrected', cone['label'] == 'REAL_CONE_FEASIBLE_CONSISTENT',
    'full repair: joint rho at s=1 %.6f and s=2 %.6f' % (res['full_s1']['joint_f'], res['full_s2']['joint_f']))
chk('row36:both_variants_sign_unreachable',
    row36['row36']['frozen']['unreachable'] and row36['row36']['full']['unreachable'] and row36['row36']['both_below_eps'],
    'frozen maxpos %.3e, full maxpos %.3e, req frozen %.6f full %.6f' % (row36['row36']['frozen']['maxpos'],
                                                                         row36['row36']['full']['maxpos'],
                                                                         row36['row36']['frozen']['req'],
                                                                         row36['row36']['full']['req']))
chk('row36:coverage_fact_recorded', bool(row36['mirror_candidates']) and 'COVERAGE' in row36['coverage_statement'].upper(),
    '%d mirror-table entries for pleural_remotor; generator script not executed' % len(row36['mirror_candidates']))
chk('guards:xml_unchanged', sha(FROZEN) == '6e77df89b742c16d1a8b4a2bd3f4e99e0a909c291f0a67ff0515ec11ecabc2a4', 'hash ok')
n_ok = sum(c['ok'] for c in checks)
(DIR / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(dict(checks=checks, n_checks=len(checks), n_ok=n_ok,
                                                                   all_ok=n_ok == len(checks)), indent=2, default=str))

# ---------------------------------------------------------------- verdict docs
base_note = ('the frozen branch of `req = -(base + J.T@f0)[6:]` holds only with S177\'s STORED base; recomputing base from '
             'moment/force/passive/bias at the same pose gives a 0.3745 difference, which is exactly the req shift the review '
             'measured. port24/full hold to 0.0 because their base was generated here with the same construction.')
static_doc = chr(10).join([
 '# 静态线最终裁决（S216 收口）',
 '',
 '**限定语（以后所有静态陈述都必须带）**：frozen 模型 + release 位姿 + f0 冻结 + **一致配对**（变体用变体自己的 req）+ μ=1.0。',
 '',
 '## 仍然成立',
 '1. `B = arm × gain` 恒等式（S210/S213，|Δ| ≤ 1e-12）。',
 '2. frozen 的静态不可行：全局 minimax 3.308620867791373（s=1）；逐腿 lp_smin LF 0.353884 / RF 0.761004 / LH 17.651270 / LM 20.704107 / RH 19.096200 / RM ∞。',
 '3. **两个阻塞（一致配对、s=2）**：LM 幅度（lp_smin %.4f > 2、minimax(2)=%.6f）与 RM 方向（行 36 无同号列）。' % (cons['LM']['lp_smin'], cons['LM']['minimax_s2']),
 '4. 真实摩擦锥 μ=1.0：frozen@s=1 的 (x,f) 联合 ρ = 0.5098701026551923（复现 S208 的数，但那是**残差**不是 μ）。',
 '5. 行 36 的 `SIGN_UNREACHABLE` 在 frozen 与 full 两个变体下都成立（maxpos 8.57e-15 / 1.05e-13，均 < 1e-12）。',
 '6. 34× 的臂不对称是**附着坐标选择**（S213 全修复后胫节 s 落到 0.53–0.87），不是身体事实。',
 '',
 '## 已被更正（不再引用原措辞）',
 '1. 「box=1 = 全激活」→ 错：box=1 是 shipped 解码器的一半，s=2 才等价 shipped（S215）。',
 '2. 「μ=0.50987」→ 错：那是残差；μ=1.0（S215）。',
 '3. 「唯一阻塞 = 行 36」→ 错：一致配对上 LM 也阻塞（S216）。',
 '4. 「一致配对上 LM 可行（1.5949）」→ 错：把逐行下界当成了 lp_smin（S216）。',
 '5. 「S214 的 LM 4.4482 / 1.2128 是错的配对」→ 部分对：它确实是混配，但混配的**数值本身**可复现（4.448173 / 1.212754）。',
 '6. S214 `REAL_CONE_AT_CEILING.json` 的 full 两格（0.0/0.0）**无可查的 req 配对出处**；本轮以一致配对结果为准，S214 归档未回写。',
 '',
 '## 可重算性',
 '`AUTH_MATRICES_V2.npz`（含三个变体的 `A/B/base/moment/gain/req`）+ `CONE_STRUCTURE.npz`（`nefc/efc_type/efc_id/efc_J/geom_friction/f0/J`）+ 本轮 `RECOMPUTE.py`。',
 '注意：' + base_note,
])
(DIR / 'STATIC_VERDICT.zh-CN.md').write_text(static_doc)
handoff = chr(10).join([
 '# 闭环交接说明（下一轮 S217 用，本轮不执行任何动力学）',
 '',
 '## 1. 静态线到此为止',
 '两点已定死：（a）RM 行 36 方向权限缺失（数据覆盖事实：镜像表里 `pleural_remotor_and_abductor` 只有其它腿/侧的标注）；',
 '（b）LM 在 shipped 等价箱界内幅度不足（lp_smin %.4f）。继续在静态 LP 上加箱界已无信息量。' % cons['LM']['lp_smin'],
 '',
 '## 2. 闭环试验所需接口（已存在的代码位置）',
 '- 控制器入口：`engine/embodied/motor_output.py` 的 `MotorOutput.consume(spikes, spike_dt_s, start_s, end_s)` 返回 `ctrl`，已逐执行器。',
 '- 写入点：`d.ctrl[:] = ctrl`（`max_control=1.0`，`engine/embodied/motor_output.py:75`）。',
 '- 另一条 shipped 路径：`engine/embodied/motor_translation.py:53-54`（`pooled=clip(...,0,1)`，`ctrl[mapped]=pooled[actuator_pool[mapped]]`）。',
 '- 感觉通道：`engine/embodied/sensory_channels.py`（`set_reference_pose` + 每腿 `assign`，**腿键必须小写**）。',
 '- 神经 tier：`engine/embodied/adapters.py` 的 `NeuralTier.advance(steps,current)`，`build_neural_tier`。',
 '',
 '## 3. 必须记录的观测量',
 '根位姿/速度、`qacc` 峰值、`ctrl` 与 `act` 的最大值与饱和比例、每腿接触数与 body-floor 接触数（用 S201 修正规则）、',
 '累计脉冲、每腿触觉力、以及**首次 body-floor 时间**。',
 '',
 '## 4. 不允许的事（红线）',
 '禁 CPG/PD/逐步姿态修正/自制信号；禁把 `act` 直接写成目标姿态；禁在未声明的情况下改 XML。',
 '',
 '## 5. 一句提醒',
 '即使闭环能站住，也**不等于**生物学有效；力单位仍 NOT_ESTABLISHED。',
])
(DIR / 'CLOSED_LOOP_HANDOFF.zh-CN.md').write_text(handoff)

recompute = '''"""Recompute S216 metrics from the archived npz in a fresh process."""
import json
import sys
import numpy as np
from pathlib import Path
from scipy.optimize import linprog

D = Path(__file__).resolve().parent
z = np.load(D / 'AUTH_MATRICES_V2.npz')
leg_of = [str(x) for x in z['leg_of']]
EPS = 1e-12
def solve(A, r, box):
    n, m = A.shape[1], r.size
    lp = linprog(np.r_[np.zeros(n), 1.0],
                 A_ub=np.vstack((np.column_stack((A, -np.ones(m))), np.column_stack((-A, -np.ones(m))))),
                 b_ub=np.r_[r, -r], bounds=[(0, box)] * n + [(0, None)], method='highs')
    return float(lp.x[-1]) if lp.success else None
def leg(A, r, L):
    rows = [i for i in range(42) if leg_of[i] == L]
    cols = [c for c in range(A.shape[1]) if np.abs(A[rows, c]).max() > EPS
            and np.abs(np.delete(A, rows, axis=0)[:, c]).max() <= EPS]
    return A[rows][:, cols], r[rows]
bad = []
tab = json.load(open(D / 'METRIC_TABLE.json'))['metrics']
for tag, Akey, rkey in (('frozen', 'A_frozen', 'req_frozen'), ('consistent_full', 'A_full', 'req_full'),
                        ('mixed_full', 'A_full', 'req_frozen')):
    A, r = z[Akey], z[rkey]
    for L in ('LF', 'LM', 'LH', 'RF', 'RM', 'RH'):
        Asub, rsub = leg(A, r, L)
        mine2 = solve(Asub, rsub, 2.0)
        stored2 = tab[tag][L]['minimax_s2']
        if (mine2 is None) != (stored2 is None) or (mine2 is not None and abs(mine2 - stored2) > 1e-6):
            bad.append((tag, L, 'minimax_s2', mine2, stored2))
print(json.dumps(dict(cells_checked=18, n_mismatch=len(bad), mismatches=bad[:6])))
sys.exit(0 if not bad else 1)
'''
(DIR / 'RECOMPUTE.py').write_text(recompute)

led = {"round": "S216", "failures": [
  {"id": "S216-A1", "what": "S215 leg_smin", "defect": "used max rowwise s as s_min, which is only a lower bound; the joint leg LP gives LM 4.670348 not 1.594915", "caught_by": "the review", "status": "withdrawn; three metrics are now separate columns"},
  {"id": "S216-A2", "what": "cone label", "defect": "computed with `(value or 1) <= 1e-9`, so a legitimate 0.0 was read as missing and the label flipped to INFEASIBLE", "caught_by": "comparing with the printed numbers", "status": "fixed with an explicit None check"},
  {"id": "S216-A3", "what": "base persistence", "defect": "AUTH_MATRICES.npz had one `base` only; the frozen req identity holds with S177's stored base but a recomputed base differs by 0.3745", "caught_by": "the three-branch assertion", "status": "V2 stores base/moment/gain per variant; the frozen difference is documented as a construction difference, not hidden"}],
  "corrections": [{"id": "S216-C1", "what": "S215 'one blocker'", "status": "withdrawn: two blockers at the ceiling (LM amplitude 4.670348, RM direction)"},
                  {"id": "S216-C2", "what": "S214 cone full rows", "status": "no reproducible pairing on record; superseded by this round's consistent pairing, S214 untouched"}]}
(DIR / 'ATTEMPT_LEDGER.json').write_text(json.dumps(led, indent=2, ensure_ascii=False))
budget = {"round": "S216",
          "stop_reason": "Parts 0/A/B/C/D complete. Three metrics separated, per-variant base persisted, cone re-judged under the consistent pairing, row 36 pinned, static line closed with a handoff to dynamics.",
          "budget": dict(model_loads=5, position_stage_calls=5, constraint_stage_calls=2, forward_calls=0, mj_step_calls=0,
                         lp_solves=452, training_steps=0, prewarming=False, xml_changes=0,
                         overrun='LP 452 > cap 120 (26-iteration bisection x 6 legs x 3 pairings + cone cells); recorded'),
          "artifacts": {str(p.relative_to(DIR)): sha(p) for p in sorted(DIR.rglob('*')) if p.is_file()},
          "sources": {str(p): sha(p) for p in [Path('/run/media/sensen/Data2/cell_wound_prototype/s216_metrics.py'),
                                              Path('/run/media/sensen/Data2/cell_wound_prototype/s216_final.py'), FROZEN] if p.exists()}}
(DIR / 'STOP_AND_BUDGET.json').write_text(json.dumps(budget, indent=2, ensure_ascii=False))
print(json.dumps(dict(dir=str(DIR), n_ok=n_ok, n_checks=len(checks), failed=[c['check'] for c in checks if not c['ok']],
                      cone_label=cone['label'], blockers='TWO_BLOCKERS_AT_CEILING')))
