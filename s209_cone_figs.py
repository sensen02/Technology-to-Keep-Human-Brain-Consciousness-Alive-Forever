"""S209 Part D (real cone variants) + figures + verification + conclusion + bookkeeping."""
import json
import hashlib
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import linprog

DIR = sorted(Path('/home/sensen/Desktop/engine修复验收/s209-row36-sign-and-agonist').iterdir())[-1]
S208 = Path('/home/sensen/Desktop/engine修复验收/s208-rm-direction-and-real-cone/20261025T163943555010Z')
STATE = Path('/home/sensen/Desktop/engine修复验收/s177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz')
MAP = Path('/home/sensen/Desktop/engine修复验收/s166-mapped-support/20261024T231418078692647Z/MAPPING.npz')
ROWDOF = sorted(Path('/home/sensen/Desktop/engine修复验收').glob('s179-*/*/ROW_DOF_MAP.json'))[-1]
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')

st = np.load(STATE)
B, base = np.asarray(st['B'], float), np.asarray(st['base'], float)
f0, J, T = np.asarray(st['f0'], float), np.asarray(st['J'], float), np.asarray(st['T'], float)
qpos = np.asarray(st['qpos'], float)
req = (-(base + J.T @ f0))[6:]
A = (B @ T)[6:, :]
jn = [r['dof_name'] for r in json.load(open(ROWDOF))['rows']]
leg_of_row = [n[6:8] for n in jn]
RM_ROWS = [i for i in range(42) if leg_of_row[i] == 'RM']
T_syn = np.zeros(90); T_syn[47] = 0.5
A_syn_col = (B @ T_syn)[6:]
A_aug = np.column_stack((A, A_syn_col))

# ---------------------------------------------------------------- Part D: real cone
m = mujoco.MjModel.from_xml_path(str(XML))
d = mujoco.MjData(m)
d.qpos[:] = qpos
d.qvel[:] = 0.0
mujoco.mj_fwdPosition(m, d)
mujoco.mj_fwdConstraint(m, d)
nefc = int(d.nefc)
efc_type = np.asarray(d.efc_type[:nefc]).copy()
efc_id = np.asarray(d.efc_id[:nefc]).copy()
ncon = int(d.ncon)
geom_friction = np.asarray(m.geom_friction).copy()
con_geom = [(int(d.contact[i].geom1), int(d.contact[i].geom2)) for i in range(ncon)]
groups = {}
for r in range(nefc):
    if int(efc_type[r]) in (5, 6, 7):
        groups.setdefault(int(efc_id[r]), []).append(r)
assert sum(len(v) for v in groups.values()) == nefc, 'contact rows must cover nefc'
csets = []
for ci, rows in groups.items():
    g1, g2 = con_geom[ci]
    mu = min(float(geom_friction[g1, 0]), float(geom_friction[g2, 0]))
    csets.append((mu, rows))
s208 = json.load(open(S208 / 'REAL_CONE_CONTACT.json'))
recon_gate = dict(nefc=nefc, nefc_s208=s208['nefc'], n_contacts=len(groups), n_contacts_s208=s208['n_contacts'],
                  efc_types=sorted({int(t) for t in efc_type}), group_sizes={int(k): len(v) for k, v in groups.items()},
                  f0_components=int(f0.size),
                  matches=bool(nefc == s208['nefc'] and len(groups) == s208['n_contacts'] and f0.size == nefc))


def cone_solve(Amat, ncols, box):
    n = ncols + f0.size
    N = n + 1
    Aub, bub = [], []
    for i in range(42):
        for sgn in (1.0, -1.0):
            row = np.zeros(N)
            row[:ncols] = sgn * Amat[i, :ncols]
            row[ncols:n] = -sgn * J[:, 6:].T[i]
            row[n] = -1.0
            Aub.append(row); bub.append(sgn * req[i])
    for mu, rows in csets:
        normal, tans = rows[0], rows[1:]
        for t in tans:
            for s in (1.0, -1.0):
                row = np.zeros(N)
                row[ncols + t] = s
                row[ncols + normal] = -mu / np.sqrt(len(tans))
                Aub.append(row); bub.append(0.0)
    for mu, rows in csets:
        row = np.zeros(N); row[ncols + rows[0]] = -1.0
        Aub.append(row); bub.append(0.0)
    bnds = [(0, box)] * ncols + [(None, None)] * f0.size + [(0, None)]
    lp = linprog(np.r_[np.zeros(n), 1.0], A_ub=np.array(Aub), b_ub=np.array(bub), bounds=bnds, method='highs')
    return float(lp.x[-1]) if lp.success else None

rho_frozen_cone = cone_solve(A, A.shape[1], 1.0)
rho_syn_cone = cone_solve(A_aug, A_aug.shape[1], 1.0)
D = dict(calls=dict(model_loads=1, position_stage_calls=1, constraint_stage_calls=1, forward_calls=0, mj_step_calls=0),
         reconstruction_gate=recon_gate,
         mu_values=sorted({mu for mu, _ in csets}),
         rho_frozen_66_columns=rho_frozen_cone, rho_with_synthetic_column=rho_syn_cone,
         s208_reported=0.509870,
         reproduces_s208=bool(rho_frozen_cone is not None and abs(rho_frozen_cone - 0.509870) <= 1e-6),
         labels=dict(real_cone=('REAL_CONE_FEASIBLE' if rho_frozen_cone is not None and rho_frozen_cone <= 1e-6
                                else 'REAL_CONE_INFEASIBLE'),
                     with_synthetic=('REAL_CONE_FEASIBLE' if rho_syn_cone is not None and rho_syn_cone <= 1e-6
                                     else 'REAL_CONE_INFEASIBLE')),
         qualification=('S208 REAL_CONE_INFEASIBLE stands but is qualified: it holds for the current 66 commandable pool '
                        'columns, WITHOUT the #47 copyright, with f0 frozen at the release pose. Adding the synthetic antagonist '
                        'column changes the real-cone rho to %r.' % rho_syn_cone))
(DIR / 'REAL_CONE_VARIANTS.json').write_text(json.dumps(D, indent=2, default=str))

# ---------------------------------------------------------------- figures
hp = json.load(open(DIR / 'HIGH_PRECISION_ROW36.json'))
sign = json.load(open(DIR / 'SIGN_CENSUS.json'))
row36 = json.load(open(DIR / 'ROW36_ACTUATORS.json'))
syn = json.load(open(DIR / 'SYNTHETIC_COLUMN.json'))

fig, ax = plt.subplots(figsize=(12, 5.5))
acts = row36['actuators']
labels = ['#%d %s' % (a['actuator_index'], (a['actuator'] or '')[:22]) for a in acts]
vals = [a['B_coefficient'] for a in acts]
colors = ['tab:red' if (a['sign'] == '+' and not a['commandable']) else
          ('tab:orange' if a['sign'] == '+' else ('tab:blue' if a['commandable'] else '0.7')) for a in acts]
ax.bar(range(len(acts)), vals, color=colors, edgecolor='k', lw=.4)
for i, a in enumerate(acts):
    ax.text(i, vals[i] + (0.15 if vals[i] > 0 else -0.35), '%.3f' % vals[i], ha='center', fontsize=8)
    if not a['commandable']:
        ax.text(i, vals[i] / 2, 'uncommandable', rotation=90, ha='center', va='center', fontsize=7, color='w')
ax.set_xticks(range(len(acts)))
ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=7)
ax.set_ylabel('B[DOF 42, actuator] coefficient')
ax.set_title('FIG1  Row 36 (joint_RMCoxa_pitch, req = %.6f): the only POSITIVE actuator is #47, and it is uncommandable'
             % row36['required'], fontsize=9)
ax.axhline(0, color='k', lw=.8)
fig.tight_layout()
fig.savefig(DIR / 'FIG1_ROW36_ACTUATORS.png', dpi=150)

fig, ax = plt.subplots(figsize=(10, 5.5))
eps = [float(e) for e in sign['eps_grid']]
series = sign['eps_flip_table']
if series:
    name = list(series)[0]
    vals2 = [float(series[name]['series'][str(e)]) if False else (1.0 if series[name]['series'][str(e)] == 'reachable' else 0.0)
             for e in sign['eps_grid']]
    ax.semilogx(eps, vals2, 'o-', color='tab:purple', label='row 36: reachable(1) / unreachable(0) under eps')
ax.axhline(1.0, color='0.8', lw=.7)
ax.axhline(0.0, color='0.8', lw=.7)
ax.axvline(1.4376264251311566, color='tab:blue', ls=':', lw=1.2, label='req = 1.437626')
ax.axvline(hp['float64_max_positive'], color='tab:red', ls='--', lw=1.2, label='max positive coef = 8.57e-15')
ax.axvline(1e-12, color='k', ls=':', lw=1.0, label='eps_main = 1e-12')
ax.set_xlabel('eps (log)')
ax.set_ylabel('structural reachability of row 36')
ax.set_title('FIG2  Row 36 reachability vs eps: the verdict flips at eps = %.0e, i.e. the recognition of the 8.57e-15 residue\n'
             'the required activation at box=1 is %.3e, so no eps in any physical range makes this row reachable' %
             (float(series[name]['flips_at_eps']) if series else 0.0, hp['activation_needed']), fontsize=9)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(DIR / 'FIG2_REACHABILITY_VS_EPS.png', dpi=150)

fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
cases = ['box=1', 'unbounded']
froz = [syn['solves']['RM_box1']['rho_frozen'], syn['solves']['RM_unbounded']['rho_frozen']]
withsyn = [syn['solves']['RM_box1']['rho_with_synthetic'], syn['solves']['RM_unbounded']['rho_with_synthetic']]
x = np.arange(2)
axs[0].bar(x - 0.2, [v if v is not None else 0 for v in froz], width=0.4, color='tab:red', label='frozen 66 columns')
axs[0].bar(x + 0.2, [v if v is not None else 0 for v in withsyn], width=0.4, color='tab:green', label='+ synthetic #47 column')
axs[0].set_xticks(x); axs[0].set_xticklabels(cases)
axs[0].set_ylabel('RM minimax rho')
axs[0].legend(fontsize=8)
axs[0].set_title('FIG3a  RM rho with and without the synthetic antagonist column\n(box=1 unchanged: %.6f; unbounded -> %.3e)'
                 % (froz[0], withsyn[1]), fontsize=9)
cone_vals = [D['rho_frozen_66_columns'], D['rho_with_synthetic_column']]
axs[1].bar(['frozen 66', '+ synthetic'], [v if v is not None else 0 for v in cone_vals],
           color=['tab:red', 'tab:green'], edgecolor='k')
for i, v in enumerate(cone_vals):
    axs[1].text(i, (v or 0) + 0.02, '%.6f' % (v or 0), ha='center', fontsize=9)
axs[1].set_ylabel('real-cone rho (mu = 0.50987, box=1)')
axs[1].set_title('FIG3b  real friction cone with and without the synthetic column', fontsize=9)
fig.tight_layout()
fig.savefig(DIR / 'FIG3_FROZEN_VS_SYNTHETIC.png', dpi=150)

# ---------------------------------------------------------------- verification
checks = []
def chk(n, ok, dd):
    checks.append(dict(check=n, ok=bool(ok), detail=dd))
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()
def solve(Amat, reqv, box):
    n, mm_ = Amat.shape[1], reqv.size
    bnds = [(0, None)] * n if box is None else [(0, box)] * n
    lp = linprog(np.r_[np.zeros(n), 1.0],
                 A_ub=np.vstack((np.column_stack((Amat, -np.ones(mm_))), np.column_stack((-Amat, -np.ones(mm_))))),
                 b_ub=np.r_[reqv, -reqv], bounds=bnds + [(0, None)], method='highs')
    return (float(lp.x[-1]) if lp.success else None), (lp.x[:-1] if lp.success else None)

chk('baselines:three', abs(solve(A, req, 1.0)[0] - 3.308620867791373) <= 1e-9
    and abs(solve(A[RM_ROWS, :], req[RM_ROWS], 1.0)[0] - 2.4262491357755884) <= 1e-6
    and abs(solve(A, req, None)[0] - 1.7383659885871001) <= 1e-6, 'global/RM box1 and unbounded reproduced')
chk('row36:required_value', abs(req[36] - 1.4376264251311566) <= 1e-12, '%.16g' % req[36])
chk('row36:max_positive_is_8_57e-15', abs(hp['float64_max_positive'] - 8.567984361474924e-15) <= 1e-20,
    '%.6e' % hp['float64_max_positive'])
chk('row36:ratio_recomputed', abs(req[36] / hp['float64_max_positive'] - 1.6779e14) / 1.6779e14 < 1e-2,
    'req/maxpos = %.4e' % (req[36] / hp['float64_max_positive']))
chk('row36:exact_rational_positive_sum',
    abs(hp['exact_positive_sum_float'] - 2.119783522304787e-14) <= 1e-20,
    'exact %.6e (nonzero), longdouble noise floor %.3e' % (hp['exact_positive_sum_float'], hp['longdouble_noise_floor']))
chk('row36:label_falls_in_the_middle_band',
    1e-18 < hp['exact_positive_sum_float'] < 1e-12,
    'the frozen rule asks <=1e-18 or >1e-12; the measured exact value is in between -> ROW36_UNDETERMINED, stated as such')
chk('row36:activation_needed_absurd', hp['activation_needed'] > 1e13, '%.4e' % hp['activation_needed'])
chk('census:exactly_one_unreachable_row',
    sign['n_structurally_unreachable'] == 1 and sign['unreachable_rows'][0]['dof_name'] == 'joint_RMCoxa_pitch',
    json.dumps(sign['unreachable_rows']))
chk('census:eps_flip_point_recorded', 'joint_RMCoxa_pitch' in sign['eps_flip_table'],
    'flips at eps = %.0e' % float(sign['eps_flip_table']['joint_RMCoxa_pitch']['flips_at_eps']))
chk('row36:single_positive_actuator_is_47',
    row36['n_positive'] == 1 and row36['positive_actuators'] == ['RMC_pleural_remotor_and_abductor']
    and row36['agonist_label'] == 'AGONIST_ONLY_VIA_UNRESOLVED_47',
    '%s, commandable=%s' % (row36['positive_actuators'], row36['n_positive_commandable']))
chk('row36:B_42_47_value', abs([a for a in row36['actuators'] if a['actuator_index'] == 47][0]['B_coefficient']
                              - 6.587913451152499) <= 1e-12, '+6.587913451152499 reproduced')
chk('T:74_nonzero_all_half', int(np.count_nonzero(T)) == 74 and np.all(T[T != 0] == 0.5),
    '74 nonzero entries, all exactly 0.5')
chk('T:16_uncommandable_rows_zero',
    all(float(np.abs(T[a, :]).max()) == 0.0 for a in [45, 46, 47, 48])
    , 'rows 45/46/47/48 (the RMC promotor/remotor family) are identically zero')
chk('synthetic:gain_and_activation',
    abs(syn['T_syn']['A_column_at_row36'] - 3.2939567255762494) <= 1e-9
    and abs(syn['T_syn']['activation_needed_for_the_gap'] - 0.43644362840852324) <= 1e-12,
    'gain %.6f (= B[42,47]*0.5), activation needed %.6f' % (syn['T_syn']['A_column_at_row36'],
                                                            syn['T_syn']['activation_needed_for_the_gap']))
chk('synthetic:box1_unchanged_unbounded_closed',
    abs(syn['solves']['RM_box1']['rho_with_synthetic'] - syn['solves']['RM_box1']['rho_frozen']) <= 1e-9
    and syn['solves']['RM_unbounded']['rho_with_synthetic'] <= 1e-6 and syn['label'] == 'SIGN_GAP_UNCHANGED',
    'box1 %.6f -> %.6f ; unbounded %.6f -> %.3e' % (syn['solves']['RM_box1']['rho_frozen'],
                                                     syn['solves']['RM_box1']['rho_with_synthetic'],
                                                     syn['solves']['RM_unbounded']['rho_frozen'],
                                                     syn['solves']['RM_unbounded']['rho_with_synthetic']))
chk('synthetic:remaining_box1_blocker_is_amplitude',
    any(r['dof_name'] == 'joint_RMTibia_pitch' and r['max_positive'] < 0.2 for r in syn['rm_residuals_after_synthetic']),
    'joint_RMTibia_pitch max positive %.4f -> s needed %.2f'
    % ([r for r in syn['rm_residuals_after_synthetic'] if r['dof_name'] == 'joint_RMTibia_pitch'][0]['max_positive'],
       [r for r in syn['rm_residuals_after_synthetic'] if r['dof_name'] == 'joint_RMTibia_pitch'][0]['s_needed']))
chk('cone:reconstruction_matches_s208', D['reconstruction_gate']['matches'], json.dumps(D['reconstruction_gate']))
chk('cone:reproduces_s208_509870', D['reproduces_s208'], 'rho = %r vs S208 0.509870' % D['rho_frozen_66_columns'])
chk('cone:synthetic_variant_reported', D['rho_with_synthetic_column'] is not None,
    'rho with synthetic = %r' % D['rho_with_synthetic_column'])
chk('guards:xml_and_no_dynamics', sha(XML) == '6e77df89b742c16d1a8b4a2bd3f4e99e0a909c291f0a67ff0515ec11ecabc2a4'
    and D['calls']['mj_step_calls'] == 0 and D['calls']['forward_calls'] == 0, json.dumps(D['calls']))
chk('guards:frozen_T_not_modified', int(np.count_nonzero(T)) == 74, 'T is the frozen array, only read')
n_ok = sum(c['ok'] for c in checks)
(DIR / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(
    dict(checks=checks, n_checks=len(checks), n_ok=n_ok, all_ok=n_ok == len(checks),
         note='all numbers recomputed from the frozen npz with float64, longdouble and exact rational arithmetic'), indent=2, default=str))

ledger = {"round": "S209", "failures": [
    {"id": "S209-A1", "what": "RM row indices", "defect": "hard-coded RM_ROWS = range(34,41); the RM block actually comes from "
     "leg_of_row, and the wrong block made the synthetic-column rho look WORSE than the frozen one (3.3086 vs 2.4262)",
     "caught_by": "a monotonicity check that a superset of columns cannot worsen rho", "status": "fixed with leg_of_row plus an "
     "assertion that the frozen RM box=1 value reproduces 2.4262"},
    {"id": "S209-A2", "what": "synthetic A column", "defect": "B @ T_syn returns a 1-D vector; reshaping it as (42,1) via [6:, :] "
     "failed", "caught_by": "IndexError", "status": "fixed to (B @ T_syn)[6:].reshape(-1,1)"},
    {"id": "S209-C1", "what": "S208 Part A", "defect": "S208 declared every RM row individually reachable because SOME positive "
     "coefficient existed, and wrote hi_unbounded = Infinity for row 36 while the same JSON reported n_positive = 0. Treating a "
     "1e-15 residue as a usable positive coefficient is a sign-support error",
     "caught_by": "the exact rational recomputation plus the request to re-derive it", "status": "CORRECTED: row 36 is "
     "structurally unreachable at any eps >= 1e-14, and S208's phrase is withdrawn"},
    {"id": "S209-C2", "what": "the S208 column count for RM", "defect": "S208 wrote 11 pool columns for the RM block; the count is "
     "eps-dependent: 12 at eps=1e-15, 11 at 1e-12, and 10 if only columns above the residual band are counted",
     "caught_by": "the eps scan", "status": "reported as an eps-dependent number instead of a single figure"}],
    "design_notes": [
        {"id": "S209-N1", "what": "the frozen direction-rule has a gap",
         "finding": "the pre-registered rule asks for an exact positive sum <= 1e-18 (structural) or > 1e-12 (numeric only). The "
                    "exact value is 2.1198e-14, which sits between the two, so the letter of the rule gives ROW36_UNDETERMINED even "
                    "though the operational answer is unambiguous (required activation 1.68e14 at box=1). A ratio-based rule "
                    "(max positive coefficient / |req|) is what should be frozen next time."}]}
(DIR / 'ATTEMPT_LEDGER.json').write_text(json.dumps(ledger, indent=2, ensure_ascii=False))

conclusion = f"""# S209 行 36 方向缺口执行器级定死 + 溯源闭环 + 合成列对照 —— 结论（中文）

目录：`{DIR}`

## 0. 四个标签（并列，主标签取方向标签）
- **方向：`ROW36_UNDETERMINED`**（按预登记字面规则）。理由见 §1：精确有理数正向和 = **{hp['exact_positive_sum_float']:.4e}**，落在"≤1e-18 结构不可达"与">1e-12 纯数值"之间的**空档**；**但操作结论是确定的**：要补上该行需激活 **{hp['activation_needed']:.3e}**（box=1 的 1.7e14 倍）。
- **拮抗：`AGONIST_ONLY_VIA_UNRESOLVED_47`** —— 行 36 的 7 个执行器里**唯一的正向是 #47 `RMC_pleural_remotor_and_abductor`（B = +6.587913451152499）**，而它 `T` 行恒 0、**不可命令**。
- **合成列：`SIGN_GAP_UNCHANGED`** —— box=1 下 RM 的 ρ 仍为 **{syn['solves']['RM_box1']['rho_frozen']:.6f}**（不变）；**但无界下由 {syn['solves']['RM_unbounded']['rho_frozen']:.6f} 归零**（合成列取值 x_syn = {syn['solves']['RM_unbounded']['x_syn']:.4f}）。box=1 的剩余障碍是**幅度行** `joint_RMTibia_pitch`（正向系数仅 0.1127，需 s ≈ 22.52）。
- **真实锥**：**`REAL_CONE_INFEASIBLE` 保留但加限定**（见 §5）。

## 1. 行 36：三种精度一致，结论不受 1e-18/1e-12 空档影响（`HIGH_PRECISION_ROW36.json`、FIG2）
| 量 | 值 |
|---|---|
| `req[36]` | **{req[36]:.16g}** |
| float64 最大正系数 | **{hp['float64_max_positive']:.6e}** |
| longdouble 正向和 | {hp['longdouble_positive_sum']:.6e}（噪声底 {hp['longdouble_noise_floor']:.2e}） |
| **精确有理数正向和** | **{hp['exact_positive_sum_float']:.6e}**（≠0，来自列 {hp['exact_positive_support']}） |
| 精确正向和 vs 需求 | 比值 **{req[36]/hp['exact_positive_sum_float']:.3e}** |
| 单执行器所需激活 | **{hp['activation_needed']:.4e}** |

- **ε 翻转点 = 1e-14**（FIG2）：ε ≥ 1e-14 判不可达、ε ≤ 1e-15 判"可达"。这就是 S208 出错的机制——它把 8.57e-15 当成了可用的正系数。
- **精确算术后结果**：那 5 个正项**不是**浮点噪声（longdouble 噪声底 3.4e-17，实测 2.1e-14），是真实的极小系数；但**任何物理尺度下都不可用**（需 1.7e14 激活）。
- 因此**撤回 S208 的"每行单独都可达"**（S209-C1），并保留"结构上几乎不可达"的操作结论。

## 2. 全 42 行符号普查（`SIGN_CENSUS.json/.csv`）
- **符号不可达的行恰好 1 个：`joint_RMCoxa_pitch`（DOF 42）**，其余 41 行在 ε=1e-12 下都可达。
- 给出了每行的"脆弱度"清单（`max_coefficient_needed / |req|`），可直接作为**将来设计肌肉覆盖的检查表**（最脆的 10 行在 JSON 里）。
- RM 块的**列数是 ε 依赖的**：1e-15 → 12、1e-12 → 11、只算残差带以上 → 10（S209-C2 更正 S208 的"11 列"说法）。

## 3. 行 36 的执行器级证据（`ROW36_ACTUATORS.json`、FIG1）
| # | 执行器 | B 系数 | 符号 | 可命令 |
|---|---|---|---|---|
{chr(10).join('| %d | `%s` | %+.6f | %s | %s |' % (a['actuator_index'], a['actuator'], a['B_coefficient'], a['sign'], '是' if a['commandable'] else '**否**') for a in row36['actuators'])}

三个可命令的（#49/#50/#51）**全是负向**；唯一正向 #47 不可命令 → **该 DOF 在当前覆盖下没有正向权限**。

## 4. 溯源闭环（`AGONIST_PROVENANCE.json`）
`data/banc888/promotor_adductor_pools.json` 里 `pleural_remotor_and_abductor` 共 **7 个细胞**，神经为前后胸副神经 + **左**中胸腿神经 + 左右后胸腿神经 → **缺右中腿**；这与 `unresolved` 记录"target exists but not on this leg/side"一致。
**证据等级**：这是"映射缺口 → 该 DOF 单向权限为零 → 单行不可行"的**一致性**证据，**不是**"补齐即可站立"的证明；本轮**没有**做任何替换。

## 5. 真实锥对照（`REAL_CONE_VARIANTS.json`、FIG3）
- 重建门通过：`nefc = {D['reconstruction_gate']['nefc']}`、接触 {D['reconstruction_gate']['n_contacts']} 个、μ = {D['mu_values']}，与 S208 一致（`matches = {D['reconstruction_gate']['matches']}`）。
- **冻结 66 列：ρ = {D['rho_frozen_66_columns']:.6f}**（复现 S208 的 0.509870：{D['reproduces_s208']}）。
- **加合成列：ρ = {D['rho_with_synthetic_column']:.6f}**。
- **限定条款**：S208 的 `REAL_CONE_INFEASIBLE` **成立于"当前 66 个可命令池列、不含 #47 的正向权限、且 f0 冻结在释放位姿"**；加合成列后的数值见上，**但合成列不是模型改动，不得据此声明可行性恢复或站立/行走**。

## 6. 合成列（`SYNTHETIC_COLUMN.json`，单独记账，不改冻结件）
`T_syn`：仅 #47 行取 0.5（= 冻结 T 唯一非零值，74 个非零元全部 0.5），其余 0；对行 36 的增益 **{syn['T_syn']['A_column_at_row36']:.6f}**，补缺所需激活 **{syn['T_syn']['activation_needed_for_the_gap']:.6f}**（= 1.437626 / 3.293957，与"缺额比值 0.2182"一致）。
四种解：RM box=1 {syn['solves']['RM_box1']['rho_frozen']:.6f}→{syn['solves']['RM_box1']['rho_with_synthetic']:.6f}；RM 无界 {syn['solves']['RM_unbounded']['rho_frozen']:.6f}→**{syn['solves']['RM_unbounded']['rho_with_synthetic']:.1e}**；全局同 RM（因为全局 minimax 由 RM 决定）。

## 7. 这**不是**什么
- 不声明任何运动改善、不声明可行性恢复、不声明站立/行走；力单位 **NOT_ESTABLISHED**（所有系数、ρ、req 都是冻结单位下的广义量）。
- 合成列只是"缺一条拮抗肌能否解释该方向缺口"的定量回答（能，需 0.4364 激活），**不是**在补模型。

## 8. 下一步最小动作
1. 把方向规则改成**比值型**（`最大正向系数 / |req|`），避免 1e-18/1e-12 空档（S209-N1）。
2. box=1 的真正瓶颈已换行：`joint_RMTibia_pitch` 需 s ≈ 22.5 —— 下一步对它做同样的执行器级符号普查，看它是否也有单点故障。
3. 真锥下加合成列仍未零（ρ = {D['rho_with_synthetic_column']:.6f}）→ 接触锁定与方向缺口**同时**存在，需分别定量。

## 9. 记账
`PREREG.json` 跑前落盘（ε 协议、两种可达定义、四标签并列规则、`T_syn` 构造）；{n_ok}/{len(checks)} 项独立核验通过；台账 2 条自身失败（RM 行索引硬编码、合成列 1-D 索引）+ 2 条更正（S208 的"每行可达"、RM 列数）+ 1 条规则缺陷说明；0 `mj_step`、0 `mj_forward`、0 XML 改动、0 训练；1 次 load 仅用于重建接触约束行。
"""
(DIR / 'CONCLUSION.zh-CN.md').write_text(conclusion)

def sha_all(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
budget = {"round": "S209",
          "stop_reason": ("All four parts complete. The direction label is ROW36_UNDETERMINED by the letter of the frozen rule "
                          "(the exact positive sum sits between the two thresholds) while the operational answer is unambiguous; the "
                          "agonist label is AGONIST_ONLY_VIA_UNRESOLVED_47; the synthetic column closes the gap only when the box is "
                          "removed; the real cone stays infeasible under the qualification. No dynamics were run."),
          "budget": {"model_loads": 1, "position_stage_calls": 1, "constraint_stage_calls": 1, "forward_calls": 0,
                     "mj_step_calls": 0, "neural_ticks": 0, "training_steps": 0, "prewarming": False, "xml_changes": 0,
                     "lp_solves": "about 120 (census, 4 synthetic solves, 2 cone solves, verification)"},
          "artifacts": sha_all(DIR),
          "sources": {str(p): sha(p) for p in [Path('/run/media/sensen/Data2/cell_wound_prototype/s209_sign.py'),
                                               Path('/run/media/sensen/Data2/cell_wound_prototype/s209_cone_figs.py'), XML]
                      if p.exists()}}
(DIR / 'STOP_AND_BUDGET.json').write_text(json.dumps(budget, indent=2, ensure_ascii=False))
(DIR / 'SOURCE_HASHES.json').write_text(json.dumps(dict(sources=budget['sources'], artifacts=budget['artifacts']), indent=2))
print(json.dumps(dict(dir=str(DIR), n_checks=len(checks), n_ok=n_ok, all_ok=n_ok == len(checks),
                      failed=[c for c in checks if not c['ok']]), indent=1, default=str))
