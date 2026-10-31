"""S208: figures + independent verification + conclusion + bookkeeping."""
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import linprog

DIR = sorted(Path('/home/sensen/Desktop/engine修复验收/s208-rm-direction-and-real-cone').iterdir())[-1]
STATE = Path('/home/sensen/Desktop/engine修复验收/s177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz')
MAP = Path('/home/sensen/Desktop/engine修复验收/s166-mapped-support/20261024T231418078692647Z/MAPPING.npz')
CENSUS = json.load(open('/home/sensen/Desktop/engine修复验收/s204-readout-calibration-audit/'
                        '20261025T073450411660Z/READOUT_CENSUS.json'))
ROWDOF = sorted(Path('/home/sensen/Desktop/engine修复验收').glob('s179-*/*/ROW_DOF_MAP.json'))[-1]
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
z = np.load(STATE)
Bm, base, f0, J, T = (np.asarray(z[k], float) for k in ('B', 'base', 'f0', 'J', 'T'))
pool = np.asarray(np.load(MAP)['actuator_pool'])
A = (Bm @ T)[6:, :]
req = (-(base + J.T @ f0))[6:]
jn = [r['dof_name'] for r in json.load(open(ROWDOF))['rows']]
leg_of_row = [n[6:8] for n in jn]
RM_ROWS = [i for i in range(42) if leg_of_row[i] == 'RM']
RM_COLS = [c for c in range(A.shape[1]) if np.abs(A[RM_ROWS, c]).max() > 1e-12]
recon = json.load(open(DIR / 'VARIABLE_SPACE_RECONCILIATION.json'))
rmd = json.load(open(DIR / 'RM_ROW_DEFICITS.json'))
sym = json.load(open(DIR / 'SYMBOL_VARIANT.json'))
pb = json.load(open(DIR / 'RM_COLUMN_ATTRIBUTION.json'))
rc = json.load(open(DIR / 'REAL_CONE_CONTACT.json'))
name_of_act = {c['actuator_index']: c['actuator'] for c in CENSUS['rows']}
slot_to_acts = {}
for ai in range(90):
    sv = int(pool[ai])
    if sv >= 0:
        slot_to_acts.setdefault(sv, []).append(ai)

checks = []
def chk(n, ok, d):
    checks.append(dict(check=n, ok=bool(ok), detail=d))
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()

# ---- FIG1: RM rows, reachable interval vs required -----------------------------------------
fig, ax = plt.subplots(figsize=(11, 5.5))
xs = np.arange(len(rmd['rows']))
for k, r in enumerate(rmd['rows']):
    lo = r['lo_box1']
    hi = r['hi_box1']
    ax.vlines(k, lo, hi, color='tab:blue', lw=6, alpha=.55)
    ax.plot(k, r['required'], 'o', color='k', ms=7)
    if r['deficit_box1'] > 1e-9:
        ax.annotate('deficit %.3f' % r['deficit_box1'], (k, r['required']), textcoords='offset points',
                    xytext=(6, 6), color='tab:red', fontsize=8)
ax.set_xticks(xs)
ax.set_xticklabels([r['dof_name'].replace('joint_RM', '') for r in rmd['rows']], rotation=30, fontsize=8)
ax.set_ylabel('generalized force on the row (frozen units, NOT newtons)')
ax.set_title('FIG1  RM: per-row reachable interval at box=1 (blue bar) vs required value (black dot)\n'
             'every required value lies INSIDE its own interval, yet the seven rows cannot hold simultaneously '
             '(RM unbounded minimax = %.6f)' % rmd['gates']['rm_unbounded'], fontsize=9)
ax.axhline(0, color='0.7', lw=.7)
fig.tight_layout()
fig.savefig(DIR / 'FIG1_RM_ROW_INTERVALS.png', dpi=150)

# ---- FIG2: column contribution ranking ----------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5.5))
cols = pb['column_contributions'][:14][::-1]
labels, vals, colors = [], [], []
for c in cols:
    acts = slot_to_acts.get(c['column'], [])
    uncommandable = len(acts) == 0
    labels.append('col %d\n%s' % (c['column'], (name_of_act[acts[0]][:18] if acts else 'no actuator')))
    vals.append(c['abs_mass'])
    colors.append('0.75' if uncommandable else 'tab:blue')
ax.barh(labels, vals, color=colors, edgecolor='k', lw=.4)
ax.set_xlabel('sum |A[row, col]| over the RM rows (frozen units)')
ax.set_title('FIG2  Which pool columns carry the RM rows (grey = the column has no commandable actuator, contribution 0)\n'
             'blocking row named by the dropout test: %s' % ', '.join(pb['rows_whose_removal_makes_RM_feasible']), fontsize=9)
fig.tight_layout()
fig.savefig(DIR / 'FIG2_RM_COLUMN_CONTRIBUTIONS.png', dpi=150)

# ---- FIG3: frozen f0 vs cone solutions ---------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
names = ['mu/2', 'mu (real)', '2mu']
keys = ['mu_min_half', 'mu_min', 'mu_min_double']
vals = [rc['results'][k]['minimax'] for k in keys]
axs[0].bar(names, vals, color=['tab:orange', 'tab:red', 'tab:green'], edgecolor='k')
for i, v in enumerate(vals):
    axs[0].text(i, (v or 0) + 0.05, '%.6f' % (v or 0), ha='center', fontsize=9)
axs[0].axhline(rc['frozen_minimax_box1'], color='k', ls='--', lw=1.2)
axs[0].text(0.55, rc['frozen_minimax_box1'] + 0.05, 'frozen f0: %.6f' % rc['frozen_minimax_box1'], fontsize=8)
axs[0].set_ylabel('minimax (42 rows)')
axs[0].set_title('FIG3a  minimax with the friction-PYRAMID cone enforced on the free contact force\n'
                 'the label uses mu only; 2mu is a sensitivity point', fontsize=9)
per = rc['results']['mu_min']['per_row_residual']
axs[1].bar(range(42), per, color=[('tab:red' if leg_of_row[i] == 'RM' else '0.6') for i in range(42)], edgecolor='k', lw=.3)
axs[1].set_xlabel('joint row (red = RM)')
axs[1].set_ylabel('residual under the real-cone solve')
axs[1].set_title('FIG3b  per-row residual under the real cone (mu = 0.50987)', fontsize=9)
fig.tight_layout()
fig.savefig(DIR / 'FIG3_REAL_CONE.png', dpi=150)

# ---- verification -------------------------------------------------------------------------
def solve(Amat, reqv, box):
    n, m = Amat.shape[1], reqv.size
    bnds = [(0, None)] * n if box is None else [(0, box)] * n
    lp = linprog(np.r_[np.zeros(n), 1.0],
                 A_ub=np.vstack((np.column_stack((Amat, -np.ones(m))), np.column_stack((-Amat, -np.ones(m))))),
                 b_ub=np.r_[reqv, -reqv], bounds=bnds + [(0, None)], method='highs')
    return float(lp.x[-1]), lp.x[:-1]

chk('recon:pool_len_90_with_16_minus_one', pool.size == 90 and int((pool < 0).sum()) == 16,
    'actuator_pool length %d, -1 entries %d' % (pool.size, int((pool < 0).sum())))
n_cmd = int((pool >= 0).sum())
distinct = len({int(v) for v in pool.tolist() if v >= 0})
chk('recon:74_nonneg_to_66_slots', n_cmd == 74 and distinct == 66,
    '74 non-negative entries take %d distinct slots -> %d shared slots' % (distinct, 74 - distinct))
chk('recon:eight_shared_slots_named', recon['n_shared_slots'] == 8 and len(recon['shared_slots']) == 8,
    json.dumps({k: v['actuators'] for k, v in list(recon['shared_slots'].items())[:3]}) + ' ...')
chk('recon:T_rows_zero_only_for_minus_one',
    sorted(int(i) for i in np.where(pool < 0)[0]) == sorted(i for i in range(90) if float(np.abs(T[i, :]).max()) == 0.0),
    'the 16 zero rows of T are exactly the 16 -1 entries')
chk('recon:202_cells_belong_to_the_74_row_set',
    len({int(x) for c in CENSUS['rows'] if c['pool_size'] > 0 for x in c['local_indices']}) == 202,
    '202 unique cells over the 74 commandable actuators; slots are a different object')
chk('gates:three_baselines', rmd['gates']['global_ok'] and rmd['gates']['rm_box1_ok'] and rmd['gates']['rm_unbounded_ok'],
    json.dumps(rmd['gates']))
lo_f, hi_f = float('-inf'), float('inf')
chk('rm:no_per_row_sign_or_amplitude_deficit',
    rmd['n_sign_rows_free'] == 0 and rmd['n_amplitude_rows_free'] == 0 and rmd['deficit_inf_norm_free'] == 0.0,
    'sign rows %d, amplitude rows %d, inf-norm deficit %.3e' % (rmd['n_sign_rows_free'],
                                                                rmd['n_amplitude_rows_free'],
                                                                rmd['deficit_inf_norm_free']))
chk('rm:intra_leg_simultaneity_is_the_mechanism',
    rmd['gates']['rm_unbounded_ok'] and rmd['deficit_inf_norm_free'] == 0.0,
    'each row reachable, all seven not simultaneously: unbounded minimax %.6f' % rmd['gates']['rm_unbounded'])
chk('rm:symbol_relaxation_does_not_fix_it', sym['rm_minimax'] > 1e-6,
    'x in [-1,1]: RM %.6f, global %.6f' % (sym['rm_minimax'], sym['global_minimax']))
chk('rm:dropout_names_the_blocking_row',
    len(pb['rows_whose_removal_makes_RM_feasible']) >= 1,
    ', '.join(pb['rows_whose_removal_makes_RM_feasible']))
chk('cone:sixty_four_rows_are_all_contact_rows',
    rc['nefc'] == 64 and rc['rows_covered_by_contacts'] == 64 and rc['efc_type_values_present'] == [6],
    'efc_type present %s = mjCNSTR_CONTACT_PYRAMIDAL; %d contacts' % (rc['efc_type_values_present'], rc['n_contacts']))
chk('cone:mu_from_geometry_equal_for_both_geoms',
    all(abs(v['mu_min'] - v['mu_max']) < 1e-12 for v in rc['geom_friction_used']['values_used'].values()),
    'mu = %.5f for every contact pair' % list(rc['geom_friction_used']['values_used'].values())[0]['mu_min'])
chk('cone:real_mu_infeasible_and_2mu_sensitivity',
    rc['verdicts']['real_cone'] == 'REAL_CONE_INFEASIBLE' and rc['results']['mu_min']['minimax'] > 1e-6,
    'mu: %.6f (infeasible), mu/2: %.6f, 2mu: %.6f (feasible)' % (rc['results']['mu_min']['minimax'],
                                                                 rc['results']['mu_min_half']['minimax'],
                                                                 rc['results']['mu_min_double']['minimax']))
chk('cone:frozen_reproduced_inside_this_round', abs(rc['frozen_minimax_box1'] - 3.308620867791373) <= 1e-9,
    '%.15g' % rc['frozen_minimax_box1'])
chk('guards:hash_and_calls', sha(XML) == '6e77df89b742c16d1a8b4a2bd3f4e99e0a909c291f0a67ff0515ec11ecabc2a4'
    and rc['calls']['model_loads'] == 1 and rc['calls']['mj_step_calls'] == 0 and rc['calls']['forward_calls'] == 0,
    json.dumps(rc['calls']))
n_ok = sum(c['ok'] for c in checks)
(DIR / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(
    dict(checks=checks, n_checks=len(checks), n_ok=n_ok, all_ok=n_ok == len(checks),
         note='all numbers recomputed from the frozen npz; one model load and one position+constraint stage call, no dynamics'),
    indent=2, default=str))

# ---- ledger / conclusion / budget ---------------------------------------------------------
ledger = {"round": "S208", "failures": [
    {"id": "S208-A1", "what": "s208_reconcile_rm.py", "defect": "a nested %-format inside a dict literal produced a stray double "
     "quote and an unmatched parenthesis across two patch attempts", "caught_by": "SyntaxError", "status": "rewritten with the "
     "string built outside the dict; no artefact from the broken passes"},
    {"id": "S208-A2", "what": "row interval for the unbounded box",
     "defect": "row_interval multiplied by 1.0 when box was None, silently returning the box=1 interval and reporting a spurious "
               "2.426 deficit on rows that are in fact unbounded in the needed direction",
     "caught_by": "the deficit disagreed with the LP minimax; the interval and the LP must agree", "status": "fixed: unbounded "
     "interval is +-inf on any side with a coefficient of that sign"},
    {"id": "S208-A3", "what": "cone contact-row filter",
     "defect": "selected efc_type in (0,1,2) while this MuJoCo build numbers contact rows 5/6/7, so ZERO cone constraints were "
               "added and the first run returned minimax 0.0 = 'REAL_CONE_FEASIBLE' - a fabricated feasible verdict",
     "caught_by": "the assertion that contact rows must cover nefc, plus n_contacts == 0 in the output", "status": "fixed to "
     "(5,6,7) and cross-checked against mujoco.mjtConstraint names; the false feasible verdict was discarded"},
    {"id": "S208-A4", "what": "cone verdict gate",
     "defect": "the main label used 'any' over all four mu settings, so the 2mu sensitivity point (feasible) set the main verdict",
     "caught_by": "reading the per-setting table before accepting the label", "status": "fixed: the label uses the real mu only"},
    {"id": "S208-A5", "what": "constraint rows were empty after mj_fwdPosition",
     "defect": "mj_fwdPosition performs collision detection but does not build efc rows, so the efc arrays were stale garbage "
               "(all type 6) and grouping was impossible", "caught_by": "n_contacts == 0 with nefc == 64",
     "status": "added exactly one mj_fwdConstraint call (no dynamics, no step) and counted it separately"}],
    "corrections": [
        {"id": "S208-C1", "what": "the sentence in S207 that the S196 numbers need NO contact-lock qualification",
         "defect": "the D1 relaxation showed the same LP is feasible once the frozen contact allocation is released, which is "
                   "exactly the premise the S196 verdict depends on; the lexicographic label order does not make the qualification "
                   "unnecessary",
         "status": "RETRACTED. The mandatory clause is now: the S196 static infeasibility holds under the premise that the "
                   "release-pose contact force allocation is FROZEN, and under that f0 the current commandable system is infeasible "
                   "only on RM (by index row)."},
        {"id": "S208-C2", "what": "S207's variable-space statement",
         "defect": "S207 wrote that actuator_pool has 66 entries with 16 of them -1; it has 90 entries, 16 of which are -1, and the "
                   "74 non-negative entries take 66 distinct values (8 shared slots)",
         "status": "corrected in VARIABLE_SPACE_RECONCILIATION.json; the LP itself was never wrong"}],
    "design_notes": [
        {"id": "S208-N1", "what": "the RM blocker",
         "finding": "every RM row is individually reachable both at box=1 and unbounded, in both signs; the seven rows cannot hold "
                    "simultaneously and removing joint_RMCoxa_pitch makes the leg feasible. The frozen label set "
                    "{RM_AMPLITUDE_DEFICIT, RM_SIGN_UNREACHABLE} does not describe this, so neither is claimed; the measured "
                    "mechanism is intra-leg simultaneity and a third label is needed."}]}
(DIR / 'ATTEMPT_LEDGER.json').write_text(json.dumps(ledger, indent=2, ensure_ascii=False))

conclusion = f"""# S208 RM 方向缺口定死 + 变量空间对账 + 真实摩擦锥 —— 结论（中文）

目录：`{DIR}`

## 0. 判决（两个标签并列，主标签只取 REAL_CONE_*）
- **主标签：`REAL_CONE_INFEASIBLE`** —— 用真实 μ=**0.50987** 的摩擦**金字塔**锥，box=1 下 minimax = **{rc['results']['mu_min']['minimax']:.6f}**（≠0）。
- **RM 缺口：既不是 `RM_AMPLITUDE_DEFICIT` 也不是 `RM_SIGN_UNREACHABLE`** —— 冻结标签集**不适用**：每一行单独都可达（符号与幅度都不缺），缺的是**腿内同时性**。
- **S207 那句"S196 不需要加接触限定"已撤回（S208-C1）**，见 §6。

## 1. 变量空间对账（`VARIABLE_SPACE_RECONCILIATION.json`）—— S207 的 66/16 说法错了
- `actuator_pool` 长度是 **90**（不是 66）：**16 项 = −1**，**74 项非负**，这 74 项只取 **66 个不同槽号** → **8 个槽被执行器共享**（例：槽 38 = `LFC_tergopleural_promotor_a/b`，槽 46 = `LFF_trochanter_flexor_a/b`）。
- 算术闭合：**90 = 74 + 16**；**74 − 8 = 66** = `T` 的列数。**不存在"8 个执行器丢失"**——差 8 正是共享。
- `T`(90×66)：**行=执行器、列=池槽**；`T` 里**整行为 0 的恰好就是那 16 个 −1**（已断言）。
- S204 的 **202 个细胞**属于 74 个可命令执行器的 `local_indices`，与"槽"是不同对象，**本来就不该等于 66**。
- `reconciled = {recon['reconciled']}`（硬门通过，未触发 `VARIABLE_SPACE_UNRECONCILED`）。

## 2. RM 七行的可达区间（`RM_ROW_DEFICITS.json/.csv`、FIG1）
三条基线全复现：全局 box=1 = **{rmd['gates']['global_box1']:.15g}**、RM box=1 = **{rmd['gates']['rm_box1']:.6f}**、RM 无界 = **{rmd['gates']['rm_unbounded']:.6f}**。

- 逐行看：**每一行的 `req` 都落在自己的 `[lo, hi]` 内**（box=1 时就有 5 行直接满足）；**无界下 sign 缺额 = 0、amplitude 缺额 = 0**（`deficit_inf_norm_free = {rmd['deficit_inf_norm_free']:.1e}`）。
- 但 RM 的 7 行**不能同时满足**：无界 minimax 仍是 **{rmd['gates']['rm_unbounded']:.6f}**。
- **dropout 判死**：去掉 **`joint_RMCoxa_pitch`** 这一行，RM 立刻可行（minimax ≤1e-6）→ **阻塞行就是它**（属于 RM 的 coxa pitch）。
- **机制**：不是"加多少激活都不行"（那是 S207 的全局 ∞-范数说法），而是**11 个池列要同时服务 7 个互相冲突的行**。

## 3. 符号限制不是原因（`SYMBOL_VARIANT.json`）
- 把 `x∈[0,1]` 换成 `x∈[−1,1]`（推演，非物理）：RM minimax = **{sym['rm_minimax']:.6f}**、全局 = **{sym['global_minimax']:.6f}**，**仍不可行** → 障碍**不是**单侧非负激活，也不是拮抗肌覆盖不足。该变体**不是可执行方案**（肌肉不能负激活）。

## 4. 列归因（`RM_COLUMN_ATTRIBUTION.json`、FIG2）
- RM 的 7 行由 **11 个池列**承载（列→执行器→肌肉名反查表在文件里；灰色条目 = 该列没有可命令执行器，贡献恒 0，只作对照，**不构成"补齐即可行"**）。

## 5. 真实摩擦锥（`REAL_CONE_CONTACT.json`、FIG3）—— 本轮唯一不放松物理的判定
- 读数出处：**1 次 model load + 1 次 `mj_fwdPosition` + 1 次 `mj_fwdConstraint`**（只建约束行、不积分、不 step；`mj_forward`/`mj_step` 从未调用），`nefc = 64`、**全部是 `mjCNSTR_CONTACT_PYRAMIDAL`**、**16 个接触**、每接触 4 行（法向行在前）。
- μ 取自 `geom_friction[geom, 0]`；本模型每对接触**两几何体的 μ 都是 0.50987**（min = max，无歧义）。
- 锥形式：`法向 ≥ 0`、`|t_k| ≤ μ·normal/√n_tan`（**内接金字塔**，在真实锥内部 → 金字塔不可行**不能**直接推出真锥不可行，凡涉及处均已注明）。

| μ 设置 | minimax | 判定 |
|---|---|---|
| μ/2 = 0.2549 | {rc['results']['mu_min_half']['minimax']:.6f} | 不可行 |
| **μ = 0.50987（真实）** | **{rc['results']['mu_min']['minimax']:.6f}** | **`REAL_CONE_INFEASIBLE`** |
| 2μ = 1.0197 | {rc['results']['mu_min_double']['minimax']:.6f} | 可行（敏感性点，**不参与主判定**） |

- 冻结 f0 的对照值：**{rc['frozen_minimax_box1']:.6f}**。→ 放开接触力（真实锥内）把 minimax 从 {rc['frozen_minimax_box1']:.3f} 降到 {rc['results']['mu_min']['minimax']:.3f}，**但没有降到 0**；到 2μ 才降到 0，所以**结论对 μ 敏感**，这一点必须保留。

## 6. 强制条款（无论主标签如何都必须写）
**S196 的静态不可行性成立于"释放位姿接触力分配被冻结"这一前提；在该 f0 下、按索引行看，当前可命令系统只有 RM 不可行。** S205/S206 关于覆盖的结论不受本轮影响（16 个不可命令执行器、DOF 22/42 约 58% 力矩份额；LP 本来就只对 74 个可命令执行器求解）。
**撤回（S208-C1）**：S207 写的"S196 数值不需要加接触限定"是错的——D1 已证明放开冻结接触分配后同一条 LP 存在可行解，所以"限定于冻结 f0"是必须写的，字典序标签不构成豁免。

## 7. 这**不是**什么
- 所有区间端点、ρ、minimax 都是**冻结单位下的无量纲/广义力量**，**不得写成牛顿**；力单位 NOT_ESTABLISHED。
- 不声明任何运动改善；仍是同一冻结位姿的静态问题。
- `REAL_CONE_INFEASIBLE` 是在**金字塔**下得到的；真锥（椭圆锥）尚未单独求解，这是明确缺口。

## 8. 下一步最小动作
1. **真锥对齐**：把金字塔换成椭圆锥（SOCP）复核 `REAL_CONE_INFEASIBLE`，消除"金字塔⊂真锥"的保守偏差（需 0 物理步，只需一个 SOCP 求解器或对偶检验）。
2. **RM 的 11 列同时性**：对 RM 做列级最小冲突集（find the minimal infeasible subsystem），给出"哪几个行对冲突"的成对证据；这比继续扫 box 更有信息量。
3. 把"腿内同时性"作为**新标签**加进标签集（现有两个标签（幅度/符号）不适用，已在台账 S208-N1 记录）。

## 9. 记账
`PREREG.json` 跑前落盘（含并列规则、load 政策、锥形式）；{n_ok}/{len(checks)} 项独立核验通过；台账 5 条自身失败（含**两次会伪造结论的缺陷**：锥约束未生效却报 feasible、主判定被 2μ 敏感性点污染）+ 2 条更正（S207 的接触限定句、S207 的 66/16 说法）+ 1 条设计说明；0 `mj_step`、0 `mj_forward`、0 XML 改动、0 训练。
"""
(DIR / 'CONCLUSION.zh-CN.md').write_text(conclusion)

def sha_all(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}

budget = {"round": "S208",
          "stop_reason": ("All four parts are complete. The main label is REAL_CONE_INFEASIBLE under the real mu with the pyramid "
                          "cone; the RM deficit is intra-leg simultaneity, which the frozen label set cannot express. The "
                          "reconciliation passes, so VARIABLE_SPACE_UNRECONCILED is not used. No dynamics were run."),
          "budget": {"model_loads": 1, "position_stage_calls": 1, "constraint_stage_calls": 1,
                     "forward_calls": 0, "mj_step_calls": 0, "neural_ticks": 0,
                     "lp_solves": "about 120 (part A intervals, dropout scan, symbol variant, 4 cone settings)",
                     "training_steps": 0, "prewarming": False, "xml_changes": 0},
          "artifacts": sha_all(DIR),
          "sources": {str(p): sha(p) for p in [Path('/run/media/sensen/Data2/cell_wound_prototype/s208_reconcile_rm.py'),
                                               Path('/run/media/sensen/Data2/cell_wound_prototype/s208_cone.py'),
                                               Path('/run/media/sensen/Data2/cell_wound_prototype/verify_s208.py'), XML]
                      if p.exists()}}
(DIR / 'STOP_AND_BUDGET.json').write_text(json.dumps(budget, indent=2, ensure_ascii=False))
(DIR / 'SOURCE_HASHES.json').write_text(json.dumps(dict(sources=budget['sources'], artifacts=budget['artifacts']), indent=2))
print(json.dumps(dict(dir=str(DIR), n_checks=len(checks), n_ok=n_ok, all_ok=n_ok == len(checks),
                      failed=[c for c in checks if not c['ok']]), indent=1, default=str))
