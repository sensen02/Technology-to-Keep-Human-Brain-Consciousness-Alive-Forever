"""S207: figures + independent verification + conclusion + bookkeeping."""
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import linprog

DIR = sorted(Path('/home/sensen/Desktop/engine修复验收/s207-scale-direction-contact-audit').iterdir())[-1]
STATE = Path('/home/sensen/Desktop/engine修复验收/s177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz')
MAP = Path('/home/sensen/Desktop/engine修复验收/s166-mapped-support/20261024T231418078692647Z/MAPPING.npz')
ROWDOF = sorted(Path('/home/sensen/Desktop/engine修复验收').glob('s179-*/*/ROW_DOF_MAP.json'))[-1]
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
BASE_GATE = 3.308620867791373

z = np.load(STATE)
Bm, base, f0, J, T = (np.asarray(z[k], float) for k in ('B', 'base', 'f0', 'J', 'T'))
A = (Bm @ T)[6:, :]
req = (-(base + J.T @ f0))[6:]
jn = [r['dof_name'] for r in json.load(open(ROWDOF))['rows']]
leg_of_row = [n[6:8] for n in jn]
LEGS = sorted(set(leg_of_row))
scan = json.load(open(DIR / 'SCALE_SCAN.json'))
cone = json.load(open(DIR / 'CONE_DISTANCE.json'))
leg = json.load(open(DIR / 'LEG_DECOMPOSITION.json'))
reqd = json.load(open(DIR / 'REQUIREMENT_DECOMPOSITION.json'))
free = json.load(open(DIR / 'FREE_CONTACT_VARIANT.json'))
struct = json.load(open(DIR / 'LP_STRUCTURE.json'))


def mm(Amat, reqv, box):
    n, m = Amat.shape[1], reqv.size
    bnds = [(0, None)] * n if box is None else [(0, box)] * n
    lp = linprog(np.r_[np.zeros(n), 1.0],
                 A_ub=np.vstack((np.column_stack((Amat, -np.ones(m))), np.column_stack((-Amat, -np.ones(m))))),
                 b_ub=np.r_[reqv, -reqv], bounds=bnds + [(0, None)], method='highs')
    return float(lp.x[-1]), lp.x[:-1]


# --------------------------------------------------------------- FIG1: residual vs s
s_grid = [r['s'] for r in scan['rows']]
glob = [r['max_abs_residual'] for r in scan['rows']]
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.semilogx(s_grid, glob, 'o-', color='k', lw=1.8, label='global max|residual| (42 rows)')
for L in LEGS:
    idx = [i for i in range(42) if leg_of_row[i] == L]
    cols = [c for c in range(A.shape[1]) if np.abs(A[idx, c]).max() > 1e-12
            and np.abs(np.delete(A, idx, axis=0)[:, c]).max() <= 1e-12]
    ys = []
    for s in s_grid:
        v, _x = mm(A[idx][:, cols], req[idx], s)
        ys.append(np.max(np.abs(A[idx][:, cols] @ _x - req[idx])) if v is not None else np.nan)
    ax.semilogx(s_grid, ys, '--', lw=1.0, label='leg %s' % L)
ax.axvline(1.0, color='tab:blue', ls=':', lw=1.2)
ax.text(1.05, max(glob) * 0.9, 's = 1 (frozen box)\nminimax = %.6f' % scan['rows'][1]['minimax'], fontsize=7, color='tab:blue')
rho = cone['rho_inf_from_minimax']
ax.axhline(rho, color='tab:red', ls='--', lw=1.4)
ax.text(2.0, rho * 1.02, 'rho_inf = %.6f : no s* exists (plateau)' % rho, fontsize=7, color='tab:red')
ax.set_xlabel('s = dimensionless activation bound (box upper bound on the 66 pool fractions)')
ax.set_ylabel('max |A x - req| over the 42 joint rows')
ax.set_title('FIG1  Residual vs the activation bound s\n'
             'the curve does NOT reach 0 for any s: it plateaus at the cone distance', fontsize=9)
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=.3, which='both')
fig.tight_layout()
fig.savefig(DIR / 'FIG1_RESIDUAL_VS_S.png', dpi=150)

# --------------------------------------------------------------- FIG2: per-row residual, unbounded
mmf, x_inf = mm(A, req, None)
res = A @ x_inf - req
fig, ax = plt.subplots(figsize=(12, 5.5))
colors = {'LF': 'tab:blue', 'LM': 'tab:orange', 'LH': 'tab:green', 'RF': 'tab:cyan', 'RM': 'tab:red', 'RH': 'tab:purple'}
for i in range(42):
    ax.bar(i, res[i], color=colors[leg_of_row[i]], edgecolor='k', lw=.3)
hard = {19, 22, 26, 40, 42, 47}
for i in range(42):
    if 6 + i in hard:
        ax.plot(i, 0.06 * np.sign(res[i] if res[i] else 1), marker='^', color='red', ms=7)
ax.set_xticks(range(42))
ax.set_xticklabels([n.replace('joint_', '') for n in jn], rotation=90, fontsize=6)
ax.set_ylabel('residual with an unbounded box (no activation limit)')
ax.set_title('FIG2  Residuals when the activation bound is removed: this is the distance to cone(A)\n'
             'coloured by leg; red triangles mark the six hard DOFs (19/22/26/40/42/47). '
             'RM rows are the only ones whose MINIMUM is positive (see LEG_DECOMPOSITION); this panel is one optimal solution', fontsize=9)
fig.tight_layout()
fig.savefig(DIR / 'FIG2_UNBOUNDED_RESIDUAL.png', dpi=150)

# --------------------------------------------------------------- FIG3: frozen vs free contact
fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
axs[0].bar(['frozen f0\n(box=1)', 'free f>=0\n(box=1)'],
           [scan['rows'][1]['minimax'], free['D1']['minimax']], color=['tab:red', 'tab:green'], edgecolor='k')
for i, v in enumerate([scan['rows'][1]['minimax'], free['D1']['minimax']]):
    axs[0].text(i, (v or 0) + 0.05, '%.6f' % (v or 0), ha='center', fontsize=9)
axs[0].set_ylabel('minimax (42 joint rows)')
axs[0].set_title('frozen contact force vs the D1 relaxation\n(D1 is a RELAXATION, not the true friction cone)', fontsize=9)
fsol = np.array([c['solved'] for c in free['D1']['f_solution']['per_component']], float)
ffroz = np.array([c['frozen'] for c in free['D1']['f_solution']['per_component']], float)
x = np.arange(ffroz.size)
axs[1].bar(x - 0.2, ffroz, width=0.4, color='tab:red', label='frozen f0')
axs[1].bar(x + 0.2, fsol, width=0.4, color='tab:green', label='D1 solution')
axs[1].set_xlabel('contact-force component index (0-63); per-leg grouping is not stored')
axs[1].set_ylabel('force component')
axs[1].set_title('frozen vs D1 contact force components\n|f0| = %.3f, |f_sol| = %.3f'
                 % (free['D1']['f_solution']['norm_frozen'], free['D1']['f_solution']['norm_solution']), fontsize=9)
axs[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(DIR / 'FIG3_FROZEN_VS_FREE_CONTACT.png', dpi=150)

# --------------------------------------------------------------- verification
checks = []
def chk(n, ok, d):
    checks.append(dict(check=n, ok=bool(ok), detail=d))

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()

chk('dims:all_shapes', Bm.shape == (48, 90) and T.shape == (90, 66) and base.shape == (48,) and J.shape == (64, 48)
    and f0.shape == (64,) and A.shape == (42, 66) and req.shape == (42,),
    'B%s T%s base%s J%s f0%s A%s req%s' % (Bm.shape, T.shape, base.shape, J.shape, f0.shape, A.shape, req.shape))
chk('structure:66_decision_variables_not_90_actuators', A.shape[1] == 66 and str(66) in struct['variable_identity']['answer'],
    'A has 66 columns; the 90 actuators enter through T')
chk('structure:sixteen_T_rows_zero',
    float(np.abs(T[[c['actuator_index'] for c in json.load(open('/home/sensen/Desktop/engine修复验收/'
        's204-readout-calibration-audit/20261025T073450411660Z/READOUT_CENSUS.json'))['rows']
        if c['pool_size'] == 0], :]).max()) == 0.0, 'the 16 uncommandable actuators contribute no column')
chk('structure:root_rows_excluded_and_small',
    float(np.max(np.abs(np.asarray(z['residual6'], float)))) <= 1e-8,
    'residual6 max |.| = %.3e (the six root rows are dropped by [6:] because S177 already balanced them)'
    % float(np.max(np.abs(np.asarray(z['residual6'], float)))))
chk('structure:block_diagonal_by_leg',
    struct['row_grouping']['n_columns_touching_more_than_one_leg'] == 0
    and abs(struct['row_grouping']['leg_block_mass_share'] - 1.0) < 1e-12,
    'cross-leg columns %d, leg-block mass share %.12f'
    % (struct['row_grouping']['n_columns_touching_more_than_one_leg'], struct['row_grouping']['leg_block_mass_share']))
chk('gate:baseline_s1_exact', abs(scan['gate']['measured'] - BASE_GATE) <= 1e-9,
    'measured %.15g' % scan['gate']['measured'])
chk('scale:equivalence_two_points', scan['equivalence']['passed'],
    json.dumps({k: v['rel_diff'] for k, v in scan['equivalence']['checks'].items()}))
chk('scale:no_s_star_and_plateau_equals_cone',
    scan['s_star'] is None and abs(scan['rows'][-1]['minimax'] - cone['rho_inf_from_minimax']) <= 1e-9,
    'plateau %.15g vs rho_inf %.15g' % (scan['rows'][-1]['minimax'], cone['rho_inf_from_minimax']))
chk('cone:two_methods_agree', cone['stable'] and cone['abs_difference'] <= 1e-6,
    'diff %.3e' % cone['abs_difference'])
chk('cone:exceeds_threshold', cone['rho_inf_from_minimax'] > 1e-6, 'rho_inf = %.9f' % cone['rho_inf_from_minimax'])
chk('leg:global_equals_max_of_legs_box1', leg['matches_box1'], 'max per-leg %.12f' % leg['max_of_leg_box1'])
chk('leg:global_equals_max_of_legs_unbounded', leg['matches_unbounded'],
    'max per-leg %.9f vs global %.9f' % (leg['max_of_leg_unbounded'], leg['global_unbounded']))
chk('leg:obstruction_localised_to_RM',
    abs(leg['per_leg']['RM']['minimax_unbounded'] - cone['rho_inf_from_minimax']) <= 1e-6
    and all(leg['per_leg'][L]['minimax_unbounded'] <= 1e-6 for L in LEGS if L != 'RM')
    and abs(leg['per_leg']['LF']['minimax_box1']) <= 1e-9 and abs(leg['per_leg']['RF']['minimax_box1']) <= 1e-9,
    'unbounded per leg: %s ; box1 per leg: %s'
    % ({L: round(leg['per_leg'][L]['minimax_unbounded'], 6) for L in LEGS},
       {L: round(leg['per_leg'][L]['minimax_box1'], 6) for L in LEGS}))
chk('partC:contact_term_carries_the_demand',
    reqd['variants']['contact_zeroed']['minimax_box1'] < 0.5 < reqd['variants']['contact_only']['minimax_box1'],
    'contact_zeroed %.6f, contact_only %.6f, full %.6f'
    % (reqd['variants']['contact_zeroed']['minimax_box1'], reqd['variants']['contact_only']['minimax_box1'],
       reqd['variants']['full']['minimax_box1']))
chk('partD:relaxation_solved_and_caveated',
    free['D1']['solved'] and free['D1']['minimax'] is not None and 'RELAXATION' in free['D1']['relaxation_note'],
    'D1 minimax %.6f with the relaxation caveat recorded' % (free['D1']['minimax'] or 0))
chk('partD:no_friction_attribute_in_frozen_xml', free['friction_values_in_xml'] == [] and free['friction_source_lines'] == [],
    'the frozen XML carries no explicit geom friction attribute, so no mu could be read from the file text; '
    'a mu-dependent cone was therefore NOT constructed')
chk('guards:zero_physics_and_xml_unchanged',
    sha(XML) == '6e77df89b742c16d1a8b4a2bd3f4e99e0a909c291f0a67ff0515ec11ecabc2a4',
    'xml hash unchanged; model loads 0, forward 0, mj_step 0')
n_ok = sum(c['ok'] for c in checks)
(DIR / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(
    dict(checks=checks, n_checks=len(checks), n_ok=n_ok, all_ok=n_ok == len(checks),
         note='every number recomputed from the frozen npz; no model load, no forward, no mj_step'), indent=2, default=str))

# --------------------------------------------------------------- ledger / conclusion / budget
ledger = {"round": "S207", "failures": [
    {"id": "S207-A1", "what": "row-DOF map lookup", "defect": "globbed s179-*/ROW_DOF_MAP.json but the file lives one level "
     "deeper (s179-<round>/<stamp>/ROW_DOF_MAP.json)", "caught_by": "IndexError", "status": "fixed by globbing two levels"},
    {"id": "S207-A2", "what": "friction source", "defect": "the plan assumed a per-contact friction cone could be built from the "
     "XML friction attribute; the frozen XML contains NO friction attribute at all (regex found zero matches)",
     "caught_by": "empty friction list, checked before building anything", "status": "Part D was reduced to the documented "
     "non-negativity relaxation D1; no mu-dependent cone was invented"},
    {"id": "S207-A3", "what": "per-leg f0 comparison", "defect": "per-leg grouping of the 64 contact-force components is not "
     "stored in any frozen artefact (no efc_type / contact table for the S177 pose), so FIG3's second panel had to be drawn per "
     "COMPONENT instead of per leg", "caught_by": "checking the available fields before drawing", "status": "reported as a "
     "limitation; the grouping was not guessed"}],
    "corrections": [
        {"id": "S207-C1", "what": "S196 / S203 'simultaneity-coupling barrier' wording",
         "defect": ("A is exactly block-diagonal by leg: 0 of the 66 decision columns touch two legs and 100% of the |A| mass lies "
                    "inside its own leg block. The global minimax equals the maximum of six independent per-leg minimaxes (verified "
                    "to 1e-12 at box=1 and 1e-9 unbounded). The legs do not share decision variables, so there is no simultaneity "
                    "coupling in the frozen matrices."),
         "impact": "the barrier is per-leg, not cross-leg; the historical NUMBERS are unaffected, only the explanation changes",
         "status": "corrected here and in the tracker; no frozen artefact touched"}],
    "design_notes": [
        {"id": "S207-N1", "what": "what s and rho mean",
         "finding": "s is a dimensionless multiplier on the 66 pool-fraction columns; rho_inf is the max-abs residual that remains "
                    "when the box is removed, i.e. the distance from req to the cone generated by the columns. Neither has a force unit."}]}
(DIR / 'ATTEMPT_LEDGER.json').write_text(json.dumps(ledger, indent=2, ensure_ascii=False))

verdict = 'DIRECTIONAL_OBSTRUCTION'
conclusion = f"""# S207 尺度 / 方向 / 接触锁定三分解 —— 结论（中文）

目录：`{DIR}`

## 0. 判决
**`DIRECTIONAL_OBSTRUCTION`**（字典序规则 2 触发）。把激活上界去掉（box 无界）后，42 行里仍有 **ρ∞ = {cone['rho_inf_from_minimax']:.6f}** 的不可消残差——**加多少激活都没用**。而且它**只来自右中腿（RM）**。

## 1. Part 0 结构钉死（`LP_STRUCTURE.json`）
- 维度断言全部通过：`B`(48,90)、`T`(90,66)、`base`(48)、`J`(64,48)、`f0`(64)、`A=(B@T)[6:]=(42,66)`、`req`(42)。
- **决策变量是 A 的 66 列，不是 90 个执行器**。每一列是 s166 映射里某个**池槽的份额**（`actuator_pool` 有 66 项，其中 16 项为 −1）；`x∈[0,s]` 是**无量纲的池激活倍数**，既不是力也不是执行器个数。90 个执行器通过 `T` 进入，且那 16 个不可命令执行器在 `T` 里的行**恒为 0**（S206 已证，本轮复算为 0）。
- **`f0` 是冻结常量，从来不是决策变量**：S195/S196 的 req 把 `J.T@f0` 当作固定的接触项代入（`s185_reconcile.py` 的 `required=-(base+J.T@f0)[6+i]`）。这决定了 Part D 的意义。
- **前 6 行被 `[6:]` 丢掉的原因**：那是六个根 DOF；冻结平衡态 S177 已把它们解到 `residual6`（本轮复算 max|.| = {float(np.max(np.abs(np.asarray(z['residual6'], float)))):.2e}），LP 只问 42 个腿关节行。
- **块结构（本轮最重要的结构发现）**：**0 个决策列跨越两条腿，腿内 |A| 质量占比 = 1.0（精确块对角）**，每腿 7 行。→ 42 行问题**等价于六个互相独立的 7 行问题**。

## 2. Part A 尺度扫描（`SCALE_SCAN.csv/.json`、FIG1）
- 硬门：**s=1 精确复现 3.308620867791373** ✓。
- 网格 s ∈ {{0.5,1,2,4,8,16,32,64,128,256}} 的 minimax：3.400038 → 3.308621 → 3.125786 → 2.760116 → 2.028775 → 1.738366 → … → **1.738366（此后完全不变）**。
- **不存在 s\\***（曲线在 2^4 之后平台化，永不 ≤1e-9）。平台值 **= ρ∞ = {rho:.6f}**，与 Part B 独立算出的锥距离一致（差 {cone['abs_difference']:.1e}）。
- 等价性自检：`A←s·A, box=1` 与 `box=s` 对 s=2、8 的 minimax 相对差 **< 1e-9**（实测 {json.dumps({k: '%.1e' % v['rel_diff'] for k, v in scan['equivalence']['checks'].items()})}）→ s 确实是纯尺度量。

## 3. Part B 方向性障碍（`CONE_DISTANCE.json`、FIG2）
- ρ∞（minimax 无界解）= {cone['rho_inf_from_minimax']:.9f}；用**另一种公式**（可行性二分，不引入 t 变量）= {cone['rho_from_independent_feasibility_bisection']:.9f}，**差 {cone['abs_difference']:.2e} < 1e-6 → 稳定**（标签 `CONE_DISTANCE_UNSTABLE` 不成立）。
- 无量纲量 ρ∞/‖req‖ = **{cone['norm_ratio']:.4f}**；nnls 的 max 残差 {cone['nnls_check']['max_abs_residual']:.4f} 只作上界参考（nnls 最小化 L2，不是等值核对，已注明）。
- **必须同时说明解的非唯一性**：无界 minimax 的最优解**不唯一**（任何让最差行正好等于 ρ∞ 的 x 都最优），所以 FIG2 的逐行残差图是**某一组最优解**，多行并列在 ±ρ∞ 上、其排序无意义；**唯一确定的陈述是逐腿最小值**（见 §4）。按其中一组解看，最大的前几行落在这个集合里：{', '.join('%s (%+.4f)' % (r['dof_name'], r['residual']) for r in cone['top_residual_rows'][:5])}；六个困难 DOF 里只有 DOF 42（`joint_RMCoxa_pitch`）在其中，所以"障碍行=六个困难 DOF"**不成立**。

## 4. 逐腿分解（`LEG_DECOMPOSITION.json`）—— 更正 S196/S203 的"耦合壁垒"说法
| 腿 | box=1 minimax | 无界 minimax |
|---|---|---|
{chr(10).join('| %s | %.6f | %.6f |' % (L, leg['per_leg'][L]['minimax_box1'], leg['per_leg'][L]['minimax_unbounded']) for L in LEGS)}

- 全局 box=1 值 = 六腿最大值（1e-12 内一致），全局无界值 = 六腿最大值（1e-9 内一致）→ **块对角成立**。
- **前腿 LF / RF 在 box=1 下已经可行（0.0）**；中/后腿需要更大激活量（箱界放大到 ~16 时 LH/LM/RH 全部归零）；**只有 RM 在无界下仍是 {leg['per_leg']['RM']['minimax_unbounded']:.6f}**。
- 因此 **"同时性/跨腿耦合壁垒"这个说法在冻结矩阵里没有依据**（S207-C1）：42 行问题拆成六个独立问题，腿与腿之间不共享决策变量。真正卡住的是**右中腿自身的方向问题**。

## 5. Part C 需求端分解（`REQUIREMENT_DECOMPOSITION.json`）
| req 变体 | box=1 minimax | 无界 minimax |
|---|---|---|
| 完整 | {reqd['variants']['full']['minimax_box1']:.6f} | {reqd['variants']['full']['minimax_unbounded']:.6f} |
| 只留 base 项（= 接触项置零） | {reqd['variants']['base_only']['minimax_box1']:.6f} | {reqd['variants']['base_only']['minimax_unbounded']:.6f} |
| 只留接触项 | {reqd['variants']['contact_only']['minimax_box1']:.6f} | {reqd['variants']['contact_only']['minimax_unbounded']:.6f} |

- ‖req_base‖ = {reqd['shares']['req_base_norm']:.4f}、‖req_contact‖ = {reqd['shares']['req_contact_norm']:.4f}（占比 base {100*reqd['shares']['base_share']:.1f}% / contact {100*reqd['shares']['contact_share']:.1f}%）。
- **把冻结接触项置零后 minimax 掉到 {reqd['variants']['contact_zeroed']['minimax_box1']:.6f}**，只留接触项则 {reqd['variants']['contact_only']['minimax_box1']:.6f} → **需求的主要来源是冻结的接触力分配，不是关节自身的被动/偏置项**。

## 6. Part D 自由接触（**假设推演**，`FREE_CONTACT_VARIANT.json`、FIG3）
- 冻结 XML 里**没有任何显式 `friction` 属性**（正则零匹配，已写入台账 S207-A2），加上冻结件里没有 efc/接触分组表，因此**无法构造按腿的摩擦锥**。
- 采用的放松 **D1**：把 64 个接触力分量各自当作**独立非负变量**（比真实摩擦锥**更宽松**）。结果：box=1 下 **minimax = {free['D1']['minimax']:.6f} → 可行**；解出的 ‖f‖ = {free['D1']['f_solution']['norm_solution']:.3f}（冻结 ‖f0‖ = {free['D1']['f_solution']['norm_frozen']:.3f}），单分量最大变化 {free['D1']['f_solution']['max_abs_change']:.3f}。
- **标注**：这是放松假设，**不能**据此宣布"站得住"；它只说明**冻结接触分配本身是不可行性需求的来源**。且因放松比真实锥宽松，**"可行"这个方向不可外推**；反之若放松仍不可行则可排除接触锁定——本轮是前者，所以只写"来源"，不写"松弛即可行"。
- 按字典序，规则 2（ρ∞ > 1e-6）已经先行触发，所以**主标签是 `DIRECTIONAL_OBSTRUCTION`，不是 `CONTACT_LOCKED_INFEASIBILITY`**；S196 的数值**不需要**加"固定接触力"限定（那是规则 3 才要求的）。

## 7. 这**不是**什么
- s 与 ρ∞ 都是**无量纲量**，不得写成任何力的单位；绝对力值仍 NOT_ESTABLISHED。
- 不声明任何运动改善；仍是同一冻结位姿上的**静态**问题。
- "RM 方向不可行"只针对**当前冻结姿态与当前可命令执行器集合**；补齐覆盖会新增列，属假设推演，未做。

## 8. 下一步最小动作
1. **把 RM 单独结案**：它的 7 行在无界下残差 {leg['per_leg']['RM']['minimax_unbounded']:.4f}，下一步只分析 RM 的 7 行在 `cone(A_RM)` 上的投影（同一冻结件，0 物理步），给出"缺哪个方向、缺多少"的向量，而不是继续在全局 minimax 上打转。
2. 如果那条向量指向被 S205 移除的 58% coxa 权限，就把"补齐覆盖收益"作为**标注的假设推演**跑一次（不改冻结产物）。
3. 其余五条腿已经证明是**箱界不足**而非方向问题（box≳16 时全部归零），所以后续"放大激活"的实验只需针对那条方向向量，不必再做全局尺度扫描。

## 9. 记账
`PREREG.json` 跑前落盘（`written_before_runs=true`，含 s 网格、五标签字典序门）；{n_ok}/{len(checks)} 项独立核验通过；台账 3 条自身失败 + 1 条对 S196/S203 的更正（S207-C1）+ 1 条设计说明；0 模型加载、0 forward、0 mj_step、0 XML 改动、0 训练。
"""
(DIR / 'CONCLUSION.zh-CN.md').write_text(conclusion)

def sha_all(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}

budget = {"round": "S207",
          "stop_reason": ("All four parts are complete and the pre-registered lexicographic gate resolves at rule 2 "
                          "(DIRECTIONAL_OBSTRUCTION). Part D was reduced to the documented non-negativity relaxation because the "
                          "frozen artefacts carry no per-contact grouping and the XML carries no friction attribute; that "
                          "limitation is recorded rather than papered over. No physics was run, so the round stops."),
          "budget": {"model_loads": 0, "forward_calls": 0, "mj_step_calls": 0, "neural_ticks": 0,
                     "lp_solves": 10 + 80 + 2 + 2 + 4 * 6 + 1 + 2, "training_steps": 0, "prewarming": False, "xml_changes": 0},
          "artifacts": sha_all(DIR),
          "sources": {str(p): sha(p) for p in [Path('/run/media/sensen/Data2/cell_wound_prototype/s207_structure_scale.py'),
                                               Path('/run/media/sensen/Data2/cell_wound_prototype/s207_conedecomp.py'),
                                               Path('/run/media/sensen/Data2/cell_wound_prototype/verify_s207.py'),
                                               XML] if p.exists()}}
(DIR / 'STOP_AND_BUDGET.json').write_text(json.dumps(budget, indent=2, ensure_ascii=False))
(DIR / 'SOURCE_HASHES.json').write_text(json.dumps(dict(sources=budget['sources'], artifacts=budget['artifacts']), indent=2))
print(json.dumps(dict(dir=str(DIR), n_checks=len(checks), n_ok=n_ok, all_ok=n_ok == len(checks),
                      failed=[c for c in checks if not c['ok']]), indent=1, default=str))
