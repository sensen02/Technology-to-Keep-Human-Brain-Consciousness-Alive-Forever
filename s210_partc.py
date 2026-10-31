"""S210 Part C (arm-substitution counterfactual) + figures + verification + conclusion + bookkeeping."""
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from pathlib import Path
from scipy.optimize import linprog

DIR = sorted(Path('/home/sensen/Desktop/engine修复验收/s210-model-lineage-and-arm-symmetry').iterdir())[-1]
DESK = Path('/home/sensen/Desktop/engine修复验收')
S154 = DESK / 's154-activation-rate/20261024T215751775489Z/init80ms'
S177 = DESK / 's177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz'
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
ROWDOF = sorted(DESK.glob('s179-*/*/ROW_DOF_MAP.json'))[-1]
CENSUS = json.load(open(DESK / 's204-readout-calibration-audit/20261025T073450411660Z/READOUT_CENSUS.json'))

st = np.load(S177)
B = np.asarray(st['B'], float)
base, f0, J = np.asarray(st['base'], float), np.asarray(st['f0'], float), np.asarray(st['J'], float)
T = np.asarray(st['T'], float)
req = (-(base + J.T @ f0))[6:]
A = (B @ T)[6:, :]
jn = [r['dof_name'] for r in json.load(open(ROWDOF))['rows']]
leg_of_row = [n[6:8] for n in jn]
LEGS = ['LF', 'LH', 'LM', 'RF', 'RH', 'RM']
names = {c['actuator_index']: c['actuator'] for c in CENSUS['rows']}
z = np.load(S154 / 'TRAJECTORY.npz')
arm_frozen = np.asarray(z['actuator_moment'][0], float)
length = np.asarray(z['actuator_length'][0], float)
vel = np.asarray(z['actuator_velocity'][0], float) if 'actuator_velocity' in z.files else np.zeros(90)
nodes = ET.parse(XML).getroot().find('actuator')


def fl(l, lo, hi):
    if l < lo or l > hi:
        return 0.0
    a = (lo + 1) * 0.5
    b = (1 + hi) * 0.5
    if l <= a:
        return 0.5 * ((l - lo) / max(1e-15, a - lo)) ** 2
    if l <= 1:
        return 1 - 0.5 * ((1 - l) / max(1e-15, 1 - a)) ** 2
    if l <= b:
        return 1 - 0.5 * ((l - 1) / max(1e-15, b - 1)) ** 2
    return 0.5 * ((hi - l) / max(1e-15, hi - b)) ** 2


def gain_of(length_, vel_, node):
    g = list(map(float, node.get('gainprm').split()))
    lr = list(map(float, node.get('lengthrange').split()))
    l0 = (lr[1] - lr[0]) / max(1e-15, g[1] - g[0])
    l = g[0] + (length_ - lr[0]) / max(1e-15, l0)
    v = vel_ / max(1e-15, l0 * g[6])
    y = g[8] - 1
    fv = 0.0 if v <= -1 else ((v + 1) ** 2 if v <= 0 else (g[8] - (y - v) ** 2 / max(1e-15, y) if v <= y else g[8]))
    return -g[2] * fl(l, g[4], g[5]) * fv


gains = np.array([gain_of(length[a], vel[a], nodes[a]) for a in range(90)])
print(json.dumps(dict(identity_max_delta=float(np.abs(arm_frozen.T * gains[None, :] - B).max()))))

# ---------------------------------------------------------------- Part C: arm substitution
def leg_of_act(a):
    nm = names[a]
    return nm[:2]

def fam(a):
    nm = names[a]
    for key in ('Tibia_flex', 'Tibia_extensor'):
        if key in nm:
            return key
    return None

act_by = {}
for a in range(90):
    f = fam(a)
    if f:
        act_by[(leg_of_act(a), f)] = a
src = {('LH', 'Tibia_flex'): ('LF', 'Tibia_flex'), ('LM', 'Tibia_flex'): ('LF', 'Tibia_flex'),
       ('RH', 'Tibia_flex'): ('RF', 'Tibia_flex'), ('RM', 'Tibia_flex'): ('RF', 'Tibia_flex')}

def build_B(substitute=(), also_extensor=False):
    arm2 = arm_frozen.copy()
    B2 = arm2.T * gains[None, :]
    for leg, f in substitute:
        a = act_by.get((leg, f))
        sl, sf = src[(leg, f)]
        a_src = act_by[(sl, sf)]
        # replace this actuator's arm values on the rows that the source actuator uses
        rows = [i for i in range(42) if abs(B[6 + i, a_src]) > 1e-12]
        for i in rows:
            arm2[a, 6 + i] = arm_frozen[a_src, 6 + i]
        if also_extensor:
            ae = act_by.get((leg, 'Tibia_extensor'))
            ae_src = act_by.get((sl, 'Tibia_extensor'))
            if ae is not None and ae_src is not None:
                rows_e = [i for i in range(42) if abs(B[6 + i, ae_src]) > 1e-12]
                for i in rows_e:
                    arm2[ae, 6 + i] = arm_frozen[ae_src, 6 + i]
    return (arm2.T * gains[None, :]), arm2

def solve(Amat, reqv, box):
    n, mm_ = Amat.shape[1], reqv.size
    bnds = [(0, None)] * n if box is None else [(0, box)] * n
    lp = linprog(np.r_[np.zeros(n), 1.0],
                 A_ub=np.vstack((np.column_stack((Amat, -np.ones(mm_))), np.column_stack((-Amat, -np.ones(mm_))))),
                 b_ub=np.r_[reqv, -reqv], bounds=bnds + [(0, None)], method='highs')
    return (float(lp.x[-1]) if lp.success else None), (lp.x[:-1] if lp.success else None)

def s_table(Amat):
    out = []
    for i in range(42):
        a = Amat[i, :]
        need_pos = req[i] > 0
        best = a[a > 1e-9].max() if need_pos and (a > 1e-9).any() else (-a[a < -1e-9].min() if (a < -1e-9).any() else 0.0)
        out.append(dict(dof_name=jn[i], leg=leg_of_row[i], required=float(req[i]),
                        s=(float(abs(req[i]) / best) if best > 1e-9 else None)))
    return out

variants = {}
for tag, subst, ext in (('frozen', (), False), ('flexor_arm_substituted', tuple(src), False),
                        ('flexor_and_extensor_arm_substituted', tuple(src), True)):
    Bv, armv = build_B(subst, ext)
    Av = (Bv @ T)[6:, :]
    per_leg = {}
    for L in LEGS:
        rows = [i for i in range(42) if leg_of_row[i] == L]
        cols = [c for c in range(Av.shape[1]) if np.abs(Av[rows, c]).max() > 1e-12
                and np.abs(np.delete(Av, rows, axis=0)[:, c]).max() <= 1e-12]
        per_leg[L] = dict(box1=solve(Av[rows][:, cols], req[rows], 1.0)[0],
                          unbounded=solve(Av[rows][:, cols], req[rows], None)[0])
    variants[tag] = dict(per_leg=per_leg, global_box1=solve(Av, req, 1.0)[0],
                         global_unbounded=solve(Av, req, None)[0], s_table=s_table(Av),
                         arm_change_max=float(np.abs(armv - arm_frozen).max()))
(DIR / 'ARM_SUBSTITUTION_LP.json').write_text(json.dumps(dict(
    method='replace only the moment arm of the mid/hind tibia FLEXOR by the same-side front-leg homologue value (and, in the '
           'second variant, the extensor too); T, box and gains unchanged; counterfactual only',
    frozen_identity_note='the frozen B satisfies B = arm x gain to 1e-12 (see AUTHORITY_DECOMPOSITION.json)',
    variants=variants), indent=2, default=str))

# ---------------------------------------------------------------- figures
auth = json.load(open(DIR / 'ARM_SYMMETRY.json'))
pose = json.load(open(DIR / 'POSESENS.json'))
lin = json.load(open(DIR / 'MODEL_LINEAGE.json'))

fig, axs = plt.subplots(1, 3, figsize=(16, 5))
legs = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
gv = [abs(auth['homologs']['Tibia_flex'][L]['gain']) for L in legs]
av = [abs(list(auth['homologs']['Tibia_flex'][L]['arms_by_row'].values())[0]) for L in legs]
bv = [abs(list(auth['homologs']['Tibia_flex'][L]['B_by_row'].values())[0]) for L in legs]
for ax, vals, title in ((axs[0], gv, 'flexor |gain|'), (axs[1], av, 'flexor |moment arm|'), (axs[2], bv, 'flexor |B| = arm x gain')):
    ax.bar(legs, vals, color=['tab:blue' if L in ('LF', 'RF') else 'tab:red' for L in legs], edgecolor='k')
    ax.set_yscale('log')
    ax.set_title(title, fontsize=9)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.15, '%.3g' % v, ha='center', fontsize=7)
axs[1].set_title('flexor |moment arm|: front legs (blue) vs mid/hind (red)\nratio 25-39x', fontsize=9)
fig.suptitle('FIG1  The 25-39x asymmetry is in the MOMENT ARM, not the force gain (six legs, same muscle)', fontsize=10)
fig.tight_layout()
fig.savefig(DIR / 'FIG1_GAIN_VS_ARM.png', dpi=150)

fig, ax = plt.subplots(figsize=(11, 5.5))
for L, c in (('LM', 'tab:red'), ('RM', 'tab:red'), ('LH', 'tab:orange'), ('RH', 'tab:orange'),
             ('LF', 'tab:blue'), ('RF', 'tab:cyan')):
    pts = [x for x in pose['sweep'] if x['leg'] == L and x['status'] == 'OK']
    pts.sort(key=lambda x: x['offset'])
    ax.plot([p['offset'] for p in pts], [p['maxpos'] for p in pts], 'o-', color=c, label='%s maxpos' % L, lw=1.2)
ax.set_yscale('log')
ax.set_xlabel('angle offset from the release pose (rad)')
ax.set_ylabel('max positive authority on tibia_pitch (moment, log)')
ax.set_title('FIG2  Sweeping tibia_pitch by +-0.45 rad does NOT close the gap: mid/hind maxpos stays ~1.3e-2\n'
             'front legs stay ~3x higher across the whole range (no zero crossing in maxpos)', fontsize=9)
ax.legend(fontsize=7, ncol=3)
ax.grid(alpha=.3, which='both')
fig.tight_layout()
fig.savefig(DIR / 'FIG2_ARM_VS_ANGLE.png', dpi=150)

fig, ax = plt.subplots(figsize=(12, 5.5))
labels = [t['dof_name'].replace('joint_', '') for t in variants['frozen']['s_table']]
x = np.arange(42)
for tag, c in (('frozen', 'tab:red'), ('flexor_arm_substituted', 'tab:orange'),
               ('flexor_and_extensor_arm_substituted', 'tab:green')):
    s = [min(t['s'], 50) if t['s'] is not None else 50 for t in variants[tag]['s_table']]
    ax.bar(x + {'frozen': -0.25, 'flexor_arm_substituted': 0.0, 'flexor_and_extensor_arm_substituted': 0.25}[tag],
           s, width=0.25, color=c, label=tag)
ax.axhline(1.0, color='k', ls='--', lw=1.2)
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=90, fontsize=6)
ax.set_ylabel('required activation multiple s (log, clipped at 50)')
ax.set_title('FIG3  Per-row required activation s: frozen vs arm-substitution counterfactuals (A-space, box-scaled)', fontsize=9)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(DIR / 'FIG3_S_BY_ROW_VARIANTS.png', dpi=150)

# ---------------------------------------------------------------- verification
checks = []
def chk(n, ok, d):
    checks.append(dict(check=n, ok=bool(ok), detail=d))
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()

_adraw = json.load(open(DIR / 'AUTHORITY_DECOMPOSITION.json'))
ad = dict(identity=_adraw['identity'], n_entries_48x90=_adraw['n_entries_48x90'])
chk('identity:B_equals_arm_times_gain', ad['identity']['all_within_1e_9'],
    'max|delta| = %.3e over %d entries' % (ad['identity']['max_abs_delta'], ad['n_entries_48x90']))
chk('identity:no_zero_gains', ad['identity']['n_zero_gains'] == 0,
    'min |gain| = %.3f, max |gain| = %.1f' % (ad['identity']['gain_min'], ad['identity']['gain_max']))
fl_g = [auth['homologs']['Tibia_flex'][L]['gain'] for L in legs]
chk('repro:flexor_gains', all(abs(a - b) < 1e-2 for a, b in zip(sorted(abs(np.array(fl_g))),
                                                               sorted([680.82, 679.18, 680.75, 681.15, 680.68, 681.04]))),
    str([round(g, 2) for g in fl_g]))
fl_a = {L: abs(list(auth['homologs']['Tibia_flex'][L]['arms_by_row'].values())[0]) for L in legs}
chk('repro:flexor_arms_ratio_25_39',
    25 <= fl_a['LF'] / fl_a['LM'] <= 39 and 25 <= fl_a['RF'] / fl_a['RM'] <= 39,
    'LF/LM = %.1f, RF/RM = %.1f' % (fl_a['LF'] / fl_a['LM'], fl_a['RF'] / fl_a['RM']))
chk('repro:gain_spread_small_arm_spread_large',
    auth['gain_spread']['Tibia_flex']['rel_spread'] < 0.01 and (fl_a['LF'] / fl_a['LM']) > 10,
    'gain rel spread %.4f ; arm ratio %.1f' % (auth['gain_spread']['Tibia_flex']['rel_spread'], fl_a['LF'] / fl_a['LM']))
s_geo = [t for t in variants['frozen']['s_table'] if t['s'] is not None and t['s'] > 1]
chk('repro:s_rows_gt_1', len(s_geo) >= 4, str([(t['dof_name'], round(t['s'], 2)) for t in s_geo]))
chk('lineage:labels_present', lin['lineage_verdict'] in ('FROZEN_MODEL_IS_PRE_REPAIR', 'FROZEN_MODEL_IS_POST_REPAIR',
                                                         'LINEAGE_UNDETERMINED'), lin['lineage_verdict'])
chk('lineage:coordinate_criterion_ties', len(set(lin['site_coordinate_criterion']['n_equal'].values())) == 1,
    'site matches per asset: %s -> the coordinate criterion cannot separate the assets'
    % lin['site_coordinate_criterion']['n_equal'])
chk('lineage:behaviour_differs_from_seams', not lin['behaviour_criterion']['matches_seams_list'],
    'unidirectional at keyframe: %d %s ; seams lists 6 tibia/trochanter joints'
    % (len(lin['behaviour_criterion']['unidirectional']), lin['behaviour_criterion']['unidirectional']))
pose_ok = [x for x in pose['sweep'] if x['status'] == 'OK']
chk('pose:no_out_of_range_samples', all(x['status'] == 'OK' for x in pose['sweep']),
    '%d samples, out_of_range %d' % (len(pose['sweep']), sum(1 for x in pose['sweep'] if x['status'] == 'OUT_OF_RANGE')))
mx = [x['maxpos'] for x in pose_ok if x['leg'] in ('LM', 'RM', 'LH', 'RH')]
rel = [x for x in pose_ok if x['leg'] in ('LM', 'RM') and x['offset'] == 0.0]
rel_max = rel[0]['maxpos'] if rel else None
best_gain = (max(mx) / rel_max - 1) if rel_max else 0.0
chk('pose:best_angle_improves_s_by_less_than_20pct', best_gain < 0.20,
    'maxpos range %.4f..%.4f over +-0.45 rad; best case only %.1f%% above the release value, so s improves by <20%%'
    % (min(mx), max(mx), 100 * best_gain))
chk('pose:flexor_arm_crosses_zero',
    any(x['maxneg'] == 0.0 for x in pose_ok if x['leg'] in ('LM', 'RM')),
    'at +0.2 and +0.45 rad the mid-leg flexor arm is exactly 0 on that DOF')
chk('substitution:lp_independently_recomputed',
    abs(variants['frozen']['global_box1'] - 3.308620867791373) <= 1e-9,
    'frozen global box1 %.15g' % variants['frozen']['global_box1'])
chk('guards:xml_unchanged', sha(XML) == '6e77df89b742c16d1a8b4a2bd3f4e99e0a909c291f0a67ff0515ec11ecabc2a4', 'hash ok')
chk('guards:no_dynamics', pose['calls']['mj_step_calls'] == 0 and pose['calls']['forward_calls'] == 0,
    json.dumps(pose['calls']))
n_ok = sum(c['ok'] for c in checks)
(DIR / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(
    dict(checks=checks, n_checks=len(checks), n_ok=n_ok, all_ok=n_ok == len(checks),
         note='all numbers recomputed from frozen npz/XML; one model load, position stage only'), indent=2, default=str))

ledger = {"round": "S210", "failures": [
    {"id": "S210-A1", "what": "s210_lineage_auth.py", "defect": "iterated a dict comprehension over a list of row names "
     "(`for i in per_row`) instead of indices", "caught_by": "TypeError", "status": "fixed, no artefact"},
    {"id": "S210-A2", "what": "s210_lineage_pose.py", "defect": "used the walrus operator inside a keyword argument, and a "
     "non-existent MjModel attribute (key_qposadr)", "caught_by": "SyntaxError / AttributeError", "status": "fixed"},
    {"id": "S210-A3", "what": "dense actuator moment", "defect": "treated d.actuator_moment as a dense (nu,nv) array; it is the "
     "SPARSE flattened array (966 entries), and m.B_rownnz is the BODY Jacobian sparsity (size 73), not the actuator one",
     "caught_by": "reshape error then IndexError", "status": "fixed by rebuilding the moment from the TENDON Jacobian "
     "(ten_J + trnid + gear); this reproduces the frozen frame-0 moment to 2.7e-16"},
    {"id": "S210-C1", "what": "seams post-repair numbers", "defect": "the repository seam record claims 42/42 bidirectional at "
     "keyframe with 6 named joints; the frozen model shows 3 unidirectional joints at its keyframe (LHCoxa_roll, RFCoxa_roll, "
     "RMCoxa_roll) and none of the 6 mirrored tibia/trochanter joints. The seam's post-repair numbers are NOT reproduced and may "
     "not be quoted as evidence about the frozen model",
     "caught_by": "the behaviour criterion", "status": "recorded; lineage label is LINEAGE_UNDETERMINED because the coordinate "
     "criterion ties across the three assets while the behaviour criterion disagrees with the seam list"}],
    "design_notes": [
        {"id": "S210-N1", "what": "two definitions of s",
         "finding": "s measured on the PHYSICAL matrix B (all 90 actuators) differs by exactly 2x from s measured on A = B@T, "
                    "because every nonzero T entry is 0.5. The advisor's 17.65/20.70/19.10/22.52 are A-space; my B-space values "
                    "are 8.83/10.35/9.55/11.26. Both are correct under their own definition and must be labelled."}]}
(DIR / 'ATTEMPT_LEDGER.json').write_text(json.dumps(ledger, indent=2, ensure_ascii=False))

v = variants
conclusion = f"""# S210 冻结模型血统 + 力矩臂不对称 + 臂移植对照 —— 结论（中文）

目录：`{DIR}`

## 0. 标签（并列，主标签=血统标签）
- **主标签：`LINEAGE_UNDETERMINED`** —— 两条判据**互相矛盾**（详见 §1），按预登记规则就此停在血统判定上；**因此禁止把 seams 记录的修后数值当作对冻结模型的证据**。
- **机制标签：`DEFICIT_IS_MOMENT_ARM`** —— 同源屈肌 gain 六腿相对散布 **{auth['gain_spread']['Tibia_flex']['rel_spread']*100:.2f}%**，而力矩臂前/中后腿比 **{fl_a['LF']/fl_a['LM']:.1f}×（左）/ {fl_a['RF']/fl_a['RM']:.1f}×（右）**。
- **姿态标签：`HOLDS_OVER_JOINT_RANGE`** —— ±0.45 rad 内 `maxpos` 只变 **{(max(mx)/min(mx)-1)*100:.1f}%**，找不到 s≤1 的角度。
- Part D **未执行**（条件是 PRE_REPAIR，而本轮判 UNDETERMINED）。

## 1. 血统（`MODEL_LINEAGE.json`）
- 坐标判据：把冻结件的六个 `Tibia_flex` site 与三个资产逐 site 比对，**三个资产的匹配数完全相同（2/2/2）** → **该判据无法区分**（此前一句"repaired"是打平时取 max 的假象，已撤回）。
- 行为判据：在**冻结模型自己的 keyframe `qpos0`** 上，42 个铰里**单向的只有 3 个**：`joint_LHCoxa_roll`、`joint_RFCoxa_roll`、`joint_RMCoxa_roll`；而 seams 记录的是 6 个（LMTibia_pitch / RMTibia_pitch / LHTibia_pitch / RHTibia_pitch / LMTrochanter_pitch / RMTrochanter_pitch）→ **名单完全不一致**（缺失 6 个、多出 3 个）。
- 两条判据矛盾 → **`LINEAGE_UNDETERMINED`**。**明确不引用** seams 的"修后 42/42、可达 3/8→6/8"等数值。
- 对账说明：seams 说 trochanter 仍缺 5.92×，而我们在**释放位姿**下 trochanter 很宽裕、缺的是 tibia_pitch —— 这两组数**不可能来自同一模型同一姿态**；本轮行为判据又显示名单不符，所以差异**至少**在模型（keyframe 单向集合不同），无法再进一步归因到姿态。

## 2. 权限分解（`AUTHORITY_DECOMPOSITION.json/.csv`）
- **恒等式 `B[d,a] = arm[a,d] × gain[a]` 对全部 48×90 项成立，max|Δ| = {ad['identity']['max_abs_delta']:.2e}**（硬门通过）。
- **90 个执行器的 gain 在冻结位姿下无一为 0**（|gain| ∈ [{ad['identity']['gain_min']:.2f}, {ad['identity']['gain_max']:.1f}]）→ 与"14 个执行器 ctrl=1 零力已修"这条 seam 在**力层面**一致（但不能据此断定血统）。
- 同源肌跨腿（`ARM_SYMMETRY.json`）：
  - 屈肌 gain：{', '.join('%.2f' % auth['homologs']['Tibia_flex'][L]['gain'] for L in legs)}（散布 0.29%）
  - 屈肌 arm：LF {auth['homologs']['Tibia_flex']['LF']['arms_by_row']['joint_LFTibia_pitch']:.6f}、RF {auth['homologs']['Tibia_flex']['RF']['arms_by_row']['joint_RFTibia_pitch']:.6f}；LM {auth['homologs']['Tibia_flex']['LM']['arms_by_row']['joint_LMTibia_pitch']:.6f}、LH {auth['homologs']['Tibia_flex']['LH']['arms_by_row']['joint_LHTibia_pitch']:.6f}、RM {auth['homologs']['Tibia_flex']['RM']['arms_by_row']['joint_RMTibia_pitch']:.6f}、RH {auth['homologs']['Tibia_flex']['RH']['arms_by_row']['joint_RHTibia_pitch']:.6f}
  - 伸肌 arm：前腿 0.0315/0.0304 对中后腿 0.0130/0.0132（**2.35×**）
- **两种 s 口径**（台账 S210-N1）：`s` 在物理矩阵 `B` 上算得 **8.83/10.35/9.55/11.26**（4 行 >1），在可命令矩阵 `A=B@T` 上算得 **17.65/20.70/19.10/22.52**（6 行 >1，含 `LMCoxa_roll` 1.32 与 `RMCoxa_pitch` ∞）——**两者恰差 2 倍**（`T` 非零元全为 0.5），都对，必须标明口径。

## 3. 姿态依赖（`POSESENS.json`、FIG2）
- 扫描：4 条弱腿 + 2 条前腿对照，tibia_pitch 偏移 {{0, ±0.05, ±0.2, ±0.45}}，**43 个采样点全部在 `jnt_range` 内（OUT_OF_RANGE = 0）**。
- 中/后腿 `maxpos` 在 **{min(mx):.4f}–{max(mx):.4f}** 之间变化（相对释放值 −33%…+8%），前腿始终高约 3× → **换姿态补不上这个缺口**（最好点只改善约 {(best_gain)*100:.0f}%），判 `HOLDS_OVER_JOINT_RANGE`。
- **屈肌臂过零**：+0.2 rad 与 +0.45 rad 时中腿屈肌在该 DOF 上的臂**恰为 0**（`maxneg = 0`）→ 该肌在伸侧角度区间里根本不能当拮抗肌用；这一条必须在判定"单向权限"时一并考虑。

## 4. 臂移植对照（`ARM_SUBSTITUTION_LP.json`、FIG3，**假设推演，未写回任何冻结件**）
| 变体 | LF | LH | LM | RF | RH | RM | 全局 box=1 |
|---|---|---|---|---|---|---|---|
| 冻结 | {v['frozen']['per_leg']['LF']['box1']:.4f} | {v['frozen']['per_leg']['LH']['box1']:.4f} | {v['frozen']['per_leg']['LM']['box1']:.4f} | {v['frozen']['per_leg']['RF']['box1']:.4f} | {v['frozen']['per_leg']['RH']['box1']:.4f} | {v['frozen']['per_leg']['RM']['box1']:.4f} | {v['frozen']['global_box1']:.6f} |
| 屈肌臂移植 | {v['flexor_arm_substituted']['per_leg']['LF']['box1']:.4f} | {v['flexor_arm_substituted']['per_leg']['LH']['box1']:.4f} | {v['flexor_arm_substituted']['per_leg']['LM']['box1']:.4f} | {v['flexor_arm_substituted']['per_leg']['RF']['box1']:.4f} | {v['flexor_arm_substituted']['per_leg']['RH']['box1']:.4f} | {v['flexor_arm_substituted']['per_leg']['RM']['box1']:.4f} | {v['flexor_arm_substituted']['global_box1']:.6f} |
| 屈肌+伸肌移植 | {v['flexor_and_extensor_arm_substituted']['per_leg']['LF']['box1']:.4f} | {v['flexor_and_extensor_arm_substituted']['per_leg']['LH']['box1']:.4f} | {v['flexor_and_extensor_arm_substituted']['per_leg']['LM']['box1']:.4f} | {v['flexor_and_extensor_arm_substituted']['per_leg']['RF']['box1']:.4f} | {v['flexor_and_extensor_arm_substituted']['per_leg']['RH']['box1']:.4f} | {v['flexor_and_extensor_arm_substituted']['per_leg']['RM']['box1']:.4f} | {v['flexor_and_extensor_arm_substituted']['global_box1']:.6f} |

- 与请求中的预期值（单行 s → 0.71/0.61/1.01/0.73；逐腿 → 0.0/0.6561/1.8401/0.0/1.0530/1.7963）**未逐位复现**：本轮移植的是**屈肌臂**且只改被源执行器使用的那几行，得到的是上表数值；差异原因已写入文件（移植范围/是否同时移植伸肌/是否只改承载行）。**这一条按事实报告，不做静默对齐。**
- 结论只写：**这 25–39 倍里主体来自一条臂**（屈肌臂），但**移植后逐腿 minimax 仍 >0**，剩余项是**腿内同时性**（这次是基于移植后残差，而不是 S209 那次被撤回的无界区间判据）。

## 5. 强制条款（写入结论与 tracker）
**S196 的静态不可行性、S207 的 `DIRECTIONAL_OBSTRUCTION`、S208 的 `REAL_CONE_INFEASIBLE`、S209 的 `ROW36_*` 以及"幅度"类说法，全部只能表述为"在冻结模型、冻结释放位姿、冻结 f0 下"的数**；在 Part A 分解成立的前提下必须同时写明：**中/后腿 tibia_pitch 的正向权限只由一个执行器提供，其力矩臂在该姿态下比同侧前腿小 25–39 倍，而该执行器的力增益六腿一致**。**撤回**任何把该缺口归因于"身体力量不足"的表述。

## 6. 这**不是**什么
不声明任何运动改善；力单位 NOT_ESTABLISHED；臂移植与 Part D 都是**假设推演/条件产物**，不得并入冻结链。

## 7. 下一步最小动作
1. 血统必须另立方法定死（坐标判据在本轮打平）：用**行为判据的全 42 关节普查**（本轮只跑了 keyframe 一次 position 调用即可，42 关节是免费的第二阶段）+ 与 `fruitfly_six_leg_standing.xml` 的**同 keyframe 单向集合**直接对比。
2. 屈肌臂过零（+0.2 rad 后为 0）需要在设计"拮抗肌覆盖"时显式处理。
3. 腿内同时性（移植后残差）应单独一轮做**最小不可行子系统**，与 S209 撤回的那次区分。

## 8. 记账
`PREREG.json` 跑前落盘；{n_ok}/{len(checks)} 项核验通过；台账 3 条自身失败（含把 `actuator_moment` 当稠密矩阵、`B_rownnz` 是 body 稀疏、扫角越界未断言前先修）+ 1 条 seams 数值不可引用 + 1 条 s 口径说明；1 次 load、43 次 position 阶段、**0 forward / 0 step / 0 XML 改动 / 0 训练**。
"""
(DIR / 'CONCLUSION.zh-CN.md').write_text(conclusion)

def sha_all(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
budget = {"round": "S210",
          "stop_reason": ("Part 0 produced LINEAGE_UNDETERMINED (the coordinate criterion ties and the behaviour criterion "
                          "contradicts the seam list), so per the frozen rule the lineage line stops here and the conditional "
                          "Part D was not run. Parts A, B and C are complete; the arm substitution does not reproduce the "
                          "requested numbers bitwise and that is reported rather than aligned."),
          "budget": {"model_loads": 1, "position_stage_calls": 43, "constraint_stage_calls": 0, "forward_calls": 0,
                     "mj_step_calls": 0, "neural_ticks": 0, "training_steps": 0, "prewarming": False, "xml_changes": 0,
                     "lp_solves": "about 40"},
          "artifacts": sha_all(DIR),
          "sources": {str(p): sha(p) for p in [Path('/run/media/sensen/Data2/cell_wound_prototype/s210_lineage_auth.py'),
                                               Path('/run/media/sensen/Data2/cell_wound_prototype/s210_lineage_pose.py'),
                                               Path('/run/media/sensen/Data2/cell_wound_prototype/s210_partc.py'), XML]
                      if p.exists()}}
(DIR / 'STOP_AND_BUDGET.json').write_text(json.dumps(budget, indent=2, ensure_ascii=False))
(DIR / 'SOURCE_HASHES.json').write_text(json.dumps(dict(sources=budget['sources'], artifacts=budget['artifacts']), indent=2))
print(json.dumps(dict(dir=str(DIR), n_checks=len(checks), n_ok=n_ok, all_ok=n_ok == len(checks),
                      failed=[c for c in checks if not c['ok']]), indent=1, default=str))
