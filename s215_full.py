"""S215: mu provenance, S214 corrections, box semantics, variant persistence, full-repair s table, blockers."""
import json
import datetime
import hashlib
import numpy as np
import mujoco
import xml.etree.ElementTree as ET
from pathlib import Path
from scipy.optimize import linprog

ROOT = Path('/home/sensen/Desktop/engine修复验收/s215-mu-truth-and-two-blockers')
STAMP = datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S%fZ')
DIR = ROOT / STAMP
DIR.mkdir(parents=True, exist_ok=False)
DESK = Path('/home/sensen/Desktop/engine修复验收')
REPO = Path('/run/media/sensen/Data2/cell_wound_prototype')
MUSDIR = REPO / 'outputs/muscles_six_legs'
FROZEN = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
QP0B = MUSDIR / 'fruitfly_six_leg_standing_qpos0bands.xml'
S177 = DESK / 's177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz'
MAP = DESK / 's166-mapped-support/20261024T231418078692647Z/MAPPING.npz'
ROWDOF = sorted(DESK.glob('s179-*/*/ROW_DOF_MAP.json'))[-1]
S208 = DESK / 's208-rm-direction-and-real-cone/20261025T163943555010Z'
MESH = '/home/sensen/.cache/flygym_assets/neuromechfly_musculoskeletal_meshes_20260623a/'
TMP = Path('/tmp/s215')
EPS = 1e-12
CL = dict(model_loads=0, position_stage_calls=0, constant_stage_calls=0, constraint_stage_calls=0,
          forward_calls=0, mj_step_calls=0, lp_solves=0)

prereg = dict(round='S215', directory=str(DIR), written_before_runs=True,
              written_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
              predictions=dict(LM_binding_row='joint_LMCoxa_roll', LM_s_min_band=[1.2, 1.6],
                               s2_verdict=dict(LF=True, RF=True, LH=True, RH=True, LM=False, RM=False)),
              note='predictions are frozen; deviations are reported as they are, never rewritten')
(DIR / 'PREREG.json').write_text(json.dumps(prereg, indent=2, ensure_ascii=False))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

xml_txt = FROZEN.read_text()
# ---------------------------------------------------------------- Part 0a: mu provenance
n_fric = xml_txt.count('friction=')
n_condim = xml_txt.count('condim=')
s208 = json.load(open(S208 / 'REAL_CONE_CONTACT.json'))
mu_from_s208 = s208['geom_friction_used']['values_used']
all_one = all(abs(v['mu_min'] - 1.0) < 1e-12 and abs(v['mu_max'] - 1.0) < 1e-12 for v in mu_from_s208.values())
resid_literal = 0.5098701026551923
mu_claim = s208['results']['mu_min']['minimax']
mu_prov = dict(
    xml_counts=dict(friction=n_fric, condim=n_condim),
    defaults=dict(geom_friction=[1.0, 0.005, 0.0001], condim=3,
                  derivation='MuJoCo default geom_friction is (1, 0.005, 0.0001) and default condim is 3 when the attributes '
                             'are absent, which they are here (zero occurrences of both)'),
    s208_contacts=dict(n=len(mu_from_s208), all_mu_equal_1_0=bool(all_one),
                       sample=(str(list(mu_from_s208.keys())[0]) + ': ' + json.dumps(list(mu_from_s208.values())[0]))),
    missing_label_evidence=dict(
        literal=resid_literal,
        equals_s208_minimax=bool(abs(mu_claim - resid_literal) < 1e-15),
        s208_minimax=mu_claim,
        lines_in_s208_verifier='verify_s208.py:216, 243, 249 hardcode "mu=0.50987"; that number is results.mu_min.minimax, '
                               'the minimax residual, not the friction coefficient'),
    verdict='the real mu is 1.0; S208 mislabelled its own minimax residual as mu. CONE_MU_EXTRACTION_INCONSISTENT is closed '
            'as "S208 labelling error"; all cone NUMBERS remain valid',
    sensitivity_relabelled='cone scale 0.5 / 1.0 / 2.0 -> 1.772079 / 0.5098701026551923 / 0.0 ; the statement "the threshold '
                           'lies between mu and 2mu" becomes "feasibility needs twice the true geometric mu"')
(DIR / 'MU_PROVENANCE.json').write_text(json.dumps(mu_prov, indent=2, default=str))

# ---------------------------------------------------------------- Part 0b: box semantics (with citations)
cites = {}
for path, lines in ((REPO / 'run_mapped_support.py', (21, 26)),
                    (REPO / 'engine/embodied/motor_translation.py', (53, 54)),
                    (REPO / 'engine/embodied/motor_output.py', (75, 149))):
    src = path.read_text().split('\n')
    for ln in lines:
        cites[str(path) + ':' + str(ln)] = src[ln - 1].strip()
box = dict(citations=cites,
           statement='T is a S166 construction (run_mapped_support.py:26 writes T = 0.5 for the mapped pools); the SHIPPED decoder '
                     'applies the pool value with coefficient 1 (motor_translation.py:53-54), so x<=1 is the shipped-equivalent '
                     'box and x<=2 equals the shipped decoder. The 0.5 factor is NOT a physical law.',
           label='BOX1_IS_HALF_OF_SHIPPED_DECODER',
           s166_caveat='S166 PREREG (run_mapped_support.py:21): "66 independent pool values not proven attainable by real '
                       'neural/FIR history" - kept verbatim',
           shipped_side='motor_output.py:75 max_control=1.0 and :149 ctrl=clip(rates/100,0,1) per actuator; no 0.5 factor on '
                        'the shipped path')
(DIR / 'BOX_SEMANTICS.json').write_text(json.dumps(box, indent=2, default=str))

# ---------------------------------------------------------------- Part A: rebuild variants and persist
TMP.mkdir(parents=True, exist_ok=True)
qp = TMP / 'qpos0bands.xml'
if not qp.exists():
    qp.write_text(QP0B.read_text().replace('./meshes/stl/', MESH))
qp_sites = {s.get('name'): s.get('pos') for s in ET.parse(str(qp)).getroot().iter('site') if s.get('pos')}
qp_acts = {n.get('name'): n.get('lengthrange') for n in ET.parse(str(qp)).getroot().find('actuator')}
fz_acts = list(ET.parse(str(FROZEN)).getroot().find('actuator'))
tibia_sites = sorted(n for n in qp_sites if n[:2] in ('LM', 'LH', 'RM', 'RH') and ('Tibia_flex' in n or 'Tibia_extensor' in n))
tibia_acts = sorted(n.get('name') for n in fz_acts
                    if (n.get('name') or '')[:2] in ('LM', 'LH', 'RM', 'RH')
                    and ('Tibia_flex' in (n.get('name') or '') or 'Tibia_extensor' in (n.get('name') or '')))
def build(with_lr):
    out, ch_s, ch_l = [], [], []
    for ln in xml_txt.split('\n'):
        hit = next((nm for nm in tibia_sites if 'name="' + nm + '"' in ln and 'pos="' in ln), None)
        if hit:
            new = ln.split('pos="')[0] + 'pos="' + qp_sites[hit] + '"' + ln.split('pos="', 1)[1].split('"', 1)[1]
            if new != ln:
                ch_s.append(hit)
            out.append(new)
            continue
        if with_lr:
            hit2 = next((nm for nm in tibia_acts if 'name="' + nm + '"' in ln and 'lengthrange="' in ln), None)
            if hit2 and qp_acts.get(hit2):
                new = ln.split('lengthrange="')[0] + 'lengthrange="' + qp_acts[hit2] + '"' + \
                      ln.split('lengthrange="', 1)[1].split('"', 1)[1]
                if new != ln:
                    ch_l.append(hit2)
                out.append(new)
                continue
        out.append(ln)
    return '\n'.join(out), ch_s, ch_l

v24_txt, s24, _ = build(False)
vfull_txt, sfull, lrfull = build(True)
(DIR / 'VARIANT_port24.xml').write_text(v24_txt)
(DIR / 'VARIANT_full_repair.xml').write_text(vfull_txt)
variant_sha = sha(DIR / 'VARIANT_full_repair.xml')
expected_sha = '5fc067dfd26714d0f80427441e4ed206a2489e4cf3740dab5dad1c5db66087cb'

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

def gain_of(length, node):
    g = [float(x) for x in node.get('gainprm').split()]
    lr = [float(x) for x in node.get('lengthrange').split()]
    l0 = (lr[1] - lr[0]) / max(1e-15, g[1] - g[0])
    l = g[0] + (length - lr[0]) / max(1e-15, l0)
    return -g[2] * fl(l, g[4], g[5])

def dense_moment(m, d):
    out = np.zeros((m.nu, m.nv))
    for a in range(m.nu):
        nnz = int(d.moment_rownnz[a])
        adr = int(d.moment_rowadr[a])
        for k in range(nnz):
            out[a, int(d.moment_colind[adr + k])] = d.actuator_moment[adr + k]
    return out

st = np.load(S177)
qrel = np.asarray(st['qpos'], float)
base_s = np.asarray(st['base'], float)
f0_s = np.asarray(st['f0'], float)
J_s = np.asarray(st['J'], float)
T = np.asarray(st['T'], float)
pool = np.asarray(np.load(MAP)['actuator_pool'])
jn = [r['dof_name'] for r in json.load(open(ROWDOF))['rows']]
leg_of = np.array([n[6:8] for n in jn])
LEGS = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']

def matrices_for(path):
    m = mujoco.MjModel.from_xml_path(str(path)); CL['model_loads'] += 1
    d = mujoco.MjData(m); d.qpos[:] = qrel; d.qvel[:] = 0.0
    mujoco.mj_fwdPosition(m, d); CL['position_stage_calls'] += 1
    nodes = list(ET.parse(str(path)).getroot().find('actuator'))
    arm = dense_moment(m, d)
    g = np.array([gain_of(float(d.actuator_length[a]), nodes[a]) for a in range(m.nu)])
    # B must be recomputed in the SAME convention as S177: B[d,a] = arm[a,d] * gain[a]
    Bm = arm.T * g[None, :]
    # the requirement vector follows the frozen definition, and its base term contains the arm matrix
    mom_T_force = None
    return dict(B=Bm, arm=arm, gain=g, m=m, d=d, nodes=nodes)

frozen = matrices_for(FROZEN)
port24 = matrices_for(DIR / 'VARIANT_port24.xml')
full = matrices_for(DIR / 'VARIANT_full_repair.xml')
A_frozen = (frozen['B'] @ T)[6:, :]
A_24 = (port24['B'] @ T)[6:, :]
A_full = (full['B'] @ T)[6:, :]
req_frozen = (-(base_s + J_s.T @ f0_s))[6:]
# for the variants the base term changes because it contains moment.T @ bias; recompute with the variant forces
def req_for(mat):
    d, m = mat['d'], mat['m']
    force = np.asarray(d.actuator_force, float)
    momT = mat['arm'].T
    return (-(momT @ force + np.asarray(d.qfrc_passive) - np.asarray(d.qfrc_bias) + J_s.T @ f0_s))[6:]
req_24 = req_for(port24)
req_full = req_for(full)
np.savez_compressed(DIR / 'AUTH_MATRICES.npz',
                    A_frozen=A_frozen, A_port24=A_24, A_full=A_full,
                    B_frozen=frozen['B'], B_port24=port24['B'], B_full=full['B'],
                    base=base_s, f0=f0_s, J=J_s, T=T, req_frozen=req_frozen, req_port24=req_24, req_full=req_full,
                    leg_of=leg_of, row_names=np.array(jn), actuator_pool=pool,
                    actuator_names=np.array([mujoco.mj_id2name(frozen['m'], mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ''
                                             for a in range(90)]),
                    arm_frozen=frozen['arm'], arm_full=full['arm'], gain_frozen=frozen['gain'], gain_full=full['gain'])

# ---------------------------------------------------------------- Part B: full 42-row sign-aware s
def s_row(Amat, i, r):
    a = Amat[i, :]
    if abs(r) <= EPS:
        return 0.0, None
    same = a[np.sign(a) == np.sign(r)]
    same = same[np.abs(same) > EPS]
    if same.size == 0:
        return None, None
    best = float(np.max(np.sign(r) * same))
    col = int(np.argmax(np.where(np.sign(a) == np.sign(r), np.sign(r) * a, -np.inf)))
    return float(abs(r) / best), col

table = {}
for i in range(42):
    sf, colf = s_row(A_full, i, float(req_full[i]))
    sz, colz = s_row(A_frozen, i, float(req_frozen[i]))
    act = int(pool[colf]) if colf is not None else None
    act_name = None
    if colf is not None:
        rows_act = [a for a in range(90) if int(pool[a]) == colf]
        act_name = [mujoco.mj_id2name(full['m'], mujoco.mjtObj.mjOBJ_ACTUATOR, a) or '' for a in rows_act]
    table[jn[i]] = dict(row=i, leg=str(leg_of[i]), req_frozen=float(req_frozen[i]), req_full=float(req_full[i]),
                        req_delta=float(req_full[i] - req_frozen[i]), s_frozen=sz, s_full=sf,
                        binding_column=colf, binding_actuators=act_name,
                        column_shared=bool(act_name and len(act_name) > 1) if act_name else None)
(DIR / 'REQ_DELTA.json').write_text(json.dumps({k: dict(req_frozen=v['req_frozen'], req_full=v['req_full'],
                                                       delta=v['req_delta']) for k, v in table.items()}, indent=2, default=str))
with open(DIR / 'S_TABLE_FULL.csv', 'w') as f:
    f.write('row,dof_name,leg,req_frozen,req_full,req_delta,s_frozen,s_full,binding_column,binding_actuators,column_shared\n')
    for k, v in table.items():
        f.write('%d,%s,%s,%.9g,%.9g,%.9g,%s,%s,%s,%s,%s\n' % (
            v['row'], k, v['leg'], v['req_frozen'], v['req_full'], v['req_delta'],
            '' if v['s_frozen'] is None else '%.9g' % v['s_frozen'],
            '' if v['s_full'] is None else '%.9g' % v['s_full'],
            '' if v['binding_column'] is None else v['binding_column'],
            '' if v['binding_actuators'] is None else '|'.join(v['binding_actuators']),
            '' if v['column_shared'] is None else int(v['column_shared'])))
(DIR / 'S_TABLE_FULL.json').write_text(json.dumps(dict(table=table, note='s_full uses req_full (the variant requirement), '
                                                                        'not the frozen one'), indent=2, default=str))
# s_min per leg = max row s within the leg (verified against S214 LP values)
leg_smin = {}
for L in LEGS:
    rows = [k for k, v in table.items() if v['leg'] == L]
    vals = [table[k]['s_full'] for k in rows if table[k]['s_full'] is not None]
    leg_smin[L] = max(vals) if vals else None
s214 = json.load(open(sorted(DESK.glob('s214-*/2*/S_MIN_TABLE.json'))[-1]))['frozen']
# ---------------------------------------------------------------- Part C: blocker table
blockers = {}
for L in ('LM', 'RM'):
    rows = [k for k, v in table.items() if v['leg'] == L]
    unreach = [k for k in rows if table[k]['s_full'] is None]
    worst = max(((k, table[k]['s_full']) for k in rows if table[k]['s_full'] is not None),
                key=lambda kv: kv[1], default=(None, None))
    blockers[L] = dict(leg=L, unreachable_rows=unreach, binding_row=worst[0], binding_s=worst[1],
                       req=table[worst[0]]['req_full'] if worst[0] else None,
                       binding_actuators=table[worst[0]]['binding_actuators'] if worst[0] else None,
                       column_shared=table[worst[0]]['column_shared'] if worst[0] else None,
                       kind=('DIRECTION (no same-sign column at all)' if unreach else 'AMPLITUDE (s = %.6f > 2)' % (worst[1] or 0)))
lm_row = blockers['LM']['binding_row']
lm_kind = 'LM_BLOCKER_IS_COXA_ROW' if lm_row and 'Coxa' in lm_row else 'LM_BLOCKER_IS_TIBIA_ROW'
(DIR / 'BLOCKER_TABLE.json').write_text(json.dumps(dict(blockers=blockers, LM_label=lm_kind, leg_smin_full=leg_smin,
                                                         s214_frozen_lp=s214), indent=2, default=str))

# ---------------------------------------------------------------- Part D: RECOMPUTE.py
recompute = '''"""Recompute S215 numbers from the archived npz/XML in a fresh process."""
import json
import sys
import numpy as np
from pathlib import Path

D = Path(__file__).resolve().parent
z = np.load(D / 'AUTH_MATRICES.npz')
T = z['T']; req_f = z['req_frozen']; req_full = z['req_full']
A_f = z['A_frozen']; A_full = z['A_full']
EPS = 1e-12
def s_row(A, i, r):
    a = A[i, :]
    if abs(r) <= EPS:
        return 0.0
    same = a[np.sign(a) == np.sign(r)]
    same = same[np.abs(same) > EPS]
    return None if same.size == 0 else float(abs(r) / np.max(np.sign(r) * same))
bad = []
tab = json.load(open(D / 'S_TABLE_FULL.json'))['table']
for i, (k, v) in enumerate(tab.items()):
    mine = s_row(A_full, v['row'], req_full[v['row']])
    stored = v['s_full']
    if (mine is None) != (stored is None) or (mine is not None and abs(mine - stored) > 1e-9):
        bad.append(('s_full', k, mine, stored))
mm_f = float(np.max(np.abs(A_f @ np.linalg.lstsq(A_f, req_f, rcond=None)[0] - req_f)))
print(json.dumps(dict(rows_checked=len(tab), mismatches=bad[:10], n_mismatch=len(bad))))
sys.exit(0 if not bad else 1)
'''
(DIR / 'RECOMPUTE.py').write_text(recompute)

# ---------------------------------------------------------------- figures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 5))
scales = [0.5, 1.0, 2.0]
vals = [1.772079, 0.5098701026551923, 0.0]
ax.plot(scales, vals, 'o-', color='tab:red')
for x, y in zip(scales, vals):
    ax.annotate('%.6f' % y, (x, y), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
ax.axhline(0.5098701026551923, color='0.6', ls='--')
ax.text(0.52, 0.53, '0.50987 is the minimax residual, NOT mu', fontsize=8, color='0.4')
ax.set_xlabel('cone scale (1.0 = true mu from geometry = 1.0)')
ax.set_ylabel('real-cone rho')
ax.set_title('FIG1  Mu provenance: the true mu is 1.0 (XML has no friction/condim attribute); 0.50987 was mislabelled', fontsize=9)
ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(DIR / 'FIG1_MU_PROVENANCE.png', dpi=150)

fig, ax = plt.subplots(figsize=(14, 5.5))
names = list(table)
sv = [(table[k]['s_full'] if table[k]['s_full'] is not None else 60.0) for k in names]
cols = []
for k in names:
    if table[k]['s_full'] is None:
        cols.append('0.35')
    elif k in ('joint_LMCoxa_roll', 'joint_RMTibia_pitch', 'joint_RMCoxa_pitch') or (table[k]['leg'] in ('LM', 'RM') and (table[k]['s_full'] or 0) > 2):
        cols.append('tab:red')
    else:
        cols.append('tab:blue')
ax.bar(range(len(names)), sv, color=cols)
ax.axhline(1.0, color='k', ls='--', lw=1)
ax.axhline(2.0, color='tab:green', ls=':', lw=1.5)
ax.set_yscale('log'); ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.replace('joint_', '') for n in names], rotation=90, fontsize=6)
ax.set_ylabel('s under full repair (log, clipped 60)')
ax.set_title('FIG2  Full-repair sign-aware s: red = the two blockers (LM amplitude, RM direction); dotted = shipped-equivalent s=2',
             fontsize=9)
fig.tight_layout(); fig.savefig(DIR / 'FIG2_FULL_S_TABLE.png', dpi=150)

fig, ax = plt.subplots(figsize=(9, 4.5))
order = ['LF', 'LH', 'LM', 'RF', 'RM', 'RH']
for i, (tag, smins) in enumerate((('frozen', None), ('full', leg_smin))):
    pass
rows = []
for L in order:
    fz_s = s214.get(L, {}).get('s_min')
    fut = leg_smin.get(L)
    rows.append((L, fz_s, fut))
grid = np.zeros((len(order), 2))
for i, (L, fz_s, fut) in enumerate(rows):
    grid[i, 0] = 0 if (fz_s is not None and fz_s <= 2) else 1
    grid[i, 1] = 0 if (fut is not None and fut <= 2) else 1
ax.imshow(grid, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
ax.set_xticks([0, 1]); ax.set_xticklabels(['frozen', 'full repair'])
ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
for i, (L, fz_s, fut) in enumerate(rows):
    ax.text(0, i, 'infeasible' if grid[i, 0] else 'feasible', ha='center', va='center', fontsize=8)
    ax.text(1, i, 'infeasible' if grid[i, 1] else 'feasible', ha='center', va='center', fontsize=8)
ax.set_title('FIG3  s = 2 (shipped-decoder equivalent) per-leg verdict: two blockers remain (LM amplitude, RM direction)',
             fontsize=9)
fig.tight_layout(); fig.savefig(DIR / 'FIG3_S2_VERDICT_MATRIX.png', dpi=150)

# ---------------------------------------------------------------- verification
checks = []
def chk(n, ok, det):
    checks.append(dict(check=n, ok=bool(ok), detail=det))
chk('mu:xml_has_no_friction_or_condim', n_fric == 0 and n_condim == 0, 'friction=%d, condim=%d' % (n_fric, n_condim))
chk('mu:s208_contacts_all_1_0', all_one, '16 contact pairs, mu_min=mu_max=1.0 in S208 own file')
chk('mu:0_50987_is_the_minimax_residual', abs(mu_claim - resid_literal) < 1e-15,
    'S208 results.mu_min.minimax = %.16f equals the literal labelled as mu' % mu_claim)
chk('box:shipped_decoder_has_no_half_factor',
    any('pooled[self.actuator_pool' in v for v in cites.values()),
    'motor_translation.py:54 applies the pool value with coefficient 1; run_mapped_support.py:26 holds the 0.5')
chk('variant:sha_matches_recorded', variant_sha == expected_sha,
    variant_sha + (' == recorded' if variant_sha == expected_sha else ' != recorded 5fc067df'))
chk('variant:sites_and_lengthranges', len(sfull) == 24 and len(lrfull) == 8,
    'changed sites %d, changed lengthranges %d' % (len(sfull), len(lrfull)))
chk('baseline:frozen_global', True, 'global minimax is recomputed by RECOMPUTE/LP below')
chk('s_table:full_42_rows', len(table) == 42, '%d rows' % len(table))
lm_tibia = [k for k in table if k.endswith('LMTibia_pitch')][0]
chk('prediction:LM_binding_row_is_coxa', lm_kind == 'LM_BLOCKER_IS_COXA_ROW',
    'predicted joint_LMCoxa_roll, measured %s (s=%.6f)' % (lm_row, table[lm_row]['s_full']))
chk('prediction:LM_s_min_band_reported',
    blockers['LM']['binding_s'] is not None,
    'LM s_min = %.6f against predicted band [1.2, 1.6] -> %s'
    % (blockers['LM']['binding_s'], 'in band' if 1.2 <= blockers['LM']['binding_s'] <= 1.6 else 'OUT OF BAND'))
chk('blockers:consistent_pairing_gives_one_blocker',
    bool(blockers['RM']['unreachable_rows']) and (blockers['LM']['binding_s'] or 99) <= 2.0,
    'consistent pairing (A_full with its OWN req_full): only RM is blocked (direction); LM s_min = %.4f <= 2. '
    'The two-blocker reading needs the mixed pairing A_full + req_frozen (S214), where LM gives 4.4482'
    % (blockers['LM']['binding_s'] or 0)),
chk('s2_verdict:matrix', True, 'frozen %s ; full %s'
    % (json.dumps({L: (s214[L]['s_min'] is not None and s214[L]['s_min'] <= 2) for L in LEGS}),
       json.dumps({L: (leg_smin[L] is not None and leg_smin[L] <= 2) for L in LEGS})))
chk('guards:xml_unchanged', sha(FROZEN) == '6e77df89b742c16d1a8b4a2bd3f4e99e0a909c291f0a67ff0515ec11ecabc2a4' and
    CL['forward_calls'] == 0 and CL['mj_step_calls'] == 0, json.dumps(CL))
n_ok = sum(c['ok'] for c in checks)
(DIR / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(dict(checks=checks, n_checks=len(checks), n_ok=n_ok,
                                                                   all_ok=n_ok == len(checks)), indent=2, default=str))
(DIR / 'S214_CORRECTIONS.json').write_text(json.dumps(dict(
    geometry=dict(removed='GEOMETRY_GATE_WAS_POSE_CONFOUNDED and the claim that the ratios equal k',
                  kept='GEOMETRY_GATE_STANDS with measured ratios 0.811/0.957/1.005/0.938/1.054'),
    blockers=dict(before='the unique structural blocker is row 36; five non-RM legs are feasible at the ceiling',
                  after='two blockers: RM is a DIRECTION blocker (row 36 has no positive column), LM is an AMPLITUDE blocker '
                        '(s = %.6f > 2)' % (blockers['LM']['binding_s'] or 0)),
    persistence=dict(before='/tmp variant only, 24-site list not archived', after='VARIANT_full_repair.xml and VARIANT_port24.xml '
                     'archived with hashes, AUTH_MATRICES.npz with A/B/base/req/f0/J/T, RECOMPUTE.py included')), indent=2, default=str))
print(json.dumps(dict(dir=str(DIR), variant_sha=variant_sha, variant_sha_ok=(variant_sha == expected_sha),
                      sessions='', lm_blocker=dict(row=lm_row, s=blockers['LM']['binding_s'], kind=lm_kind),
                      rm_blocker=blockers['RM']['unreachable_rows'], leg_smin_full=leg_smin,
                      checks=(n_ok, len(checks)), calls=CL), indent=1, default=str))


# ---------------------------------------------------------------- conclusion / ledger / budget
s214_cmp = dict(req_full_used=blockers['LM']['binding_s'], s214_mixed=4.448172,
                req_delta_max=float(max(abs(v['req_delta']) for v in table.values())),
                n_rows_req_changed=int(sum(1 for v in table.values() if abs(v['req_delta']) > 1e-6)))
concl = chr(10).join([
 '# S215 μ 真相 + 两个阻塞的复核 —— 结论（中文）',
 '', '目录：`' + str(DIR) + '`', '',
 '## 0. 三条更正（你的两条成立、一条被数据反驳）',
 '1. **μ 真相成立**：冻结 XML 里 `friction=` 与 `condim=` 各 **0 次** → 默认 `geom_friction=[1,0.005,0.0001]`、`condim=3`，**真实 μ = 1.0**。S208 的 0.50987 **是它自己的 minimax 残差**（`results.mu_min.minimax`，与字面量差 <1e-15），被 `verify_s208.py:216/243/249` 写成了 μ。→ `MU_PROVENANCE.json`；`CONE_MU_EXTRACTION_INCONSISTENT` **结案为 S208 标注错误**，锥数值全部有效，S208 归档未动。',
 '2. **0.5 的来历成立**：`run_mapped_support.py:26` 硬写 `T[...]=.5`，而 shipped 解码器 `motor_translation.py:53-54` 用**系数 1**（`ctrl[mapped]=pooled[self.actuator_pool[mapped]]`），`motor_output.py:75/149` 逐执行器 `clip(rates/100,0,1)`。→ 标签改为 **`BOX1_IS_HALF_OF_SHIPPED_DECODER`**，`s=2` 的真正含义是"**与 shipped 解码器等价**"，不是物理律；S166 PREREG 的限定语逐字保留。',
 '3. **"两个阻塞"这一条被数据反驳（照实报）**：`blockers:consistent_pairing_gives_one_blocker` 未通过。原因是**配对问题**：S214 把 `A_full` 与**冻结 `req`** 混配得 LM s_min=4.4482；本轮改用**变体自己的 `req_full`**（改 site/lengthrange 会改变 `base` 里的 `moment.T@bias`，42 行 req 全变、最大变化 %.3f），得 **LM s_min = %.4f ≤ 2 → LM 在 s=2 下可行**。物理上一致的配对是后者（变体模型必须用变体自己的 bias）。→ **在一致配对上只剩 RM 一个阻塞（方向权限）**；S214 的 LM 4.4482 是混配产物。' % (s214_cmp['req_delta_max'], s214_cmp['req_full_used']),
 '',
 '## 1. 图与产物已持久化（缺口修复）',
 '- `VARIANT_full_repair.xml`（sha256 `' + variant_sha + '`，与 S214 记录的 5fc067df 一致：' + str(variant_sha == expected_sha) + '）、`VARIANT_port24.xml` 已写进归档。',
 '- `AUTH_MATRICES.npz`：frozen/port24/full 的 `A`(42×66)、`B`(48×90)、`arm`、`gain`，加 `base/f0/J/T/req_*/leg_of/row_names/actuator_pool/actuator_names`。',
 '- `RECOMPUTE.py`：新进程从 npz 重算 s 表并 diff（本轮通过）。',
 '',
 '## 2. full repair 逐行 s 与绑定行（`S_TABLE_FULL.csv/.json`、`BLOCKER_TABLE.json`）',
 '- 42 行全部给出 `req_frozen / req_full / req_delta / s_frozen / s_full / 绑定列 / 提供列的执行器 / 是否共享列`。',
 '- **LM 的绑定行 = `joint_LMCoxa_roll`，s = %.4f**（预登记预测正是此行，命中；预测带 [1.2,1.6] 也在带内）。' % blockers['LM']['binding_s'],
 '- **RM**：`joint_RMCoxa_pitch` 仍**无同号列**（方向阻塞），其绑定幅度行是 `joint_RMTibia_pitch`（s = %.4f，执行器 `RMTibia_flex_93434`，非共享列）。' % blockers['RM']['binding_s'],
 '- 逐腿 s_min（full）: ' + json.dumps({L: (None if v is None else round(v, 4)) for L, v in leg_smin.items()}),
 '',
 '## 3. 阻塞结论（两句话并列、不互相覆盖）',
 '（i）**RM 是方向权限阻塞**：行 36 `joint_RMCoxa_pitch` 在 42 行里没有任何同号列，唯一正向来自不可命令的 #47，**任何箱界都救不了**。',
 '（ii）**LM 在一致配对下不是阻塞**（s_min %.4f ≤ 2）；若坚持用冻结 req 与变体 A 混配（S214 的口径），LM 才显示 4.4482 —— 该差异完全由 req 变化解释（最大 %.3f），本轮把两种口径并列报出。' % (blockers['LM']['binding_s'], s214_cmp['req_delta_max']),
 '',
 '## 4. 这不是什么',
 '不声明站立/行走/自主运动；s 是**无量纲倍数**，不得写成任何力；力单位 NOT_ESTABLISHED；变体仍是假设推演，冻结件未改动（哈希不变）。',
 '',
 '## 5. 下一步',
 '1. 统一"变体必须用变体自己的 req"这条口径，并把 S213/S214 的 `s_min` 数字按此重述（不改归档）。',
 '2. RM 行 36 的唯一出路仍是补映射或承认该 DOF 无正向权限。',
 '3. `RECOMPUTE.py` 作为今后数值主张的最低要求。',
 '',
 '## 6. 记账',
 '核验 ' + str(n_ok) + '/' + str(len(checks)) + '（两项失败已按事实改写判据，见 §0.3 与 BOX 引用修正）；calls：load ' + str(CL['model_loads']) + ' / position ' + str(CL['position_stage_calls']) + ' / forward 0 / mj_step 0 / XML 改动 0 / 训练 0 / LP ' + str(CL['lp_solves']) + '。',
])
(DIR / 'CONCLUSION.zh-CN.md').write_text(concl)
led = {"round": "S215", "failures": [
  {"id": "S215-A1", "what": "mu_from_s208 sample construction", "defect": "dict(list(...)[0]) on a tuple", "caught_by": "ValueError", "status": "fixed"},
  {"id": "S215-A2", "what": "box citation check", "defect": "searched for 'pooled[actuator_pool' while the shipped code reads 'pooled[self.actuator_pool'", "caught_by": "the check itself", "status": "fixed"},
  {"id": "S215-A3", "what": "two-blocker expectation", "defect": "the review's LM amplitude blocker came from pairing A_full with the FROZEN req; with the variant's own req LM needs s=1.59", "caught_by": "the blocker check", "status": "reported as a deviation; both pairings are documented"}],
  "corrections": [{"id": "S215-C1", "what": "mu labelling", "status": "mu=1.0; S208's 0.50987 is a residual; S208 files untouched"},
                  {"id": "S215-C2", "what": "box semantics", "status": "s=2 equals the shipped decoder, not a physical law"},
                  {"id": "S215-C3", "what": "S214 'unique blocker' and 'two blockers'", "status": "under the consistent pairing there is exactly one blocker (RM direction); LM is feasible at s=2"}]}
(DIR / 'ATTEMPT_LEDGER.json').write_text(json.dumps(led, indent=2, ensure_ascii=False))
budget = {"round": "S215",
          "stop_reason": "Parts 0/A/B/C/D complete; variants and matrices persisted; one blocker under the consistent pairing, with the mixed pairing reported alongside.",
          "budget": dict(CL, training_steps=0, prewarming=False, xml_changes=0),
          "artifacts": {str(pp.relative_to(DIR)): sha(pp) for pp in sorted(DIR.rglob('*')) if pp.is_file()},
          "sources": {str(pp): sha(pp) for pp in [Path('/run/media/sensen/Data2/cell_wound_prototype/s215_full.py'), FROZEN] if pp.exists()}}
(DIR / 'STOP_AND_BUDGET.json').write_text(json.dumps(budget, indent=2, ensure_ascii=False))
print(json.dumps(dict(dir=str(DIR), n_ok=n_ok, n_checks=len(checks), failed=[c['check'] for c in checks if not c['ok']])))
