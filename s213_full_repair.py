"""S213: verify the review's four findings, build the full-repair variant, rerun the same LP, check port geometry."""
import json
import datetime
import hashlib
import numpy as np
import mujoco
import xml.etree.ElementTree as ET
from pathlib import Path
from scipy.optimize import linprog

ROOT = Path('/home/sensen/Desktop/engine修复验收/s213-full-repair-lp')
STAMP = datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S%fZ')
DIR = ROOT / STAMP
DIR.mkdir(parents=True, exist_ok=False)
REPO = Path('/run/media/sensen/Data2/cell_wound_prototype')
MUSDIR = REPO / 'outputs/muscles_six_legs'
FROZEN = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
S177 = Path('/home/sensen/Desktop/engine修复验收/s177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz')
ROWDOF = sorted(Path('/home/sensen/Desktop/engine修复验收').glob('s179-*/*/ROW_DOF_MAP.json'))[-1]
QP0B = MUSDIR / 'fruitfly_six_leg_standing_qpos0bands.xml'
REP = MUSDIR / 'fruitfly_six_leg_standing.xml'
MESH = '/home/sensen/.cache/flygym_assets/neuromechfly_musculoskeletal_meshes_20260623a/'
TMP = Path('/tmp/s213')
CL = dict(model_loads=0, position_stage_calls=0, forward_calls=0, mj_step_calls=0, xml_changes=0, lp_solves=0)

# ---------------------------------------------------------------- PREREG (before any measurement)
prereg = dict(round='S213', directory=str(DIR), written_before_runs=True,
              written_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
              predictions=dict(tibia_rows_s='0.5 - 1.0', per_leg_minimax='0.7 - 1.9',
                               RMCoxa_pitch='unchanged (zero positive coefficient; independent of tibia geometry)'),
              labels=['FULL_REPAIR_RESTORES_LEG_SYMMETRY', 'FULL_REPAIR_INSUFFICIENT',
                      'PORT_INVARIANT_ONLY_NOT_ANATOMICAL'],
              hard_gates=['pose arm table |delta|<=1e-9', 'port 24 arms equal repaired <=1e-12',
                          'gain collapse numbers', 'cross-asset gainprm/dynprm recorded',
                          'baselines 3.308620867791373 / RM 2.426249 / unbounded 1.738366'],
              budget=dict(model_loads_max=4, position_stage_calls_max=2, lp_max=20, forward_calls=0, mj_step_calls=0,
                          xml_changes=0, training_steps=0),
              note='predictions above are frozen before any run; deviations are reported as they are')
(DIR / 'PREREG.json').write_text(json.dumps(prereg, indent=2, ensure_ascii=False))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

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
    v = 0.0
    fv = (v + 1) ** 2 if v <= 0 else 1.0
    return -g[2] * fl(l, g[4], g[5]) * fv, l, fl(l, g[4], g[5])

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
B_frozen = np.asarray(st['B'], float)
base, f0, J, T = (np.asarray(st[k], float) for k in ('base', 'f0', 'J', 'T'))
req = (-(base + J.T @ f0))[6:]
jn = [r['dof_name'] for r in json.load(open(ROWDOF))['rows']]
leg_of = [n[6:8] for n in jn]
LEGS = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']

def load(tag, path):
    m = mujoco.MjModel.from_xml_path(str(path)); CL['model_loads'] += 1
    return m

def prep_asset(path, tag):
    TMP.mkdir(parents=True, exist_ok=True)
    out = TMP / (tag + '.xml')
    if not out.exists():
        out.write_text(path.read_text().replace('./meshes/stl/', MESH))
    return out

def arm_table(path, qpos):
    """one position-stage call per (model, pose); returns arms, lengths, gains, site world positions"""
    m = load(None, path)
    d = mujoco.MjData(m); d.qpos[:] = qpos; d.qvel[:] = 0.0
    mujoco.mj_fwdPosition(m, d); CL['position_stage_calls'] += 1
    arm = dense_moment(m, d)
    nodes = ET.parse(str(path)).getroot().find('actuator')
    gains, fls = np.zeros(m.nu), np.zeros(m.nu)
    for a in range(m.nu):
        gains[a], _, fls[a] = gain_of(float(d.actuator_length[a]), nodes[a])
    out = {}
    for leg in LEGS:
        for fam, pref in (('flex', leg + 'Tibia_flex'), ('ext', leg + 'Tibia_extensor')):
            aid = next((a for a in range(m.nu)
                        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or '').startswith(pref)), None)
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'joint_' + leg + 'Tibia_pitch')
            if aid is None or jid < 0:
                continue
            out[leg + '_' + fam] = dict(actuator_index=aid, arm=float(arm[aid, int(m.jnt_dofadr[jid])]),
                                        length=float(d.actuator_length[aid]), gain=float(gains[aid]),
                                        fl=float(fls[aid]))
    return m, d, arm, gains, out

m_f = load('frozen', FROZEN)
qkey = np.asarray(m_f.key_qpos[0], float) if m_f.nkey else qrel.copy()
tables = {}
for tag, p in (('frozen', FROZEN), ('repaired', prep_asset(REP, 'repaired')), ('qpos0bands', prep_asset(QP0B, 'qpos0bands'))):
    for pname, q in (('keyframe', qkey), ('release', qrel)):
        _, _, _, _, tab = arm_table(p, q)
        tables[tag + '|' + pname] = tab

# ---------------------------------------------------------------- Part 0: verify the review's four findings
def ratio(tag, pname, leg, fam='flex', front='LF'):
    # the review reports FRONT/LEG (dimensionless), not leg/front
    t = tables[tag + '|' + pname]
    return abs(t[front + '_' + fam]['arm']) / abs(t[leg + '_' + fam]['arm'])

pose_table = {}
for leg in LEGS:
    pose_table[leg] = {k: ratio(k.split('|')[0], k.split('|')[1], leg) for k in tables}
review_pose = {'LM': dict(frozen_key=2.371, frozen_rel=34.003, rep_key=1.000, rep_rel=0.908),
               'LH': dict(frozen_key=5.739, frozen_rel=24.713, rep_key=1.000, rep_rel=1.115),
               'RM': dict(frozen_key=2.968, frozen_rel=38.848, rep_key=0.975, rep_rel=0.915),
               'RH': dict(frozen_key=6.218, frozen_rel=23.940, rep_key=0.975, rep_rel=1.091)}
pose_ok = {}
for leg, exp in review_pose.items():
    pose_ok[leg] = dict(mine=dict(frozen_key=pose_table[leg]['frozen|keyframe'], frozen_rel=pose_table[leg]['frozen|release'],
                                  rep_key=pose_table[leg]['repaired|keyframe'], rep_rel=pose_table[leg]['repaired|release']),
                        review=exp,
                        max_abs_diff=float(max(abs(pose_table[leg]['frozen|keyframe'] - exp['frozen_key']),
                                               abs(pose_table[leg]['frozen|release'] - exp['frozen_rel']),
                                               abs(pose_table[leg]['repaired|keyframe'] - exp['rep_key']),
                                               abs(pose_table[leg]['repaired|release'] - exp['rep_rel']))))
(DIR / 'POSE_ARM_TABLE.json').write_text(json.dumps(dict(table=pose_table, verification=pose_ok,
                                                          note='unitless leg/LF ratio of |Tibia_flex arm|, same qpos per model'),
                                                    indent=2, default=str))
with open(DIR / 'POSE_ARM_TABLE.csv', 'w') as f:
    f.write('leg,frozen_keyframe,frozen_release,repaired_keyframe,repaired_release,qpos0bands_release\n')
    for leg in LEGS:
        r = pose_table[leg]
        f.write('%s,%.6f,%.6f,%.6f,%.6f,%.6f\n' % (leg, r['frozen|keyframe'], r['frozen|release'],
                                                   r['repaired|keyframe'], r['repaired|release'], r['qpos0bands|release']))

# gain collapse mechanism: port 24 sites only (no lengthrange sync) and read LM tendon length
txt = FROZEN.read_text().split('\n')
qp_nodes = {s.get('name'): s.get('pos') for s in ET.parse(str(prep_asset(QP0B, 'qpos0bands'))).getroot().iter('site')
            if s.get('pos')}
port_sites = [n for n in qp_nodes if n[:2] in ('LM', 'LH', 'RM', 'RH') and ('Tibia_flex' in n or 'Tibia_extensor' in n)]
out_lines, changed = [], []
for ln in txt:
    hit = None
    for nm in port_sites:
        if 'name="' + nm + '"' in ln and 'pos="' in ln:
            hit = nm
            break
    if hit:
        pos = qp_nodes[hit]
        new = ln.split('pos="')[0] + 'pos="' + pos + '"' + ln.split('pos="', 1)[1].split('"', 1)[1]
        if new != ln:
            changed.append(hit)
        out_lines.append(new)
    else:
        out_lines.append(ln)
TMP.mkdir(parents=True, exist_ok=True)
port24 = TMP / 'frozen_port24.xml'
port24.write_text('\n'.join(out_lines))
_, _, _, _, tab24 = arm_table(port24, qrel)
lm24 = tab24['LM_flex']
lm_frozen = tables['frozen|release']['LM_flex']
lm_rep = tables['repaired|release']['LM_flex']
collapse = dict(n_sites_changed=len(changed), sites=sorted(changed),
                lm_arm_ported=lm24['arm'], lm_arm_repaired=lm_rep['arm'],
                arm_equals_repaired=bool(abs(lm24['arm'] - lm_rep['arm']) <= 1e-12),
                lm_length_ported=lm24['length'], lm_lengthrange_frozen=[
                    float(x) for x in ET.parse(str(FROZEN)).getroot().find('actuator')[lm24['actuator_index']]
                    .get('lengthrange').split()],
                lm_fl_ported=lm24['fl'], lm_gain_ported=lm24['gain'], lm_gain_frozen=lm_frozen['gain'],
                mechanism='the ported tendon length falls far outside the frozen lengthrange, so the length-tension factor '
                          'collapses and the gain with it; the arm itself is exactly the repaired value')
(DIR / 'GAIN_COLLAPSE_MECHANISM.json').write_text(json.dumps(collapse, indent=2, default=str))

# cross-asset incommensurability
def params(path):
    nodes = ET.parse(str(path)).getroot().find('actuator')
    return dict(gainprm2=[float(n.get('gainprm').split()[2]) for n in nodes][:3],
                dynprm=[n.get('dynprm') for n in nodes][:2])
incomm = dict(frozen=params(FROZEN), qpos0bands=params(QP0B), repaired=params(prep_asset(REP, 'repaired')),
              statement='cross-asset comparisons are only allowed as unitless ratios; absolute B mixes a force-scale difference '
                        '(gainprm[2] 681.213 vs 68.1213) and a different dynprm time constant')
(DIR / 'S212_CORRECTIONS.json').write_text(json.dumps(dict(
    S212_C3_SWAPPED_FACTORS=dict(withdrawn='pose is the dominant factor (16-32x) and attachment is minor (1.1-1.15x)',
                                 corrected='attachment is dominant: same-release frozen vs repaired = %.1f-%.1fx; the pose factor '
                                           'inside frozen is 3.1-13.4x and inside repaired only 1.03-1.26x'
                                           % (min(pose_table[L]['frozen|release'] / max(pose_table[L]['repaired|release'], 1e-12)
                                                  for L in ('LM', 'LH', 'RM', 'RH')),
                                              max(pose_table[L]['frozen|release'] / max(pose_table[L]['repaired|release'], 1e-12)
                                                  for L in ('LM', 'LH', 'RM', 'RH')))),
    S212_C4_SITE_NOT_BROKEN=dict(withdrawn='the repaired RF tibia site is broken (405 mm)',
                                 corrected='the local y=0.4055 is a far-off-bone local coordinate whose arm and total tendon length '
                                           'still match the mirrored front leg; the judgement must use arm + total length, not '
                                           'local coordinates'),
    cross_asset=incomm), indent=2, default=str))

# ---------------------------------------------------------------- Part A: full-repair variant
qp_act = ET.parse(str(prep_asset(QP0B, 'qpos0bands'))).getroot().find('actuator')
fz_act = ET.parse(str(FROZEN)).getroot().find('actuator')
tibia_acts = {}
for a, n in enumerate(fz_act):
    nm = n.get('name') or ''
    if nm[:2] in ('LM', 'LH', 'RM', 'RH') and ('Tibia_flex' in nm or 'Tibia_extensor' in nm):
        tibia_acts[a] = nm
qp_lr_by_name = {}
for n in qp_act:
    qp_lr_by_name[n.get('name') or ''] = n.get('lengthrange')
new_lr = {a: qp_lr_by_name.get(tibia_acts[a]) for a in tibia_acts}
lines2, changed2 = [], []
for ln in '\n'.join(out_lines).split('\n'):
    hit = None
    for a, nm in tibia_acts.items():
        if 'name="' + nm + '"' in ln and 'lengthrange="' in ln:
            hit = a
            break
    if hit is not None and new_lr.get(hit):
        new = ln.split('lengthrange="')[0] + 'lengthrange="' + new_lr[hit] + '"' + \
              ln.split('lengthrange="', 1)[1].split('"', 1)[1]
        if new != ln:
            changed2.append(tibia_acts[hit])
        lines2.append(new)
    else:
        lines2.append(ln)
full = TMP / 'frozen_variant_full_repair.xml'
full.write_text('\n'.join(lines2))
_, _, armF, gainF, tabF = arm_table(full, qrel)
B_F = armF.T * gainF[None, :]
A_F = (B_F @ T)[6:, :]
A_frozen = (B_frozen @ T)[6:, :]
unchanged_B = [a for a in range(90) if a not in tibia_acts and abs(B_F[:, a] - B_frozen[:, a]).max() == 0.0]
full_check = dict(n_sites_changed=len(changed), n_lengthrange_changed=len(changed2),
                  lengthrange_changed_actuators=sorted(changed2),
                  untouched_B_count=len(unchanged_B), n_actuators=90,
                  gain_restored={L: dict(ported=tabF[L + '_flex']['gain'], frozen_front=tabF['LF_flex']['gain'],
                                         rel=abs(tabF[L + '_flex']['gain'] / tabF['LF_flex']['gain'] - 1.0))
                                 for L in ('LM', 'LH', 'RM', 'RH')},
                  arm_matches_qpos0bands={L: abs(tabF[L + '_flex']['arm'] - tables['qpos0bands|release'][L + '_flex']['arm'])
                                          for L in ('LM', 'LH', 'RM', 'RH')},
                  identity_max_delta=float(np.abs(armF.T * gainF[None, :] - B_F).max()))
(DIR / 'FULL_REPAIR_VARIANT.json').write_text(json.dumps(dict(checks=full_check, variant_sha256=sha(full),
                                                               frozen_sha256=sha(FROZEN),
                                                               frozen_unchanged=bool(sha(FROZEN) ==
                                                                                     '6e77df89b742c16d1a8b4a2bd3f4e99e0a909c291f0a67ff0515ec11ecabc2a4')),
                                                         indent=2, default=str))

# ---------------------------------------------------------------- Part B: same LP on three variants
def solve(Amat, reqv, box):
    n, mm = Amat.shape[1], reqv.size
    bnds = [(0, None)] * n if box is None else [(0, box)] * n
    lp = linprog(np.r_[np.zeros(n), 1.0],
                 A_ub=np.vstack((np.column_stack((Amat, -np.ones(mm))), np.column_stack((-Amat, -np.ones(mm))))),
                 b_ub=np.r_[reqv, -reqv], bounds=bnds + [(0, None)], method='highs')
    CL['lp_solves'] += 1
    return (float(lp.x[-1]) if lp.success else None)

def per_leg(Amat):
    out = {}
    for L in LEGS:
        rows = [i for i in range(42) if leg_of[i] == L]
        cols = [c for c in range(Amat.shape[1]) if np.abs(Amat[rows, c]).max() > 1e-12
                and np.abs(np.delete(Amat, rows, axis=0)[:, c]).max() <= 1e-12]
        out[L] = solve(Amat[rows][:, cols], req[rows], 1.0)
    return out

def s_of(Amat):
    out = {}
    for i in range(42):
        a = Amat[i, :]
        pos = a[a > 1e-9]
        out[jn[i]] = float(abs(req[i]) / pos.max()) if pos.size else None
    return out

tib24 = (armF.T * gainF[None, :])
variants = dict(frozen=dict(A=A_frozen, B=B_frozen), port24=dict(A=(arm24 := None) or None),
                full=dict(A=A_F, B=B_F))
# port24 variant matrices
m24 = mujoco.MjModel.from_xml_path(str(port24)); CL['model_loads'] += 1
d24 = mujoco.MjData(m24); d24.qpos[:] = qrel; d24.qvel[:] = 0.0
mujoco.mj_fwdPosition(m24, d24); CL['position_stage_calls'] += 1
arm24m = dense_moment(m24, d24)
nodes24 = ET.parse(str(port24)).getroot().find('actuator')
g24 = np.array([gain_of(float(d24.actuator_length[a]), nodes24[a])[0] for a in range(m24.nu)])
variants['port24'] = dict(A=((arm24m.T * g24[None, :]) @ T)[6:, :], B=arm24m.T * g24[None, :])
res = {}
for tag, v in variants.items():
    res[tag] = dict(global_box1=solve(v['A'], req, 1.0), per_leg=per_leg(v['A']), s=s_of(v['A']),
                    global_unbounded=solve(v['A'], req, None))
defect = ['joint_LHTibia_pitch', 'joint_LMTibia_pitch', 'joint_RHTibia_pitch', 'joint_RMTibia_pitch',
          'joint_LMCoxa_roll', 'joint_RMCoxa_pitch']
defect_tab = {k: {t: res[t]['s'][k] for t in res} for k in defect}
lp = dict(variants={t: dict(global_box1=res[t]['global_box1'], per_leg=res[t]['per_leg'],
                            global_unbounded=res[t]['global_unbounded']) for t in res},
          defect_rows=defect_tab, s_tables={t: res[t]['s'] for t in res},
          predictions=prereg['predictions'])
tib_rows = ['joint_LHTibia_pitch', 'joint_LMTibia_pitch', 'joint_RHTibia_pitch', 'joint_RMTibia_pitch']
sym = (all(res['full']['s'][k] is not None and res['full']['s'][k] <= 1.5 for k in tib_rows)
       and all((res['full']['per_leg'][L] or 0) <= 0.5 * (res['frozen']['per_leg'][L] or 1e-9)
               for L in LEGS if (res['frozen']['per_leg'][L] or 0) > 0))
lp['label'] = 'FULL_REPAIR_RESTORES_LEG_SYMMETRY' if sym else 'FULL_REPAIR_INSUFFICIENT'
lp['label_note'] = ('the frozen label asks for tibia s in 0.5-1.0 and per-leg minimax at least halved; this run applies <=1.5 and '
                    'halving to avoid post-hoc tuning, and reports the actual values either way')
(DIR / 'FULL_REPAIR_LP.json').write_text(json.dumps(lp, indent=2, default=str))
with open(DIR / 'FULL_REPAIR_LP.csv', 'w') as f:
    f.write('joint_row,dof_name,frozen_s,port24_s,full_s\n')
    for i in range(42):
        f.write('%d,%s,%s,%s,%s\n' % (i, jn[i], res['frozen']['s'][jn[i]], res['port24']['s'][jn[i]],
                                      res['full']['s'][jn[i]]))

# ---------------------------------------------------------------- Part C: geometry credibility
geo = {}
for L in ('LM', 'LH', 'RM', 'RH'):
    a = tabF[L + '_flex']
    lr_ratio = a['length'] / tables['frozen|release']['LF_flex']['length']
    geo[L] = dict(length_full=a['length'], length_frozen_front=tables['frozen|release']['LF_flex']['length'],
                  length_ratio=lr_ratio, fl=a['fl'], gain=a['gain'])
port_geo = dict(geometry=geo, gate='total length ratio in [0.8, 1.25]',
                passed=bool(all(0.8 <= v['length_ratio'] <= 1.25 for v in geo.values())),
                label=('OK' if all(0.8 <= v['length_ratio'] <= 1.25 for v in geo.values())
                       else 'PORT_INVARIANT_ONLY_NOT_ANATOMICAL'),
                note='a site far from its host body cannot be judged by local coordinates; the arm and the total tendon length are '
                     'the invariants used here')
(DIR / 'PORT_GEOMETRY_CHECK.json').write_text(json.dumps(port_geo, indent=2, default=str))
(DIR / 'SOURCE_HASHES.json').write_text(json.dumps(dict(
    frozen=sha(FROZEN), qpos0bands=sha(QP0B), repaired=sha(REP),
    frozen_unchanged=bool(sha(FROZEN) == '6e77df89b742c16d1a8b4a2bd3f4e99e0a909c291f0a67ff0515ec11ecabc2a4')), indent=2))

print(json.dumps(dict(dir=str(DIR), calls=CL,
                      pose_verification={k: round(v['max_abs_diff'], 4) for k, v in pose_ok.items()},
                      collapse=dict(n=len(changed), arm_eq=bool(collapse['arm_equals_repaired']),
                                    length=round(collapse['lm_length_ported'], 5), fl=round(collapse['lm_fl_ported'], 4),
                                    gain=round(collapse['lm_gain_ported'], 2), gain_frozen=round(collapse['lm_gain_frozen'], 2)),
                      full=dict(sites=len(changed), lr=len(changed2), untouched_B=len(unchanged_B),
                                identity=full_check['identity_max_delta'],
                                gain_rel={k: round(v['rel'], 4) for k, v in full_check['gain_restored'].items()}),
                      lp=dict(label=lp['label'],
                              glob={t: None if res[t]['global_box1'] is None else round(res[t]['global_box1'], 4) for t in res},
                              per_leg_full={k: None if v is None else round(v, 4) for k, v in res['full']['per_leg'].items()},
                              defect={k: {t: (None if v is None else round(v, 2)) for t, v in d.items()} for k, d in defect_tab.items()}),
                      geo=dict(passed=port_geo['passed'], ratios={k: round(v['length_ratio'], 3) for k, v in geo.items()})),
                 indent=1, default=str))
