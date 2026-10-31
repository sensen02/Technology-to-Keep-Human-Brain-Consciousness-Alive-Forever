"""S212 Part A' (instances) + Part B (two-pose arms, multiplicative decomposition) + Part C (site-port variant LP)."""
import json
import hashlib
import numpy as np
import mujoco
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import linprog

DIR = sorted(Path('/home/sensen/Desktop/engine修复验收/s212-attachment-port-decomposition').iterdir())[-1]
DESK = Path('/home/sensen/Desktop/engine修复验收')
S177 = DESK / 's177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz'
ROWDOF = sorted(DESK.glob('s179-*/*/ROW_DOF_MAP.json'))[-1]
REPO = Path('/run/media/sensen/Data2/cell_wound_prototype')
MUSDIR = REPO / 'outputs/muscles_six_legs'
FROZEN = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
ASSETS = dict(frozen=FROZEN, repaired=MUSDIR / 'fruitfly_six_leg_standing.xml',
              norepair=MUSDIR / 'fruitfly_six_leg_standing_noarmrepair.xml',
              qpos0bands=MUSDIR / 'fruitfly_six_leg_standing_qpos0bands.xml')
MESH = '/home/sensen/.cache/flygym_assets/neuromechfly_musculoskeletal_meshes_20260623a/'
TMP = Path('/tmp/s212')
CL = dict(model_loads=0, position_stage_calls=0, constraint_stage_calls=0, forward_calls=0, mj_step_calls=0)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prep(tag):
    if tag == 'frozen':
        return FROZEN
    TMP.mkdir(parents=True, exist_ok=True)
    out = TMP / (tag + '.xml')
    if not out.exists():
        out.write_text(ASSETS[tag].read_text().replace('./meshes/stl/', MESH))
    return out

st = np.load(S177)
qpos_rel = np.asarray(st['qpos'], float)
B_frozen = np.asarray(st['B'], float)
base, f0, J, T = (np.asarray(st[k], float) for k in ('base', 'f0', 'J', 'T'))
req = (-(base + J.T @ f0))[6:]
jn = [r['dof_name'] for r in json.load(open(ROWDOF))['rows']]
leg_of_row = [n[6:8] for n in jn]
LEGS = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']

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

def gain_of(length, vel, node):
    g = [float(x) for x in node.get('gainprm').split()]
    lr = [float(x) for x in node.get('lengthrange').split()]
    l0 = (lr[1] - lr[0]) / max(1e-15, g[1] - g[0])
    l = g[0] + (length - lr[0]) / max(1e-15, l0)
    v = vel / max(1e-15, l0 * g[6])
    y = g[8] - 1
    fv = 0.0 if v <= -1 else ((v + 1) ** 2 if v <= 0 else (g[8] - (y - v) ** 2 / max(1e-15, y) if v <= y else g[8]))
    return -g[2] * fl(l, g[4], g[5]) * fv

def dense_moment(m, d):
    out = np.zeros((m.nu, m.nv))
    for a in range(m.nu):
        nnz = int(d.moment_rownnz[a]); adr = int(d.moment_rowadr[a])
        for k in range(nnz):
            out[a, int(d.moment_colind[adr + k])] = d.actuator_moment[adr + k]
    return out

def probe(path_xml, qpos):
    m = mujoco.MjModel.from_xml_path(str(path_xml)); CL['model_loads'] += 1
    d = mujoco.MjData(m); d.qpos[:] = qpos; d.qvel[:] = 0.0
    mujoco.mj_fwdPosition(m, d); CL['position_stage_calls'] += 1
    return m, d, dense_moment(m, d)

# ---------------------------------------------------------------- Part A': the concrete instances
def site_pos(path, name):
    for s in ET.parse(path).getroot().iter('site'):
        if (s.get('name') or '') == name:
            return np.array([float(x) for x in s.get('pos').split()])
    return None

inst = {}
for leg, k_exp in (('LM', 1.3071), ('LH', 1.3408), ('RF', 1.0)):
    nm = leg + 'Tibia_flex_93434_pp21995'
    fz = site_pos(FROZEN, nm)
    rp = site_pos(ASSETS['repaired'], nm)
    if fz is None or rp is None:
        continue
    ratio = fz / rp
    inst[leg] = dict(site=nm, frozen=fz.tolist(), repaired=rp.tolist(), ratios=ratio.tolist(),
                     all_components_equal=bool(np.max(np.abs(ratio - ratio[0])) <= 1e-4),
                     ratio_value=float(ratio.mean()), expected_k=k_exp,
                     matches_expected=bool(abs(ratio.mean() - k_exp) <= 5e-4))

# ---------------------------------------------------------------- Part B: two poses
m0 = mujoco.MjModel.from_xml_path(str(FROZEN)); CL['model_loads'] += 1
qpos_key = np.asarray(m0.key_qpos[0], float) if m0.nkey else qpos_rel.copy()
poses = dict(keyframe=qpos_key, release=qpos_rel)
arms = {}
gains_tab = {}
for tag in ASSETS:
    for pname, q in poses.items():
        m, d, arm = probe(prep(tag), q)
        nodes = ET.parse(str(prep(tag))).getroot().find('actuator')
        g = np.array([gain_of(float(d.actuator_length[a]), 0.0, nodes[a]) for a in range(m.nu)])
        per = {}
        for leg in LEGS:
            for fam, pref in (('flex', leg + 'Tibia_flex'), ('ext', leg + 'Tibia_extensor')):
                aid = next((a for a in range(m.nu)
                            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or '').startswith(pref)), None)
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'joint_' + leg + 'Tibia_pitch')
                if aid is None or jid < 0:
                    continue
                per[leg + '_' + fam] = dict(arm=float(arm[aid, int(m.jnt_dofadr[jid])]), gain=float(g[aid]))
        arms[tag + '|' + pname] = per
        if pname == 'release':
            gains_tab[tag] = g
        if tag == 'frozen' and pname == 'release':
            arm_rel, gain_rel, nodes_frozen = arm, g, nodes
            d_rel = d

# multiplicative decomposition of the release-pose collapse
decomp = {}
for leg, front in (('LM', 'LF'), ('LH', 'LF'), ('RM', 'RF'), ('RH', 'RF')):
    f_key = 'frozen|release'
    r_key = 'repaired|release'
    fz_leg = abs(arms[f_key][leg + '_flex']['arm']); fz_front = abs(arms[f_key][front + '_flex']['arm'])
    rp_leg = abs(arms[r_key][leg + '_flex']['arm']); rp_front = abs(arms[r_key][front + '_flex']['arm'])
    kf_leg = abs(arms['frozen|keyframe'][leg + '_flex']['arm']); kf_front = abs(arms['frozen|keyframe'][front + '_flex']['arm'])
    direct = fz_front / fz_leg if fz_leg else None
    attach = rp_front / rp_leg if rp_leg else None
    pose = (fz_leg / kf_leg) / (fz_front / kf_front) if (kf_leg and kf_front) else None
    prod = (attach * pose) if (attach and pose) else None
    decomp[leg] = dict(direct_ratio=direct, attachment_factor=attach, pose_factor=pose, product=prod,
                       rel_error=(abs(prod - direct) / direct if (prod and direct) else None))
(DIR / 'ARM_DECOMPOSITION.json').write_text(json.dumps(dict(poses={k: v.tolist() for k, v in poses.items()},
                                                            arms=arms, decomposition=decomp,
                                                            calls=dict(CL)), indent=2, default=str))

# ---------------------------------------------------------------- Part C: site-port variant
txt = FROZEN.read_text()
lines = txt.split('\n')
changed = []
out_lines = []
rep_sites = {}
for s in ET.parse(str(prep('repaired'))).getroot().iter('site'):
    nm = s.get('name') or ''
    if nm[:2] in ('LM', 'LH', 'RM', 'RH') and ('Tibia_flex' in nm or 'Tibia_extensor' in nm):
        rep_sites[nm] = s.get('pos')
for ln in lines:
    hit = None
    for nm, pos in rep_sites.items():
        if 'name="' + nm + '"' in ln and 'pos="' in ln:
            hit = (nm, pos)
            break
    if hit:
        nm, pos = hit
        new = ln.split('pos="')[0] + 'pos="' + pos + '"' + ln.split('pos="', 1)[1].split('"', 1)[1]
        if new != ln:
            changed.append(nm)
        out_lines.append(new)
    else:
        out_lines.append(ln)
variant_txt = '\n'.join(out_lines)
TMP.mkdir(parents=True, exist_ok=True)
variant = TMP / 'frozen_tibia_sites_from_repaired.xml'
variant.write_text(variant_txt)
m_v, d_v, arm_v = probe(variant, qpos_rel)
nodes_v = ET.parse(str(variant)).getroot().find('actuator')
gain_v = np.array([gain_of(float(d_v.actuator_length[a]), 0.0, nodes_v[a]) for a in range(m_v.nu)])
same_gain = [a for a in range(m_v.nu) if abs(gain_v[a] - gain_rel[a]) < 1e-12]
B_v = arm_v.T * gain_v[None, :]
A_v = (B_v @ T)[6:, :]
A_frozen = (B_frozen @ T)[6:, :]

def solve(Amat, reqv, box):
    n, mm_ = Amat.shape[1], reqv.size
    bnds = [(0, None)] * n if box is None else [(0, box)] * n
    lp = linprog(np.r_[np.zeros(n), 1.0],
                 A_ub=np.vstack((np.column_stack((Amat, -np.ones(mm_))), np.column_stack((-Amat, -np.ones(mm_))))),
                 b_ub=np.r_[reqv, -reqv], bounds=bnds + [(0, None)], method='highs')
    return (float(lp.x[-1]) if lp.success else None), (lp.x[:-1] if lp.success else None)

def per_leg(Amat):
    out = {}
    for L in LEGS:
        rows = [i for i in range(42) if leg_of_row[i] == L]
        cols = [c for c in range(Amat.shape[1]) if np.abs(Amat[rows, c]).max() > 1e-12
                and np.abs(np.delete(Amat, rows, axis=0)[:, c]).max() <= 1e-12]
        out[L] = solve(Amat[rows][:, cols], req[rows], 1.0)[0]
    return out

def s_rows(Amat, cols=None):
    out = {}
    for i in range(42):
        a = Amat[i, :]
        pos = a[a > 1e-9]
        best = float(pos.max()) if pos.size else 0.0
        out[jn[i]] = dict(s=(float(abs(req[i]) / best) if best > 1e-9 else None), maxpos=best, req=float(req[i]))
    return out

variants = dict(
    frozen=dict(global_box1=solve(A_frozen, req, 1.0)[0], per_leg=per_leg(A_frozen), s=s_rows(A_frozen)),
    ported=dict(global_box1=solve(A_v, req, 1.0)[0], per_leg=per_leg(A_v), s=s_rows(A_v)))
defect = ['joint_LHTibia_pitch', 'joint_LMTibia_pitch', 'joint_RHTibia_pitch', 'joint_RMTibia_pitch',
          'joint_LMCoxa_roll', 'joint_RMCoxa_pitch']
defect_rows = {k: dict(frozen=variants['frozen']['s'][k], ported=variants['ported']['s'][k]) for k in defect}
all_le_1 = all((variants['ported']['s'][k]['s'] is not None and variants['ported']['s'][k]['s'] <= 1.0) for k in defect)
port_lp = dict(changed_sites=sorted(changed), n_changed=len(changed), variant_sha256=sha(variant),
               frozen_sha256=sha(FROZEN), frozen_unchanged=bool(sha(FROZEN) ==
                                                               '6e77df89b742c16d1a8b4a2bd3f4e99e0a909c291f0a67ff0515ec11ecabc2a4'),
               unchanged_gain_count=len(same_gain), n_actuators=int(m_v.nu),
               B_port_minus_frozen_max=float(np.abs(B_v - B_frozen).max()),
               variants=variants, defect_rows=defect_rows,
               label=('ATTACHMENT_PORT_CLOSES_GAP' if all_le_1 and variants['ported']['global_box1'] <
                      0.7 * variants['frozen']['global_box1'] else
                      ('ATTACHMENT_PORT_PARTIAL' if variants['ported']['global_box1'] < variants['frozen']['global_box1']
                       else 'ATTACHMENT_PORT_NO_EFFECT')),
               note='hypothetical port only; the repaired attachment geometry is itself suspect and this is not a repair of the '
                    'frozen model, and no movement claim follows')
(DIR / 'SITE_PORT_VARIANT.json').write_text(json.dumps(port_lp, indent=2, default=str))
(DIR / 'ARM_PORT_LP.json').write_text(json.dumps(dict(exercise=variants['frozen'], ported=variants['ported'],
                                                      label=port_lp['label']), indent=2, default=str))
(DIR / 'SITE_INSTANCES.json').write_text(json.dumps(inst, indent=2, default=str))
print(json.dumps(dict(dir=str(DIR), instances={k: dict(ratio=round(v['ratio_value'], 4), expected=v['expected_k'],
                                                       ok=v['matches_expected']) for k, v in inst.items()},
                      decomp={k: dict(direct=None if v['direct_ratio'] is None else round(v['direct_ratio'], 2),
                                      attach=None if v['attachment_factor'] is None else round(v['attachment_factor'], 2),
                                      pose=None if v['pose_factor'] is None else round(v['pose_factor'], 2),
                                      rel_err=None if v['rel_error'] is None else round(v['rel_error'], 4))
                              for k, v in decomp.items()},
                      port=dict(label=port_lp['label'], n_changed=len(changed), unchanged_gain=len(same_gain),
                                global_frozen=variants['frozen']['global_box1'], global_ported=variants['ported']['global_box1'],
                                per_leg_frozen={k: (None if v is None else round(v, 4)) for k, v in variants['frozen']['per_leg'].items()},
                                per_leg_ported={k: (None if v is None else round(v, 4)) for k, v in variants['ported']['per_leg'].items()}),
                      calls=CL), indent=1, default=str))
