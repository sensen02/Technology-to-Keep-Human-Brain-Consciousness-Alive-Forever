"""S210 Part 0 (site-coordinate criterion + behaviour criterion at keyframe) + Part B (pose sweep)."""
import json
import hashlib
import numpy as np
import mujoco
import xml.etree.ElementTree as ET
from pathlib import Path

DIR = sorted(Path('/home/sensen/Desktop/engine修复验收/s210-model-lineage-and-arm-symmetry').iterdir())[-1]
DESK = Path('/home/sensen/Desktop/engine修复验收')
S177 = DESK / 's177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz'
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
MUSCLES = Path('/run/media/sensen/Data2/cell_wound_prototype/outputs/muscles_six_legs')
ASSETS = dict(repaired=MUSCLES / 'fruitfly_six_leg_standing.xml',
              norepair=MUSCLES / 'fruitfly_six_leg_standing_noarmrepair.xml',
              qpos0bands=MUSCLES / 'fruitfly_six_leg_standing_qpos0bands.xml',
              frozen=XML)
SIX = ['LHTibia_pitch', 'LMTibia_pitch', 'RHTibia_pitch', 'RMTibia_pitch', 'LMTrochanter_pitch', 'RMTrochanter_pitch']

def sites(path, prefix):
    root = ET.parse(path).getroot()
    out = {}
    for s in root.iter('site'):
        nm = s.get('name') or ''
        if nm.startswith(prefix):
            out[nm] = [float(x) for x in s.get('pos').split()]
    return out

# ---------------------------------------------------------------- Part 0a: site coordinates
suffixes = ('_bifemlh_r-P2', '_pp21995')
coord = {}
for tag, path in ASSETS.items():
    per = {}
    for leg in ('LF', 'LH', 'LM', 'RF', 'RH', 'RM'):
        for suf in ('_bifemlh_r-P2', '_pp21995'):
            key = f'{leg}Tibia_flex_93434{suf}'
            s = sites(path, key)
            if key in s:
                per[key] = s[key]
    coord[tag] = per
matches = {}
for leg in ('LF', 'LH', 'LM', 'RF', 'RH', 'RM'):
    for suf in ('_bifemlh_r-P2', '_pp21995'):
        key = f'{leg}Tibia_flex_93434{suf}'
        if key not in coord['frozen']:
            continue
        fz = np.array(coord['frozen'][key])
        row = {}
        for tag in ('repaired', 'norepair', 'qpos0bands'):
            if key in coord[tag]:
                row[tag] = dict(max_abs_diff=float(np.max(np.abs(fz - np.array(coord[tag][key])))),
                                equal_to_1e_6=bool(np.max(np.abs(fz - np.array(coord[tag][key]))) <= 1e-6))
        matches[key] = row
n_eq = {tag: sum(1 for k, v in matches.items() if v.get(tag, {}).get('equal_to_1e_6')) for tag in ('repaired', 'norepair', 'qpos0bands')}
coord_verdict = max(n_eq, key=n_eq.get)

# ---------------------------------------------------------------- Part 0b + Part B: model work
st = np.load(S177)
qpos_release = np.asarray(st['qpos'], float)
m = mujoco.MjModel.from_xml_path(str(XML))
qpos0 = np.asarray(m.key_qpos[0], float) if m.nkey else qpos_release.copy()
cap = dict(model_loads=1, position_stage_calls=0, constraint_stage_calls=0, forward_calls=0, mj_step_calls=0)


def joint_ranges(i, jname):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
    lo, hi = m.jnt_range[jid]
    return float(lo), float(hi), bool(m.jnt_limited[jid])


def arm_matrix(q):
    """dense actuator moment at q via the tendon Jacobian (trntype=3); one position-stage call."""
    d = mujoco.MjData(m)
    d.qpos[:] = q
    d.qvel[:] = 0.0
    mujoco.mj_fwdPosition(m, d)
    cap['position_stage_calls'] += 1
    flat = np.asarray(d.ten_J)
    arm = np.zeros((m.nu, m.nv))
    for a in range(m.nu):
        t = int(m.actuator_trnid[a, 0])
        g = float(m.actuator_gear[a, 0])
        nnz = int(m.ten_J_rownnz[t])
        adr = int(m.ten_J_rowadr[t])
        for k in range(nnz):
            arm[a, int(m.ten_J_colind[adr + k])] += g * flat[adr + k]
    return arm


ARM_KEY = arm_matrix(qpos0)

# behaviour criterion: unidirectional joints at the keyframe
per_joint = {}
for j in range(m.njnt):
    jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
    if not jn.startswith('joint_') or m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
        continue
    dof = int(m.jnt_dofadr[j])
    a = ARM_KEY[:, dof]
    per_joint[jn] = dict(min=float(a.min()), max=float(a.max()),
                         unidirectional=bool(a.max() <= 1e-12 or a.min() >= -1e-12))
uni_at_key = sorted([k for k, v in per_joint.items() if v['unidirectional']])
seams_six = sorted(['joint_' + s for s in SIX])
behaviour = dict(qpos_source='keyframe qpos0' if m.nkey else 'release pose (model has no keyframe)',
                 n_hinges=len(per_joint), unidirectional=sorted(uni_at_key),
                 matches_seams_list=bool(sorted(uni_at_key) == seams_six),
                 seams_list=seams_six,
                 missing_from_seams=[x for x in seams_six if x not in uni_at_key],
                 extra_not_in_seams=[x for x in uni_at_key if x not in seams_six])
lineage_verdict = 'FROZEN_MODEL_IS_POST_REPAIR' if len(uni_at_key) <= 6 and coord_verdict == 'repaired' else (
    'FROZEN_MODEL_IS_PRE_REPAIR' if coord_verdict == 'norepair' else 'LINEAGE_UNDETERMINED')

# ---------------------------------------------------------------- Part B: pose sweep
weak = ['LH', 'LM', 'RH', 'RM']
control = ['LF', 'RF']
OFFS = [0.0, 0.05, 0.2, 0.45]
sweep = []
for leg in weak + control:
    jn = f'joint_{leg}Tibia_pitch'
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
    dof = int(m.jnt_dofadr[jid])
    adr = int(m.jnt_qposadr[jid])
    base_ang = float(qpos_release[adr])
    lo, hi, limited = joint_ranges(jid, jn)
    for off in OFFS:
        for sign in ((0.0,) if off == 0.0 else (1.0, -1.0)):
            ang = base_ang + sign * off
            in_range = bool((not limited) or (lo - 1e-12 <= ang <= hi + 1e-12))
            rec = dict(leg=leg, joint=jn, offset=sign * off, angle=ang, release_angle=base_ang,
                       in_range=in_range, jnt_range=[lo, hi], jnt_limited=limited)
            if not in_range:
                rec['status'] = 'OUT_OF_RANGE'
                sweep.append(rec)
                continue
            q = qpos_release.copy()
            q[adr] = ang
            a = arm_matrix(q)[:, dof]
            pos = a[a > 1e-12]
            neg = a[a < -1e-12]
            rec.update(status='OK', maxpos=float(pos.max()) if pos.size else 0.0,
                       maxneg=float(neg.min()) if neg.size else 0.0,
                       min_abs_arm=float(np.abs(a).min()))
            sweep.append(rec)
(DIR / 'POSESENS.json').write_text(json.dumps(dict(calls=cap, behaviour=behaviour, per_joint=per_joint,
                                                   sweep=sweep, offsets=OFFS, weak=weak, control=control),
                                             indent=2, default=str))
(DIR / 'MODEL_LINEAGE.json').write_text(json.dumps(dict(
    frozen_xml=dict(path=str(XML), sha256=hashlib.sha256(XML.read_bytes()).hexdigest()),
    assets={k: dict(path=str(v), sha256=hashlib.sha256(v.read_bytes()).hexdigest()) for k, v in ASSETS.items()},
    site_coordinate_criterion=dict(per_site=matches, n_equal={k: int(v) for k, v in n_eq.items()},
                                   verdict=coord_verdict,
                                   note='the frozen XML sites are compared against the repaired, the no-arm-repair and the '
                                        'qpos0bands assets; the verdict is whichever asset they match within 1e-6'),
    behaviour_criterion=behaviour,
    lineage_verdict=lineage_verdict), indent=2, default=str))
print(json.dumps(dict(coord_verdict=coord_verdict, n_equal=n_eq, behaviour=dict(
    n_uni=len(uni_at_key), matches_seams=behaviour['matches_seams_list'],
    missing=behaviour['missing_from_seams'][:3], extra=behaviour['extra_not_in_seams'][:3]),
    lineage=lineage_verdict, calls=cap), indent=1, default=str))
