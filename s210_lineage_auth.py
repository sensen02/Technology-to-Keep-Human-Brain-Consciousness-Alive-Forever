"""S210 Part 0 (lineage) + Part A (authority decomposition, B = arm x gain). No model load."""
import json
import datetime
import hashlib
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path('/home/sensen/Desktop/engine修复验收/s210-model-lineage-and-arm-symmetry')
STAMP = datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S%fZ')
DIR = ROOT / STAMP
DIR.mkdir(parents=True, exist_ok=False)

DESK = Path('/home/sensen/Desktop/engine修复验收')
S154 = DESK / 's154-activation-rate/20261024T215751775489Z/init80ms'
S177 = DESK / 's177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz'
S27 = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z')
XML = S27 / 'MODEL_CALIB_H_VERIFIED.xml'
MUSCLES = Path('/run/media/sensen/Data2/cell_wound_prototype/outputs/muscles_six_legs')
REPAIRED = MUSCLES / 'fruitfly_six_leg_standing.xml'
NOREPAIR = MUSCLES / 'fruitfly_six_leg_standing_noarmrepair.xml'
QPOS0BANDS = MUSCLES / 'fruitfly_six_leg_standing_qpos0bands.xml'
ROWDOF = sorted(DESK.glob('s179-*/*/ROW_DOF_MAP.json'))[-1]
CENSUS = json.load(open(DESK / 's204-readout-calibration-audit/20261025T073450411660Z/READOUT_CENSUS.json'))

EPS = 1e-9
prereg = dict(
    round='S210', directory=str(DIR), written_before_runs=True,
    written_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
    purpose='pin the frozen model lineage, decompose authority into arm x gain, test pose dependence, and test an arm-substitution counterfactual',
    labels=dict(lineage=['FROZEN_MODEL_IS_PRE_REPAIR', 'FROZEN_MODEL_IS_POST_REPAIR', 'LINEAGE_UNDETERMINED'],
                mechanism=['DEFICIT_IS_MOMENT_ARM', 'DEFICIT_IS_GAIN', 'DEFICIT_IS_MAPPING'],
                pose=['POSE_ARTIFACT', 'HOLDS_OVER_JOINT_RANGE', 'OUT_OF_RANGE'],
                rule='all labels are reported side by side; the lineage label is the main label; none may mask another'),
    sweep=dict(dof='tibia_pitch of the four weak legs', offsets=[0.0, 0.05, 0.1, 0.2, 0.3, 0.45],
               rule='both signs of every offset are sampled; any sample outside jnt_range is recorded OUT_OF_RANGE and excluded '
                    'from the conclusion; the front legs are swept as the control'),
    arm_substitution=('only the moment arms of the four/five weak-leg tibia actuators are replaced by the same-side front-leg '
                      'homologue values (LH, LM <- LF; RH, RM <- RF, handed by side); T and the box are unchanged; this is a '
                      'COUNTERFACTUAL and is never written back to any frozen artefact'),
    budget=dict(model_loads=2, position_stage_calls_max=60, constraint_stage_calls_max=4, forward_calls=0, mj_step_calls=0,
                lp_max=80, neural_ticks=0, training_steps=0, xml_changes=0),
    forbidden=dict(words=['改善', 'improved'], note='no movement claim; force units NOT_ESTABLISHED'),
)
(DIR / 'PREREG.json').write_text(json.dumps(prereg, indent=2, ensure_ascii=False))

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()

# ---------------------------------------------------------------- Part 0: lineage chain
ledger_candidates = sorted(S27.parent.glob('MODEL_LEDGER_VERIFIED.json')) + sorted(DESK.glob('**/MODEL_LEDGER_VERIFIED.json'))
ledger_path = ledger_candidates[-1] if ledger_candidates else None
chain = []
if ledger_path:
    led = json.loads(ledger_path.read_text())
    cur = led
    for k in ('chain', 'parents', 'ancestors', 'history'):
        if isinstance(cur, dict) and k in cur:
            chain = cur[k]
            break
lineage_files = {}
for p in (XML, REPAIRED, NOREPAIR, QPOS0BANDS):
    lineage_files[str(p)] = dict(sha256=sha(p), exists=p.exists(), size=p.stat().st_size if p.exists() else None)

# ---------------------------------------------------------------- Part A: B = arm x gain
z = np.load(S154 / 'TRAJECTORY.npz')
st = np.load(S177)
arm = np.asarray(z['actuator_moment'][0], float)      # (90, 48): [actuator, dof]
length = np.asarray(z['actuator_length'][0], float)   # (90,)
vel = np.asarray(z['actuator_velocity'][0], float) if 'actuator_velocity' in z.files else np.zeros(90)
T = np.asarray(st['T'], float)
B = np.asarray(st['B'], float)
base = np.asarray(st['base'], float)
f0 = np.asarray(st['f0'], float)
J = np.asarray(st['J'], float)
qpos = np.asarray(st['qpos'], float)
req = (-(base + J.T @ f0))[6:]
jn = [r['dof_name'] for r in json.load(open(ROWDOF))['rows']]
leg_of_row = [n[6:8] for n in jn]
name_of_act = {c['actuator_index']: c['actuator_name'] if 'actuator_name' in c else c['actuator'] for c in CENSUS['rows']}

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


def gain_of(length, vel, node):
    g = list(map(float, node.get('gainprm').split()))
    lr = list(map(float, node.get('lengthrange').split()))
    l0 = (lr[1] - lr[0]) / max(1e-15, g[1] - g[0])
    l = g[0] + (length - lr[0]) / max(1e-15, l0)
    v = vel / max(1e-15, l0 * g[6])
    y = g[8] - 1
    if v <= -1:
        fv = 0.0
    elif v <= 0:
        fv = (v + 1) ** 2
    elif v <= y:
        fv = g[8] - (y - v) ** 2 / max(1e-15, y)
    else:
        fv = g[8]
    return -g[2] * fl(l, g[4], g[5]) * fv, l, fv

gains = np.zeros(90)
lnorms = np.zeros(90)
fvs = np.zeros(90)
for a in range(90):
    gains[a], lnorms[a], fvs[a] = gain_of(length[a], vel[a], nodes[a])

B_recon = arm.T * gains[None, :]                       # (48, 90)
delta = np.abs(B_recon - B)
identity = dict(max_abs_delta=float(delta.max()), all_within_1e_9=bool(delta.max() <= 1e-9),
                n_zero_gains=int((gains == 0.0).sum()),
                gain_min=float(np.abs(gains).min()), gain_max=float(np.abs(gains).max()))

# cross-leg homologue table
def group_of(name):
    for key in ('Tibia_flex', 'Tibia_extensor'):
        if key in name:
            return key
    return None

homologs = {}
for a in range(90):
    g = group_of(name_of_act[a])
    if g:
        homologs.setdefault(g, {})[name_of_act[a][:2]] = a
rows_auth = []
for i in range(42):
    for a in range(90):
        if abs(B[6 + i, a]) > EPS:
            rows_auth.append(dict(row=i, dof=6 + i, dof_name=jn[i], leg=leg_of_row[i], actuator_index=a,
                                  actuator=name_of_act[a], arm=float(arm[a, 6 + i]), gain=float(gains[a]),
                                  B=float(B[6 + i, a]), t_row_sum=float(np.abs(T[a, :]).sum()),
                                  commandable=bool(np.abs(T[a, :]).sum() > 0),
                                  same_sign_as_req=bool(np.sign(B[6 + i, a]) == np.sign(req[i]))))
homo_table = {}
for g, legs in homologs.items():
    homo_table[g] = {}
    for leg, a in sorted(legs.items()):
        homo_table[g][leg] = dict(actuator_index=a, actuator=name_of_act[a], gain=float(gains[a]),
                                  arm=float(arm[a, 6 + 13]) if False else None)
    for leg, a in sorted(legs.items()):
        idx = [i for i in range(42) if abs(B[6 + i, a]) > EPS]
        homo_table[g][leg]['B_by_row'] = {jn[i]: float(B[6 + i, a]) for i in idx}
        homo_table[g][leg]['arms_by_row'] = {jn[i]: float(arm[a, 6 + i]) for i in idx}
arm_spread = {}
gain_spread = {}
for g, legs in homo_table.items():
    for row_name in {k for leg in legs.values() for k in leg.get('arms_by_row', {})}:
        vals = [leg['arms_by_row'][row_name] for leg in legs.values() if row_name in leg.get('arms_by_row', {})]
        if len(vals) >= 2:
            arm_spread[f'{g}@{row_name}'] = dict(min=float(np.min(vals)), max=float(np.max(vals)),
                                                ratio=float(abs(np.max(vals)) / abs(np.min(vals))) if np.min(vals) != 0 else None)
    gv = [leg['gain'] for leg in legs.values()]
    gain_spread[g] = dict(values=gv, min=float(np.min(gv)), max=float(np.max(gv)),
                          rel_spread=float((np.max(gv) - np.min(gv)) / abs(np.mean(gv))))

# per-row required multiple s = |req| / max coefficient of the needed sign
def s_row(i):
    a = B[6 + i, :]
    need_pos = req[i] > 0
    best = a[a > EPS].max() if need_pos and (a > EPS).any() else (-a[a < -EPS].min() if (a < -EPS).any() else 0.0)
    return (float(abs(req[i]) / best) if best > EPS else None), best

s_table = []
for i in range(42):
    s, best = s_row(i)
    s_table.append(dict(row=i, dof=6 + i, dof_name=jn[i], leg=leg_of_row[i], required=float(req[i]),
                        max_coefficient_needed=float(best), s_needed=s,
                        needs_more_than_1=bool(s is not None and s > 1.0)))
(DIR / 'AUTHORITY_DECOMPOSITION.json').write_text(json.dumps(dict(
    identity=identity, n_entries_48x90=int(B.size), rows=rows_auth,
    note='B[d,a] = arm[a,d] * gain[a] with arm from the frozen frame-0 actuator_moment and gain from the frozen XML muscle '
         'parameters evaluated at the frozen length/velocity'), indent=2, default=str))
with open(DIR / 'AUTHORITY_DECOMPOSITION.csv', 'w') as f:
    f.write('row,dof,dof_name,leg,actuator_index,actuator,arm,gain,B,commandable,same_sign_as_req\n')
    for r in rows_auth:
        f.write('%d,%d,%s,%s,%d,%s,%.9g,%.9g,%.9g,%d,%d\n' % (r['row'], r['dof'], r['dof_name'], r['leg'],
                                                              r['actuator_index'], r['actuator'], r['arm'],
                                                              r['gain'], r['B'], int(r['commandable']),
                                                              int(r['same_sign_as_req'])))
(DIR / 'ARM_SYMMETRY.json').write_text(json.dumps(dict(homologs=homo_table, arm_spread=arm_spread,
                                                        gain_spread=gain_spread, s_table=s_table), indent=2, default=str))
print(json.dumps(dict(dir=str(DIR), identity=identity,
                      s_over_1=[(t['dof_name'], round(t['s_needed'], 2)) for t in s_table if t['needs_more_than_1']],
                      flex_gain={L: round(homo_table['Tibia_flex'][L]['gain'], 2) for L in homo_table.get('Tibia_flex', {})},
                      flex_arm={L: {k: round(v, 6) for k, v in homo_table['Tibia_flex'][L]['arms_by_row'].items()}
                                for L in homo_table.get('Tibia_flex', {})},
                      gain_spread=gain_spread, arm_spread_keys=list(arm_spread)[:4]), indent=1, default=str))
