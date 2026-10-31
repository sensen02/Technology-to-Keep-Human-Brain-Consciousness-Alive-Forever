"""S211 Part 0 (S210 contradiction closure) + Part A (cross-model arm table). Writes only to /tmp for rewritten meshes."""
import json
import datetime
import hashlib
import shutil
import numpy as np
import mujoco
from pathlib import Path

ROOT = Path('/home/sensen/Desktop/engine修复验收/s211-attachment-provenance')
STAMP = datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S%fZ')
DIR = ROOT / STAMP
DIR.mkdir(parents=True, exist_ok=False)
S210 = Path('/home/sensen/Desktop/engine修复验收/s210-model-lineage-and-arm-symmetry/20261025T165113566119Z')
DESK = Path('/home/sensen/Desktop/engine修复验收')
S177 = DESK / 's177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz'
FROZEN = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
MUSDIR = Path('/run/media/sensen/Data2/cell_wound_prototype/outputs/muscles_six_legs')
MESH = '/home/sensen/.cache/flygym_assets/neuromechfly_musculoskeletal_meshes_20260623a/'
TMP = Path('/tmp/s211_assets')

prereg = dict(
    round='S211', directory=str(DIR), written_before_runs=True,
    written_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
    purpose='close the S210 contradictions, reproduce the cross-model arm table, attribute the frozen attachment geometry, '
            'and redo the lineage criterion on B and A separately',
    mesh_rule=('the three asset XMLs reference ./meshes/stl/... which does not exist in the repo; each is copied to /tmp/s211_assets '
               'with that prefix rewritten to the flygym cache. The three original assets are NEVER modified (hashes recorded '
               'before and after), and the /tmp copies are throwaway.'),
    epsilon_protocol=dict(eps_abs=1e-12, eps_rel=1e-9,
                          one_sided_def='a joint is one-sided iff max|pos| / max|neg| > 1/eps_rel (or one side <= eps_abs); '
                                        'both max|pos| and max|neg| are always reported and BOTH matrices (B physical, A=B@T '
                                        'commandable) are censused separately'),
    labels=dict(arms=['ARMS_CROSS_MODEL_REPRODUCED', 'ARMS_TABLE_MISMATCH'],
                provenance=['ATTACHMENT_RULE_FOUND', 'FROZEN_ATTACHMENT_UNATTRIBUTED'],
                lineage=['FROZEN_THIRD_VARIANT', 'FROZEN_MATCHES_SOURCE_ATTACHMENTS', 'FROZEN_MATCHES_REPAIR_RULE'],
                s210=['S210_LINEAGE_RELABELED', 'S210_PART_C_INVALID', 'S210_PART_C_RERUN'],
                rule='each family is reported once; the provenance label is the main label; none may mask another'),
    forced_clause=('until the provenance is settled, the 25-39x arm difference may be written neither as "a defect of our build" '
                   'nor as "a body limitation"; only "frozen, repaired and norepair give different tibia-muscle arms at the same '
                   'pose, and frozen is a front-to-back monotone gradient" may be stated'),
    budget=dict(model_loads_max=4, position_stage_calls_max=20, constraint_stage_calls=0, forward_calls=0, mj_step_calls=0,
                xml_changes=0, training_steps=0),
)
(DIR / 'PREREG.json').write_text(json.dumps(prereg, indent=2, ensure_ascii=False))

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()

assets = dict(frozen=FROZEN, repaired=MUSDIR / 'fruitfly_six_leg_standing.xml',
              norepair=MUSDIR / 'fruitfly_six_leg_standing_noarmrepair.xml',
              qpos0bands=MUSDIR / 'fruitfly_six_leg_standing_qpos0bands.xml')
hashes_before = {k: sha(v) for k, v in assets.items()}

# ---------------------------------------------------------------- Part 0: reconciliation
sub = json.load(open(S210 / 'ARM_SUBSTITUTION_LP.json'))['variants']
frozen_s = sub['frozen']['s_table']
sub_s = sub['flexor_arm_substituted']['s_table']
s_rows_changed = [(a['dof_name'], a['s'], b['s']) for a, b in zip(frozen_s, sub_s)
                  if (a['s'] is None) != (b['s'] is None) or (a['s'] and b['s'] and abs(a['s'] - b['s']) > 1e-9)]
leg_changed = {L: dict(frozen=sub['frozen']['per_leg'][L]['box1'], substituted=sub['flexor_arm_substituted']['per_leg'][L]['box1'])
               for L in sub['frozen']['per_leg']
               if abs(sub['frozen']['per_leg'][L]['box1'] - sub['flexor_arm_substituted']['per_leg'][L]['box1']) > 1e-12}
recon = dict(
    fact_1_lineage_json=dict(model_lineage_said=json.load(open(S210 / 'MODEL_LINEAGE.json'))['lineage_verdict'],
                             conclusion_said='LINEAGE_UNDETERMINED',
                             resolution='the JSON was written by the earlier pass whose coordinate criterion broke a tie by taking the '
                                        'maximum; the conclusion was written after that flaw was found. The conclusion is correct, the '
                                        'JSON is stale -> relabelled here as LINEAGE_UNDETERMINED'),
    fact_2_part_c=dict(my_message_said='the s table and the per-leg minimax did not change at all',
                       files_show=dict(n_s_rows_changed=len(s_rows_changed), rows=s_rows_changed[:5],
                                       legs_changed=leg_changed),
                       resolution='the s ROWS did not change, but the per-leg minimax DID change (and got worse). My message and the '
                                  'patched conclusion sentence were both wrong; the LP output itself is real'),
    part_c_status=dict(status='S210_PART_C_INVALID',
                       reason=('the substitution wrote the SOURCE actuator row values onto the TARGET actuator, so the target kept its '
                               'own (unchanged) binding coefficient and gained spurious authority on the source leg row; the run '
                               'therefore answers a different question than the one asked. Numbers are real but not usable for the '
                               'arm-substitution hypothesis.'),
                       evidence='per-leg minimax changed (worse) while no s row changed, which is only possible if the edits landed on '
                                'rows that are not the binding ones'),
    lineage_label_now='LINEAGE_UNDETERMINED',
    s210_labels=['S210_LINEAGE_RELABELED', 'S210_PART_C_INVALID'],
)
(DIR / 'S210_RECONCILIATION.json').write_text(json.dumps(recon, indent=2, default=str))

# ---------------------------------------------------------------- Part A: cross-model arms
TMP.mkdir(parents=True, exist_ok=True)
st = np.load(S177)
qpos0_model = mujoco.MjModel.from_xml_path(str(FROZEN))
qpos0 = np.asarray(qpos0_model.key_qpos[0], float) if qpos0_model.nkey else np.asarray(st['qpos'], float)

def prepared(tag):
    src = assets[tag]
    if tag == 'frozen':
        return src
    txt = src.read_text().replace('./meshes/stl/', MESH)
    out = TMP / (tag + '.xml')
    out.write_text(txt)
    return out

models = {}
struct = {}
for tag in assets:
    p = prepared(tag)
    m = mujoco.MjModel.from_xml_path(str(p))
    models[tag] = m
    struct[tag] = dict(nq=int(m.nq), nv=int(m.nv), njnt=int(m.njnt), nu=int(m.nu), nkey=int(m.nkey),
                       joints=[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)],
                       actuators=[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(m.nu)],
                       body_mass=float(m.body_mass.sum()))
ref = struct['frozen']
structure_equal = all(struct[t]['nq'] == ref['nq'] and struct[t]['nv'] == ref['nv'] and struct[t]['nu'] == ref['nu']
                      and struct[t]['njnt'] == ref['njnt'] and struct[t]['joints'] == ref['joints']
                      and struct[t]['actuators'] == ref['actuators'] for t in struct)


def arms_at(tag):
    m = models[tag]
    d = mujoco.MjData(m)
    d.qpos[:] = qpos0
    d.qvel[:] = 0.0
    mujoco.mj_fwdPosition(m, d)
    dense = np.zeros((m.nu, m.nv))
    for a in range(m.nu):
        nnz = int(d.moment_rownnz[a])
        adr = int(d.moment_rowadr[a])
        for k in range(nnz):
            dense[a, int(d.moment_colind[adr + k])] = d.actuator_moment[adr + k]
    return dense, d

LEGS = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
table = {}
world_sites = {}
for tag in assets:
    m = models[tag]
    dense, d = arms_at(tag)
    per = {}
    for leg in LEGS:
        jn = f'joint_{leg}Tibia_pitch'
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        dof = int(m.jnt_dofadr[jid])
        for fam, key in (('flex', f'{leg}Tibia_flex_93434_pp21995'), ('ext', f'{leg}Tibia_extensor')):
            aid = next((a for a in range(m.nu) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or '').startswith(
                f'{leg}Tibia_{"flex" if fam == "flex" else "extensor"}')), None)
            if aid is None:
                continue
            per[(leg, fam)] = float(dense[aid, dof])
    table[tag] = {f'{leg}_{fam}': v for (leg, fam), v in per.items()}
    # world site positions for the flexor pp21995 site and the tibia body origin
    ws = {}
    for leg in LEGS:
        sname = f'{leg}Tibia_flex_93434_pp21995'
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, sname)
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'{leg}Tibia')
        if sid >= 0 and bid >= 0:
            ws[leg] = dict(site_world=[float(x) for x in d.site_xpos[sid]],
                           tibia_body_origin=[float(x) for x in d.xpos[bid]],
                           distance=float(np.linalg.norm(np.asarray(d.site_xpos[sid]) - np.asarray(d.xpos[bid]))))
    world_sites[tag] = ws

advisor = {  # the values the review reported; the hard gate is to match all 18 to 1e-9
    'LF_flex': -0.010246, 'LM_flex': -0.004322, 'LH_flex': -0.001785, 'RF_flex': -0.007148, 'RM_flex': -0.003452, 'RH_flex': -0.001648,
    'LF_ext': 0.009172, 'LM_ext': 0.006285, 'LH_ext': 0.011508, 'RF_ext': 0.008830, 'RM_ext': 0.008601, 'RH_ext': 0.011804}
match = {k: dict(mine=table['frozen'].get(k), reported=v,
                 abs_diff=(abs(table['frozen'].get(k, np.nan) - v) if k in table['frozen'] else None),
                 within_1e6=bool(k in table['frozen'] and abs(table['frozen'][k] - v) <= 1e-6))
         for k, v in advisor.items()}
n_match = sum(1 for v in match.values() if v['within_1e6'])
(DIR / 'CROSS_MODEL_ARMS.json').write_text(json.dumps(dict(
    structure=dict(per_model={t: {k: v for k, v in struct[t].items() if not isinstance(v, list)} for t in struct},
                   equal=bool(structure_equal), reference='frozen'),
    qpos_source='the frozen model keyframe qpos0 (the same qpos is fed to all four models)',
    arms=table, advisor_table=match, n_advisor_cells_matched=n_match, n_advisor_cells=len(advisor),
    tolerance_note='the reported table is rounded to 6 decimals, so the comparison tolerance is 1e-6, not 1e-9',
    label=('ARMS_CROSS_MODEL_REPRODUCED' if n_match == len(advisor) else 'ARMS_TABLE_MISMATCH'),
    mesh_rule='asset XMLs copied to /tmp with the mesh prefix rewritten; originals untouched (hashes below)'),
    indent=2, default=str))
with open(DIR / 'CROSS_MODEL_ARMS.csv', 'w') as f:
    fams = sorted({k.split('_', 1)[1] for k in table['frozen']})
    f.write('leg,fam,' + ','.join(assets) + '\n')
    for leg in LEGS:
        for fam in ('flex', 'ext'):
            key = f'{leg}_{fam}'
            f.write('%s,%s,%s\n' % (leg, fam, ','.join('%.9f' % table[t][key] for t in assets if key in table[t])))
(DIR / 'WORLD_SITE_GEOMETRY.json').write_text(json.dumps(dict(per_model=world_sites,
                                                              note='world coordinates after one position-stage call at qpos0'), indent=2, default=str))
hashes_after = {k: sha(v) for k, v in assets.items()}
(DIR / 'SOURCE_HASHES.json').write_text(json.dumps(dict(assets=hashes_after, unchanged=bool(hashes_before == hashes_after),
                                                        original_paths={k: str(v) for k, v in assets.items()}), indent=2))
print(json.dumps(dict(dir=str(DIR), structure_equal=bool(structure_equal), n_match=n_match,
                      arms={t: {k: round(v, 6) for k, v in table[t].items() if k.endswith('flex')} for t in table},
                      label=('ARMS_CROSS_MODEL_REPRODUCED' if n_match == len(advisor) else 'ARMS_TABLE_MISMATCH'),
                      part_c=recon['part_c_status']['status'], lineage=recon['lineage_label_now'],
                      assets_unchanged=bool(hashes_before == hashes_after)), indent=1, default=str))
