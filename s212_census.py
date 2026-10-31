"""S212 Part 0 (S211 correction) + Part A (all-site attachment census, XML only, 0 model loads)."""
import json
import datetime
import hashlib
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path('/home/sensen/Desktop/engine修复验收/s212-attachment-port-decomposition')
STAMP = datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S%fZ')
DIR = ROOT / STAMP
DIR.mkdir(parents=True, exist_ok=False)
REPO = Path('/run/media/sensen/Data2/cell_wound_prototype')
MUSDIR = REPO / 'outputs/muscles_six_legs'
FROZEN = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
ASSETS = dict(repaired=MUSDIR / 'fruitfly_six_leg_standing.xml',
              norepair=MUSDIR / 'fruitfly_six_leg_standing_noarmrepair.xml',
              qpos0bands=MUSDIR / 'fruitfly_six_leg_standing_qpos0bands.xml',
              muscles=MUSDIR / 'fruitfly_six_leg_muscles.xml')
MIRROR = json.loads((MUSDIR / 'mirror_decisions.json').read_text())

prereg = dict(round='S212', directory=str(DIR), written_before_runs=True,
              written_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
              rules=['R1 frozen == norepair (mirror rule as written)',
                     'R2 frozen == norepair * k', 'R3 frozen == repaired',
                     'R4 frozen == repaired * k', 'R5 frozen == repaired + constant'],
              eps=dict(abs_=1e-9, rel=1e-6),
              labels=['MIRROR_RULE_ATTRIBUTED', 'FROZEN_ATTACHMENT_RULE_FOUND', 'FROZEN_ATTACHMENT_PARTIALLY_ATTRIBUTED',
                      'ATTACHMENT_PORT_CLOSES_GAP', 'ATTACHMENT_PORT_PARTIAL', 'ATTACHMENT_PORT_NO_EFFECT'],
              budget=dict(model_loads_max=4, position_stage_calls_max=10, constraint_stage_calls=0, forward_calls=0,
                          mj_step_calls=0, lp_max=30, xml_changes=0, training_steps=0),
              note='the generator tools_mirror_muscles.py is NOT executed (it would overwrite outputs/muscles_six_legs); its rule '
                   'is re-implemented offline here')
(DIR / 'PREREG.json').write_text(json.dumps(prereg, indent=2, ensure_ascii=False))

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()

def sites(path):
    out = {}
    for s in ET.parse(path).getroot().iter('site'):
        nm = s.get('name') or ''
        pos = s.get('pos')
        if not nm or pos is None:
            continue
        out[nm] = np.array([float(x) for x in pos.split()])
    return out

S = {k: sites(v) for k, v in ASSETS.items()}
S['frozen'] = sites(FROZEN)
maps = MIRROR['non_one_to_one_mappings']
scales = sorted({round(e['scale'], 6) for e in maps})

# ---------------------------------------------------------------- Part 0: S211 correction
common = sorted(set(S['frozen']) & set(S['repaired']))
mirrored = [e for e in maps if e['site'] in S['frozen']]
recon = dict(
    withdrew=['S211 claimed the per-leg scale table is not in any artefact. It is: outputs/muscles_six_legs/'
              'mirror_decisions.json -> non_one_to_one_mappings, every entry carries site / source_body / target_body / scale / '
              'moved_mm.'],
    scale_field=dict(n_entries=len(maps), n_distinct_scales=len(scales), scales=scales),
    generator=dict(script=str(REPO / 'tools_mirror_muscles.py'), body_map_line=106,
                   k_line=281, transform_lines='272-300',
                   rule='p_world_src = ps + Rs @ (k * local); p_world_tgt = t_T + R_T @ p_world_src; '
                        'new_pos = Rt.T @ (p_world_tgt - pt) with k = segment length ratio (isotropic)',
                   executed=False, why='running it would overwrite outputs/muscles_six_legs assets'),
    label_mirror='MIRROR_RULE_ATTRIBUTED',
)
(DIR / 'S211_RECONCILIATION.json').write_text(json.dumps(recon, indent=2, ensure_ascii=False))

# ---------------------------------------------------------------- Part A: rule scoring
def band(name):
    for b in ('Tibia', 'Femur', 'Trochanter', 'Coxa', 'Thorax'):
        if b in name:
            return b
    return 'other'

rows = []
for e in maps:
    nm = e['site']
    if nm not in S['frozen']:
        continue
    fz = S['frozen'][nm]
    k = float(e['scale'])
    row = dict(site=nm, band=band(nm), leg=nm[:2], k=k, target_body=e['target_body'],
               frozen=fz.tolist())
    for tag in ('repaired', 'norepair', 'qpos0bands', 'muscles'):
        if nm in S[tag]:
            row['d_' + tag] = float(np.max(np.abs(fz - S[tag][nm])))
            row['R4_' + tag] = float(np.max(np.abs(fz - S[tag][nm] * k)))
            row['R2_' + tag] = float(np.max(np.abs(fz - S[tag][nm] * k)))
    rows.append(row)
census = {}
for tag in ('repaired', 'norepair', 'qpos0bands', 'muscles'):
    key_d = 'd_' + tag
    key_r = 'R4_' + tag
    present = [r for r in rows if key_d in r]
    census[tag] = dict(n_compared=len(present),
                       n_identical=sum(1 for r in present if r[key_d] <= 1e-6),
                       n_scaled_match=sum(1 for r in present if r[key_r] <= 1e-6),
                       by_band={})
    for b in sorted({r['band'] for r in present}):
        sub = [r for r in present if r['band'] == b]
        census[tag]['by_band'][b] = dict(n=len(sub),
                                         identical=sum(1 for r in sub if r[key_d] <= 1e-6),
                                         scaled=sum(1 for r in sub if r[key_r] <= 1e-6))
# left/right split for the scaled rule against the repaired asset
lr = {}
for side in ('L', 'R'):
    sub = [r for r in rows if 'R4_repaired' in r and r['leg'][0] == side]
    lr[side] = dict(n=len(sub), scaled_match=sum(1 for r in sub if r['R4_repaired'] <= 1e-6),
                    y_flip_match=sum(1 for r in sub if 'repaired' in r and
                                     float(np.max(np.abs(np.array(r['frozen']) -
                                                         np.array(S['repaired'][r['site']]) * np.array([r['k'], -r['k'], r['k']])))) <= 1e-6))
(DIR / 'FROZEN_ATTACHMENT_CENSUS.json').write_text(json.dumps(dict(
    rule_labels=['MIRROR_RULE_ATTRIBUTED'], census=census, left_right=lr, rows=rows,
    note='R4 is the candidate rule "frozen site == repaired site * k (per-site isotropic scale)"; the right-leg y component is '
         'tested separately with an explicit sign flip'), indent=2, default=str))
with open(DIR / 'FROZEN_ATTACHMENT_CENSUS.csv', 'w') as f:
    cols = ['site', 'leg', 'band', 'k', 'target_body'] + [c for c in ('d_repaired', 'R4_repaired', 'd_norepair', 'R4_norepair',
                                                                      'd_muscles', 'R4_muscles') if any(c in r for r in rows)]
    f.write(','.join(cols) + '\n')
    for r in rows:
        f.write(','.join(str(r.get(c, '')) for c in cols) + '\n')
best = max(census.items(), key=lambda kv: kv[1]['n_scaled_match'])
total = sum(1 for r in rows if 'R4_repaired' in r)
cover = lr['L']['scaled_match'] + lr['R']['y_flip_match']
label = 'FROZEN_ATTACHMENT_RULE_FOUND' if total and cover / total >= 0.95 else 'FROZEN_ATTACHMENT_PARTIALLY_ATTRIBUTED'
print(json.dumps(dict(dir=str(DIR), n_mirror_sites=len(maps), n_compared=total,
                      census={k: dict(n=v['n_compared'], identical=v['n_identical'], scaled=v['n_scaled_match'])
                              for k, v in census.items()},
                      left_right=lr, coverage=cover / total if total else None, label=label), indent=1))
