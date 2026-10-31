"""External muscle-source inventory and adjudication (read-only, no model load).

Checks what actually exists on disk for per-leg Drosophila leg-muscle attachments,
records path + sha256 for each item, and adjudicates against the released external
source now downloaded next to this script. No MuJoCo model is loaded, no forward,
no body integration, no neural step.
"""
from pathlib import Path
import hashlib, json, subprocess
D = Path('/home/sensen/Desktop/engine修复验收/s5-external-muscle-source/20261028T120000Z')


def sha(p):
    p = Path(p)
    if p.is_dir():
        h = hashlib.sha256()
        for f in sorted(p.rglob('*')):
            if f.is_file():
                h.update(str(f.relative_to(p)).encode()); h.update(f.read_bytes())
        return h.hexdigest()
    return hashlib.sha256(p.read_bytes()).hexdigest()


items = [
    ('FlyBody released MuJoCo model', '/run/media/sensen/Data2/flybody_src/flybody/fruitfly/assets/fruitfly.xml'),
    ('FlyBody model builder', '/run/media/sensen/Data2/flybody_src/flybody/fruitfly/build_fruitfly/fruitfly.xml'),
    ('FlyBody README', '/run/media/sensen/Data2/flybody_src/README.md'),
    ('FlyBody data downloader', '/run/media/sensen/Data2/flybody_src/flybody/download_data.py'),
    ('project upstream converted FlyMimic MuJoCo model', '/run/media/sensen/Data2/cell_wound_prototype/data/upstream_flymimic/best_combined_cvt3.xml'),
    ('project upstream torque variant', '/run/media/sensen/Data2/cell_wound_prototype/data/upstream_flymimic/best_combined_cvt3_torque.xml'),
    ('project upstream validation report', '/run/media/sensen/Data2/cell_wound_prototype/data/upstream_flymimic/best_combined_full.pdf'),
    ('project kinematic leg model CSV', '/run/media/sensen/Data2/cell_wound_prototype/data/kinematic_leg_model/KinematicModel_InitalPosture_Coords3D-2.csv'),
    ('project kinematic leg model readme', '/run/media/sensen/Data2/cell_wound_prototype/data/kinematic_leg_model/ReadMe-2.txt'),
    ('NeuroMechFly trimmed snapshot', '/run/media/sensen/Data2/engine修复验收/s2-annotated-pose-sources/20261022T170005Z/NeuroMechFly'),
    ('NeuroMechFly sample CSV', '/run/media/sensen/Data2/engine修复验收/s2-annotated-pose-sources/20261022T170005Z/AN_SS45605.csv'),
    ('project six-leg muscle XML', '/run/media/sensen/Data2/cell_wound_prototype/outputs/muscles_six_legs/fruitfly_six_leg_muscles.xml'),
    ('project mirror decisions', '/run/media/sensen/Data2/cell_wound_prototype/outputs/muscles_six_legs/mirror_decisions.json'),
    ('project muscle anatomy resolution report', '/run/media/sensen/Data2/cell_wound_prototype/outputs/muscle_anatomy_resolution/REPORT.zh-CN.md'),
    ('project muscle anatomy resolution data', '/run/media/sensen/Data2/cell_wound_prototype/outputs/muscle_anatomy_resolution/resolution.json'),
    ('project mn-muscle map per leg', '/run/media/sensen/Data2/cell_wound_prototype/outputs/mn_muscle_map_per_leg.json'),
    ('downloaded FlyMimic OpenSim full model', str(D / 'best_combined_full.osim')),
    ('downloaded FlyMimic OpenSim model', str(D / 'best_combined.osim')),
    ('downloaded FlyMimic README', str(D / 'README_FlyMimic.md')),
]
rows = []
for name, p in items:
    p = Path(p)
    if not p.exists():
        rows.append(dict(item=name, path=str(p), sha256=None, exists=False))
        continue
    txt = ''
    if p.is_file() and p.suffix in ('.xml', '.osim', '.md', '.txt'):
        txt = p.read_text(errors='ignore')
    muscles = None
    if txt:
        muscles = txt.count('<Millard2012EquilibriumMuscle') + txt.count('<spatial') + txt.count('muscle')
    rows.append(dict(item=name, path=str(p), sha256=sha(p), exists=True,
                     is_dir=p.is_dir(), size=None if p.is_dir() else p.stat().st_size,
                     muscle_token_hits=muscles))
out = dict(scope='read-only inventory', counts=dict(model_load=0, forward=0, body_step=0, neural_step=0), items=rows)
(D / 'INVENTORY.json').write_text(json.dumps(out, indent=2) + '\n')
for r in rows:
    print(f"{r['item'][:52]:54s} {'DIR' if r.get('is_dir') else (r.get('size') or 0):>8} {str(r['sha256'])[:16] if r['sha256'] else 'MISSING'}")
