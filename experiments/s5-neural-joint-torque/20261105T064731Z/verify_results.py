"""Reuse audited replay verifier with explicit full/dominant mapping selection."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import hashlib,json
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-neural-joint-torque/20261105T062800Z/verify_and_render.py')
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T064731Z')
text=SOURCE.read_text()
old="proposals=np.array([np.clip(limit/100*np.exp(z[66:])*(S@(np.exp(z[:66])*rr)),-limit,limit) for rr in n['source_rates']])"
new="matrix = np.array(js(O/'DOMINANT_COVERAGE.json')['sign_matrix']) if 'dominant_replay' in r['label'] else S\n        proposals=np.array([np.clip(limit/100*np.exp(z[66:])*(matrix@(np.exp(z[:66])*rr)),-limit,limit) for rr in n['source_rates']])"
assert text.count(old)==1
text=text.replace(old,new).replace('assert len(checked)==38','assert len(checked)==40').replace('total=38','total=40').replace('/s5-neural-joint-torque/20261105T062800Z','/s5-neural-joint-torque/20261105T064731Z')
# Only the derived module namespace is executed; archived verifier source is unchanged.
namespace={'__name__':'full_mapping_verifier','__file__':str(SOURCE)}
exec(compile(text,str(Path(__file__)), 'exec'),namespace)
namespace['main']()
(O/'VERIFIER_ADAPTER.json').write_text(json.dumps(dict(source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),changes=['output directory','40 instead of 38 evaluations','select archived dominant matrix for two reproduction arms'],importance='核心必需'),indent=2)+'\n')
