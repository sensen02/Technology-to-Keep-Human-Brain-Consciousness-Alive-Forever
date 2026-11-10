"""Verify both escalation stages and diagnostic interventions serially."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MUJOCO_GL','egl')
import importlib.util,json,hashlib
from pathlib import Path
PREVIOUS=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-neural-joint-torque/20261105T153224Z/verify_and_report.py')
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-neural-joint-torque/20261105T062800Z/verify_and_render.py')
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T153657Z')
spec=importlib.util.spec_from_file_location('verify_prior_escalation',PREVIOUS);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
v.main()
text=SOURCE.read_text()
old="proposals=np.array([np.clip(limit/100*np.exp(z[66:])*(S@(np.exp(z[:66])*rr)),-limit,limit) for rr in n['source_rates']])"
new="identity=js(folder/'READOUT_IDENTITY.json'); assert sha(identity['file'])==identity['sha256']\n        config=js(identity['file']); matrix=np.array(config['matrix'])\n        proposals=np.array([np.clip(limit/100*np.exp(z[66:])*(matrix@(np.exp(z[:66])*rr)),-limit,limit) for rr in n['source_rates']])\n        if config['sensory_cut']: assert not np.any(n['currents'])"
assert text.count(old)==1
text=text.replace(old,new).replace('assert len(checked)==38','assert len(checked)==56').replace('total=38','total=56').replace('/s5-neural-joint-torque/20261105T062800Z','/s5-neural-joint-torque/20261105T153657Z')
namespace={'__name__':'incumbent_verifier','__file__':str(SOURCE)};exec(compile(text,str(Path(__file__)),'exec'),namespace);namespace['main']()
(O/'VERIFIER_ADAPTER.json').write_text(json.dumps(dict(source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),changes=['exact per-candidate matrix checked','56 evaluations','sensory cut zero current verified'],importance='核心必需'),indent=2)+'\n')
print('ALL 118 ROLLOUTS VERIFIED',flush=True)
