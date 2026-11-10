import subprocess,sys,json,hashlib
from pathlib import Path
D=Path(__file__).parent;O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z/synaptic_scale/20261106T193236Z');O.mkdir(parents=True,exist_ok=True);assert not (O/'PREREGISTRATION.json').exists()
p=dict(importance='核心必需',efficacy_scales=[.25,.5,.75],dt_ms=[.03125,.015625],sensory_deltas_nA=[-.005,0,.005],neural_runs=18,physics_runs=0,rule='all predeclared numerical gates plus baseline motor activity and >=1e-4s response to each signed sensory input at both dt; no silent successes; preserve all conditions',source_sha256=hashlib.sha256((D/'synaptic_scale_20261106T193236Z.py').read_bytes()).hexdigest())
(O/'PREREGISTRATION.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
for scale in ['.25','.5','.75']:subprocess.run([sys.executable,'-B',str(D/'synaptic_scale_20261106T193236Z.py'),scale],check=True)
