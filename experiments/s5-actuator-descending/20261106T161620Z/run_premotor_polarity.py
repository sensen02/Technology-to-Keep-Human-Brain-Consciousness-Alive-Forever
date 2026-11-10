import subprocess,sys,json,hashlib
from pathlib import Path
D=Path(__file__).parent;R=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');O=R/'premotor_polarity/20261106T225055Z';O.mkdir(parents=True,exist_ok=True);assert not (O/'VERIFICATION.json').exists()
new=D/'premotor_polarity_20261106T225055Z.py';offs=['-0.06','-0.03','0.0','0.03','0.06']
p=dict(importance='核心必需',stage='C6 premotor polarity map',conditions=['normal','sensory_cut'],offsets_rad=[float(o) for o in offs],premotor_nA=.1,bias_nA=.03,dt_ms=.0078125,duration_ms=150,neural_runs=10,physics_runs=0,criterion='GRADED_RESTORING: net_drive must strictly decrease across the offsets with span>=1e-3, with both pools active at delta=0',source_sha256=hashlib.sha256(new.read_bytes()).hexdigest())
(O/'PREREGISTRATION.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
for c in ['normal','sensory_cut']:
 for o in offs:subprocess.run([sys.executable,'-B',str(new),c,o],check=True)
