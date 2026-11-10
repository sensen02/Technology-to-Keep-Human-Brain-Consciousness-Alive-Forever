import subprocess,sys,json,hashlib
from pathlib import Path
D=Path(__file__).parent;R=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');O=R/'premotor/20261106T225055Z';O.mkdir(parents=True,exist_ok=True);assert not (O/'VERIFICATION.json').exists()
new=D/'premotor_scan_20261106T225055Z.py'
configs=[(p,c) for p in ['flex','ext'] for c in ['0.05','0.1','0.3']]
p=dict(importance='核心必需',stage='C5 premotor recruitment',configs=[dict(pool=a,current_nA=float(c)) for a,c in configs],bias_nA=.03,offset_rad=0.,dt_ms=.0078125,duration_ms=150,top_n=6,neural_runs=len(configs),physics_runs=0,rationale='direct descending and direct sensory routes are sub-threshold onto the flexor pool; the motor pools are dominated by 193 VNC-intrinsic partners, so test whether their strongest excitatory-dominant inputs can recruit each pool',criterion='report per-pool counts; no selection on physical performance',source_sha256=hashlib.sha256(new.read_bytes()).hexdigest())
(O/'PREREGISTRATION.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
for a,c in configs:subprocess.run([sys.executable,'-B',str(new),a,c],check=True)
