import subprocess,sys,json,hashlib
from pathlib import Path
D=Path(__file__).parent;R=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');O=R/'recruit/20261106T225055Z';O.mkdir(parents=True,exist_ok=True);assert not (O/'PREREGISTRATION.json').exists()
dns=[.1,.15,.2,.3,.4,.5];new=D/'recruit_scan_20261106T225055Z.py'
p=dict(importance='核心必需',stage='C1 fallback recruitment scan',descending_values_nA=dns,bias_nA=.03,offset_rad=0.,dt_ms=.0078125,duration_ms=150,neural_runs=len(dns),physics_runs=0,purpose='the flexor pool produced no spikes at any tested sensory background; determine the declared descending tonic at which it is recruited before mapping polarity again',pass_rule='a level is usable only if BOTH pools fire in the 100-150ms tail window; otherwise it cannot express reciprocal graded control',source_sha256=hashlib.sha256(new.read_bytes()).hexdigest())
(O/'PREREGISTRATION.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
for dn in dns:subprocess.run([sys.executable,'-B',str(new),str(dn)],check=True)
