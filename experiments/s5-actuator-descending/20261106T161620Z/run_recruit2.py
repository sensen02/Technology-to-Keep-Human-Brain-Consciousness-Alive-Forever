import subprocess,sys,json,hashlib
from pathlib import Path
D=Path(__file__).parent;R=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');O=R/'recruit2/20261106T225055Z';O.mkdir(parents=True,exist_ok=True);assert not (O/'VERIFICATION.json').exists()
new=D/'recruit2_20261106T225055Z.py'
configs=[('pool','1.0'),('pool','2.0'),('pool','4.0'),('cell:5940','0.5'),('cell:5940','1.0'),('cell:5940','2.0'),('cell:9101','0.5'),('cell:9101','1.0'),('cell:9101','2.0')]
p=dict(importance='核心必需',stage='C3 recruitment escalation',configs=[dict(mode=m,current_nA=float(c)) for m,c in configs],bias_nA=.03,offset_rad=0.,dt_ms=.0078125,duration_ms=150,neural_runs=len(configs),physics_runs=0,rationale='only 3 annotated descending cells have a direct edge onto any motor cell, each a single sub-threshold 0.0002 uS event; 5940 DNa01 is the sole descending route to a flexor and 9101/1183 DNp10 the only extensor routes. Escalating current is a declared engineering input used to test whether the real pathway can be recruited at all',criterion='a usable reciprocal interface requires the flexor pool to spike; report counts without selecting on physical performance',source_sha256=hashlib.sha256(new.read_bytes()).hexdigest())
(O/'PREREGISTRATION.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
for m,c in configs:subprocess.run([sys.executable,'-B',str(new),m,c],check=True)
