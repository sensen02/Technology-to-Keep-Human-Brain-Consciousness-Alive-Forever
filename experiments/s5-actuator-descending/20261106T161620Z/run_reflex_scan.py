import subprocess,sys,json,hashlib
from pathlib import Path
D=Path(__file__).parent;R=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');O=R/'reflex/20261106T225055Z';O.mkdir(parents=True,exist_ok=True);assert not (O/'VERIFICATION.json').exists()
new=D/'reflex_scan_20261106T225055Z.py'
configs=[(g,c) for g in ['flex_sens','ext_sens','both_sens'] for c in ['0.05','0.1','0.3']]
p=dict(importance='核心必需',stage='C4 reflex recruitment',configs=[dict(group=g,current_nA=float(c)) for g,c in configs],bias_nA=.03,offset_rad=0.,dt_ms=.0078125,duration_ms=150,neural_runs=len(configs),physics_runs=0,rationale='the frozen matrix contains exactly seven sensory cells with a direct edge onto the motor pools: SNpp50 cells 7414/6469/6232/1408 onto extensor 10765, SNpp51 cell 10026 and SNpp39 cell 10043 onto flexors. This is a real monosynaptic proprioceptor-motor pathway whose direction is encoded by the wiring itself',criterion='reciprocal interface requires flex-projecting drive to spike the flexor pool and ext-projecting drive to spike the extensor pool; report without selecting on physical performance',source_sha256=hashlib.sha256(new.read_bytes()).hexdigest())
(O/'PREREGISTRATION.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
for g,c in configs:subprocess.run([sys.executable,'-B',str(new),g,c],check=True)
