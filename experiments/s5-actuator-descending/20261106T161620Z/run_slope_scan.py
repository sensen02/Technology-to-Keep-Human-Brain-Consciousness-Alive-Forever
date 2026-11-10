import subprocess,sys,json,hashlib
from pathlib import Path
D=Path(__file__).parent;R=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');O=R/'slope/20261106T225055Z';O.mkdir(parents=True,exist_ok=True);assert not (O/'VERIFICATION.json').exists()
new=D/'slope_scan_20261106T225055Z.py'
adds=[-0.02,-0.01,-0.005,0.0,0.005,0.01,0.02,0.03,0.05];pres=[0.1,0.2]
p=dict(importance='核心必需',stage='C7 claw-to-motor slope',added_claw_values_nA=adds,premotor_values_nA=pres,bias_nA=.03,dt_ms=.0078125,duration_ms=150,neural_runs=len(adds)*len(pres),physics_runs=0,criterion='net_drive must respond monotonically to the added claw current for a proportional pathway to exist; two premotor baselines are compared to show whether the operating point matters',source_sha256=hashlib.sha256(new.read_bytes()).hexdigest())
(O/'PREREGISTRATION.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
for pre in pres:
    for a in adds:subprocess.run([sys.executable,'-B',str(new),str(a),str(pre)],check=True)
