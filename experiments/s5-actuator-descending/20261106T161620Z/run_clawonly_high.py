import subprocess,sys,json,hashlib
from pathlib import Path
D=Path(__file__).parent;R=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');O=R/'clawonly/20261106T225055Z';new=D/'clawonly_20261106T225055Z.py';offs=['-0.06','-0.03','0.0','0.03','0.06']
p=dict(importance='核心必需',stage='C8b claw-only map at a high-rate operating point',premotor_nA=.3,conditions=['normal','sensory_cut'],offsets_rad=[float(o) for o in offs],bias_nA=.03,dt_ms=.0078125,duration_ms=150,neural_runs=10,physics_runs=0,rationale='at premotor .1 the extensor pool emits only about 12 spikes in 150ms, so the steady estimate is quantisation-limited; .3 puts both pools at high rate (about 82 flex / 34 ext) to separate circuit behaviour from readout resolution',criterion='GRADED_RESTORING on net_drive; the same offsets and gain as C8',source_sha256=hashlib.sha256(new.read_bytes()).hexdigest())
(O/'PREREGISTRATION_PRE0.3.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
for c in ['normal','sensory_cut']:
    for o in offs:subprocess.run([sys.executable,'-B',str(new),c,o,'0.3'],check=True)
