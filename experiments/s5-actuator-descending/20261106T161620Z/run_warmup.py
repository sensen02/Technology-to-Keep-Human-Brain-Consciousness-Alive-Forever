import subprocess,sys,json,hashlib
from pathlib import Path
D=Path(__file__).parent;R=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');O=R/'warmup/20261106T225055Z';O.mkdir(parents=True,exist_ok=True);assert not (O/'VERIFICATION.json').exists()
new=D/'warmup_20261106T225055Z.py'
p=dict(importance='核心必需',stage='D3 warm-up or crutch',modes=['sustained','warmup100','none'],offsets_rad=[-.06,0.,.06],sensory_gain=1.0,premotor_nA=.3,warmup_window_ms=[0,100],duration_ms=200,readout_window_ms=[100,200],dt_ms=.0078125,neural_runs=9,physics_runs=0,criterion='warm-up established only if post-withdrawal motor firing resembles the sustained case and exceeds the no-premotor case; otherwise it is a crutch',source_sha256=hashlib.sha256(new.read_bytes()).hexdigest())
(O/'PREREGISTRATION.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
for m in ['sustained','warmup100','none']:
    for o in [-.06,0.,.06]:subprocess.run([sys.executable,'-B',str(new),m,str(o)],check=True)
