import subprocess,sys,json,hashlib
from pathlib import Path
D=Path(__file__).parent;R=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');O=R/'clawgain/20261106T225055Z';O.mkdir(parents=True,exist_ok=True);assert not (O/'VERIFICATION.json').exists()
new=D/'clawgain_20261106T225055Z.py';offs=['-0.06','-0.03','0.0','0.03','0.06'];pres=['0.3'];gains=['0.3','1.0']
p=dict(importance='核心必需',stage='C9 sensory gain adequate for motor modulation',gains=gains,premotor_nA=pres,offsets_rad=[float(o) for o in offs],bias_nA=.03,dt_ms=.0078125,duration_ms=150,neural_runs=len(gains)*len(pres)*(len(offs)+1),physics_runs=0,rationale='stored spikes show claw cells encode position monotonically (135 to 243 spikes) but the motor pools are flat because the injected premotor current dominates; test whether a larger declared sensory gain can make motor firing rate depend on joint angle',criterion='GRADED_RESTORING on net_drive, and additionally the motor spike counts themselves must vary with offset',source_sha256=hashlib.sha256(new.read_bytes()).hexdigest())
(O/'PREREGISTRATION.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
for g in gains:
    for pre in pres:
        for c in ['normal','sensory_cut']:
            for o in (offs if c=='normal' else ['0.0']):
                subprocess.run([sys.executable,'-B',str(new),c,o,pre,g],check=True)
