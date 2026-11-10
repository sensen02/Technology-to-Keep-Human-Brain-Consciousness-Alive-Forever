import subprocess,sys,json,hashlib
from pathlib import Path
D=Path(__file__).parent;O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z/sensory_grid/20261106T185758Z');O.mkdir(parents=True,exist_ok=True)
assert not (O/'PREREGISTRATION.json').exists()
old=D/'sensory_bias_body.py';new=D/'sensory_grid_20261106T185758Z.py'
p=dict(importance='核心必需',background_nA=[.02,.05],dynamic_gain=[.1,.3],velocity_rad_s=[0,-.1,.1],dt_ms=[.5,.0625],conditions=['normal','sensory_cut'],physical_runs=48,selection='all four dt/direction pairs must reduce paired disturbance IAE vs sensory cut; preserve all results; only strict improvement is initial screening, not gait validation',change_ledger=dict(old_file=str(old),old_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),new_file=str(new),new_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),reason='declared offline parameter grid lowering dynamic sensory gain; no connectivity/readout/body changes; omit repeated zero-output from search, require it for followup'),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(O/'PREREGISTRATION.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
for bias in ['.02','.05']:
 for gain in ['.1','.3']:
  for vel in ['0','-.1','.1']:
   subprocess.run([sys.executable,'-B',str(new),bias,vel,gain],check=True)
