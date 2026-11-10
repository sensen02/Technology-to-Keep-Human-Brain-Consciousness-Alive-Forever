import subprocess,sys,json,hashlib
from pathlib import Path
D=Path(__file__).parent;O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z/readout_window/20261106T190609Z');O.mkdir(parents=True,exist_ok=True);assert not (O/'PREREGISTRATION.json').exists()
old=D/'sensory_grid_20261106T185758Z.py';new=D/'readout_window_20261106T190609Z.py'
p=dict(importance='核心必需',tau_ms=[20.,50.],background_nA=.05,dynamic_gain=.1,velocity_rad_s=[0,-.1,.1],dt_ms=[.5,.0625],conditions=['normal','sensory_cut'],physical_runs=24,selection='all four dt/direction paired disturbance IAEs must improve vs sensory_cut, with range/force/finite/warning checks; no selection of timestep',change_ledger=dict(old_file=str(old),old_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),new_file=str(new),new_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),reason='isolate longer rate-filter windows at fixed declared bias/gain; old configuration remains failed, no connection or actuator changes'))
(O/'PREREGISTRATION.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
for tau in ['20','50']:
 for vel in ['0','-.1','.1']:subprocess.run([sys.executable,'-B',str(new),tau,vel],check=True)
