"""Read-only actual1ms cycle replay compatibility; no physics construction/step."""
import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path
import final_native_audit as a
sys.path.insert(0,str(a.RECORDER))
import student_wave as sw
run=a.ROOT/'20261110T125712Z/EVAL_ROLE_2000'
report=a.replay_cycles(run,a.read_json(run/'RESULT.json'),sw)
report.update(source=str(run),physical_steps=0,historical_writes=0)
path=Path(__file__).parent/'CYCLE_REPLAY_CHECK.json'
with path.open('x') as f:json.dump(report,f,indent=2)
print(json.dumps({'output':str(path),**report}),flush=True)
