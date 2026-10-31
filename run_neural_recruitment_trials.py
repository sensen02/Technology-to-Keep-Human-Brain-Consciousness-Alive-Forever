"""Two declared pure spike-decoder parameter trials; same real physical walking warmup.
No injected current, activation preload, replay, external assistance after withdrawal.
"""
from pathlib import Path
import os,json,ast,hashlib,traceback
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from dataclasses import asdict
from engine.embodied.motor_output import MotorOutputConfig
import run_physical_walk_prewarm as w
P=Path(__file__).parent
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-neural-recruitment')
OLD=P/'run_physical_walk_prewarm.py'
BASE=Path('/home/sensen/Desktop/engine修复验收/s5-fast-neural-takeover/20261028T061000Z')
OLD_SHA='f35115cea666dbea18bb8e20aaa9c9ad00a5d7dbfe4699e91017c872ca4777c1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def normalized_run(source):
 tree=ast.parse(source)
 fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='run')
 # Explicit textual reversal makes the previous default executable structure comparable.
 return ast.dump(fn,include_attributes=False)
def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 previous=(BASE/'run_physical_walk_prewarm.py').read_text();assert sha(BASE/'run_physical_walk_prewarm.py')==OLD_SHA
 current=OLD.read_text();reversed_source=current.replace(',neural_motor_config=None):','):').replace("__import__('dataclasses').asdict(neural_motor_config) if neural_motor_config is not None else 'unchanged default tau5ms/full-scale100Hz/gain1'","'unchanged default tau5ms/full-scale100Hz/gain1'").replace(',motor_config=neural_motor_config,',',').replace(",P/'engine/embodied/continuous_handover.py',P/'engine/embodied/motor_output.py',P/'engine/embodied/runtime_pool_guard.py',P/'engine/embodied/sensory_channels.py',P/'engine/neural_cond.py'",'')
 assert ast.dump(ast.parse(previous),include_attributes=False)==ast.dump(ast.parse(reversed_source),include_attributes=False),'Unexpected non-parameter logic change'
 source_hashes={str(p):sha(p) for p in [Path(__file__),OLD,P/'engine/embodied/continuous_handover.py',P/'engine/embodied/motor_output.py',P/'engine/embodied/runtime_pool_guard.py',P/'engine/embodied/sensory_channels.py',P/'engine/neural_cond.py']}
 ledger=dict(old_file=str(BASE/'run_physical_walk_prewarm.py'),old_sha256=OLD_SHA,new_file=str(OLD),new_sha256=sha(OLD),reason='Optional neural MotorOutputConfig; default None preserves default decoder; additionally preregister execution dependencies',changes=['optional neural_motor_config=None','serialize provided dataclass in preregistration','pass optional config to continuous neural helper','add dependency hashes'],executable_AST_identical_after_declared_reversal=True,default_none_preserves_5ms100Hz_decoder=True)
 runs=[('20261028T062000Z',MotorOutputConfig(filter_tau_s=.001,rate_for_full_control_hz=100.)),('20261028T063000Z',MotorOutputConfig(filter_tau_s=.001,rate_for_full_control_hz=20.))]
 all_results=[]
 for label,config in runs:
  config.validate();o=ROOT/label;pre=ROOT/(label+'_PREDECLARED.json');assert not o.exists() and not pre.exists()
  w.dump(pre,dict(stage='Sequential independent real runs: 1s physics warmup then .25s pure neural',body_dt_s=.0000125,neural_dt_s=.0005,command_dt_s=.001,motor_config=asdict(config),seed=0,background_current=0.,source_hashes=source_hashes,source_ledger=ledger,success_gate='continuous boundary, no post-handoff assistance/external force; compare onset nonfoot and matched250ms load; never certify walking from decoder output only'))
  try:
   w.run(o,1.,100.,50.,.0000125,neural_duration=.25,neural_command_dt_s=.001,neural_motor_config=config)
   w.dump(o/'SOURCE_PARAMETERIZATION_LEDGER.json',ledger)
   w.dump(o/'SOURCE_UNCHANGED.json',dict(passed=all(sha(Path(k))==v for k,v in source_hashes.items()),source_hashes=source_hashes))
   (o/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
   all_results.append(dict(directory=str(o),exit_code=0))
  except BaseException as exc:
   fail=dict(directory=str(o),exit_code=1,error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());all_results.append(fail);w.dump(ROOT/(label+'_WRAPPER_FAILURE.json'),fail)
   # Keep the second preregistered independent parameter trial, even if the first fails.
 w.dump(ROOT/'TWO_TRIAL_EXECUTION.json',all_results)
 print(json.dumps(all_results,indent=2),flush=True)
 if any(x['exit_code'] for x in all_results):raise SystemExit(1)
if __name__=='__main__':main()
