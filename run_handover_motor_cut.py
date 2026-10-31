"""One preregistered actual readout-cut causal trial; network/proposals still computed."""
from pathlib import Path
import json,hashlib,ast
import run_physical_walk_prewarm as w
P=Path(__file__).parent
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-causal-motorcut/20261028T066000Z')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 source=[Path(__file__),Path(w.__file__),P/'engine/embodied/continuous_handover.py',P/'engine/embodied/motor_output.py',P/'engine/neural_cond.py',P/'engine/embodied/sensory_channels.py',P/'engine/embodied/runtime_pool_guard.py']
 import run_neural_muscle_closure as c
 source.append(c.POOLS)
 hashes={str(x):sha(x) for x in source}
 # Reverse only declared edits and compare entire AST to preserved previous source.
 oldroot=Path('/home/sensen/Desktop/engine修复验收/s5-neural-recruitment/20261028T062000Z')
 oldrunner=oldroot/'run_physical_walk_prewarm.py';oldhelper=Path('/home/sensen/Desktop/engine修复验收/s5-physical-warmup/20261028T058000Z/continuous_handover.py')
 current=Path(w.__file__).read_text();rev=current.replace(',neural_motor_output_connected=True):','):').replace('neural_motor_output_connected=bool(neural_motor_output_connected),','').replace(',motor_output_connected=neural_motor_output_connected)',')')
 assert ast.dump(ast.parse(rev),include_attributes=False)==ast.dump(ast.parse(oldrunner.read_text()),include_attributes=False)
 helper=(P/'engine/embodied/continuous_handover.py').read_text();revh=helper.replace('retain_currents=False, motor_output_connected=True):','retain_currents=False):').replace('        if not motor_output_connected:\n            applied[:] = 0  # causal readout cut only; network and decoded proposals continue\n','').replace('                      motor_output_connected=bool(motor_output_connected),\n','')
 assert ast.dump(ast.parse(revh),include_attributes=False)==ast.dump(ast.parse(oldhelper.read_text()),include_attributes=False)
 pre=O.parent/(O.name+'_PREDECLARED.json');pre.parent.mkdir(parents=True,exist_ok=True);assert not pre.exists() and not O.exists()
 ledger=dict(changes=['optional motor_output_connected=True helper','cut applied command only; decode and neural steps unchanged','optional runner neural_motor_output_connected=True and prereg flag'],default_true_AST_identical_after_declared_reversal=True,old_runner=str(oldrunner),old_runner_sha256=sha(oldrunner),new_runner=str(Path(w.__file__)),new_runner_sha256=sha(Path(w.__file__)),old_helper=str(oldhelper),old_helper_sha256=sha(oldhelper),new_helper=str(P/'engine/embodied/continuous_handover.py'),new_helper_sha256=sha(P/'engine/embodied/continuous_handover.py'))
 w.dump(pre,dict(scope='Actual readout-cut versus061 connected causal comparison. Neural circuit still runs and decoded proposals saved; only applied motor control zero.',warmup_s=1.,neural_s=.25,command_dt_s=.001,neural_dt_s=.0005,body_dt_s=.0000125,motor_output_connected=False,seed=0,background_current=0,source_hashes=hashes,source_ledger=ledger,baseline='/home/sensen/Desktop/engine修复验收/s5-fast-neural-takeover/20261028T061000Z',success_gate='Preserved same prewarm+handover, actual decoded proposals and spikes present, applied control zero. Compare collapse and state/force divergence; no autonomous success claim.'))
 w.run(O,1.,100.,50.,.0000125,neural_duration=.25,neural_command_dt_s=.001,neural_motor_output_connected=False)
 w.dump(O/'SOURCE_PARAMETERIZATION_LEDGER.json',ledger);w.dump(O/'SOURCE_UNCHANGED.json',dict(passed=all(sha(Path(k))==v for k,v in hashes.items()),source_hashes=hashes));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());(O/'continuous_handover.py').write_bytes((P/'engine/embodied/continuous_handover.py').read_bytes())
 print('MOTOR_CUT_ACTUAL_COMPLETE',O,flush=True)
if __name__=='__main__':main()
