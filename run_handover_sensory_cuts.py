"""Two causal sensory-channel deletions on fresh real warmup, never a locomotor controller."""
from pathlib import Path
import json,hashlib,ast,traceback
import run_physical_walk_prewarm as w
import run_neural_muscle_closure as c
P=Path(__file__).parent
R=Path('/home/sensen/Desktop/engine修复验收/s5-sensory-channel-cuts')
OLD=Path('/home/sensen/Desktop/engine修复验收/s5-neural-causal-motorcut/20261028T066000Z')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 runner=Path(w.__file__);helper=P/'engine/embodied/continuous_handover.py'
 rev=runner.read_text().replace(',neural_sensory_channel_weights=None):','):').replace('neural_sensory_channel_weights=neural_sensory_channel_weights,','').replace(',sensory_channel_weights=neural_sensory_channel_weights)',')')
 assert ast.dump(ast.parse(rev),include_attributes=False)==ast.dump(ast.parse((OLD/runner.name).read_text()),include_attributes=False)
 revh=helper.read_text().replace(', sensory_channel_weights=None):','):').replace(', per_channel_weight=sensory_channel_weights)',')').replace(', motor_config, sensory_channel_weights)',', motor_config)')
 assert ast.dump(ast.parse(revh),include_attributes=False)==ast.dump(ast.parse((OLD/helper.name).read_text()),include_attributes=False)
 sources=[Path(__file__),runner,helper,P/'engine/embodied/sensory_channels.py',P/'engine/embodied/motor_output.py',P/'engine/embodied/runtime_pool_guard.py',P/'engine/neural_cond.py',c.POOLS,w.STANCE,w.REF,w.BASE/'INTEGRATED_MODEL.xml']
 hashes={str(x):sha(x) for x in sources};ledger=dict(changes=['optional sensory_channel_weights=None helper runtime and encoder constructor pass','optional neural_sensory_channel_weights=None runner prereg and pass'],old_runner=str(OLD/runner.name),old_runner_sha256=sha(OLD/runner.name),new_runner=str(runner),new_runner_sha256=sha(runner),old_helper=str(OLD/helper.name),old_helper_sha256=sha(OLD/helper.name),new_helper=str(helper),new_helper_sha256=sha(helper),default_none_executable_AST_identical_after_declared_reversal=True)
 R.mkdir(parents=True,exist_ok=True);results=[]
 for label,weights in [('20261028T068000Z',dict(bristle_touch=0.,campaniform_strain=0.)),('20261028T069000Z',dict(claw_position=0.,hook_direction=0.,hairplate_angle=0.))]:
  o=R/label;pre=R/(label+'_PREDECLARED.json');assert not o.exists() and not pre.exists()
  w.dump(pre,dict(stage='Real1s warmup then250ms fresh resting sensory-channel-cut neural takeover',channel_weights=weights,unlisted_channel_weights=1.,motor_output_connected=True,filter_tau_s=.005,rate_for_full_control_hz=100.,command_dt_s=.001,neural_dt_s=.0005,body_dt_s=.0000125,background_current=0.,seed=0,source_hashes=hashes,source_ledger=ledger,hypothesis='Identify feedback contribution to early and total motor recruitment, not certify autonomy from spiking.',success_gate='Same physical warmup+boundary, exact declared channel weights, causal commands, no auxiliary/external forces; report actual behavior including failure.'))
  try:
   w.run(o,1.,100.,50.,.0000125,neural_duration=.25,neural_command_dt_s=.001,neural_sensory_channel_weights=weights)
   w.dump(o/'SOURCE_PARAMETERIZATION_LEDGER.json',ledger);w.dump(o/'SOURCE_UNCHANGED.json',dict(passed=all(sha(Path(k))==v for k,v in hashes.items()),source_hashes=hashes))
   for x in [Path(__file__),runner,helper]:(o/x.name).write_bytes(x.read_bytes())
   results.append(dict(directory=str(o),exit_code=0))
  except BaseException as e:
   fail=dict(directory=str(o),exit_code=1,error=repr(e),traceback=traceback.format_exc());w.dump(R/(label+'_WRAPPER_FAILURE.json'),fail);results.append(fail)
 w.dump(R/'TWO_TRIAL_EXECUTION.json',results);print(json.dumps(results,indent=2),flush=True)
 if any(x['exit_code'] for x in results):raise SystemExit(1)
if __name__=='__main__':main()
