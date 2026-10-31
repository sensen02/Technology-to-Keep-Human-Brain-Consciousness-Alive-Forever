"""Two real continuous handovers: source-real DNp09 inclusion without/with declared pulse."""
from pathlib import Path
import json,hashlib,ast,traceback
from dataclasses import asdict
import run_physical_walk_prewarm as w
import run_neural_muscle_closure as c
from engine.embodied.descending_command import DescendingCommand
P=Path(__file__).parent;R=Path('/home/sensen/Desktop/engine修复验收/s5-descending-command');OLD=Path('/home/sensen/Desktop/engine修复验收/s5-sensory-channel-cuts/20261028T068000Z')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 R.mkdir(parents=True,exist_ok=True)
 runner=Path(w.__file__);helper=P/'engine/embodied/continuous_handover.py';module=P/'engine/embodied/descending_command.py'
 oldrunner=OLD/runner.name;oldhelper=OLD/helper.name
 current=runner.read_text();rev=current.replace(',neural_descending_config=None):','):').replace("neural_descending_config=__import__('dataclasses').asdict(neural_descending_config) if neural_descending_config is not None else None,",'').replace(',descending_command=neural_descending_config)',')')
 import re
 rev=re.sub(r'\n +if neural_descending_config is not None:\n.*?\n +manifest=c.build_motor_manifest', ';manifest=c.build_motor_manifest',rev,flags=re.S)
 assert ast.dump(ast.parse(rev),include_attributes=False)==ast.dump(ast.parse(oldrunner.read_text()),include_attributes=False),'Unlisted runner change'
 h=helper.read_text();revh=h.replace(', descending_command=None):','):').replace('        descending_trace = []\n','');start=revh.index('            external_current = np.zeros_like(current)\n');end=revh.index('            spikes[j] = net.step',start);revh=revh[:start]+revh[end:];revh=revh.replace('current_nA=current + external_current','current_nA=current');start=revh.index('        if descending_command is not None:\n');end=revh.index('        if retain_currents:',start);revh=revh[:start]+revh[end:]
 assert ast.dump(ast.parse(revh),include_attributes=False)==ast.dump(ast.parse(oldhelper.read_text()),include_attributes=False),'Unlisted helper change'
 files=[Path(__file__),runner,helper,module,P/'test_descending_command.py',P/'engine/embodied/adapters.py',P/'engine/neck_cut_data.py',P/'engine/neural_cond.py',P/'engine/embodied/runtime_pool_guard.py',P/'engine/embodied/motor_output.py',P/'engine/embodied/sensory_channels.py',c.POOLS,P/'data/flywire/banc_connectome.npz',w.STANCE,w.REF,w.BASE/'INTEGRATED_MODEL.xml'];hashes={str(x):sha(x) for x in files}
 ledger=dict(old_runner=str(oldrunner),old_runner_sha256=sha(oldrunner),new_runner=str(runner),new_runner_sha256=sha(runner),old_helper=str(oldhelper),old_helper_sha256=sha(oldhelper),new_helper=str(helper),new_helper_sha256=sha(helper),new_module=str(module),new_module_sha256=sha(module),default_none_AST_identical_after_declared_reversal=True,changes=['optional descending config enables source-real append-only tier augmentation before manifest','optional exact-identity neuronal current; zero after50ms','explicit per-neural-step command trace and target spikes; sensory current remains separately recorded'])
 w.dump(R/'ISOLATED_CALIBRATION.json',dict(scope='Three independent single-neuron networks, never reused as body network, no connectivity/body/physics',amplitude_nA=[.3,.5,1.],duration_s=.05,neural_dt_s=.0005,spikes=[14,17,20],network_instances=3,neural_steps=300,body_steps=0,chosen_amplitude_nA=.3,reason='Smallest measured amplitude already evokes spikes; engineering stimulation assumption, not physiology'))
 results=[]
 for label,amp in [('20261028T071000Z',0.),('20261028T072000Z',.3)]:
  config=DescendingCommand(amp,.05);o=R/label;pre=R/(label+'_PREDECLARED.json');assert not o.exists() and not pre.exists()
  w.dump(pre,dict(scope='REAL1s physical walking warmup then250ms neural on same body; augmented12002 existing12000 preserved plus actual cache DNp09pair',descending_command=asdict(config),external_neuronal_start_not_muscle_pattern=True,withdrawal_relative_s=.05,body_dt_s=.0000125,neural_dt_s=.0005,command_dt_s=.001,motor_config='unchanged tau5ms100Hz',seed=0,background_current=0.,source_hashes=hashes,source_ledger=ledger,success_gate='Correct source identities/edges, exact continuous handover no auxiliary forces, report support/walking after pulse removal; no success from target spiking alone'))
  try:
   w.run(o,1.,100.,50.,.0000125,neural_duration=.25,neural_command_dt_s=.001,neural_descending_config=config)
   w.dump(o/'SOURCE_PARAMETERIZATION_LEDGER.json',ledger);w.dump(o/'SOURCE_UNCHANGED.json',dict(passed=all(sha(Path(k))==v for k,v in hashes.items()),source_hashes=hashes))
   for x in [Path(__file__),helper,module,P/'test_descending_command.py']:(o/x.name).write_bytes(x.read_bytes())
   results.append(dict(directory=str(o),exit_code=0))
  except BaseException as e:
   fail=dict(directory=str(o),exit_code=1,error=repr(e),traceback=traceback.format_exc());w.dump(R/(label+'_WRAPPER_FAILURE.json'),fail);results.append(fail)
 w.dump(R/'TWO_TRIAL_EXECUTION.json',results);print(json.dumps(results,indent=2),flush=True)
 if any(r['exit_code'] for r in results):raise SystemExit(1)
if __name__=='__main__':main()
