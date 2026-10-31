"""One declared physical muscle-warmup and activation-memory engineering experiment."""
from pathlib import Path
import json,hashlib,ast
import numpy as np
import run_physical_walk_prewarm as w
import run_neural_muscle_closure as c
P=Path(__file__).parent
R=Path('/home/sensen/Desktop/engine修复验收/s5-muscle-preactivation')
O=R/'20261028T078000Z'
FIT=Path('/home/sensen/Desktop/engine修复验收/s5-settled-muscle-support/20261028T076000Z/FIXED_POSE_ACTIVATION_CANDIDATE.npz')
OLD=Path('/home/sensen/Desktop/engine修复验收/s5-walk-settle-neural/20261028T074000Z/run_physical_walk_prewarm.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def default_equivalence(current,old):
 tree=ast.parse(current);prior=ast.parse(old)
 build=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='build');build.args.args=build.args.args[:-1];build.args.defaults=[]
 build.body=[x for x in build.body if not(isinstance(x,ast.If) and ast.unparse(x.test)=='muscle_activation_timeconstants is not None')]
 run=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='run');run.args.args=run.args.args[:-3];run.args.defaults=run.args.defaults[:-3]
 run.body=[x for x in run.body if not(isinstance(x,ast.If) and ast.unparse(x.test)=='warmup_actuator_command is not None')]
 for x in ast.walk(run):
  if isinstance(x,ast.Call):x.keywords=[k for k in x.keywords if k.arg not in ['muscle_activation_timeconstants','warmup_actuator_command','warmup_activation_start_s']]
  if isinstance(x,ast.For):x.body=[v for v in x.body if not(isinstance(v,ast.If) and ast.unparse(v.test).startswith('warmup_actuator_command is not None'))]
 assert ast.dump(tree,include_attributes=False)==ast.dump(prior,include_attributes=False),'Unexpected executable change outside declared optional model/warmup arguments'
def main():
 R.mkdir(parents=True,exist_ok=True);source=Path(w.__file__);default_equivalence(source.read_text(),OLD.read_text())
 fit=np.load(FIT);x=fit['pool_activation'];G=fit['G'];cmd=G@x;np.testing.assert_array_equal(cmd,fit['actuator_control'])
 assert G.shape==(90,66) and x.shape==(66,) and int(np.sum(G.sum(1)==0))==16
 assert np.all((cmd>=0)&(cmd<=1))
 sources=[Path(__file__),source,FIT,OLD,P/'engine/embodied/continuous_handover.py',P/'engine/embodied/motor_output.py',P/'engine/embodied/sensory_channels.py',P/'engine/embodied/runtime_pool_guard.py',P/'engine/neural_cond.py',P/'engine/embodied/adapters.py',P/'engine/neck_cut_data.py',c.POOLS,P/'data/flywire/banc_connectome.npz',w.STANCE,w.REF,w.BASE/'INTEGRATED_MODEL.xml']
 hashes={str(p):sha(p) for p in sources};ledger=dict(old_file=str(OLD),old_sha256=sha(OLD),new_file=str(source),new_sha256=sha(source),default_none_AST_identical_after_declared_reversal=True,changes=['optional build muscle activation rise/decay timeconstants only','optional90 vector muscle control during assisted warmup fromdeclaredstart','preregister engineering parameters and explicit muscle warmup'],physiology='NOT_ESTABLISHED')
 pre=R/(O.name+'_PREDECLARED.json');assert not O.exists() and not pre.exists()
 w.dump(pre,dict(scope='Real1s walk+150ms settle+50ms assisted muscle preactivation then250ms neural. Two engineering interventions, not isolated causal effects.',muscle_activation_timeconstants_s=[.005,.030],old_native_timeconstants_s=[.0001,.0004],warmup_activation_start_s=1.15,warmup_actuator_command=cmd,warmup_pool_activation=x,source_fit='Fixed074pose tied66 best bounded activation, not an equilibrium or controller solution',neural_start_s=1.2,neural_duration_s=.25,command_dt_s=.001,body_dt_s=.0000125,neural_dt_s=.0005,post_neural_auxiliary=False,post_neural_static_muscle_command=False,source_hashes=hashes,source_ledger=ledger,success_gate='Actualwalking/warmupnonfootabsent, preservedphysicalactboundary, no posthandover auxiliary/current preload, assesslast100ms beyondactivation-memory transient; targetactivationmemoryalone isnot neuralwalking'))
 w.run(O,1.,100.,50.,.0000125,neural_duration=.25,neural_command_dt_s=.001,warmup_settle_s=.15,warmup_hold_s=.05,muscle_activation_timeconstants=(.005,.030),warmup_actuator_command=cmd,warmup_activation_start_s=1.15)
 w.dump(O/'SOURCE_PARAMETERIZATION_LEDGER.json',ledger);w.dump(O/'SOURCE_UNCHANGED.json',dict(passed=all(sha(Path(k))==v for k,v in hashes.items()),source_hashes=hashes));w.dump(O/'WARMUP_MUSCLE_COMMAND.json',dict(command=cmd,pool_activation=x,G=G,fit_file=str(FIT),fit_sha256=sha(FIT),unresolved_control_zero=True,pool_groups=66))
 for p in [Path(__file__),source]:(O/p.name).write_bytes(p.read_bytes())
 (O/'OLD_RUNNER.py').write_bytes(OLD.read_bytes());print('MUSCLE_PREACTIVATION_ACTUAL_COMPLETE',O,flush=True)
if __name__=='__main__':main()
