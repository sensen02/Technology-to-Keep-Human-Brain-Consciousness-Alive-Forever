"""One prospective actual walking -> candidate hinge stance -> neural handover."""
from pathlib import Path
import ast,json,hashlib
import numpy as np
import run_physical_walk_prewarm as w
import run_neural_muscle_closure as c
P=Path(__file__).parent
O=Path('/home/sensen/Desktop/engine修复验收/s5-compatible-stance-handover/20261028T080000Z')
C=Path('/home/sensen/Desktop/engine修复验收/s5-muscle-compatible-stance/20261028T079000Z/CANDIDATE.npz')
OLD=Path('/home/sensen/Desktop/engine修复验收/s5-muscle-preactivation/20261028T078000Z/run_physical_walk_prewarm.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def reverse(tree):
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
 assert fn.args.args[-1].arg=='warmup_settle_target';fn.args.args.pop();fn.args.defaults.pop()
 class R(ast.NodeTransformer):
  def visit_Assign(self,node):
   if any(isinstance(t,ast.Name) and t.id in ['settle_target','settle_joint_ids'] for t in node.targets):return None
   return self.generic_visit(node)
  def visit_If(self,node):
   if 'settle_target' in ast.unparse(node.test):return None
   return self.generic_visit(node)
  def visit_Expr(self,node):
   if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='dump' and 'SETTLE_TARGET.json' in ast.unparse(node):return None
   return self.generic_visit(node)
  def visit_Call(self,node):
   node=self.generic_visit(node);node.keywords=[k for k in node.keywords if k.arg!='warmup_settle_target'];return node
  def visit_Name(self,node):
   return ast.Subscript(value=ast.Name(id='q',ctx=ast.Load()),slice=ast.Name(id='qa',ctx=ast.Load()),ctx=node.ctx) if node.id=='settle_target' else node
 return ast.fix_missing_locations(R().visit(tree))
def main():
 source=Path(w.__file__);now=source.read_text();old=OLD.read_text();assert ast.dump(reverse(ast.parse(now)),include_attributes=False)==ast.dump(ast.parse(old),include_attributes=False),'Unexpected change outside candidate target option'
 z=np.load(C);target=np.array(z['qpos_hinges'],dtype=float);assert target.shape==(42,) and np.isfinite(target).all()
 # Explicit address/name identity checking is completed before running when candidate includes them.
 ga=np.load('/home/sensen/Desktop/engine修复验收/s5-walk-settle-neural/20261028T074000Z/GAIT.npz');qa=ga['qpos_addresses']
 if 'qpos_addresses' in z.files:np.testing.assert_array_equal(z['qpos_addresses'],qa)
 files=[Path(__file__),source,C,OLD,P/'engine/embodied/continuous_handover.py',P/'engine/embodied/motor_output.py',P/'engine/embodied/sensory_channels.py',P/'engine/embodied/runtime_pool_guard.py',P/'engine/neural_cond.py',P/'engine/embodied/adapters.py',P/'engine/neck_cut_data.py',P/'data/flywire/banc_connectome.npz',c.POOLS,w.STANCE,w.REF,w.BASE/'INTEGRATED_MODEL.xml'];hashes={str(x):sha(x) for x in files}
 ledger=dict(old_file=str(OLD),old_sha256=sha(OLD),new_file=str(source),new_sha256=sha(source),default_none_AST_identical_after_declared_reversal=True,changes=['optional warmup_settle_target=None','finite42 native hinge range validation and target sidecar','settling target substituted only during assisted transition, sensory reference unchanged'])
 pre=O.parent/(O.name+'_PREDECLARED.json');pre.parent.mkdir(parents=True,exist_ok=True);assert not pre.exists() and not O.exists()
 w.dump(pre,dict(scope='Actual1s original walking then150ms candidate hinge-target transition and50ms hold then250ms pure neural; no direct state/root writes',candidate=str(C),candidate_sha256=sha(C),target=target,qpos_addresses=qa,walking_s=1.,settle_s=.15,assisted_hold_s=.05,neural_s=.25,neural_command_dt_s=.001,muscle_activation_timeconstants=None,warmup_actuator_command=None,descending_pulse=False,sensory_reference='Original prewarm qpos, not candidate',source_hashes=hashes,source_ledger=ledger,success_gate='Same original walking prefix and model; candidate physically reached without nonfoot contact; continuous pure neural body support outcome. Relative static LP improvement is not equilibrium or behavior proof.'))
 w.run(O,1.,100.,50.,.0000125,neural_duration=.25,neural_command_dt_s=.001,warmup_settle_s=.15,warmup_hold_s=.05,warmup_settle_target=target)
 w.dump(O/'SOURCE_PARAMETERIZATION_LEDGER.json',ledger);w.dump(O/'SOURCE_UNCHANGED.json',dict(passed=all(sha(Path(k))==v for k,v in hashes.items()),source_hashes=hashes))
 for x in [Path(__file__),C]:(O/x.name).write_bytes(x.read_bytes())
 print('ACTUAL_COMPATIBLE_STANCE_COMPLETE',O,flush=True)
if __name__=='__main__':main()
