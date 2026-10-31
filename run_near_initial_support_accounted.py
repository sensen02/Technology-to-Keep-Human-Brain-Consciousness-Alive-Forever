"""S172 opt-in accounted S169 derivative. No model loads on import/default invocation.
Original physics/range/acceptance AST retained; execution requires --execute-search.
This turn tests only synthetic callback accounting, never this physics entrypoint.
"""
from pathlib import Path
import ast,inspect,argparse,hashlib
import numpy as np
from search_budget_accounting import ExplicitJacobian,BudgetExhausted
SOURCE=Path(__file__).with_name('newrun_near_initial_support.py')
SOURCE_SHA256='795023e2b987e90ca7e62158cbb26a79ca8fb36f5f0e2b0c5cd21f7429353ec5'
SEARCH_CAP=1990
TOTAL_CAP=2000
RESERVED_VALIDATION=10
MAX_JACOBIANS=20
class LimitedJacobian(ExplicitJacobian):
 def jac(self,x):
  if self.jacobians>=MAX_JACOBIANS:
   self.emit(role='COMPLETE_JACOBIAN_LIMIT',limit=MAX_JACOBIANS)
   raise BudgetExhausted('Maximum complete Jacobians reached')
  return super().jac(x)
def build():
 raw=SOURCE.read_bytes();assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA256,'Original immutable S169 source changed'
 tree=ast.parse(raw)
 class Rewrite(ast.NodeTransformer):
  def visit_Assign(self,node):
   self.generic_visit(node)
   if any(isinstance(t,ast.Name) and t.id=='CAP' for t in node.targets):node.value=ast.Constant(value=SEARCH_CAP)
   if any(isinstance(t,ast.Name) and t.id=='files' for t in node.targets):
    node.value=ast.BinOp(left=node.value,op=ast.Add(),right=ast.parse("[ROOT/'newrun_near_initial_support.py',ROOT/'search_budget_accounting.py']",mode='eval').body)
   return node
  def visit_Call(self,node):
   self.generic_visit(node)
   if isinstance(node.func,ast.Name) and node.func.id=='least_squares':node.func=ast.Name(id='_budgeted_solve',ctx=ast.Load())
   return node
  def visit_FunctionDef(self,node):
   self.generic_visit(node)
   if node.name=='objective':
    body=[]
    for line in node.body:
     body.append(line)
     if isinstance(line,ast.Assign) and isinstance(line.value,ast.Call) and isinstance(line.value.func,ast.Name) and line.value.func.id=='lsq_linear':body.extend(ast.parse("ledger['inner_solves']=ledger.get('inner_solves',0)+1").body)
    node.body=body
   return node
 tree=ast.fix_missing_locations(Rewrite().visit(tree));return tree

def _budgeted_solve(fun,x0,**kwargs):
 from scipy.optimize import least_squares
 v=inspect.getclosurevars(fun).nonlocals;ledger=v['ledger'];out=v['out'];lower=v['lower'];width=v['width'];q0=v['q0'];indices=v['indices']
 def posekey(x):
  q=q0.copy();q[indices]=lower+x*width;return q.tobytes()+np.asarray(x,dtype=np.float64).tobytes()
 import json
 (out/'EXPLICIT_ACCOUNTING_CONFIG.json').write_text(json.dumps(dict(qpos_indices=indices.tolist(),physical_width=width.tolist(),lower=lower.tolist(),cache_key='exact physical qpos bytes AND exact normalized x bytes; no rounding',forward_per_miss=2,inner_per_miss=1,jacobian_columns=len(x0),cached_base_full_jacobian_cost=2*len(x0),first_base_and_jacobian_cost=2*(len(x0)+1),meaning='jacobian base transitions are not a convergence or optimizer accepted-update claim'),indent=2))
 accounting=LimitedJacobian(fun,len(x0),cap=SEARCH_CAP,initial_forward=ledger['forward'],output=out/'EXPLICIT_EVALUATION_LEDGER.json',counters=lambda:(ledger['forward'],ledger.get('inner_solves',0)),key_function=posekey)
 kwargs['jac']=accounting.jac
 kwargs['max_nfev']=1000
 try:return least_squares(accounting.fun,x0,**kwargs)
 except BudgetExhausted as exc:raise fun.__globals__['BudgetStop']() from exc
 finally:accounting.emit(role='SOLVER_ACCOUNTING_FINISHED',note='base transition counts completed Jacobian bases, not iterations or convergence; trial objectives logged separately')

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute-search',action='store_true');ap.add_argument('--output',type=Path);args=ap.parse_args()
 if not args.execute_search:print('NOT_EXECUTED: synthetic regression only; no physics entrypoint invoked');return
 assert args.output is not None and args.output.is_absolute() and not args.output.exists()
 ns=dict(__name__='s172_physics_derivative',__file__=str(Path(__file__).resolve()),_budgeted_solve=_budgeted_solve)
 exec(compile(build(),str(SOURCE),'exec'),ns);ns['main'](args.output)
if __name__=='__main__':main()
