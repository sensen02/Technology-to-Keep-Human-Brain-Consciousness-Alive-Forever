"""Deterministic synthetic ENGINE-FREE engineering tests; no optimize() call."""
from pathlib import Path
import argparse,json,sys,ast
import numpy as np
from search_budget_accounting import ExplicitJacobian,BudgetExhausted
from run_near_initial_support_accounted import build

def run(out):
 out.mkdir(parents=True,exist_ok=False);tests=[]
 def record(name,condition):assert condition,name;tests.append(dict(test=name,passed=True))
 A=np.arange(12,dtype=float).reshape(3,4)/7;calls=[]
 def f(x):calls.append(x.copy());return A@x+np.array([1.,2.,3.])
 a=ExplicitJacobian(f,4,cap=11,initial_forward=1,output=out/'SYNTHETIC_LEDGER.json');x=np.array([0.,.4,.8,1.]);a.fun(x);J=a.jac(x)
 record('linear FD including zero fallback and backward bound',np.max(abs(J-A))<2e-7)
 record('full initial base+Jacobian costs 2*(n+1)',a.used==11 and len(calls)==5 and a.inner==5)
 a.fun(x.copy());a.jac(x.copy());record('identical exact cached base and all probes free',a.used==11 and len(calls)==5)
 before=len(calls)
 try:a.jac(np.array([.2,.4,.8,1.]));raise AssertionError('Expected stop')
 except BudgetExhausted:pass
 record('no probes or partial Jacobian after atomic insufficient budget',len(calls)==before and a.used==11)
 a2=ExplicitJacobian(f,4,cap=9,initial_forward=1);a2.fun(x);before=len(calls)
 try:a2.jac(x);raise AssertionError('Expected stop')
 except BudgetExhausted:pass
 record('initial Jacobian refusal before any difference probe',len(calls)==before and a2.jacobians==0)
 a3=ExplicitJacobian(f,4,cap=21,initial_forward=1);a3.fun(x);a3.fun(x+np.array([1e-14,0,0,0]));record('near but unequal poses never rounded into cache',a3.used==5 and len(a3.cache)==2)
 a4=ExplicitJacobian(lambda v:np.array([v[0]**2+3*v[1],np.sin(v[1])]),2,cap=100);xx=np.array([.3,.7]);JJ=a4.jac(xx);truth=np.array([[.6,3],[0,np.cos(.7)]]);record('nonlinear FD first-order accuracy',np.max(abs(JJ-truth))<1e-5)
 tree=build();text=ast.unparse(tree);record('adapter uses explicit wrapper not implicit2point',"_budgeted_solve(objective" in text and "ledger['inner_solves']" in text)
 # Compare physics objective and all acceptance expressions AST after removing accounting-only addition.
 original=ast.parse(Path(__file__).with_name('newrun_near_initial_support.py').read_text());old=next(n for n in ast.walk(original) if isinstance(n,ast.FunctionDef) and n.name=='objective');new=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='objective');new.body=[n for n in new.body if "ledger['inner_solves']" not in ast.unparse(n)];record('physics objective AST unchanged except inner-solve counter',ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False))
 record('no engine module loaded','mujoco' not in sys.modules)
 (out/'TEST_RESULTS.json').write_text(json.dumps(dict(passed=True,engineering_only=True,physics_runs=0,model_load=0,forward=0,optimization_experiments=0,tests=tests),indent=2));print(json.dumps(tests))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
