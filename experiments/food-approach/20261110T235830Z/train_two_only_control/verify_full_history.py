"""Independent complete BOTH-state cache proof; no updates or physics."""
import ast,json
import torch
from model_io import HERE,load_parent,sha256
from data import prepare,padded
from engine import make

def main():
 torch.set_num_threads(1);model,_=load_parent();episodes,hashes=prepare(model);p=padded(episodes);engine=make(model)
 checks={};max_mean=max_rh=0.
 with torch.no_grad():
  batched,rh=engine(p['features'],p['gate'],p['means'],p['valid'])
  for j,e in enumerate(episodes):
   mean,hist=model.sequence(e['x'][:,None],e['u'][:,None])
   checks[e['name']+'_all2880_base_hidden_exact']=torch.equal(hist[0][:,0],e['hidden'])
   mm=float((mean[:,0]-batched[:,j]).abs().max());rr=float((hist[1][:,0]-rh[:,j]).abs().max())
   max_mean=max(max_mean,mm);max_rh=max(max_rh,rr)
   checks[e['name']+'_all2880_means_tol']=mm<=1e-12
   checks[e['name']+'_all2880_residual_hidden_tol']=rr<=1e-12
 def defs(path):return [ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
 old=HERE.parent/'train/engine.py';checks['primary_engine_class_and_make_AST_exact']=defs(old)==defs(HERE/'engine.py')
 sources={str(path):sha256(path) for path in (HERE/'verify_full_history.py',HERE/'engine.py',HERE/'model_io.py',HERE/'data.py',old)}
 report=dict(all_passed=all(checks.values()),checks=checks,source_hashes=sources,data_hashes=hashes,max_mean_error=max_mean,max_residual_hidden_error=max_rh,physical_steps=0,optimizer_updates=0,scope='independent fullbothstate andprimaryengine mathematical binding, not physical behavior')
 with (HERE/'FULL_HISTORY_CHECKS.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(report,indent=2))
 if not report['all_passed']:raise ValueError('completebothstate proof failed')

if __name__=='__main__':main()
