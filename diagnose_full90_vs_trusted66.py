"""Saved-only LP comparison: physical all90 authority vs admissible trusted66 runtime outputs."""
from pathlib import Path
import json, numpy as np
from scipy.optimize import linprog
O=Path('/home/sensen/Desktop/engine修复验收/s5-muscle-geometry-provenance/20261028T083000Z');S=Path('/home/sensen/Desktop/engine修复验收/s5-fixed-muscle-authority/20261028T081000Z');M=Path('/home/sensen/Desktop/engine修复验收/s5-walk-settle-neural/20261028T074000Z/MAPPING.json')
def dump(n,x):
 def enc(v):return v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v)
 (O/n).write_text(json.dumps(x,indent=2,default=enc)+'\n')
def main():
 import hashlib
 pre=O/'FULL90_PREREGISTRATION.json';assert not pre.exists();dump(pre.name,dict(source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),O/'OPERATING_ARRAYS.npz',S/'STATIC_DETAIL.json',M]},scope='All90independent activations diagnostic only vs trusted66 runtime ties/unresolved16zero. Saved-matrix LP only, no new engine calls, not a permitted motor mapping.'))
 z=np.load(O/'OPERATING_ARRAYS.npz');d=json.loads((S/'STATIC_DETAIL.json').read_text());G=np.array(d['G']);A90=z['active_torque']*.9999;np.testing.assert_allclose(A90@G,d['active_force_matrix'],rtol=1e-8,atol=1e-9);J=np.array(d['contact_force_matrix']);L=np.array(d['joint_limit_force_matrix']);w=d['bodyweight_native'];base=z['moment'].T@z['force_floor'];np.testing.assert_allclose(base,d['native_actuator_floor_force'],rtol=1e-8,atol=1e-9);target=(np.array(d['qfrc_bias'])-np.array(d['qfrc_passive'])-base)/w;results=[];raw={}
 for label,A in [('trusted66',A90@G),('physical90',A90)]:
  nmus=A.shape[1]
  for inner in [False,True]:
   B=np.column_stack([A,J,L])/w;n=B.shape[1];nc=len(d['contacts']);ub=[np.c_[B,-np.ones(48)],np.c_[-B,-np.ones(48)]];rhs=[target,-target]
   for j,c in enumerate(d['contacts']):
    dirs=[(1,1),(1,-1),(-1,1),(-1,-1)] if inner else [(1,0),(-1,0),(0,1),(0,-1)];h=np.zeros((4,n+1));h[:,nmus+3*j:nmus+3*j+3]=[[sx,sy,-c['mu']] for sx,sy in dirs];ub.append(h);rhs.append(np.zeros(4))
   obj=np.zeros(n+1);obj[-1]=1;bounds=[(0,1)]*nmus+[(None,None),(None,None),(0,None)]*nc+[(0,None)]*L.shape[1]+[(0,None)];aa=np.vstack(ub);bb=np.concatenate(rhs);r=linprog(obj,A_ub=aa,b_ub=bb,bounds=bounds,method='highs');assert r.success;key=label+('_inner' if inner else '_outer');results.append(dict(label=key,residual=float(r.fun),feasible=bool(r.fun<1e-7)));raw[key]=dict(A_ub=aa,b_ub=bb,bounds=bounds,objective=obj,solution=r.x)
 rows=json.loads(M.read_text())['rows'];unresolved=[dict(actuator=r['actuator'],actuator_index=r['actuator_index'],annotation=r['source_annotation']) for r in rows if not r['local_indices']];dump('UNRESOLVED_PHYSICAL_AUTHORITY.json',unresolved);dump('FULL90_RESULTS.json',results);dump('FULL90_LP_DETAIL.json',raw);dump('FULL90_EXECUTION.json',dict(new_models=0,new_forward=0,new_body_steps=0,new_neural_steps=0,exit_code=0));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(results),flush=True)
if __name__=='__main__':main()
