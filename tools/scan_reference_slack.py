"""Within-cycle static slack scan over already-sealed postures; pure arrays, no model.
Answers: is ANY posture in the reference close to statically supportable, and where.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.optimize import linprog
C=Path('/home/sensen/Desktop/engine修复验收/s2-reference-support-candidate/20261102T182109Z')
O=Path('/home/sensen/Desktop/engine修复验收/s4-reference-slack-scan/20261102T232000Z')
FLOOR=1e-4
def ar(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def main():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'RESULT.json').exists()
 g=ar(C/'GROUP_MASKS.npz');G=g['G'];files=[Path(__file__),C/'GROUP_MASKS.npz'];hashes={str(p):sha(p) for p in files}
 rows=[];calls=0
 for pose in range(33):
  nat=ar(C/f'NATIVE_{pose}.npz');mus=ar(C/f'MUSCLE_{pose}.npz');rea=ar(C/f'REACTIONS_{pose}.npz');con=json.loads((C/f'CONTACTS_{pose}.json').read_text())
  A=nat['moment'].T*mus['gain'];base=(nat['qfrc_passive']-nat['qfrc_bias'])+nat['moment'].T@mus['bias']
  contacts=rea['native_foot_columns'];internal=rea['internal_columns']
  nint=int(internal.shape[1]);nc=contacts.shape[1]//3
  for arm,P in [('mapped66',G),('independent90',np.eye(90))]:
   n=P.shape[1];E=np.column_stack([A@P,contacts,internal]) if nint else np.column_stack([A@P,contacts])
   b=-base-A@np.full(90,FLOOR);ncols=E.shape[1]
   bounds=[(0.,1.-FLOOR)]*n+[(0.,np.inf)]*contacts.shape[1]+[(0.,np.inf)]*nint
   rs=np.maximum(1,np.maximum(abs(E).max(1),abs(b)))
   Es=E/rs[:,None];bs=b/rs;nv=ncols+1
   Ef=np.vstack([np.hstack([Es,-np.ones((len(Es),1))]),np.hstack([-Es,-np.ones((len(Es),1))])]);bf=np.concatenate([bs,-bs])
   nv2=ncols+1;bnds=bounds+[(0.,np.inf)];c=np.zeros(nv2);c[-1]=1.
   res=linprog(c,A_ub=Ef,b_ub=bf,bounds=bnds,method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9});calls+=1
   gap=float(res.x[-1]) if res.success else None
   rows.append(dict(pose=pose,arm=arm,feet=nc,internal=nint,relative_equation_slack=gap,status=int(res.status)))
  if pose%8==0:print('pose',pose,flush=True)
 best=min((r for r in rows if r['relative_equation_slack'] is not None),key=lambda r:r['relative_equation_slack'])
 out=dict(passed=True,source_sha256=hashes,scope='sealed 33 reference postures, conditional static slack with qvel zeroed; no model call',rows=rows,
  closest=best,interpretation='Slack is the smallest uniform relative violation of the 48 equilibrium rows. 0 would mean statically holdable. This is a within-cycle ranking only, not a dynamic prediction.',calls=dict(model_load=0,forward=0,body_step=0,neural_step=0,linprog=calls))
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'RESULT.json',out);print(json.dumps(best,indent=1))
if __name__=='__main__':main()
