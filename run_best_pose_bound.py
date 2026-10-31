"""S174 nested saved-matrix interface bounds, max3LPs; no engine."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,argparse
import numpy as np
from scipy.optimize import linprog
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path('/home/sensen/Desktop/engine修复验收');S=BASE/'s173-accounted-near-initial-search/20261024T235544196037334Z/RUN';M=BASE/'s166-mapped-support/20261024T231418078692647Z/MAPPING.npz'
def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main(p):
 assert not p.exists();p.mkdir(parents=True);(p/'INPUT_FREEZE').mkdir()
 files=[Path(__file__).resolve(),S/'BEST_DIAGNOSTIC_NOT_CANDIDATE.npz',S/'EVAL_303.npz',S/'SUMMARY.json',S/'EVALUATIONS.json',S/'S173_SAVED_INDEPENDENT_CHECK.json',S/'FROZEN_SEARCH.npz',S/'PREREG.json',M]
 hashes={str(f):sha(f) for f in files};dump(p/'INPUT_HASHES.json',hashes)
 for j,f in enumerate(files):shutil.copyfile(f,p/'INPUT_FREEZE'/f'{j:03d}_{f.name}')
 dump(p/'PREREG.json',dict(scope='S174/P1-S2 fixed S173best303 only, all48 Bact+base+J.Tf0=0',sequence='wide90 first; if certified infeasible reuse same y on real66 and fill82 no LP; else real66, if infeasible fill82; if real feasible embed fill witness',max_LPs=3,contact='fixed native f0 never free',tolerance=1e-8,certificate='b.y-sum(max(A.Ty,0))>0; independent exact rational binary64 check',geometry='frozen S169/S173 gates unchanged',no_model=True,no_forward=True,no_step=True,no_neural=True,no_training=True))
 assert sha(files[1])==sha(files[2]);best=load(files[1]);mapping=load(M);T=mapping['T'];missing=np.flatnonzero(mapping['actuator_pool']<0);Ts={'wide':.5*np.eye(90),'real':T,'fill':np.column_stack((T,.5*np.eye(90)[:,missing]))};b=-best['base']-best['J'].T@best['f0'];trace=[];cases=[];data={}
 def evaluate(name,y=None,embedded=None):
  tt=Ts[name];A=best['B']@tt;n=A.shape[1];z=dict(T=tt,A=A,b=b);solved=y is None and embedded is None
  if solved:
   result=linprog(np.r_[np.zeros(n),1.],A_ub=np.vstack((np.column_stack((A,-np.ones(48))),np.column_stack((-A,-np.ones(48))))),b_ub=np.r_[b,-b],bounds=[(0,1)]*n+[(0,None)],method='highs',options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9));trace.append(dict(name=name,success=bool(result.success),status=int(result.status)));dump(p/'SOLVE_TRACE.json',trace);assert result.success,result.message
   x=result.x[:n];y=result.ineqlin.marginals[:48]-result.ineqlin.marginals[48:];z.update(x=x,t=result.x[-1],dual_marginals=result.ineqlin.marginals)
  elif embedded is not None:x=embedded;z['x']=x;y=np.zeros(48)
  z['y']=y;gap=float(b@y-np.maximum(A.T@y,0).sum());z['separator_gap']=gap
  if 'x' in z:
   act=tt@z['x'];res=A@z['x']-b;z.update(activation=act,ctrl=act,act_dot=np.zeros(90),residual=res);feasible=bool(abs(res).max()<=1e-8 and z['x'].min()>=-1e-10 and z['x'].max()<=1+1e-10)
  else:feasible=False
  status='FEASIBLE_WITNESS' if feasible else ('INFEASIBLE_CERTIFICATE' if gap>1e-8 else 'NUMERICAL_NO_SOLUTION_FOUND');np.savez_compressed(p/f'CASE_{name}.npz',**z);cases.append(dict(name=name,status=status,solved=solved,certificate_source='wide' if not solved and not feasible else name));data[name]=z;return status
 status=evaluate('wide')
 if status=='INFEASIBLE_CERTIFICATE':
  for name in ['real','fill']:assert evaluate(name,y=data['wide']['y'])=='INFEASIBLE_CERTIFICATE'
 elif status=='FEASIBLE_WITNESS':
  real=evaluate('real')
  if real=='FEASIBLE_WITNESS':evaluate('fill',embedded=np.r_[data['real']['x'],np.zeros(16)])
  elif real=='INFEASIBLE_CERTIFICATE':evaluate('fill')
 assert len(trace)<=3;dump(p/'SUMMARY.json',dict(cases=cases,calls=dict(phaseI_LPs=len(trace),certificate_LPs=0,model_load=0,forward=0,step=0,neural=0),best_evaluation=303,inputs_unchanged=all(sha(f)==h for f,h in hashes.items())))
 fig,ax=plt.subplots(2,1,figsize=(13,9));ax[0].plot(data['wide']['residual']);ax[0].set_title('Wide90 phase-I diagnostic residual, NOT a witness if infeasible');ax[0].set_xlabel('48 DOFs');ax[1].plot(data['wide']['y'],'o-');ax[1].set_title('Separator y coefficients: NOT causal contribution proportions');ax[1].set_xlabel('48 DOFs');fig.suptitle('S174 fixed S173best303 native soft f0; nested-set certificates');fig.tight_layout();fig.savefig(p/'RESIDUAL_CERTIFICATE.png',dpi=150)
 print(json.dumps(cases))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);a=ap.parse_args();main(a.output or BASE/'s174-best-pose-interface-bound'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
