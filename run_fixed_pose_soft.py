"""S171 three fixed saved-pose native-soft matrix tests, no engine or integration."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,csv,hashlib,shutil
import numpy as np
from scipy.optimize import linprog
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype');BASE=Path('/home/sensen/Desktop/engine修复验收');S169=BASE/'s169-near-initial-support/20261024T233428554259Z';S166=BASE/'s166-mapped-support/20261024T231418078692647Z';S154=BASE/'s154-activation-rate/20261024T215751775489Z/init80ms'
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main(p):
 assert p.is_absolute() and not p.exists();p.mkdir(parents=True);freeze=p/'INPUT_FREEZE';freeze.mkdir()
 paths=[Path(__file__).resolve(),S169/'BEST_DIAGNOSTIC_NOT_CANDIDATE.npz',S169/'FROZEN_SEARCH.npz',S169/'PREREG.json',S169/'S169_SAVED_INDEPENDENT_CHECK.json',S166/'MAPPING.npz',S154/'NATIVE_MODEL_ARRAYS.npz',S154/'COM_CONTACT_METADATA.json']
 hashes={str(f):sha(f) for f in paths};dump(p/'INPUT_HASHES.json',hashes)
 for j,f in enumerate(paths):shutil.copyfile(f,freeze/f'{j:03d}_{f.name}')
 dump(p/'PREREG.json',dict(task='S171/P1-S2 exactly three fixed-pose native-soft feasibility problems',fixed='S169best qpos,zeroV,prescribedqacc0,B,base,J,f0 all48 equalities; nofreecontactforces',conditions=['66real pools gain.5 missing16zero','66real+.5 missing16 independent','90 independent activation .5'],variable_bounds='all normalized x in [0,1]',method='one phaseI LP each min t s.t -t<=A*x-b<=t; no other optimization',tolerance=dict(balance=1e-8,bounds=1e-10,separator_gap=1e-8),certificate='y=lowerrowdual-upperrowdual; verify b.y>sum(max(A.Ty,0)) for x box[0,1]; independent longdouble+roundoff guard',actdot='constant ctrl=act; muscle activation ODE numerator ctrl-act exactly0, positive frozen tau',geometry='unchanged frozen native gap/friction/nonfoot/limits criteria independently checked; algebraic solution alone not sufficient',restrictions='counterfactual cases2/3 notneural source/interface change; no newpose/load/forward/step/neural/training/deploy',budget=dict(phaseI_LPs=3,certificate_LPs=0,model_load=0,forward=0,step=0,neural=0)))
 z=load(paths[1]);m=load(S166/'MAPPING.npz');T=m['T'];missing=np.flatnonzero(m['actuator_pool']<0);assert T.shape==(90,66) and len(missing)==16 and np.array_equal(z['T'],T)
 Ts=[T,np.column_stack((T,.5*np.eye(90)[:,missing])),.5*np.eye(90)];B=z['B'];b=-z['base']-z['J'].T@z['f0'];assert B.shape==(48,90) and b.shape==(48,)
 names=['real66_missingzero','real66_plus_missing16','independent90_half'];summary=[];cases=[]
 for idx,(Tm,name) in enumerate(zip(Ts,names),1):
  A=B@Tm;n=A.shape[1];c=np.r_[np.zeros(n),1.];Au=np.vstack((np.column_stack((A,-np.ones(48))),np.column_stack((-A,-np.ones(48)))));bu=np.r_[b,-b]
  result=linprog(c,A_ub=Au,b_ub=bu,bounds=[(0,1)]*n+[(0,None)],method='highs',options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9));assert result.success,str(result.message)
  x=result.x[:n];t=float(result.x[-1]);act=Tm@x;res=A@x-b;y=result.ineqlin.marginals[48:]-result.ineqlin.marginals[:48];support=np.maximum(A.T@y,0).sum();gap=float(b@y-support);feasible=bool(abs(res).max()<=1e-8 and x.min()>=-1e-10 and x.max()<=1+1e-10);status='FEASIBLE_WITNESS' if feasible else ('INFEASIBLE_CERTIFICATE' if gap>1e-8 else 'NUMERICAL_NO_SOLUTION_FOUND')
  np.savez_compressed(p/f'CASE_{idx}.npz',T=Tm,A=A,b=b,x=x,t=t,y=y,activation=act,ctrl=act,act_dot=np.zeros(90),residual=res,separator_gap=gap,dual_marginals=result.ineqlin.marginals)
  row=dict(condition=idx,name=name,status=status,variables=n,minimax_t=t,max_abs_residual=float(abs(res).max()),separator_gap=gap,activation_min=float(act.min()),activation_max=float(act.max()),solver_status=int(result.status));summary.append(row);cases.append((res,act,y))
 dump(p/'SUMMARY.json',dict(cases=summary,pose='fixed S169best only',calls=dict(phaseI_LPs=3,model_load=0,forward=0,step=0,neural=0),inputs_unchanged=all(sha(f)==h for f,h in hashes.items()),interpretation='Each infeasible conclusion conditional on independently verified certificate; cases2/3 counterfactual only'))
 with (p/'RESIDUAL48.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['dof','condition1','condition2','condition3']);w.writeheader();w.writerows(dict(dof=i,**{f'condition{k+1}':float(cases[k][0][i]) for k in range(3)}) for i in range(48))
 fig,axs=plt.subplots(3,2,figsize=(16,11))
 for i,(res,act,y) in enumerate(cases):
  axs[i,0].plot(res);axs[i,0].axhline(0,color='black',lw=.5);axs[i,0].set_title(f'{i+1}: {summary[i]["status"]}, max|r|={abs(res).max():.6g}');axs[i,0].set_xlabel('48 DOFs; phase-I point is NOT a witness if infeasible')
  axs[i,1].bar(np.arange(90),act);axs[i,1].scatter(missing,act[missing],color='red',s=8,label='16 missing outputs');axs[i,1].set_ylim(0,.55);axs[i,1].legend();axs[i,1].set_title('Diagnostic LP activation, no deployment');axs[i,1].set_xlabel('90 actuators')
 fig.suptitle('S171 same pose / native soft f0 / all48 equations — three saved-matrix tests only');fig.tight_layout();fig.savefig(p/'THREE_CONDITIONS.png',dpi=150)
 assert all(sha(f)==h for f,h in hashes.items());print(json.dumps(summary))
if __name__=='__main__':
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path);args=pa.parse_args();main(args.output or BASE/'s171-fixed-pose-soft-interface'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
