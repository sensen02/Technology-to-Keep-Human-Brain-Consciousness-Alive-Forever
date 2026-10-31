"""S175 addendum: per-row activation-interval reachability + figure, saved data only."""
from pathlib import Path
import argparse,json,csv
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path('/home/sensen/Desktop/engine修复验收')
def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def main(p):
 S173=BASE/'s173-accounted-near-initial-search/20261024T235544196037334Z/RUN';S169=BASE/'s169-near-initial-support/20261024T233428554259Z'
 N=BASE/'s154-activation-rate/20261024T215751775489Z/init80ms';T=np.load(BASE/'s166-mapped-support/20261024T231418078692647Z/MAPPING.npz')['T']
 native=load(N/'NATIVE_MODEL_ARRAYS.npz');mass=native['body_mass'];W=float(mass.sum()*9801.)
 poses=[('s173_best303',S173/'BEST_DIAGNOSTIC_NOT_CANDIDATE.npz'),('s169_best91',S169/'BEST_DIAGNOSTIC_NOT_CANDIDATE.npz'),('original_release_eval1',S169/'ORIGINAL_POSE_OPTIMIZED_POOLS_DIAGNOSTIC.npz')]
 rows=[];summary=[]
 for name,path in poses:
  z=load(path);B=z['B'];bb=-(z['base']+z['J'].T@z['f0'])
  for setname,Tm in [('real66',T),('widest90',.5*np.eye(90))]:
   A=B@Tm
   for d in range(48):
    lo=float(np.minimum(A[d],0).sum());hi=float(np.maximum(A[d],0).sum());target=float(bb[d])
    reach=bool(lo-1e-9<=target<=hi+1e-9)
    rows.append(dict(pose=name,activation_set=setname,dof=d,target=target,min_possible=lo,max_possible=hi,interval_reachable=reach,
      class_='ROOT_ZERO_AUTHORITY' if d<6 else ('SIGN_OR_AUTHORITY_LIMITED' if not reach else 'interval_ok')))
   sh=sum(1 for r in rows if r['pose']==name and r['activation_set']==setname and not r['interval_reachable'])
   summary.append(dict(pose=name,activation_set=setname,rows_not_interval_reachable=sh,rows_checked=48,
     reachable_dofs=[r['dof'] for r in rows if r['pose']==name and r['activation_set']==setname and r['interval_reachable']]))
 with (p/'ROW_INTERVAL_REACHABILITY.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 json.dump(summary,(p/'ROW_INTERVAL_SUMMARY.json').open('w'),indent=2)
 z=load(S173/'BEST_DIAGNOSTIC_NOT_CANDIDATE.npz');A=z['B']@T;meta=json.loads((N/'COM_CONTACT_METADATA.json').read_text())
 fig,axs=plt.subplots(2,2,figsize=(16,11))
 r=[x for x in rows if x['pose']=='s173_best303' and x['activation_set']=='real66']
 for d in range(48):
  x=r[d];c='tab:red' if x['class_']!='interval_ok' else 'tab:blue';axs[0,0].plot([d,d],[x['min_possible'],x['max_possible']],c=c,lw=3)
  axs[0,0].plot(d,x['target'],'k.')
 axs[0,0].axhline(0,color='gray',lw=.5);axs[0,0].set_title('S173best303 real66: achievable interval (bar) vs required target (dot)\nred = target outside interval, cannot be fixed by any activation in [0,1]');axs[0,0].set_xlabel('48 generalized DOFs');axs[0,0].set_ylabel('generalized force')
 axs[0,1].semilogy(range(48),[max(abs(A[d]).max(),1e-18) for d in range(48)],'o',ms=4);axs[0,1].axvline(5.5,color='r',ls='--');axs[0,1].set_title('Max muscle authority per DOF: root 0-5 are zero');axs[0,1].set_xlabel('generalized DOF')
 cp=z['contact_pos'];xp=z['xpos'];com=(mass[:,None]*xp).sum(0)/mass.sum();axs[1,0].scatter(cp[:,0],cp[:,1],c='tab:blue',label='contact points');axs[1,0].scatter([com[0]],[com[1]],c='r',marker='*',s=200,label='COM');axs[1,0].set_title('S173best303 top view');axs[1,0].legend();axs[1,0].axis('equal')
 wr=json.loads((p/'CONTACT_WRENCH_AND_COM.json').read_text());axs[1,1].bar(range(len(wr)),[v['ratio'] for v in wr]);axs[1,1].axhline(1,color='r',ls='--');axs[1,1].set_xticks(range(len(wr)),[v['pose'] for v in wr],fontsize=8,rotation=15);axs[1,1].set_title('Total normal force / weight (1 needed for static support)')
 fig.suptitle('S175 saved mechanism audit: root rows have no muscle authority; 5 leg rows are sign/authority-limited, not amplitude-limited');fig.tight_layout();fig.savefig(p/'ROOT_WRENCH_AUDIT.png',dpi=150)
 print(json.dumps(summary))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);main(ap.parse_args().output)
