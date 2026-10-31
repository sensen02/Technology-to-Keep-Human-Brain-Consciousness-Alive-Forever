"""S175 saved-only root-wrench mechanism audit; no engine, no forward, bounded LPs."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,argparse,csv
import numpy as np
from scipy.optimize import linprog
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path('/home/sensen/Desktop/engine修复验收')
S173=BASE/'s173-accounted-near-initial-search/20261024T235544196037334Z/RUN'
S169=BASE/'s169-near-initial-support/20261024T233428554259Z'
S166=BASE/'s166-mapped-support/20261024T231418078692647Z/MAPPING.npz'
N=BASE/'s154-activation-rate/20261024T215751775489Z/init80ms'
POSE_KEYS=['s169_best91','s173_best303','original_release_eval1']
def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main(p):
 assert p.is_absolute() and not p.exists();p.mkdir(parents=True);(p/'INPUT_FREEZE').mkdir()
 files=[Path(__file__).resolve(),S169/'BEST_DIAGNOSTIC_NOT_CANDIDATE.npz',S173/'BEST_DIAGNOSTIC_NOT_CANDIDATE.npz',S169/'ORIGINAL_POSE_OPTIMIZED_POOLS_DIAGNOSTIC.npz',S166,N/'NATIVE_MODEL_ARRAYS.npz',N/'COM_CONTACT_METADATA.json',BASE/'s174-best-pose-interface-bound/20261025T000617427397Z/CASE_wide.npz',BASE/'s174-best-pose-interface-bound/20261025T000617427397Z/S174_SAVED_INDEPENDENT_CHECK.json']
 h={str(f):sha(f) for f in files};dump(p/'INPUT_HASHES.json',h)
 for j,f in enumerate(files):shutil.copyfile(f,p/'INPUT_FREEZE'/f'{j:03d}_{f.name}')
 dump(p/'PREREG.json',dict(scope='S175/P1-S2 mechanism audit of already-certified static failure; saved data only',
  question='do the 6 root DOFs have zero actuator authority, making interface amplitude/mapping irrelevant to them; are the other 42 rows satisfiable',
  poses=POSE_KEYS,max_LPs=6,lp_use='42-row subsystem minimax only (root rows excluded and reported separately)',
  claimed='authority and wrench decomposition are exact algebra on frozen saved arrays; subsystem feasibility only',
  not_claimed='no impossibility of any pose, no dynamic/standing/neural claim, no causal ranking of certificate coefficients',
  no_model=True,no_forward=True,no_step=True,no_neural=True,no_training=True))
 native=load(N/'NATIVE_MODEL_ARRAYS.npz');meta=json.loads((N/'COM_CONTACT_METADATA.json').read_text())
 mass=native['body_mass'];W=float(mass.sum()*9801.);floors=set(meta['floor_geom_ids']);gl={int(k):v for k,v in meta['geom_map'].items()}
 mapping=load(S166);Treal=mapping['T'];missing=np.flatnonzero(mapping['actuator_pool']<0)
 Ts={'real66':Treal,'widest90':.5*np.eye(90)}
 poses={'s169_best91':S169/'BEST_DIAGNOSTIC_NOT_CANDIDATE.npz','s173_best303':S173/'BEST_DIAGNOSTIC_NOT_CANDIDATE.npz','original_release_eval1':S169/'ORIGINAL_POSE_OPTIMIZED_POOLS_DIAGNOSTIC.npz'}
 auth=[];rows=[];subsystem=[];trace=[];wrench=[];posez={}
 for name,path in poses.items():
  z=load(path);posez[name]=z;B=z['B'];base=z['base'];J=z['J'];f0=z['f0'];act=z['act'];res=z['residual48'];con=J.T@f0;mus=B@act
  assert z['B'].shape==(48,90) and abs(res-(mus+base+con)).max()<1e-8
  for d in range(48):
   col=np.array([abs((B@Ts[k])[d]).max() for k in ('real66','widest90')])
   auth.append(dict(pose=name,dof=d,authority_real66=float(col[0]),authority_widest90=float(col[1]),class_='ROOT_ZERO_AUTHORITY' if d<6 else 'JOINT_CONTROLLABLE'))
  for d in range(6):
   rows.append(dict(pose=name,dof=d,muscle=B@act[d] if False else float(mus[d]),base=float(base[d]),contact=float(con[d]),residual=float(res[d])))
  xp=z['xpos'];com=(mass[:,None]*xp).sum(0)/mass.sum();root=z['qpos'][:3]
  cp=np.array([z['contact_pos'][i] for i in range(len(z['contact_geom'])) if int(z['contact_efc_address'][i])>=0 and f0[int(z['contact_efc_address'][i])]>0])
  legs_world=z['foot_load'];fz=float(legs_world[:,2].sum())
  wrench.append(dict(pose=name,weight=W,total_normal_from_legs=fz,ratio=fz/W,com_world=com.tolist(),root_world=root.tolist(),com_minus_root=(com-root).tolist(),
    com_x_inside_contact_x=bool(cp[:,0].min()<=com[0]<=cp[:,0].max()),com_y_inside_contact_y=bool(cp[:,1].min()<=com[1]<=cp[:,1].max()),
    contact_x_range=[float(cp[:,0].min()),float(cp[:,0].max())],contact_y_range=[float(cp[:,1].min()),float(cp[:,1].max())],
    gravity_torque_about_root_y_required=float(-base[4]),gravity_torque_about_root_z_required=float(-base[5]),contact_torque_y=float(con[4]),contact_torque_z=float(con[5]),
    contacts_positive_force=int(len(cp)),note='root rows have zero activation authority; only gravity+contact can satisfy them'))
  for setname,Tm in Ts.items():
   A=(B@Tm)[6:,:];bb=(-base-con)[6:];n=A.shape[1]
   r=linprog(np.r_[np.zeros(n),1.],A_ub=np.vstack((np.column_stack((A,-np.ones(42))),np.column_stack((-A,-np.ones(42))))),b_ub=np.r_[bb,-bb],bounds=[(0,1)]*n+[(0,None)],method='highs',options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
   trace.append(dict(pose=name,set=setname,success=bool(r.success),status=int(r.status)))
   t=float(r.x[-1]) if r.success else float('nan')
   subsystem.append(dict(pose=name,activation_set=setname,variables=n,joint_rows=42,minimax_t=t,joint_rows_exactly_satisfiable=bool(r.success and t<=1e-8),root_rows_excluded=6,note='excluded root rows carry the certified imbalance and are untouched by any activation'))
 assert len(trace)==6
 with (p/'ACTUATOR_AUTHORITY_BY_DOF.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(auth[0]));w.writeheader();w.writerows(auth)
 with (p/'ROOT_ROW_DECOMPOSITION.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 dump(p/'SUBSYSTEM_42ROW_FEASIBILITY.json',dict(solve_trace=trace,results=subsystem))
 dump(p/'CONTACT_WRENCH_AND_COM.json',wrench)
 z=posez['s173_best303'];B=z['B'];x=np.load(BASE/'s174-best-pose-interface-bound/20261025T000617427397Z/CASE_wide.npz')
 assert np.array_equal(x['b'],-(z['base']+z['J'].T@z['f0'])) and np.array_equal(x['A'],z['B']@x['T'])
 dump(p/'S174_CONSISTENCY.json',dict(same_A=True,same_b=True,wide_T_is_half_identity=bool(np.array_equal(x['T'],.5*np.eye(90))),certificate_dofs=np.flatnonzero(x['y']).tolist()))
 fig,axs=plt.subplots(2,2,figsize=(16,11))
 a=[v for v in auth if v['pose']=='s173_best303']
 axs[0,0].semilogy(range(48),[max(v['authority_real66'],1e-18) for v in a],'o',ms=4);axs[0,0].axvline(5.5,color='r',ls='--');axs[0,0].set_title('Max actuator authority per DOF (real66): DOF0-5 root ~0');axs[0,0].set_xlabel('generalized DOF')
 r173=[v for v in rows if v['pose']=='s173_best303'];axs[0,1].bar(np.arange(6)-.2,[v['base'] for v in r173],.4,label='gravity/passive/bias');axs[0,1].bar(np.arange(6)+.2,[v['contact'] for v in r173],.4,label='native soft contact');axs[0,1].plot(np.arange(6),[v['residual'] for v in r173],'k-o',label='residual');axs[0,1].set_xticks(range(6),['x','y','z','rotx','roty','rotz']);axs[0,1].set_title('Root rows: muscle term exactly 0');axs[0,1].legend(fontsize=8)
 xp=z['xpos'];com=(mass[:,None]*xp).sum(0)/mass.sum();cp=np.array([z['contact_pos'][i] for i in range(len(z['contact_geom']))])
 axs[1,0].scatter(cp[:,0],cp[:,1],c='tab:blue',label='contact points');axs[1,0].scatter([com[0]],[com[1]],c='r',marker='*',s=180,label='COM');axs[1,0].set_title('S173best303 top view: COM vs contact points');axs[1,0].set_xlabel('world x');axs[1,0].set_ylabel('world y');axs[1,0].legend();axs[1,0].axis('equal')
 axs[1,1].bar(np.arange(6),[v['ratio'] for v in wrench]);axs[1,1].axhline(1,color='r',ls='--');axs[1,1].set_xticks(range(6),[v['pose'] for v in wrench],fontsize=8);axs[1,1].set_title('Total normal force / weight (needs 1 for static support)')
 fig.suptitle('S175 saved mechanism audit: root DOFs have no muscle authority; balance is a contact-wrench problem');fig.tight_layout();fig.savefig(p/'ROOT_WRENCH_AUDIT.png',dpi=150)
 assert all(sha(f)==v for f,v in h.items());dump(p/'SUMMARY.json',dict(poses=POSE_KEYS,lp_calls=len(trace),root_zero_authority_dofs=list(range(6)),max_root_authority=float(max(abs(v['authority_widest90']) for v in auth if v['dof']<6)),min_joint_authority=float(min(v['authority_real66'] for v in auth if v['dof']>=6)),inputs_unchanged=True,calls=dict(model_load=0,forward=0,step=0,neural=0,optimization_LPs=len(trace))))
 print(json.dumps(subsystem));print(json.dumps(wrench))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);a=ap.parse_args();main(a.output or BASE/'s175-root-wrench-audit'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
