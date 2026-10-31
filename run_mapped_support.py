"""S166 saved-only interface topology static capacity upper bound. No engine imports."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,json,hashlib,shutil
import numpy as np
from scipy.optimize import linprog
BASE=Path('/home/sensen/Desktop/engine修复验收');S165=BASE/'s165-static-support-corrected/20261024T231400000000Z';S163=BASE/'s163-sensory-readout-audit/20261024T224712953805523Z';S154=BASE/'s154-activation-rate/20261024T215751775489Z/init80ms';SOURCE=BASE/'s162-fast-sensory/20261024T224135142077Z/BOUND_INPUT_FREEZE/0104_motor_translation.py';META=S154/'COM_CONTACT_METADATA.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def table(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main(p):
 assert p.is_absolute() and not p.exists();p.mkdir(parents=True);freeze=p/'INPUT_FREEZE';freeze.mkdir()
 files=[Path(__file__).resolve(),SOURCE,META,S163/'GRAPH_90_MUSCLE_MAPPING.csv',S163/'GRAPH_66_POOL_MEMBERSHIP.csv',S165/'AFFINITY_PROOF.json',S165/'MODEL_PROOF.json',S165/'PARENT_STATIC_INDEPENDENT_CHECK.json',S154/'TRAJECTORY.npz']
 for i in [0,15,83]:files += [S165/f'{stem}_{i:03d}.npz' for stem in ['STATIC_AFFINE','ZERO_V','DYNAMIC']]+[S165/f'rigid_upper_{i:03d}_upper{u:.1f}.npz' for u in [.5,1.]]
 hashes={str(f):sha(f) for f in files};dump(p/'INPUT_HASHES.json',hashes)
 for i,f in enumerate(files):shutil.copyfile(f,freeze/f'{i:03d}_{f.name}')
 dump(p/'PREREG.json',dict(scope='S166 P1/S2 pure offline mapped rigid capability upper bound; no body/neural',states=[0,15,83],equation='[B*T,J.T] [pools,lambda]=-base; T[i,p]=.5 for actual mapped pool, otherwise0',variables='66 postclip pool outputs [0,1] +72 native pyramid edges >=0',activation='act=ctrl=T*pools, native constant-state zero rate; unmapped16 zero, shared pools equal',tolerance_native_absolute=1e-8,objective='minimum sum of actual muscle activation via T; diagnostic selection only, not neural/controller',contact='unchanged all72 rows/48 equations from S165; rigid relaxed penetration/compliance upper bound ONLY',limitations='66 independent pool values not proven attainable by real neural/FIR history; no soft re-solve, no pose/contact/force changes',actual_calls=dict(model_load=0,forward=0,step=0,neural_ticks=0),units='native; SI NOT_ESTABLISHED'))
 rows=list(csv.DictReader((S163/'GRAPH_90_MUSCLE_MAPPING.csv').open()));assert len(rows)==90
 ap=np.array([int(r['pool_id']) for r in rows]);assert np.array_equal(ap,load(S154/'TRAJECTORY.npz')['actuator_pool']);assert len(set(ap[ap>=0]))==66 and sum(ap<0)==16
 groups=[tuple(sorted(map(int,r['local_indices'].split(';')))) if r['local_indices'] else () for r in rows];pools=sorted(set(g for g in groups if g));assert len(pools)==66
 assert np.array_equal(ap,np.array([pools.index(g) if g else -1 for g in groups]))
 T=np.zeros((90,66));T[np.flatnonzero(ap>=0),ap[ap>=0]]=.5;np.savez_compressed(p/'MAPPING.npz',T=T,actuator_pool=ap)
 shutil.copyfile(S163/'GRAPH_90_MUSCLE_MAPPING.csv',p/'FULL_ACTUATOR_MAPPING.csv');dump(p/'POOL_MEMBERSHIP.json',pools)
 aff=json.loads((S165/'AFFINITY_PROOF.json').read_text());ctrlrange=np.array(aff['ctrlrange']);limited=np.array(aff['ctrllimited'],bool);dyn=np.array(aff['dynprm']);assert np.all(dyn[:,0]==.001) and np.all(dyn[:,1]==.0004)
 out=[];poolrows=[];actrows=[];dofrows=[];contactrows=[];legrows=[]
 for i in [0,15,83]:
  a=load(S165/f'STATIC_AFFINE_{i:03d}.npz');z=load(S165/f'ZERO_V_{i:03d}.npz');assert a['J'].shape==(72,48) and np.all(z['efc_type']==6) and np.all(z['contact_row_count']==4)
  A=np.column_stack((a['B']@T,a['J'].T));b=-a['base'];bounds=np.column_stack((np.zeros(138),np.r_[np.ones(66),np.full(72,np.inf)]));scale=np.maximum(abs(A).max(1),1e-12);cost=np.r_[T.sum(0),np.zeros(72)]
  sol=linprog(cost,A_eq=A/scale[:,None],b_eq=b/scale,bounds=[(0,1)]*66+[(0,None)]*72,method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
  x=np.full(138,np.nan) if sol.x is None else sol.x;res=A@x-b;feasible=bool(np.isfinite(x).all() and abs(res).max()<=1e-8 and x.min()>=-1e-8 and x[:66].max()<=1+1e-8);act=T@x[:66]
  np.savez_compressed(p/f'STATE_{i:03d}.npz',Aeq=A,beq=b,x=x,bounds=bounds,residual=res,activation=act,pools=x[:66],**{'lambda':x[66:]})
  info=dict(sample=i,solver_status=int(sol.status),message=sol.message,feasible=feasible,max_residual=None if not feasible else float(abs(res).max()),classification='FEASIBLE_INTERFACE_RIGID_UPPER_BOUND' if feasible else 'NUMERICAL_NO_SOLUTION_PENDING_CERTIFICATE');out.append(info)
  for j in range(66):poolrows.append(dict(sample=i,pool=j,output=None if not feasible else float(x[j])))
  for j,r in enumerate(rows):actrows.append(dict(sample=i,actuator=j,name=r['actuator'],leg=r['leg'],pool=int(ap[j]),actual_activation=float(a['actual_activation'][j]),mapped_activation=None if not feasible else float(act[j]),independent90_half=float(load(S165/f'rigid_upper_{i:03d}_upper0.5.npz')['x'][j])))
  for j in range(48):dofrows.append(dict(sample=i,dof=j,residual=None if not feasible else float(res[j])))
  if not feasible:continue
  clamped=np.where(limited,np.clip(act,ctrlrange[:,0],ctrlrange[:,1]),act);clamped=np.clip(clamped,0,1);rate=(clamped-act)/np.where(clamped>act,dyn[:,0]*(.5+1.5*act),dyn[:,1]/(.5+1.5*act));assert np.array_equal(rate,np.zeros(90));assert np.all(act[ap<0]==0)
  np.savez_compressed(p/f'CONSTANT_CONTROL_{i:03d}_NOT_APPLIED.npz',ctrl=act,activation=act,clamped_ctrl=clamped,offline_act_dot=rate)
  # Independently compare chosen 90D upper-bound witnesses against topology, not optimality.
  for u in [.5,1.]:
   independent=load(S165/f'rigid_upper_{i:03d}_upper{u:.1f}.npz')['x'][:90];info[f'independent90_upper{u}_unmapped_max']=float(abs(independent[ap<0]).max());info[f'independent90_upper{u}_L1_to_mapped']=float(abs(independent-act).sum())
  info.update(pool_max=float(x[:66].max()),activation_max=float(act.max()),activation_sum=float(act.sum()),act_dot_max=0.)
  meta=json.loads(META.read_text());floors=set(meta['floor_geom_ids']);geomleg={int(k):v.upper() for k,v in meta['geom_map'].items()};legforce={leg:np.zeros(3) for leg in ['LF','LM','LH','RF','RM','RH']}
  for ci,addr in enumerate(z['contact_efc_address']):
   e=x[66+addr:66+addr+4];mu=z['contact_friction'][ci,:2];local=np.r_[e.sum(),mu[0]*(e[0]-e[1]),mu[1]*(e[2]-e[3])];margin=float(local[0]-sum(abs(local[1:]/mu)));assert margin>=-1e-8
   g1,g2=map(int,z['contact_geom_ids'][ci]);assert (g1 in floors)!=(g2 in floors);sign=1 if g1 in floors else -1;leg=geomleg[g2 if g1 in floors else g1];world=sign*z['contact_frame'][ci].T@local;legforce[leg]+=world
   contactrows.append(dict(sample=i,contact=ci,leg=leg,Fn=float(local[0]),Ft1=float(local[1]),Ft2=float(local[2]),mu1=float(mu[0]),mu2=float(mu[1]),pyramid_margin=margin,min_edge=float(e.min()),Fx=float(world[0]),Fy=float(world[1]),Fz=float(world[2])))
  for leg,force in legforce.items():legrows.append(dict(sample=i,leg=leg,Fx=float(force[0]),Fy=float(force[1]),Fz=float(force[2])))
 dump(p/'SUMMARY.json',dict(states=out,actual_calls=dict(model_load=0,forward=0,step=0,neural_ticks=0),inputs_unchanged=all(sha(f)==h for f,h in hashes.items()),limitations='interface topology bound only, not neural/FIR attainability nor native-soft standing'))
 assert all(sha(f)==h for f,h in hashes.items())
 for name,data in [('POOL_OUTPUTS.csv',poolrows),('ACTIVATION90.csv',actrows),('RESIDUAL48.csv',dofrows)]:table(p/name,data)
 if contactrows:table(p/'CONTACT_MARGINS.csv',contactrows)
 if legrows:table(p/'LEG_SUPPORT.csv',legrows)
 print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);main(pa.parse_args().output)
