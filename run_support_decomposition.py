"""S167 six saved-only constraint decomposition LPs. No engine/neural imports."""
from pathlib import Path
import argparse,json,csv,hashlib,shutil
import numpy as np
from scipy.optimize import linprog
BASE=Path('/home/sensen/Desktop/engine修复验收');S165=BASE/'s165-static-support-corrected/20261024T231400000000Z';S166=BASE/'s166-mapped-support/20261024T231418078692647Z';S154=BASE/'s154-activation-rate/20261024T215751775489Z/init80ms'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main(p):
 assert p.is_absolute() and not p.exists();p.mkdir(parents=True);freeze=p/'INPUT_FREEZE';freeze.mkdir()
 files=[Path(__file__).resolve(),S166/'MAPPING.npz',S166/'FULL_ACTUATOR_MAPPING.csv',S166/'INDEPENDENT_CHECK.json',S166/'CERTIFICATES.npz',S165/'AFFINITY_PROOF.json',S154/'COM_CONTACT_METADATA.json']
 for i in [0,15,83]:files += [S165/f'{s}_{i:03d}.npz' for s in ['STATIC_AFFINE','ZERO_V']]+[S165/f'rigid_upper_{i:03d}_upper0.5.npz',S166/f'STATE_{i:03d}.npz']
 hashes={str(f):sha(f) for f in files};dump(p/'INPUT_HASHES.json',hashes)
 for j,f in enumerate(files):shutil.copyfile(f,freeze/f'{j:03d}_{f.name}')
 dump(p/'PREREG.json',dict(scope='S167 P1/S2 exactly six offline rigid-upper-bound decomposition LPs',states=[0,15,83],conditions=dict(unbind='16missing zero;74 mapped muscles independent activation[0,.5]',fill='66real shared pools retained;16missing independently allowed[0,.5] WITHOUT neural source claim'),equation='[B*T,J.T] [v,lambda]=-base; finite v[0,1] normalized act via .5*T selector,72lambda>=0',activation='ctrl=act constant rate0 offline proof, no solution application',tolerance=1e-8,objective='min sum muscle activation diagnostic only',comparators='S166 both restricted;S165 independent90[0,.5] both relaxed, reuse existing witnesses and certificates only',limitations='rigid relaxed negativegap/compliance bound notnative-soft equilibrium or neural/FIR attainability; no new connections',calls=dict(model_load=0,forward=0,step=0,neural_ticks=0),units='native; SI NOT_ESTABLISHED'))
 mapping=load(S166/'MAPPING.npz');ap=mapping['actuator_pool'];mapped=np.flatnonzero(ap>=0);missing=np.flatnonzero(ap<0);assert len(mapped)==74 and len(missing)==16
 unbind=np.zeros((90,74));unbind[mapped,np.arange(74)]=.5
 fill=np.column_stack((mapping['T'],np.zeros((90,16))));fill[missing,66+np.arange(16)]=.5
 aff=json.loads((S165/'AFFINITY_PROOF.json').read_text());dyn=np.array(aff['dynprm']);ctrlrange=np.array(aff['ctrlrange']);limited=np.array(aff['ctrllimited'],bool)
 meta=json.loads((S154/'COM_CONTACT_METADATA.json').read_text());floors=set(meta['floor_geom_ids']);geomleg={int(k):v.upper() for k,v in meta['geom_map'].items()};muscles=list(csv.DictReader((S166/'FULL_ACTUATOR_MAPPING.csv').open()))
 results=[];actrows=[];legrows=[];contactrows=[];dofrows=[];varrows=[]
 for i in [0,15,83]:
  a=load(S165/f'STATIC_AFFINE_{i:03d}.npz');z=load(S165/f'ZERO_V_{i:03d}.npz');assert a['J'].shape==(72,48) and np.all(z['efc_type']==6) and np.all(z['contact_row_count']==4)
  for condition,T in [('unbind',unbind),('fill',fill)]:
   n=T.shape[1];A=np.column_stack((a['B']@T,a['J'].T));b=-a['base'];bounds=np.column_stack((np.zeros(n+72),np.r_[np.ones(n),np.full(72,np.inf)]));scale=np.maximum(abs(A).max(1),1e-12)
   sol=linprog(np.r_[T.sum(0),np.zeros(72)],A_eq=A/scale[:,None],b_eq=b/scale,bounds=[(0,1)]*n+[(0,None)]*72,method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
   x=np.full(n+72,np.nan) if sol.x is None else sol.x;act=T@x[:n];res=A@x-b;feasible=bool(np.isfinite(x).all() and abs(res).max()<=1e-8 and x.min()>=-1e-8 and x[:n].max()<=1+1e-8)
   np.savez_compressed(p/f'CASE_{i:03d}_{condition}.npz',T=T,Aeq=A,beq=b,bounds=bounds,x=x,activation=act,residual=res)
   info=dict(sample=i,condition=condition,solver_status=int(sol.status),message=sol.message,feasible=feasible,max_residual=None if not feasible else float(abs(res).max()));results.append(info)
   for j,r in enumerate(muscles):actrows.append(dict(sample=i,condition=condition,actuator=j,name=r['actuator'],leg=r['leg'],pool=int(ap[j]),actual_activation=float(a['actual_activation'][j]),solution_activation=None if not feasible else float(act[j])))
   for j in range(n):varrows.append(dict(sample=i,condition=condition,variable=j,normalized_value=None if not feasible else float(x[j])))
   for j in range(48):dofrows.append(dict(sample=i,condition=condition,dof=j,residual=None if not feasible else float(res[j])))
   if not feasible:continue
   clamped=np.where(limited,np.clip(act,ctrlrange[:,0],ctrlrange[:,1]),act);clamped=np.clip(clamped,0,1);rate=(clamped-act)/np.where(clamped>act,dyn[:,0]*(.5+1.5*act),dyn[:,1]/(.5+1.5*act));assert np.array_equal(rate,np.zeros(90));np.savez_compressed(p/f'CONTROL_{i:03d}_{condition}_NOT_APPLIED.npz',ctrl=act,activation=act,clamped_ctrl=clamped,offline_act_dot=rate)
   forces={leg:np.zeros(3) for leg in ['LF','LM','LH','RF','RM','RH']};minmargin=np.inf
   for ci,addr in enumerate(z['contact_efc_address']):
    e=x[n+addr:n+addr+4];mu=z['contact_friction'][ci,:2];local=np.r_[sum(e),mu[0]*(e[0]-e[1]),mu[1]*(e[2]-e[3])];margin=float(local[0]-sum(abs(local[1:]/mu)));assert margin>=-1e-8;minmargin=min(minmargin,margin)
    g1,g2=map(int,z['contact_geom_ids'][ci]);assert (g1 in floors)!=(g2 in floors);leg=geomleg[g2 if g1 in floors else g1];world=(1 if g1 in floors else -1)*z['contact_frame'][ci].T@local;forces[leg]+=world
    contactrows.append(dict(sample=i,condition=condition,contact=ci,leg=leg,Fn=float(local[0]),Ft1=float(local[1]),Ft2=float(local[2]),mu1=float(mu[0]),mu2=float(mu[1]),min_edge=float(e.min()),pyramid_margin=margin,Fx=float(world[0]),Fy=float(world[1]),Fz=float(world[2])))
   for leg,force in forces.items():legrows.append(dict(sample=i,condition=condition,leg=leg,Fx=float(force[0]),Fy=float(force[1]),Fz=float(force[2])))
   info.update(activation_sum=float(act.sum()),activation_max=float(act.max()),min_friction_margin=minmargin,act_dot_max=0.)
 for name,rows in [('ACTIVATIONS.csv',actrows),('VARIABLES.csv',varrows),('RESIDUAL48.csv',dofrows),('LEG_SUPPORT.csv',legrows),('CONTACT_MARGINS.csv',contactrows)]:
  if rows:table(p/name,rows)
 assert all(sha(f)==h for f,h in hashes.items());dump(p/'SUMMARY.json',dict(results=results,inputs_unchanged=True,calls=dict(model_load=0,forward=0,step=0,neural_ticks=0)));print(json.dumps(results))
if __name__=='__main__':
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);main(pa.parse_args().output)
