"""S170 saved-only algebraic failure attribution; no optimization or engine imports."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,csv,hashlib,shutil
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path('/home/sensen/Desktop/engine修复验收');S169=BASE/'s169-near-initial-support/20261024T233428554259Z';S167=BASE/'s167-support-decomposition/20261024T231837641256212Z';S165=BASE/'s165-static-support-corrected/20261024T231400000000Z';S166=BASE/'s166-mapped-support/20261024T231418078692647Z';S154=BASE/'s154-activation-rate/20261024T215751775489Z/init80ms'
def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def contacts(z,edges,meta,state,label):
 geom=z['contact_geom'] if 'contact_geom' in z else z['contact_geom_ids'];floors=set(meta['floor_geom_ids']);gl={int(k):v.upper() for k,v in meta['geom_map'].items()};leg={k:np.zeros(3) for k in ['LF','LM','LH','RF','RM','RH']};rows=[];occ={}
 for ci,(g1,g2) in enumerate(geom):
  g1,g2=int(g1),int(g2);key=(g1,g2);occ[key]=occ.get(key,0)+1;addr=int(z['contact_efc_address'][ci]);dim=int(z['contact_dim'][ci]);count=int(z['contact_row_count'][ci]);local=np.zeros(3);margin=0.;mine=0.
  if addr>=0:
   assert dim==3 and count==4;e=edges[addr:addr+4];mu=z['contact_friction'][ci,:2];local=np.r_[e.sum(),mu[0]*(e[0]-e[1]),mu[1]*(e[2]-e[3])];margin=float(local[0]-sum(abs(local[1:]/mu)));mine=float(e.min());assert margin>=-1e-8 and mine>=-1e-8
  else:mu=z['contact_friction'][ci,:2]
  assert (g1 in floors)!=(g2 in floors);l=gl.get(g2 if g1 in floors else g1,'NONFOOT');world=(1 if g1 in floors else -1)*z['contact_frame'][ci].T@local
  if l in leg:leg[l]+=world
  for eidx in range(count):rows.append(dict(state=state,label=label,contact=ci,geom1=g1,geom2=g2,pair_occurrence=occ[key],matching_policy='row identity only within own saved state; not tracked material contact',edge=eidx,edge_force=float(edges[addr+eidx]) if addr>=0 else 0.,leg=l,distance=float(z['contact_dist'][ci]),mu1=float(mu[0]),mu2=float(mu[1]),Fn=float(local[0]),Ft1=float(local[1]),Ft2=float(local[2]),pyramid_margin=margin,min_edge=mine,Fx=float(world[0]),Fy=float(world[1]),Fz=float(world[2])))
 return rows,leg

def main(p):
 assert p.is_absolute() and not p.exists();p.mkdir(parents=True);freeze=p/'INPUT_FREEZE';freeze.mkdir()
 paths=[Path(__file__).resolve(),S169/'BEST_DIAGNOSTIC_NOT_CANDIDATE.npz',S169/'BEST_RESIDUAL48.csv',S169/'BEST_LEG_SUPPORT.csv',S169/'FROZEN_SEARCH.npz',S169/'S169_SAVED_INDEPENDENT_CHECK.json',S166/'MAPPING.npz',S154/'COM_CONTACT_METADATA.json',S167/'INDEPENDENT_CHECK.json']
 for i in [0,15,83]:paths += [S167/f'CASE_{i:03d}_fill.npz',S165/f'STATIC_AFFINE_{i:03d}.npz',S165/f'ZERO_V_{i:03d}.npz']
 hashes={str(f):sha(f) for f in paths};dump(p/'INPUT_HASHES.json',hashes)
 for j,f in enumerate(paths):shutil.copyfile(f,freeze/f'{j:03d}_{f.name}')
 dump(p/'PREREG.json',dict(scope='S170 P1/S2 exactly saved algebraic contrasts of S169best vs three S167fill witnesses',equation='rbest-rref=mapped+missing+geometry+contact',mapped='Bbest*(abest_mapped-aref_mapped)',missing='-Bbest*aref_missing',geometry='(Bbest-Bref)*aref+(basebest-baseref)',contact='Jbest.T*f0best-Jref.T*lambda_ref; last two belong pose/native-soft law class',accept='sum closure maxabs<=1e-8 all48; immutable native mixed-DOF quantities no cross-DOF physical norm dominance',causal_limits='chosen witness/path-dependent decomposition; no separate optimization, counterfactual newpose feasibility or interventions; cancellation and ordering prevent dominance inference',contact_comparison='own-state contact/edge rows retained with geom pair/occurrence, no cross-pose material point identification assumed',calls=dict(model=0,forward=0,step=0,neural=0,optimization=0)))
 b=load(S169/'BEST_DIAGNOSTIC_NOT_CANDIDATE.npz');ap=load(S166/'MAPPING.npz')['actuator_pool'];missing=ap<0;meta=json.loads((S154/'COM_CONTACT_METADATA.json').read_text());resbest=b['B']@b['act']+b['base']+b['J'].T@b['f0'];assert abs(resbest-b['residual48']).max()<1e-8
 saved=list(csv.DictReader((S169/'BEST_RESIDUAL48.csv').open()));assert abs(np.array([float(x['best']) for x in saved])-resbest).max()<1e-8
 cr,lb=contacts(b,b['f0'],meta,'BEST','S169 native-soft prescribed qacc0');contactrows=cr;legrows=[];dof=[];summ=[];series=[]
 for i in [0,15,83]:
  a=load(S165/f'STATIC_AFFINE_{i:03d}.npz');z=load(S165/f'ZERO_V_{i:03d}.npz');ref=load(S167/f'CASE_{i:03d}_fill.npz');ar=ref['activation'];n=ref['T'].shape[1];lam=ref['x'][n:];rr=a['B']@ar+a['base']+a['J'].T@lam;assert abs(rr).max()<1e-8
  mapped=b['B']@np.where(missing,0,b['act']-ar);miss=-b['B']@np.where(missing,ar,0);geo=(b['B']-a['B'])@ar+b['base']-a['base'];con=b['J'].T@b['f0']-a['J'].T@lam;closure=resbest-rr-mapped-miss-geo-con;assert abs(closure).max()<1e-8
  np.savez_compressed(p/f'DECOMPOSITION_{i:03d}.npz',mapped=mapped,missing=miss,geometry=geo,contact=con,best_residual=resbest,reference_residual=rr,closure=closure,reference_activation=ar,best_activation=b['act'])
  for j in range(48):dof.append(dict(reference_state=i,dof=j,best_residual=float(resbest[j]),reference_residual=float(rr[j]),mapped_activation_difference=float(mapped[j]),missing_zero_difference=float(miss[j]),pose_muscle_bias_passive_gravity_difference=float(geo[j]),native_soft_vs_rigid_contact_difference=float(con[j]),pose_contact_class_sum=float(geo[j]+con[j]),closure=float(closure[j])))
  rc,lr=contacts(z,lam,meta,i,'S167 rigid fill witness');contactrows+=rc
  for l in lb:legrows.append(dict(reference_state=i,leg=l,best_Fx=float(lb[l][0]),best_Fy=float(lb[l][1]),best_Fz=float(lb[l][2]),reference_Fx=float(lr[l][0]),reference_Fy=float(lr[l][1]),reference_Fz=float(lr[l][2]),delta_Fz=float(lb[l][2]-lr[l][2])))
  summ.append(dict(reference_state=i,closure_maxabs=float(abs(closure).max()),reference_maxabs=float(abs(rr).max()),best_maxabs=float(abs(resbest).max()),diagnostic_maxabs_by_term={k:float(abs(v).max()) for k,v in dict(mapped=mapped,missing=miss,geometry=geo,contact=con).items()},warning='component maxima in mixed native generalized forces are descriptive, NOT causal ranking'))
  series.append((i,mapped,miss,geo,con))
 table(p/'RESIDUAL_DECOMPOSITION48.csv',dof);table(p/'LEG_SUPPORT_DIFFERENCE.csv',legrows);table(p/'CONTACT_EDGES_GAP_FRICTION.csv',contactrows)
 gaps=[]
 for state,label,zz in [('BEST','native-soft',b)]+[(str(i),'rigid-reference-pose',load(S165/f'ZERO_V_{i:03d}.npz')) for i in [0,15,83]]:gaps.append(dict(state=state,label=label,min_distance=float(zz['contact_dist'].min()),max_distance=float(zz['contact_dist'].max()),contacts=len(zz['contact_dist']),rows=len(zz['efc_type']),interpretation='negative gap native compliance vs rigid relaxation; these laws differ'))
 dump(p/'GAP_FRICTION_SUMMARY.json',gaps)
 fig,axs=plt.subplots(3,2,figsize=(16,12))
 for row,(i,mapped,miss,geo,con) in enumerate(series):
  for label,arr in [('Mapped activation',mapped),('Missing zero',miss),('Pose force mechanics',geo),('Soft vs rigid contact',con),('Best residual',resbest)]:axs[row,0].plot(arr,label=label)
  axs[row,0].set_title(f'48DOF exact algebraic contrast vs fill state {i}');axs[row,0].legend(fontsize=7)
  rows=[x for x in legrows if x['reference_state']==i];x=np.arange(6);axs[row,1].bar(x-.18,[v['best_Fz'] for v in rows],width=.36,label='S169 best soft f0 (unbalanced)');axs[row,1].bar(x+.18,[v['reference_Fz'] for v in rows],width=.36,label=f'S167 rigid fill {i}');axs[row,1].set_xticks(x,[v['leg'] for v in rows]);axs[row,1].legend(fontsize=8)
 fig.suptitle('S170 saved algebraic contrasts: NOT causal dominance; no new optimization or simulation');fig.tight_layout();fig.savefig(p/'FAILURE_ATTRIBUTION_COMPARISON.png',dpi=150)
 assert all(sha(f)==h for f,h in hashes.items());dump(p/'SUMMARY.json',dict(classification='两者无法由现有保存数据区分',contrasts=summ,inputs_unchanged=True,calls=dict(model_load=0,forward=0,step=0,neural_ticks=0,optimization=0),why='different selected activations/poses and native-soft vs free rigid forces change simultaneously; decomposition is exact but order/witness dependent; no causal dominance certificate'))
 print(json.dumps(summ))
if __name__=='__main__':
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);main(pa.parse_args().output)
