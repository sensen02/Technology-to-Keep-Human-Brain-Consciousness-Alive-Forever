"""Independent nativeconstraint/LP/primal/dual arithmetic audit; no model or solver."""
from pathlib import Path
import json,math,hashlib
import numpy as np
ROOT=Path('/home/sensen/Desktop/engine修复验收')
S=ROOT/'s2-native-support-constraints/20261102T172633Z'
P=ROOT/'s2-mapped-support-capacity/20261102T170741Z'
O=ROOT/'s2-native-constraints-parent-review/20261102T172633Z'
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=False)
 paths=[Path(__file__),S/'MODEL_SCHEMA.json',S/'CAPTURE_RESULT.json',S/'RESULT.json',S/'PREREGISTRATION.json']+[S/('NATIVE_'+str(i)+'.npz') for i in range(3)]+[S/('REACTIONS_'+str(i)+'.npz') for i in range(3)]+[S/('LP_'+str(i)+'_'+arm+'_'+mode+'.npz') for i in range(3) for arm in ['mapped66','independent74','independent90'] for mode in ['native','outerbox']];paths += list(S.glob('FARKAS_*.npz'));hashes={str(p):sha(p) for p in paths};dump(O/'PREREGISTRATION.json',dict(scope='Independent sparseJ/rays/limits/18eq/bounds/primal/dual scalarcheck;0solver/model',source_sha256=hashes,limits='Only conditional savedpose matrixfloating evidence,not global or biological proof.'))
 schema=json.loads((S/'MODEL_SCHEMA.json').read_text());result=json.loads((S/'RESULT.json').read_text());capture=json.loads((S/'CAPTURE_RESULT.json').read_text());assert schema['cone']==0 and schema['neq']==0 and schema['compiled_arrays_exact']==469 and capture['passed']
 maxJ=maxE=maxrhs=maxproject=maxdual=0.;poses=[];certs=[];primal=[]
 for i in range(3):
  a=arrays(S/('NATIVE_'+str(i)+'.npz'));old=arrays(P/('POSE_'+str(i)+'.npz'));reactions=arrays(S/('REACTIONS_'+str(i)+'.npz'));contacts=json.loads((S/('CONTACTS_'+str(i)+'.json')).read_text());J=np.zeros_like(a['J'])
  if schema['sparse']:
   for row in range(len(J)):
    off=int(a['efc_J_rowadr'][row]);count=int(a['efc_J_rownnz'][row]);J[row,a['efc_J_colind'][off:off+count]]=a['efc_J_raw'][off:off+count]
  else:J[:]=a['efc_J_raw'].reshape(J.shape)
  assert np.array_equal(J,a['J']) and np.array_equal(a['qpos'],old['qpos']) and not a['qvel'].any() and not a['qfrc_applied'].any() and not a['xfrc_applied'].any()
  for key,oldkey in [('moment','moment'),('length','length'),('velocity','static_velocity'),('qfrc_passive','qfrc_passive'),('qfrc_bias','qfrc_bias')]:assert np.array_equal(a[key],old[oldkey]),key
  projected=np.array([math.fsum(float(J[r,j])*float(a['efc_force'][r]) for r in range(len(J))) for j in range(48)]);maxproject=max(maxproject,float(abs(projected-a['qfrc_constraint']).max()))
  foot=[];covered=set();internal=[];ibounds=[];limitdetails=[];footindex=0
  for co in contacts:
   rows=co['rows'];covered.update(rows);assert co['foot_floor'] and not co['selfcontact'] and co['dim']==3
   assert len(rows)==4 and len(rows)==2*(co['dim']-1);foot.append(J[rows].T)
   sign=1. if co['name1']=='floor' else -1.;local=sign*np.array(co['frame'])@old['J'][footindex];expected=np.array([local[0]+float(co['friction'][t])*sg*local[t+1] for t in range(2) for sg in [1,-1]]);maxJ=max(maxJ,float(abs(expected-J[rows]).max()));footindex+=1
  model=arrays(S/'COMPILED_MODEL.npz')
  for row,typ in enumerate(a['efc_type']):
   if row in covered:continue
   assert int(typ)==3,'Actual cases contain onlyfoot pyramids andone jointlimit';joint=int(a['efc_id'][row]);dof=int(model['jnt_dofadr'][joint]);q=float(a['qpos'][int(model['jnt_qposadr'][joint])]);lo,hi=model['jnt_range'][joint];side='lower' if abs(q-lo)<=abs(q-hi) else 'upper';expected=np.zeros(48);expected[dof]=1 if side=='lower' else -1;assert np.array_equal(J[row],expected);internal.append(J[row]);ibounds.append([0.,np.inf]);covered.add(row);limitdetails.append(dict(row=row,jointid=joint,dof=dof,side=side,q=q,range=[float(lo),float(hi)],native_direction=float(J[row,dof])))
  assert covered==set(range(len(J)))
  nativefoot=np.column_stack(foot);inner=np.array(internal).reshape(-1,48).T;ib=np.array(ibounds).reshape(-1,2);assert np.array_equal(nativefoot,reactions['native_foot_columns']) and np.array_equal(inner,reactions['internal_columns']) and np.array_equal(ib,reactions['internal_bounds'])
  for arm in ['mapped66','independent74','independent90']:
   previous=arrays(P/('LP_'+str(i)+'_'+arm+'.npz'));n=previous['P'].shape[1]
   for mode in ['native','outerbox']:
    lp=arrays(S/('LP_'+str(i)+'_'+arm+'_'+mode+'.npz'));contact=nativefoot if mode=='native' else previous['A_eq'][:,n:];E=np.column_stack([previous['A_eq'][:,:n],contact,inner]);bounds=np.concatenate([previous['bounds'][:n],np.array([[0.,np.inf]]*contact.shape[1]) if mode=='native' else previous['bounds'][n:],ib]);assert np.array_equal(E,lp['E']) and np.array_equal(lp['b'],previous['b_eq']) and np.array_equal(bounds,lp['bounds']) and np.array_equal(lp['P'],previous['P'])
    U=np.zeros_like(lp['U'])
    if mode=='outerbox':
     for c,mu in enumerate(old['mu']):
      for k,(sx,sy) in enumerate([(1,0),(-1,0),(0,1),(0,-1)]):U[c*4+k,n+3*c:n+3*c+3]=[sx,sy,-float(mu)]
    assert np.array_equal(U,lp['U']) and not lp['v'].any();assert np.array_equal(lp['row_scale'],np.maximum(1,np.maximum(abs(E).max(1),abs(lp['b']))))
    outcome=next(r for r in result['rows'] if r['pose']==i and r['arm']==arm and r['contact_model']==mode)
    if outcome['witness']:
     sol=arrays(S/('WITNESS_'+str(i)+'_'+arm+'_'+mode+'.npz'));x=sol['x'];res=np.array([math.fsum(float(E[j,k])*float(x[k]) for k in range(len(x)))-float(lp['b'][j]) for j in range(48)]);assert abs(res).max()<=1e-8 and np.max(U@x-lp['v'],initial=0)<=1e-8 and np.all(x>=bounds[:,0]-1e-10) and np.all(x<=bounds[:,1]+1e-10);primal.append(dict(pose=i,arm=arm,mode=mode,eqmax=float(abs(res).max())))
  poses.append(dict(pose=i,nefc=len(J),foot_candidates=len(contacts),native_foot_reaction_columns=nativefoot.shape[1],internal_columns=inner.shape[1],limits=limitdetails))
  path=S/('FARKAS_'+str(i)+'.npz')
  if path.exists():
   c=arrays(path);lp=arrays(S/('LP_'+str(i)+'_independent90_outerbox.npz'));H=list(lp['U']);h=list(lp['v']);labels=['friction:'+str(k) for k in range(len(H))]
   for k,(lo,hi) in enumerate(lp['bounds']):
    if np.isfinite(lo):v=np.zeros(lp['E'].shape[1]);v[k]=-1;H.append(v);h.append(-float(lo));labels.append('lower:'+str(k))
    if np.isfinite(hi):v=np.zeros(lp['E'].shape[1]);v[k]=1;H.append(v);h.append(float(hi));labels.append('upper:'+str(k))
   H=np.array(H);h=np.array(h);assert np.array_equal(H,c['H']) and np.array_equal(h,c['h']) and np.array_equal(lp['E'],c['E']) and np.array_equal(lp['b'],c['b']) and labels==list(c['H_labels'])
   if 'y' in c:
    y,z=c['y'],c['z'];assert np.all(z>=-1e-10) and np.max(abs(y*c['eq_scale']-c['y_scaled']))<=1e-12 and np.max(abs(z*c['ineq_scale']-c['z_scaled']))<=1e-12
    st=np.array([math.fsum([float(c['E'][r,k])*float(y[r]) for r in range(48)]+[float(H[r,k])*float(z[r]) for r in range(len(H))]) for k in range(H.shape[1])]);margin=math.fsum([float(c['b'][r])*float(y[r]) for r in range(48)]+[float(h[r])*float(z[r]) for r in range(len(h))]);maxdual=max(maxdual,float(abs(st).max()));assert abs(st).max()<=1e-8 and margin<=-1+1e-8
    denominator=np.abs(c['E']).T@np.abs(y)+np.abs(H).T@np.abs(z);certs.append(dict(pose=i,raw_stationarity_max=float(abs(st).max()),margin=margin,dual_min=float(np.min(z,initial=0)),relative_stationarity=float(np.max(abs(st)/np.maximum(1,denominator))),equality_dual_top_dofs=[dict(dof=int(k),value=float(y[k])) for k in np.argsort(abs(y))[::-1][:8]],raw_y_max=float(abs(y).max()),raw_z_max=float(abs(z).max())))
 assert max(maxJ,maxproject,maxdual)<1e-8
 out=dict(passed=True,source_sha256=hashes,native_sparse_schema_rebuilt=True,all18_constraints_andbounds_exact=True,old_muscle_RHS_reused_exact=True,poses=poses,actual_frame_worldJ_maxerror=maxJ,actual_constraintforce_projection_maxerror=maxproject,primal_witnesses=primal,floating_outer90_certificates=certs,original_solver_statuses=[dict(pose=r['pose'],arm=r['arm'],mode=r['contact_model'],status=r['status'],witness=r['witness']) for r in result['rows']],calls=dict(model_compile=0,forward=0,body_step=0,neural_step=0,linprog=0),limits='Finiteprecisiondualseparation forfixedq/contact/actbounds only. No globalmechanicalimpossibility,softsolver/complementarity proof or neuralcontrolsuccess. Allactiveconstrainttypes inactualcases accounted.')
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'RESULT.json',out);print(json.dumps({k:v for k,v in out.items() if k not in ['source_sha256','original_solver_statuses']},indent=2))
if __name__=='__main__':main()
