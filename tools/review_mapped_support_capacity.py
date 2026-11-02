"""Read-only constraint and primal witness audit; no solver/physics calls."""
from pathlib import Path
import json,math,hashlib
import numpy as np
ROOT=Path('/home/sensen/Desktop/engine修复验收')
S=ROOT/'s2-mapped-support-capacity/20261102T170741Z'
B=ROOT/'s5-banc888-physical-handover/20261101T220018Z'
O=ROOT/'s2-mapped-capacity-parent-review/20261102T170741Z'
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=False)
 paths=[Path(__file__),S/'PREREGISTRATION.json',S/'RESULT.json',S/'POSE_CONTACTS.json',S/'GROUP_MASKS.npz',S/'AUTHORITATIVE_MAPPING.json',S/'COMPILED_MODEL.npz']+[S/('POSE_'+str(i)+'.npz') for i in range(3)]+[S/('LP_'+str(i)+'_'+arm+'.npz') for i in range(3) for arm in ['mapped66','independent74','independent90']];hashes={str(p):sha(p) for p in paths}
 dump(O/'PREREGISTRATION.json',dict(scope='Independent constraints/primalwitness arithmetic only; no solvercalled and no secondphysicsrun',source_sha256=hashes,limits='Infeasiblesolverstatus cannotbecertifiedasphysicalimpossibility; omittedjointlimitreactions andsoftcontactforceindependenceexplicit.'))
 result=json.loads((S/'RESULT.json').read_text());mapping=json.loads((B/'RUN/MAPPING.json').read_text())['rows'];g=arrays(S/'GROUP_MASKS.npz');roots=arrays(B/'IDENTITIES.npz')['root_ids'];groups={};pg=[]
 for row in mapping:
  key=tuple(row['local_indices']);assert np.array_equal(roots[list(key)],row['tier_root_ids'])
  if key and key not in groups:groups[key]=len(groups)
  pg.append(groups[key] if key else -1)
 assert np.array_equal(pg,g['pool_group']) and len(groups)==66 and pg.count(-1)==16
 expectedG=np.zeros((90,66))
 for u,pool in enumerate(pg):
  if pool>=0:expectedG[u,pool]=1.
 assert np.array_equal(g['G'],expectedG) and np.array_equal(g['P74'],np.eye(90)[:,np.array(pg)>=0]) and np.array_equal(g['P90'],np.eye(90))
 meta=json.loads((S/'POSE_CONTACTS.json').read_text());model=arrays(S/'COMPILED_MODEL.npz');old=arrays(B/'RUN/BODY.npz');slow=arrays(ROOT/'s5-slow-reference-handover/20261102T165227Z/SLOW_2P5HZ_PHASE_MATCHED/RUN/BODY.npz');summaries=[];maxA=maxbase=maxeq=maxjac=maxgain=0.;witnesses=0
 for pose,source,index in [(0,old,0),(1,old,2600),(2,slow,2500)]:
  a=arrays(S/('POSE_'+str(pose)+'.npz'));assert np.array_equal(a['qpos'],source['qpos'][index]) and np.array_equal(a['qvel_actual'],source['qvel'][index]) and not a['static_velocity'].any();assert abs(a['actual_force']-source['muscle_force'][index]).max()<1e-10
  A=np.array([[float(a['moment'][u,j])*float(a['gain'][u]) for u in range(90)] for j in range(48)]);base=np.array([float(a['qfrc_passive'][j])-float(a['qfrc_bias'][j])+math.fsum(float(a['moment'][u,j])*float(a['bias'][u]) for u in range(90)) for j in range(48)]);maxA=max(maxA,float(abs(A-a['A90']).max()));maxbase=max(maxbase,float(abs(base-a['base']).max()))
  for u in range(90):
   p=model['actuator_gainprm'][u];r0,r1,F,scale,lo,hi,vmax,fp,fvmax=map(float,p[:9]);lr=model['actuator_lengthrange'][u];L0=(lr[1]-lr[0])/(r1-r0);L=r0+(a['length'][u]-lr[0])/L0;left=(lo+1)/2;right=(hi+1)/2
   if L<lo or L>hi:fl=0.
   elif L<=left:fl=.5*((L-lo)/(left-lo))**2
   elif L<=1:fl=1-.5*((1-L)/(1-left))**2
   elif L<=right:fl=1-.5*((L-1)/(right-1))**2
   else:fl=.5*((hi-L)/(hi-right))**2
   F=scale/model['actuator_acc0'][u] if F<0 else F;maxgain=max(maxgain,abs(-F*fl-a['gain'][u]))
  nc=len(a['J']);assert nc==len(meta[pose]['contacts']) and not meta[pose]['excluded_nonfoot_floor']
  for c,contact in enumerate(meta[pose]['contacts']):
   assert 'Tars' in contact['geom'] and contact['condim']==3 and contact['mu']==a['mu'][c] and np.array_equal(contact['position'],a['positions'][c]);maxjac=max(maxjac,float(abs(a['J'][c,:,:3]-np.eye(3)).max()))
   for j in range(3):maxjac=max(maxjac,float(abs(np.cross(a['root_angular'][c,j],a['positions'][c]-a['root_position'][c])-a['J'][c,:,j+3]).max()))
  arms=[]
  for arm,P in [('mapped66',expectedG),('independent74',g['P74']),('independent90',g['P90'])]:
   lp=arrays(S/('LP_'+str(pose)+'_'+arm+'.npz'));n=P.shape[1];assert np.array_equal(P,lp['P']);eq=np.zeros_like(lp['A_eq'])
   for j in range(48):
    for col in range(n):eq[j,col]=math.fsum(float(A[j,u])*float(P[u,col]) for u in range(90))
    for c in range(nc):eq[j,n+3*c:n+3*c+3]=a['J'][c,:,j]
   rhs=-base-np.array([math.fsum(map(float,A[j]))*1e-4 for j in range(48)]);maxeq=max(maxeq,float(abs(eq-lp['A_eq']).max()),float(abs(rhs-lp['b_eq']).max()));assert maxeq<1e-10
   friction=np.zeros_like(lp['A_ub'])
   for c in range(nc):
    for k,(sx,sy) in enumerate([(1,1),(1,-1),(-1,1),(-1,-1)]):friction[c*4+k,n+3*c:n+3*c+3]=[sx,sy,-float(a['mu'][c])]
   assert np.array_equal(friction,lp['A_ub']) and not lp['b_ub'].any();assert np.array_equal(lp['row_scale'],np.maximum(1,np.maximum(abs(lp['A_eq']).max(1),abs(lp['b_eq']))))
   bounds=np.array([[0.,.9999]]*n+[[-np.inf,np.inf],[-np.inf,np.inf],[0.,np.inf]]*nc);assert np.array_equal(bounds,lp['bounds'])
   row=next(r for r in result['rows'] if r['pose']==pose and r['arm']==arm);arms.append(dict(arm=arm,reported_status=row['status'],reported_witness=row['witness']))
   if row['witness']:
    sol=arrays(S/('SOLUTION_'+str(pose)+'_'+arm+'.npz'));x=sol['x'];res=eq@x-rhs;assert abs(res).max()<=1e-8 and np.max(friction@x,initial=0)<=1e-8 and np.all(x>=bounds[:,0]-1e-10) and np.all(x<=bounds[:,1]+1e-10);witnesses+=1
   else:assert row['status']==2 and not (S/('SOLUTION_'+str(pose)+'_'+arm+'.npz')).exists()
  summaries.append(dict(pose=pose,label=meta[pose]['label'],footcontact_candidates=nc,active_jointlimit_rows_omitted=len(a['active_joint_limit_efc_rows']),arms=arms))
 assert max(maxA,maxbase,maxeq,maxjac,maxgain)<1e-9
 out=dict(passed=True,source_sha256=hashes,group66_74_90_exact=True,unresolved16_sourceverified=True,scalar_errors=dict(muscle_matrix=maxA,passive_bias_base=maxbase,equilibrium=maxeq,root_contact_Jacobian=maxjac,stationary_gain=maxgain),primal_witnesses_verified=witnesses,reported_infeasible_cases=sum(r['status']==2 for r in result['rows']),infeasibility_dual_certificates_verified=0,poses=summaries,calls=dict(model_compile=0,forward=0,body_step=0,neural_step=0,linprog=0),limits='Allreportedstatuses areoriginalsolverresults,not independentinfeasibilityproofs. Nojointlimitreactioncols; independentlyrelaxedsoftcontacts andinnerfrictioncone. Cannotcallthisfullmechanicalimpossibility or neuralcontrolsuccess.')
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'RESULT.json',out);print(json.dumps({k:v for k,v in out.items() if k!='source_sha256'},indent=2))
if __name__=='__main__':main()
