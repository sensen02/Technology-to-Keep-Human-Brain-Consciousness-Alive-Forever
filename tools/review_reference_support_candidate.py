"""Independent reference candidate selection and witness verification; no LP or physics."""
from pathlib import Path
import json,math,hashlib
import numpy as np
ROOT=Path('/home/sensen/Desktop/engine修复验收')
S=ROOT/'s2-reference-support-candidate/20261102T182109Z'
B=ROOT/'s5-banc888-physical-handover/20261101T220018Z/RUN'
O=ROOT/'s2-reference-candidate-parent-review/20261102T182109Z'
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
 paths=[Path(__file__),S/'PREREGISTRATION.json',S/'SOURCE_FROZEN.npz',S/'RESULT.json',S/'CAPTURE_RESULT.json',S/'COMPILED_MODEL.npz',S/'GROUP_MASKS.npz']+sorted(S.glob('NATIVE_*.npz'))+sorted(S.glob('MUSCLE_*.npz'))+sorted(S.glob('REACTIONS_*.npz'))+sorted(S.glob('LP_*.npz'));hashes={str(p):sha(p) for p in paths};dump(O/'PREREGISTRATION.json',dict(scope='Independent frozen33poses/nativeconstraints/LP/witness/selection review,0solve/modelintegration',source_sha256=hashes))
 result=json.loads((S/'RESULT.json').read_text());capture=json.loads((S/'CAPTURE_RESULT.json').read_text());assert capture['passed']
 source=arrays(B/'BODY.npz');frozen=arrays(S/'SOURCE_FROZEN.npz');indices=np.rint(np.linspace(2200,2600,33)).astype(int);assert np.array_equal(indices,frozen['source_indices'])
 for key in source:assert np.array_equal(frozen[key],source[key][indices]),key
 group=arrays(S/'GROUP_MASKS.npz');G=group['G'];assert G.shape==(90,66) and int((group['pool_group']<0).sum())==16
 originalgroup=arrays(ROOT/'s2-mapped-support-capacity/20261102T170741Z/GROUP_MASKS.npz');assert set(group)==set(originalgroup) and all(np.array_equal(group[k],originalgroup[k]) for k in group)
 model=arrays(S/'COMPILED_MODEL.npz');cached=ROOT/'s2-native-support-constraints/20261102T172633Z';oldmodel=arrays(cached/'COMPILED_MODEL.npz');assert len(model)==469 and set(model)==set(oldmodel) and all(np.array_equal(model[k],oldmodel[k],equal_nan=True) for k in model)
 counters=dict(newposes=0,cachedposes=0,executedLP=0,cachedLP=0,skippedLP=0,primal_witnesses=0);maxA=maxbase=maxeq=maxJ=maxproj=maxgain=0.;summaries=[];feasible66=[];feasible90=[]
 for pose,index in enumerate(indices):
  a=arrays(S/('NATIVE_'+str(pose)+'.npz'));muscle=arrays(S/('MUSCLE_'+str(pose)+'.npz'));reaction=arrays(S/('REACTIONS_'+str(pose)+'.npz'));contacts=json.loads((S/('CONTACTS_'+str(pose)+'.json')).read_text());assert np.array_equal(a['qpos'],frozen['qpos'][pose]) and not a['qvel'].any() and not a['qfrc_applied'].any() and not a['xfrc_applied'].any()
  A=a['moment'].T*muscle['gain'];base=np.array([float(a['qfrc_passive'][j])-float(a['qfrc_bias'][j])+math.fsum(float(a['moment'][u,j])*float(muscle['bias'][u]) for u in range(90)) for j in range(48)]);maxA=max(maxA,float(abs(A-muscle['A90']).max()));maxbase=max(maxbase,float(abs(base-muscle['base']).max()))
  for u in range(90):
   p=model['actuator_gainprm'][u];r0,r1,F,scale,lo,hi,vmax,fp,fvmax=map(float,p[:9]);lr=model['actuator_lengthrange'][u];L0=(lr[1]-lr[0])/(r1-r0);L=r0+(a['length'][u]-lr[0])/L0;left=(lo+1)/2;right=(hi+1)/2
   if L<lo or L>hi:fl=0.
   elif L<=left:fl=.5*((L-lo)/(left-lo))**2
   elif L<=1:fl=1-.5*((1-L)/(1-left))**2
   elif L<=right:fl=1-.5*((L-1)/(right-1))**2
   else:fl=.5*((hi-L)/(hi-right))**2
   F=scale/model['actuator_acc0'][u] if F<0 else F;maxgain=max(maxgain,abs(-F*fl-muscle['gain'][u]))
  J=a['J'];native=[];covered=set();internal=[];ibounds=[];foot=0
  for co in contacts:
   assert co['foot_floor'] and not co['selfcontact'] and co['dim']==3;rows=co['rows'];covered.update(rows);assert len(rows)==4;native.append(J[rows].T)
   sign=1. if co['name1']=='floor' else -1.;local=sign*np.asarray(co['frame'])@muscle['J'][foot];expected=np.array([local[0]+float(co['friction'][t])*sg*local[t+1] for t in range(2) for sg in [1,-1]]);maxJ=max(maxJ,float(abs(expected-J[rows]).max()));foot+=1
  for row,typ in enumerate(a['efc_type']):
   if row in covered:continue
   assert int(typ)==3;ji=int(a['efc_id'][row]);qa=int(model['jnt_qposadr'][ji]);dof=int(model['jnt_dofadr'][ji]);q=a['qpos'][qa];lo,hi=model['jnt_range'][ji];direction=1 if abs(q-lo)<=abs(q-hi) else -1;expected=np.zeros(48);expected[dof]=direction;assert np.array_equal(J[row],expected);internal.append(J[row]);ibounds.append([0.,np.inf]);covered.add(row)
  assert covered==set(range(len(J)))
  nf=np.column_stack(native) if native else np.zeros((48,0));ic=np.array(internal).reshape(-1,48).T;ib=np.array(ibounds).reshape(-1,2);assert np.array_equal(nf,reaction['native_foot_columns']) and np.array_equal(ic,reaction['internal_columns']) and np.array_equal(ib,reaction['internal_bounds'])
  projection=np.array([math.fsum(float(J[r,j])*float(a['efc_force'][r]) for r in range(len(J))) for j in range(48)]);maxproj=max(maxproj,float(abs(projection-a['qfrc_constraint']).max()))
  if pose==32:
   counters['cachedposes']+=1
   for filename in ['NATIVE','REACTIONS']:
    actual=arrays(S/(filename+'_32.npz'));original=arrays(cached/(filename+'_1.npz'));assert set(actual)==set(original) and all(np.array_equal(actual[k],original[k],equal_nan=True) for k in actual)
  else:counters['newposes']+=1
  outer_ok=False
  for arm,mode in [('independent90','outerbox'),('independent90','native'),('mapped66','native')]:
   row=next(r for r in result['rows'] if r['pose']==pose and r['arm']==arm and r['contact_model']==mode)
   lp=arrays(S/('LP_'+str(pose)+'_'+arm+'_'+mode+'.npz'));P=np.eye(90) if arm=='independent90' else G;n=P.shape[1];ct=nf if mode=='native' else muscle['J'].transpose(2,0,1).reshape(48,-1);E=np.column_stack([A@P,ct,ic]);b=-base-np.array([math.fsum(map(float,A[j]))*1e-4 for j in range(48)]);maxeq=max(maxeq,float(abs(E-lp['E']).max()),float(abs(b-lp['b']).max()));assert maxeq<1e-10
   bounds=np.concatenate([np.array([[0.,.9999]]*n),np.array([[0.,np.inf]]*ct.shape[1]) if mode=='native' else np.array([[-np.inf,np.inf],[-np.inf,np.inf],[0.,np.inf]]*foot).reshape(-1,2),ib]);assert np.array_equal(bounds,lp['bounds']) and np.array_equal(P,lp['P'])
   U=np.zeros_like(lp['U'])
   if mode=='outerbox':
    for c,mu in enumerate(muscle['mu']):
     for k,(sx,sy) in enumerate([(1,0),(-1,0),(0,1),(0,-1)]):U[4*c+k,n+3*c:n+3*c+3]=[sx,sy,-float(mu)]
   assert np.array_equal(U,lp['U']) and not lp['v'].any()
   if row['execution']=='skipped':assert mode=='native' and not outer_ok and not row['witness'] and row['status'] is None;counters['skippedLP']+=1;continue
   if row['execution']=='cached':
    prior=next(r for r in json.loads((cached/'RESULT.json').read_text())['rows'] if r['pose']==1 and r['arm']==arm and r['contact_model']==mode);assert pose==32 and row['status']==prior['status'] and row['witness']==prior['witness'];counters['cachedLP']+=1;continue
   counters['executedLP']+=1
   if row['witness']:
    sol=arrays(S/('WITNESS_'+str(pose)+'_'+arm+'_'+mode+'.npz'));x=sol['x'];res=E@x-b;assert abs(res).max()<=1e-8 and np.max(U@x,initial=0)<=1e-8 and np.all(x>=bounds[:,0]-1e-10) and np.all(x<=bounds[:,1]+1e-10);assert np.array_equal(sol['activation90'],1e-4+P@x[:n]);counters['primal_witnesses']+=1
    if mode=='native':(feasible66 if arm=='mapped66' else feasible90).append(pose)
   else:assert row['status'] in [0,2,4] # raw numericalgate fail is not a witness
   if mode=='outerbox':outer_ok=row['witness']
  summaries.append(dict(candidate=pose,sourceindex=int(index),foot_candidates=foot,active_limits=len(internal)))
 selected=min(feasible66) if feasible66 else min(feasible90) if feasible90 else None;assert (result['selected'] is None)==(selected is None)
 if selected is not None:
  assert result['selected']['pose']==selected and result['selected']['arm']==('mapped66' if feasible66 else 'independent90')
  raise RuntimeError('Eligible witness found: actual body diagnostic schema must be inspected before finalreview completion')
 assert result['body_gate']=='skipped_no_native_witness' and counters['executedLP']==result['calls']['LP'] and counters['newposes']==32 and counters['cachedposes']==1
 assert max(maxA,maxbase,maxeq,maxJ,maxproj,maxgain)<1e-8
 report=dict(passed=True,source_sha256=hashes,frozen_source33_exact=True,native_contact_and_active_limits_exact=True,originalmodel469_exact=True,counters=counters,selection=None,body_diagnostic='NOT_EXECUTED_NO_VALID_NATIVE_WITNESS',errors=dict(muscleA=maxA,affinebase=maxbase,eq=maxeq,native_vs_worldJ=maxJ,constraint_projection=maxproj,stationarygain=maxgain),poses=summaries,calls=dict(model_compile=0,forward=0,body_step=0,neural_step=0,linprog=0),limits='No solution found in finite33savedreferenceposes; solverstatuses fornewcases are not independentFarkasproof. Staticstanding is not necessary fordynamicwalking. No allpose/mechanicalimpossibility or neurocontrolsuccessclaim.')
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'RESULT.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','poses']},indent=2))
if __name__=='__main__':main()
