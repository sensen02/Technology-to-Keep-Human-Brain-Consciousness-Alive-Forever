"""S2/S4 sourced antagonist MN threshold response; new evidence only."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import sys,json,hashlib,math,argparse,inspect,traceback,time,dataclasses,shutil
from pathlib import Path
import numpy as np
from scipy import sparse
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'))
O=Path('/home/sensen/Desktop/engine修复验收/s4-antagonist-mn-support/20261102T061756Z')
E=Path('/home/sensen/Desktop/engine修复验收/s2-early-muscle-support/20261102T053639Z')
B=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
C=Path('/home/sensen/Desktop/engine修复验收/s4-bending-closedloop-training/20261102T042848Z')
A=Path('/home/sensen/Desktop/engine修复验收/s3-banc888-runtime-admission/20261101T220018Z')
LEGS=('LF','LM','LH','RF','RM','RH');DT=.0000125

def arr(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def js(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item())+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csrhash(w):
 h=hashlib.sha256()
 for k in ('data','indices','indptr'):h.update(getattr(w,k).tobytes())
 return h.hexdigest()

def static():
 assert not O.exists(),'Refuse existing evidence directory'
 O.mkdir(parents=True)
 identity=js(E/'IDENTITY_AND_MAPPING.json');mapping=js(B/'RUN/MAPPING.json');assert identity['mapping']==mapping
 roots=arr(B/'IDENTITIES.npz')['root_ids'];rows=mapping['rows'];groups=[];lookup={}
 for i,row in enumerate(rows):
  assert row['actuator_index']==i
  assert list(roots[row['local_indices']])==row['tier_root_ids']
  key=tuple(row['local_indices'])
  if not key:continue
  if key not in lookup:lookup[key]=len(groups);groups.append(dict(group=len(groups),local_indices=list(key),root_ids=roots[list(key)],muscle_indices=[],rows=[]))
  g=groups[lookup[key]];g['muscle_indices'].append(i);g['rows'].append(row)
 assert len(groups)==66
 overlaps=[]
 for i,g in enumerate(groups):
  for h in groups[i+1:]:
   shared=sorted(set(g['local_indices'])&set(h['local_indices']))
   if shared:overlaps.append(dict(group_a=g['group'],group_b=h['group'],nodes=shared,root_ids=roots[shared]))
 audits={name:arr(E/(name+'_AUDIT.npz')) for name in ['TRAIN_000_BASELINE','TRAIN_017_MINUS']}
 for variant,z in audits.items():
  t=z['relative_time_s'];ix=int(np.argmin(abs(t)));assert abs(t[ix])<1e-10
  win=(t>=-1e-10)&(t<=.005+1e-10)
  coeff=z['moment_42x90']*z['actuator_gain'][:,None,:]
  active=z['moment_42x90']*z['active_force'][:,None,:]
  for g in groups:
   ids=g['muscle_indices'];co=coeff[:,:,ids].sum(2);ac=active[:,:,ids].sum(2)
   g.setdefault('variants',{})[variant]=dict(release_coefficient=co[ix],first5ms_min=co[win].min(0),first5ms_max=co[win].max(0),first5ms_mean=co[win].mean(0),release_active_contribution=ac[ix],first5ms_active_mean=ac[win].mean(0))
 np.savez_compressed(O/'STATIC_POOL_ARRAYS.npz',root_ids=roots,group_by_muscle=np.array([lookup.get(tuple(row['local_indices']),-1) for row in rows]),**{variant+'_'+k:z[k] for variant,z in audits.items() for k in ['relative_time_s','moment_42x90','actuator_gain','active_force','activation']})
 plans=[]
 for label,joint,direction in [('ARM_LF_tibia_pair','joint_LFTibia_pitch',1),('ARM_LM_yaw_pair','joint_LMCoxa_yaw',-1)]:
  ji=identity['joint_names'].index(joint);values=np.array([g['variants']['TRAIN_000_BASELINE']['release_coefficient'][ji] for g in groups]);positive=np.flatnonzero(values>0);negative=np.flatnonzero(values<0)
  pair=[int(positive[np.argmax(values[positive])]) if len(positive) else None,int(negative[np.argmin(values[negative])]) if len(negative) else None]
  valid=all(v is not None for v in pair);conflict=[] if not valid else sorted(set(groups[pair[0]]['local_indices'])&set(groups[pair[1]]['local_indices']))
  assignments=[]
  if valid and not conflict:
   for gid,sign in zip(pair,[1,-1]):
    assignments.append(dict(group=gid,shift_mV=-1 if sign==direction else 1,local_indices=groups[gid]['local_indices']))
  plans.append(dict(label=label,joint=joint,joint_index=ji,desired_direction=direction,positive_group=pair[0],negative_group=pair[1],valid=bool(valid and not conflict),conflicting_nodes=conflict,assignments=assignments,all_nonzero_groups=[dict(group=i,release_coefficient=float(values[i])) for i in np.flatnonzero(values)]))
 if all(p['valid'] for p in plans):
  a,b=plans;shared=sorted(set(sum([v['local_indices'] for v in a['assignments']],[]))&set(sum([v['local_indices'] for v in b['assignments']],[])))
  plans.append(dict(label='ARM_COMBINED',valid=not shared,conflicting_nodes=shared,assignments=a['assignments']+b['assignments']))
 dump(O/'STATIC_POOL_SIGN_AND_OVERLAP.json',dict(method='group=sum actual matrix row*actual native gain, never name polarity; no integration/model/forward',identity_source=str(E/'IDENTITY_AND_MAPPING.json'),manifest_source=str(B/'RUN/MAPPING.json'),joint_names=identity['joint_names'],groups=groups,overlaps=overlaps,plans=plans,RH_Trochanter_pitch='no intervention: instantaneous independent90 upper bound negative in both release states; no threshold can fabricate positive fixed-state ability',units='native model; SI NOT_ESTABLISHED'))
 shutil.copyfile(__file__,O/'STATIC_EXECUTION_SOURCE.py')
 print(json.dumps(dict(plans=plans,overlap_pairs=len(overlaps)),ensure_ascii=False,indent=2))

def prepare():
 d=js(O/'STATIC_POOL_SIGN_AND_OVERLAP.json');plans=d['plans'];eps=1e-12
 for p in plans:
  if p['label']=='ARM_COMBINED':continue
  coefficients={v['group']:v['release_coefficient'] for v in p['all_nonzero_groups']}
  p['valid']=bool(p['valid'] and coefficients.get(p['positive_group'],0)>eps and coefficients.get(p['negative_group'],0)<-eps)
  p['reason']='valid signed pair' if p['valid'] else 'No sourced positive group above declared numerical-zero 1e-12; floating residual is not positive ability; descriptive only'
 plans[-1]['valid']=all(p['valid'] for p in plans[:-1]) and not plans[-1]['conflicting_nodes']
 dump(O/'ADJUDICATED_PLAN.json',dict(coefficient_numerical_zero=eps,plans=plans,not_optimized=True,original_static_output_retained=True))
 import train_bending_closedloop as old
 text=inspect.getsource(old.trial)
 replacements=[
 ("folder=O/label;folder.mkdir();z=np.asarray(z,float);assert z.shape==(12,) and np.all(abs(z)<=1)","folder=O/label;folder.mkdir();z=np.asarray(z,float);assert z.shape==(12000,) and np.all(np.isin(z,[-1.,0.,1.]))"),
 ("gain=np.exp(z[:6]*math.log(2));theta=2*z[6:]","gain=np.ones(6);theta=z.copy()"),
 ("offset[motor]=source[motor]+theta[g[motor]]","offset=source+theta;assert not source.any()"),
 ("leg_index=g)","leg_index=g,assigned_group=ASSIGNED_GROUP)"),
 ("netref.append(net);return net,enc,dec","netref.append(net);return net,enc,dec"),
 ("observer=kw['body_observer'];isrelease=kw.get('runtime_state') is not None","observer=kw['body_observer'];isrelease=kw.get('runtime_state') is not None\n  compiled={k:getattr(model,k).copy() for k in dir(model) if isinstance(getattr(model,k),np.ndarray)}\n  if not isrelease:np.savez_compressed(folder/'ACTUAL_COMPILED_INITIAL.npz',**compiled)"),
 ("if isrelease:\n   np.savez_compressed(folder/'BODY_BOUNDARY.npz',**ret['handover_after_withdrawal'])","if isrelease:\n   np.savez_compressed(folder/'BODY_BOUNDARY.npz',**ret['handover_after_withdrawal'])\n   np.savez_compressed(folder/'ACTUAL_COMPILED_FINAL.npz',**{k:getattr(model,k).copy() for k in compiled})\n   dump(folder/'MATRIX_FINAL_HASHES.json',{k:csrhash(getattr(net,k)) for k in ['We','Wi']})\n   for key in ['We','Wi']:sparse.save_npz(folder/('FINAL_'+key+'.npz'),getattr(net,key))"),
 ]
 for before,after in replacements:
  assert text.count(before)==1,before;text=text.replace(before,after)
 compile(text,str(O/'TRIAL_EXECUTION_SOURCE.py'),'exec')
 (O/'TRIAL_EXECUTION_SOURCE.py').write_text(text)
 paths=[Path(__file__),R/'tools/train_bending_closedloop.py',O/'TRIAL_EXECUTION_SOURCE.py',O/'STATIC_EXECUTION_SOURCE.py',O/'STATIC_POOL_SIGN_AND_OVERLAP.json',O/'STATIC_POOL_ARRAYS.npz',O/'ADJUDICATED_PLAN.json',B/'RUN/MAPPING.json',B/'RUN/MODEL.xml',B/'RUN/BODY.npz',B/'NEURAL_RASTER.npz',B/'ACTUAL_We.npz',B/'ACTUAL_Wi.npz',B/'IDENTITIES.npz',C/'EXACT_GATE.json',C/'FROZEN_REFERENCE.json',C/'TRAIN_000_BASELINE/BODYLOSS_NATIVE.npz',E/'IDENTITY_AND_MAPPING.json']
 paths+=[O/'OFFLINE_SELECTED_STATES.py',O/'VERIFY_ARRAYS.py',Path(old_target()),C/'TRAIN_000_BASELINE/RELEASE_FK.npz',E/'TRAIN_000_BASELINE_AUDIT.npz',E/'TRAIN_017_MINUS_AUDIT.npz',E/'COMPILED_MODEL.npz']
 paths+=list((R/'engine').rglob('*.py'))+[R/'run_physical_walk_prewarm.py',R/'run_neural_muscle_closure.py',R/'run_integrated_repaired_closure.py']+[A/n for n in ['VERSION_CONTRACT.json','MN_POOLS_SNAPSHOT888.json','SENSORY_LOOKUP_SNAPSHOT888.json']]
 assert js(C/'EXACT_GATE.json')['passed'] and max(js(C/'EXACT_GATE.json')['body_max_abs'].values())==0
 dump(O/'PREREGISTRATION.json',dict(stage='S2/S4 sourced antagonist MN response, not training',head='ffc1ecacc99698d5ac5c0cb0217a725245c7846f',source_sha256={str(p):sha(p) for p in paths},trials=[p for p in plans if p['valid']],skipped=[p for p in plans if not p['valid']],max_trials=3,actual_registered_trials=sum(p['valid'] for p in plans),config=old.CONFIG,release_s=.05,body_per_trial=108000,neural_per_trial=2300,neutral='No new neutral integrations; original round13 fullhistory .7s release cache, 2300 raster prefix; prior fresh factory-neutral exact gate reused with source/matrix checks',intervention='constant ±1mV threshold offsets on sourced nodes only from first neural t0; enhance−1 reduce+1, no currents or force/gain/decoder changes, no hidden overlap summing; fresh body/net each trial',support=dict(horizon_s=.05,deadband_weight=1e-8,meanfoot=.8,minheight=.8,early_tau_s=.02),h0q0W=js(C/'FROZEN_REFERENCE.json'),budget=dict(body=108000*sum(p['valid'] for p in plans),neural=2300*sum(p['valid'] for p in plans)),offline='at most1 extra compile, selected saved state forward/moment only after live trial; zero extra body/neural; live schedule no added forward/Jac',failure='Initial static stdout serialization failed after scientific arrays written, no physics/neural calls; original STATIC_EXECUTION_SOURCE.py retained. Static raw positive LM5e-15 adjudicated numerical zero before any experiment.'))
 shutil.copyfile(__file__,O/'EXECUTION_SOURCE.py');print('PREPARED',flush=True)

def run():
 import train_bending_closedloop as old
 pr=js(O/'PREREGISTRATION.json')
 for p,h in pr['source_sha256'].items():assert sha(p)==h,p
 scope=dict(old.__dict__);scope.update(O=O)
 exec(compile((O/'TRIAL_EXECUTION_SOURCE.py').read_text(),str(O/'TRIAL_EXECUTION_SOURCE.py'),'exec'),scope)
 groups=js(O/'STATIC_POOL_SIGN_AND_OVERLAP.json')['groups'];roots=arr(B/'IDENTITIES.npz')['root_ids']
 for plan in pr['trials']:
  z=np.zeros(12000);assigned=np.full(12000,-1,int)
  for a in plan['assignments']:
   ids=a['local_indices'];assert np.all(assigned[ids]<0);z[ids]=a['shift_mV'];assigned[ids]=a['group']
  scope['ASSIGNED_GROUP']=assigned
  scope['trial'](plan['label'],z,.05)
 score_all();print('EXPERIMENT_COMPLETE',flush=True)

def score_all():
 from engine.embodied.support_priority import evaluate_support,select_support_first
 ref=js(C/'FROZEN_REFERENCE.json');trials=js(O/'PREREGISTRATION.json')['trials'];results=[]
 data=[('BASELINE_CACHE',C/'TRAIN_000_BASELINE')]+[(p['label'],O/p['label']) for p in trials]
 target=arr(old_target());baseline_fk=arr(C/'TRAIN_000_BASELINE/RELEASE_FK.npz')
 for label,folder in data:
  rows=arr(folder/'BODYLOSS_NATIVE.npz')['rows'];rows=rows[:4001];assert rows.shape==(4001,20)
  t=np.arange(1,4001)*DT;r=rows[1:]
  support=evaluate_support(t,r[:,1:7],r[:,7],r[:,10],weight=ref['weight'],h0=ref['h0'],horizon_s=.05,assistance=r[:,19])
  fk=baseline_fk if label=='BASELINE_CACHE' else arr(folder/'RELEASE_FK.npz');ix=np.array([np.argmin(abs(fk['time']-(1.3+v))) for v in target['time_relative_s'][:7]])
  points=fk['points'][ix];u=points[...,:-2,:]-points[...,1:-1,:];v=points[...,2:,:]-points[...,1:-1,:];angles=np.arctan2(np.linalg.norm(np.cross(u,v),axis=-1),np.sum(u*v,axis=-1));motion=float(np.mean((angles-target['angles_rad'][:7])**2))
  first=lambda col:None if not np.any(col) else float(t[np.flatnonzero(col)[0]])
  result=dict(label=label,support=support,motion_loss=motion,first_nonfoot_candidate_s=first(r[:,9]>0),first_any_positive_nonfoot_s=first(r[:,7]>0),mean_positive_nonfoot_weight=float(r[:,7].mean()/ref['weight']),per_leg_mean_foot_weight=r[:,1:7].mean(0)/ref['weight'],COMz_last=float(r[-1,10]),COMz_min=float(r[:,10].min()),root_last=r[-1,11:18],ctrl_mean_square=float(r[:,18].mean()),geometry_samples=7,walking_certified=False)
  results.append(result)
  if label!='BASELINE_CACHE':
   np.savez_compressed(folder/'GEOMETRY_SAMPLES.npz',source_frames=target['source_frame'][:7],source_time=target['time_relative_s'][:7],physical_time=fk['time'][ix],points=points,angles=angles)
 dump(O/'RESULT.json',dict(records=results,selection=select_support_first(results),no_optimizer=True,no_new_neutral=True,baseline_source=str(C/'TRAIN_000_BASELINE/BODYLOSS_NATIVE.npz'),baseline_raster_source=str(B/'NEURAL_RASTER.npz')))

def old_target():return '/home/sensen/Desktop/engine修复验收/s4-invariant-motion-target/20261102T023600Z/INVARIANT_BENDING_TARGETS.npz'

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['static','prepare','run','score']);args=parser.parse_args()
 if args.mode=='static':static()
 elif args.mode=='prepare':prepare()
 elif args.mode=='run':run()
 elif args.mode=='score':score_all()
