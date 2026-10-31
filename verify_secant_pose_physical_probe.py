"""Independent offline verifier for S146 secant-pose physical probe.
No MuJoCo/torch/producer imports; reads only the actual saved schema."""
from pathlib import Path
import argparse, hashlib, json, ast
import numpy as np

ROOT=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--report',type=Path,required=True);a=p.parse_args(); out=a.output.resolve(); arm=out/'init80ms'; checks=[]
 def ck(n,v,**d): checks.append({'name':n,'passed':bool(v),**d})
 r=json.loads((arm/'RESULT.json').read_text()); prereg=json.loads((out/'PREREG.json').read_text())
 traj=load(arm/'TRAJECTORY.npz'); init=load(arm/'INITIALIZATION.npz'); rows=load(arm/'CONSTRAINT_ROWS.npz')
 ck('S146 diagnostic metadata',r['status']=='STOPPED' and r['secant_pose_diagnostic'] is True)
 ck('first failure and stop',r['stop_reasons']==['LOAD_RATIO_LT_0.8'] and abs(r['actual_duration_ms']-1.925)<=1e-10)
 ck('actual dynamic counts',r['samples']==78 and r['calls']=={'model_load':1,'forward':78,'step':77,'neural_ticks':164})
 ck('no 5ms sensory claim',r['sensory_5ms_refresh_reached'] is False and r['actual_sensory_observation_times_s']==[0.0])
 ck('not prewarm success',r['physical_prewarm_accepted'] is False and r['behavior_claim']=='NONE')
 ck('clock and arrays',len(traj['time'])==78 and np.allclose(traj['time'],np.arange(78)*25e-6,rtol=0,atol=1e-15))
 ck('all trajectory finite',all(np.isfinite(v).all() for v in traj.values()))
 # Original S142 root height is explicitly retained as the stop reference.
 offset=float(json.loads((out/'SECANT_SELECTION_BINDING.json').read_text())['offset'])
 orig=load(out/'POSE_INITIALIZATION.npz')['original_qpos']; new=load(out/'POSE_INITIALIZATION.npz')['new_qpos']
 ck('single root-z pose change',np.array_equal(new,orig+np.array([0.0,0.0,offset]+[0.0]*(len(orig)-3))) and np.array_equal(traj['qpos'][0],new))
 ck('original-height reference',abs(float(r['original_height_reference'])-float(orig[2]))==0)
 ck('zero external applied loads',not traj['qfrc_applied'].any() and not traj['xfrc_applied'].any())
 # Dynamic first failure is independently recomputed from recorded gates.
 # Recompute six gates from physical arrays, never trust the saved gate field.
 q0=traj['qpos'][0]; quat=traj['qpos'][:,3:7]; norms=np.linalg.norm(quat,axis=1); n0=np.linalg.norm(q0[3:7])
 angle=2*np.arccos(np.clip(np.abs(quat@q0[3:7])/(norms*n0),0,1))
 load_ratio=traj['foot_force'][:,:,2].sum(1)/np.abs(traj['gravity_force_world'][:,2])
 recomputed=np.column_stack((traj['qpos'][:,2]/orig[2]>=.85,angle<=.35,np.count_nonzero(traj['contacts']>0,axis=1)>=3,load_ratio>=.8,traj['nonfoot_depth']<=1e-12,traj['limit_excess']<=1e-6))
 ck('saved gates match independent recomputation',np.array_equal(recomputed,traj['six_gates']))
 first=77
 ck('first failing sample recomputed',first==77 and recomputed[:77].all() and not bool(recomputed[77,3]),first_sample=int(first),first_time_ms=float(traj['time'][first]*1000))
 ck('stop sample retained',len(traj['time'])==first+1)
 ck('native full48 force balance',np.max(np.abs(traj['qfrc_actuator']+traj['qfrc_passive']+traj['qfrc_constraint']-traj['qfrc_bias']-traj['Ma']))<=1e-10)
 errs=[]
 for i in range(78):
  j=rows[f'sample{i:03d}_efc_J_dense']; f=rows[f'sample{i:03d}_efc_force']; errs.append(np.max(np.abs(j.T@f-traj['qfrc_constraint'][i])))
 ck('all48 dense J.T force closure',max(errs)<=1e-10,max_abs=float(max(errs)))
 ck('COM acceleration closure',np.max(np.abs(traj['com_acceleration_world']-traj['external_force_world']/traj['body_mass'][0].sum()))<=1e-9)
 # Actual schema neural/FIR contract: PRE160 plus ceil(77/20), and post spike stream is saved.
 ck('PRE160 plus partial final tick',init['spikes'].shape[0]==160 and traj['spikes'].shape[0]==4 and 160+int(np.ceil(77/20))==164)
 ck('post neural arrays dimensions',traj['spikes'].shape==(4,17099) and traj['features'].shape==(4,66,5) and traj['applied'].shape==(4,90) and traj['decoded'].shape==(4,90))
 ck('FIR history and release schema',load(arm/'NEURAL_FINAL.npz')['FIR_history'].shape==(80,66) and init['history'].shape==(80,66))
 # Source freeze is checked against the producer's published immutable hashes.
 hz=json.loads((out/'SOURCE_FREEZE'/'HASHES.json').read_text()); ih=hz['input_hashes']; ck('source freeze hashes',all(Path(k).is_file() and sha(k)==v for k,v in ih.items()))
 ck('private source parses',ast.parse((out/'SOURCE_FREEZE'/'PRIVATE_RUNNER_SOURCE.py').read_text()) is not None)
 report={'passed':all(x['passed'] for x in checks),'evidence_valid':all(x['passed'] for x in checks),'prewarm_success':False,'behavior_claim':'NONE','physical_units':'NOT_ESTABLISHED','verifier':str(Path(__file__).resolve()),'verifier_sha256':sha(__file__),'checks':checks,'actual_root':str(out),'first_failure_sample':int(first),'first_failure_ms':float(traj['time'][first]*1000),'post_neural_ticks':164}
 a.report.resolve().write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'passed':report['passed'],'report':str(a.report.resolve())}))
 return 0 if report['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
