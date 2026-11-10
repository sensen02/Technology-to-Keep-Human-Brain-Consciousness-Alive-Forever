"""Independent saved-data verification of the extended and cold/frozen controls."""
from pathlib import Path
import json,numpy as np
from plant_probe import Plant,O,dump,sha,dsm
from ann_translator import NumpyFullBrainTranslator
from verify_stand import arrays

def main():
 L=O/'learning';reg=json.loads((L/'EXTENDED_PREREGISTRATION.json').read_text());results=json.loads((L/'EXTENDED_RESULTS.json').read_text());checks={};details=[];plant=Plant();model=NumpyFullBrainTranslator(L/'TRANSLATOR.npz')
 checks['model_unchanged']=sha(L/'TRANSLATOR.npz')==reg['model_sha256'];checks['runtime_unchanged']=sha(Path(__file__).with_name('learn_stand.py'))==reg['runtime_sha256'];checks['extension_code_unchanged']=sha(Path(__file__).with_name('extended_check.py'))==reg['source_sha256']
 for r in results:
  tag=r['tag'];folder=Path(r['folder']);z=arrays(folder/'DATA.npz');cmd=arrays(folder/'COMMANDS.npz');native=arrays(folder/'NATIVE.npz')['rows'];body=arrays(folder/'BODY.npz');end=arrays(folder/'END.npz')
  checks[tag+'_finite']=bool(all(np.isfinite(a).all() for a in list(z.values())+list(cmd.values())+list(body.values())+list(end.values())) and np.isfinite(native).all())
  checks[tag+'_physics_clock']=bool(np.max(abs(np.diff(native[:,0])-.0000125))<1e-12 and abs(native[-1,0]-r['seconds'])<1e-8)
  ref=cmd['q0'].copy();integration_error=0.
  for row in cmd['rows']:
   integration_error=max(integration_error,float(abs(ref-row[1:43]).max()))
   ref=np.clip(ref+.001*np.clip(row[43:85]+row[85:127],-4,4),plant.m.jnt_range[plant.jids,0],plant.m.jnt_range[plant.jids,1])
  checks[tag+'_target_integration']=integration_error<1e-12
  lo=0 if tag=='cold_ann' else 200;hist=np.array([z['frames'][np.maximum(0,np.arange(i-3,i+1))] for i in range(lo,len(z['frames']))],np.float32)
  if tag=='frozen_brain_history':hist[:]=hist[0].copy()
  predicted=[]
  for i in range(0,len(hist),32):predicted.append(4*model(hist[i:i+32],z['target_states'][lo+i:lo+i+32]))
  error=float(abs(np.concatenate(predicted)-z['actions'][lo:]).max());checks[tag+'_ANN_provenance']=error<.005
  assist=native[(native[:,0]>=.8)&(native[:,0]<1)];release=native[native[:,0]>=1]
  reference=dict(foot_support_mean=float(assist[:,1:7].sum(1).mean()/dsm.WEIGHT),com_z_mean=float(assist[:,10].mean()))
  gate=dsm.evaluate_native(reference,release);gate['checks']['D1_no_nonfoot_ground_force']=bool(release[:,7].max()==0);gate['passed']=all(gate['checks'].values())
  checks[tag+'_gate_recompute']=gate['checks']==r['validation']['checks'] and gate['passed']==r['validation']['passed']
  checks[tag+'_force_balance']=gate['checks']['D6_force_balance_consistent']
  details.append(dict(tag=tag,stand_pass=gate['passed'],whole_run_nonfoot_max=float(native[:,7].max()),foot_support=gate['release']['foot_support_mean'],minimum_com=gate['release']['com_z_min'],recorded_seconds=r['seconds'],ANN_inference_error_rad_s=error,score_seconds=r['seconds']-1,cold_teacher_calls=0 if tag=='cold_ann' else 'warmup only',target_integration_error=integration_error))
 checks['initial_three_seconds_repeat']=all(np.array_equal(arrays(L/'extended_ann_5s/BODY.npz')[k][:600],arrays(L/'eval_ann_dt0.0625/BODY.npz')[k]) for k in ['time','qpos','qvel','ctrl','com','feet'])
 v=dict(importance='核心必需',passed=all(checks.values()),checks=checks,results=details,scoring='cold run first1s is ANN settling, not teacher warmup; gates evaluated remaining2s, whole-run nonfoot separately verified')
 dump(L/'EXTENDED_VERIFICATION.json',v);print(json.dumps(v),flush=True);assert v['passed'],[k for k,val in checks.items() if not val]
if __name__=='__main__':main()
