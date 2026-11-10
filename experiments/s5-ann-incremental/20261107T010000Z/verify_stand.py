"""Verify saved ANN inference, physical replay, target integration and support."""
import sys,json
from pathlib import Path
import numpy as np
import mujoco as mj
from plant_probe import Plant,O,dump,sha,dsm
from ann_translator import NumpyFullBrainTranslator

def arrays(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def main():
 root=O/'learning';p=json.loads((root/'PREREGISTRATION.json').read_text());checks={};detail=[]
 checks['frozen_sources']=all(sha(Path(f))==h for f,h in p['hashes'].items())
 ledger=json.loads((O/'CHANGE_LEDGER.json').read_text())
 checks['supplemental_dependencies_unchanged']=all(sha(Path(f))==h for f,h in ledger['supplemental_dependency_hashes'].items())
 model=NumpyFullBrainTranslator(root/'TRANSLATOR.npz');results=json.loads((root/'EVALUATION.json').read_text());plant=Plant()
 handovers=[]
 for r in results:
  folder=Path(r['folder']);z=arrays(folder/'DATA.npz');n=arrays(folder/'NATIVE.npz')['rows'];cmd=arrays(folder/'COMMANDS.npz');b=arrays(folder/'BODY.npz');h=arrays(folder/'HANDOVER.npz');handovers.append((r,h))
  tag=r['tag'];ref=cmd['q0'].copy();max_integral=0.
  for row in cmd['rows']:
   max_integral=max(max_integral,float(abs(ref-row[1:43]).max()))
   ref=np.clip(ref+.001*np.clip(row[43:85]+row[85:127],-4,4),plant.m.jnt_range[plant.jids,0],plant.m.jnt_range[plant.jids,1])
  checks[tag+'_target_integration']=max_integral<1e-12
  checks[tag+'_finite']=bool(all(np.isfinite(x).all() for x in list(b.values())+list(z.values())+list(h.values())+list(cmd.values())) and np.isfinite(n).all() and r['warnings']==0)
  checks[tag+'_native_clock']=bool(np.max(abs(np.diff(n[:,0])-.0000125))<1e-12)
  release=n[n[:,0]>=1];foot=release[:,1:7].sum(1)/dsm.WEIGHT
  checks[tag+'_support_recompute']=abs(float(foot.mean())-r['validation']['release']['foot_support_mean'])<1e-12
  checks[tag+'_nonfoot_recompute']=bool(np.max(release[:,7])==0)==r['validation']['checks']['D1_no_nonfoot_ground_force']
  infer_error=None
  if r['mode'] in ['ann','brain_cut']:
   hs=np.array([z['frames'][np.maximum(0,np.arange(i-3,i+1))] for i in range(200,len(z['frames']))],np.float32)
   if r['mode']=='brain_cut':hs[:]=0
   pred=[]
   for start in range(0,len(hs),32):pred.append(model(hs[start:start+32],z['target_states'][200+start:200+start+32])*4)
   infer_error=float(abs(np.concatenate(pred)-z['actions'][200:]).max())
   # Runtime inference uses float32 histories; saved training/inspection frames are float16.
   checks[tag+'_inference_quantization_tolerance']=infer_error<.005
  detail.append(dict(tag=tag,all_standing_gates=r['validation']['passed'],foot_support_mean=float(foot.mean()),nonfoot_max=float(release[:,7].max()),min_com=float(release[:,10].min()),target_integral_max_error=max_integral,saved_float16_inference_error_rad_s=infer_error))
 dev=arrays(root/'dev0/DATA.npz');dev_errors=[]
 for i in range(220,len(dev['frames'])):
  hist=dev['frames'][np.arange(i-3,i+1)].astype(np.float32)[None]
  pred=model(hist,dev['target_states'][i:i+1])[0]
  dev_errors.append(float(np.mean((pred-dev['teacher_actions'][i])**2)))
 dump(root/'DEV_POST_DISTURBANCE.json',dict(importance='重要辅助',start_s=1.1,samples=len(dev_errors),mse=float(np.mean(dev_errors)),note='Excludes common pre-disturbance prefix; new physical trajectory, not full distribution guarantee.'))
 first=handovers[0][1]
 for r,h in handovers[1:]:
  keys=first.keys() if r['brain_dt_ms']==results[0]['brain_dt_ms'] else ['integration','target']
  checks[r['tag']+'_matched_warmup']=all(np.array_equal(first[k],h[k]) for k in keys)
 # Independent full native-step replay of the ANN run from declared initial keyframe.
 r=next(r for r in results if r['mode']=='ann');folder=Path(r['folder']);b=arrays(folder/'BODY.npz');c=arrays(folder/'COMMANDS.npz');d=plant.fresh();target=c['q0'].copy();errs=dict(qpos=0.,qvel=0.,ctrl=0.,time=0.)
 for k,row in enumerate(c['rows']):
  target=np.clip(target+.001*np.clip(row[43:85]+row[85:127],-4,4),plant.m.jnt_range[plant.jids,0],plant.m.jnt_range[plant.jids,1])
  for _ in range(80):
   d.ctrl[:]=np.clip(c['kp']*(target-d.qpos[plant.qa])-c['kd']*d.qvel[plant.va],-c['limits'],c['limits']);mj.mj_step(plant.m,d);mj.mj_forward(plant.m,d)
  if k%5==4:
   i=k//5
   for key,actual in [('qpos',d.qpos),('qvel',d.qvel),('ctrl',d.ctrl),('time',d.time)]:errs[key]=max(errs[key],float(np.max(abs(actual-b[key][i]))))
  if (k+1)%1000==0:print(json.dumps(dict(replay_ms=k+1,errors=errs)),flush=True)
 checks['independent_physical_replay']=max(errs.values())<1e-9
 checks['full_free_body']=bool((plant.m.nq,plant.m.nv,plant.m.nu,plant.m.nmesh,plant.m.neq)==(49,48,42,71,0) and np.array_equal(plant.m.opt.gravity,[0,0,-9801]))
 v=dict(importance='核心必需',passed=all(checks.values()),checks=checks,results=detail,replay=errs,counts_caveat='DATA counts contains completed5ms bins at policy tick; final5ms is not flushed; END net_spike_count has full count. Not used as complete spike ledger.',cold_control='not executed in initial5 evaluations; no cold-start neural success claim',image_visual_inspection='not yet performed',brain_necessity='judge using brain_cut control; do not infer from ANN input wiring')
 dump(root/'VERIFICATION.json',v);print(json.dumps(v),flush=True);assert v['passed'],[k for k,val in checks.items() if not val]
if __name__=='__main__':main()
