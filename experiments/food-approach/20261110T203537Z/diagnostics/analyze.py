"""Read-only saved fitting analysis; zero optimization or physical integration."""
from pathlib import Path
import sys,json
import numpy as np
import torch
from scipy.spatial.distance import cdist
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from fast_motor import load,sha256
from fast_data import prepare,cached


def main():
 torch.set_num_threads(1)
 ck=ROOT/'TRAIN_FAST_1000/checkpoint_001000.pt'
 model,p=load(ck,'3dd80394239504ec8b88f4cfe07c79569cf08329a9320c749c3c7b25035fda28')
 episodes,datahash=prepare(model)
 curve=p['curve'];names=[e['name'] for e in episodes]
 report=dict(scope='saved teacher-distribution fitting diagnosis, no optimization/physics; not closed-loop failure-cause proof',checkpoint=str(ck),checkpoint_sha256=sha256(ck),cases={},checkpoint_comparisons={},optimizer={},sources={str(ROOT/n):sha256(ROOT/n) for n in ('fast_motor.py','fast_data.py','train_fast.py','FAST_CHECKS.json','FAST_RANGE_FLOOR.json','TRAIN_FAST_1000/RESULT.json')},base=dict(checkpoint=p['identity']['base_checkpoint'],sha256=p['identity']['base_sha256'],frozen_all_exact=True,target_inhibition='base -softplus(raw_gain)*obs[126:168]; adapter excludes target channels and direct prior unchanged'),physical_steps=0,optimizer_updates=0)
 # Read each compressed trajectory only once; arrays remain bounded to one per case.
 for e in episodes:
  name=e['name'];directory=Path(next(q for q in datahash if q.endswith('SAMPLES_5MS.npz') and name in q)).parent
  with np.load(directory/'TRACE_1MS.npz',allow_pickle=False) as z:
   times=z['time'].copy();loads=z['loads'].copy();sources=z['control_source'].copy()
  with np.load(directory/'SAMPLES_5MS.npz',allow_pickle=False) as z: decision_times=z['actual_time'].copy()
  # time at action decision equals preceding physical endpoint, not future endpoint.
  indices=np.searchsorted(times,decision_times[960:]+1e-9,side='right')-1
  if np.max(np.abs(times[indices]-decision_times[960:]))>1e-8:raise ValueError('load alignment failed')
  # Normalize by snapshot native weight without importing physical backend.
  # Weight derives from actual sum load, robust at stable support; threshold sensitivity explicit.
  weight=24.446354755878005  # exact frozen sources/repair_core.py WEIGHT
  leg_load=loads[indices]/weight
  with torch.no_grad(): pred=cached(model,e)[960:].tanh().numpy()
  target=e['y'][960:].numpy();error=pred-target;se=error**2
  mean=e['mean'][960:].numpy();gate=e['gate'][960:].numpy()
  floor=np.maximum(np.tanh(mean-gate)-target,0)+np.minimum(np.tanh(mean+gate)-target,0)
  case=dict(MSE=float(se.mean()),absolute_error_quantiles=dict(zip(['p50','p90','p95','p99','max'],map(float,np.quantile(abs(error),[.5,.9,.95,.99,1])))),algebraic_floor_MSE=float((floor**2).mean()),floor_fraction_of_observed_MSE=float((floor**2).mean()/se.mean()),per_leg={},load_threshold_weight='exact frozen sources/repair_core.py WEIGHT=24.446354755878005; samples use immediately preceding physical endpoint',estimated_weight_native=weight)
  for leg,j in zip(['LF','LH','LM','RF','RH','RM'],range(6)):
   s=se[:,j*7:(j+1)*7];loaded=leg_load[:,j]>=.1
   case['per_leg'][leg]=dict(MSE=float(s.mean()),error_energy_fraction=float(s.sum()/se.sum()),joint_group_quantiles=np.quantile(abs(error[:,j*7:(j+1)*7]),[.5,.95,.99,1]).tolist(),loaded_ge_p1_fraction=float(loaded.mean()),loaded_MSE=float(s[loaded].mean()) if loaded.any() else None,unloaded_MSE=float(s[~loaded].mean()) if (~loaded).any() else None,loaded_error_energy_fraction=float(s[loaded].sum()/s.sum()))
  case['worst_joints']=sorted([dict(index=j,MSE=float(se[:,j].mean()),floor_MSE=float((floor[:,j]**2).mean())) for j in range(42)],key=lambda v:-v['MSE'])[:8]
  perrow=se.mean(1);worst=np.argsort(perrow)[-12:][::-1]
  # Teacher provenance only; never actor input. Report decision elapsed and bin, no phase labels invented.
  case['worst_decisions']=[dict(post_decision=int(i),actual_time=float(decision_times[960+i]),post_elapsed=float(i*.005),MSE=float(perrow[i]),command=e['u'][960+i].tolist(),load_W=leg_load[i].tolist()) for i in worst]
  # Observability diagnostic: nearest cross-time neighbors excluding +/-0.25s.
  # Use RMS standardized physical+command features vs same augmented with base hidden.
  x=e['x'][960:].numpy()[:,model.base.physical_mask.numpy().astype(bool)];u=e['u'][960:].numpy();h=e['base_hidden'][960:].numpy()
  obsdiag={}
  for label,f in [('physical_command',np.c_[x,u]),('physical_hidden_command',np.c_[x,h,u]),('hidden_command',np.c_[h,u])]:
   sd=np.maximum(f.std(0),.05);f=(f-f.mean(0))/sd
   ds=cdist(f,f,'sqeuclidean')/f.shape[1]
   idx=np.arange(len(f));ds[np.abs(idx[:,None]-idx[None,:])<=50]=np.inf
   nearest=ds.argmin(1);dist=np.sqrt(ds[idx,nearest]);targetdelta=np.mean((target-target[nearest])**2,axis=1)
   obsdiag[label]=dict(nearest_RMS_distance_quantiles=np.quantile(dist,[.1,.5,.9]).tolist(),neighbor_target_MSE_quantiles=np.quantile(targetdelta,[.1,.5,.9,.99]).tolist(),near_lt_p05_count=int((dist<.05).sum()),near_lt_p05_target_MSE_median=float(np.median(targetdelta[dist<.05])) if (dist<.05).any() else None)
  case['observability_nearest_neighbor']=obsdiag
  # Exact scalar feature collisions only: absence is not proof of learnability.
  case['exact_duplicate_feature_rows']=len(e['features'][960:])-len(np.unique(e['features'][960:].numpy(),axis=0))
  report['cases'][name]=case
 for step in (500,600,900,1000):
  path=ROOT/f'TRAIN_FAST_1000/checkpoint_{step:06d}.pt';mm,pp=load(path)
  opt=pp['optimizer'];group=opt['param_groups'][0]
  counter=[float(s['step']) for s in opt['state'].values()]
  report['checkpoint_comparisons'][str(step)]=dict(MSE={},lr=group['lr'],optimizer_step_min=min(counter),optimizer_step_max=max(counter),state_tensor_count=len(counter),finite_moments=all(bool(torch.isfinite(s[k]).all()) for s in opt['state'].values() for k in ('exp_avg','exp_avg_sq')),gradient_norm=pp['curve'][-1]['gradient_norm'])
  with torch.no_grad():
   for e in episodes:report['checkpoint_comparisons'][str(step)]['MSE'][e['name']]=float((cached(mm,e)[960:].tanh()-e['y'][960:]).square().mean())
 report['optimizer']=dict(groups=len(p['optimizer']['param_groups']),uniform_lr=[g['lr'] for g in p['optimizer']['param_groups']],all_steps1000=all(float(s['step'])==1000 for s in p['optimizer']['state'].values()),gradient_clipped_steps=sum(r.get('gradient_norm',0)>1 for r in curve),max_gradient_norm=max(r.get('gradient_norm',0) for r in curve))
 report['loss_spike_windows']={str(end):[dict(step=r['step'],loss=r['postprime_MSE'],grad=r.get('gradient_norm'),after=r.get('metrics_after_update')) for r in curve[end-12:end+1]] for end in (600,1000)}
 report['checkpoint_boundary_continuity']={str(s):dict(after=p['curve'][s]['metrics_after_update'],next_pre=p['curve'][s+1]['postprime_MSE'],exact=p['curve'][s]['metrics_after_update']==p['curve'][s+1]['postprime_MSE']) for s in range(100,1000,100)}
 report['source_data_hashes_unchanged']=all(sha256(q)==v for q,v in {**p['config']['sources'],**datahash}.items())
 report['diagnostic_source_sha256']=sha256(__file__)
 (HERE/'DIAGNOSIS.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(done=True,cases={k:{q:v[q] for q in ('MSE','algebraic_floor_MSE','floor_fraction_of_observed_MSE')} for k,v in report['cases'].items()},optimizer=report['optimizer'],checkpoint_comparisons=report['checkpoint_comparisons'])))

if __name__=='__main__':main()
