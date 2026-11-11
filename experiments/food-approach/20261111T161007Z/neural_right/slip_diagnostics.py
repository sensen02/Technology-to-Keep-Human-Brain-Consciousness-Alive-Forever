"""Read-only same-run three-window/load-bearing and same-teacher-state fit diagnostics."""
import json,ast,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
from model import HERE,BASE,DATA,sha256
LEGS=['LF','LH','LM','RF','RH','RM']
XML=BASE.parent.parent/'banc-triad-pad/20261110T010000Z/MODEL_TRIAD_x200.xml'
W=24.446354755878005

def arrays(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def main():
 torch.set_num_threads(1)
 root=HERE/'EVAL_right';t=arrays(root/'trace_1ms.npz');teacher_t=arrays(HERE.parent/'attempt02/right_yaw_m060/TRACE_1MS.npz');data=arrays(DATA)
 ck=HERE/'TRAIN/checkpoint_002000.pt';p=torch.load(ck,map_location='cpu',weights_only=False);means=p['replay']['means'][:,0].numpy();pred=np.tanh(means);latent_error=np.zeros_like(means)
 assert np.max(np.abs(data['Y42'][960:]))<1.;latent_error[960:]=means[960:]-np.arctanh(data['Y42'][960:])
 names=[n.attrib['joint'] for n in ET.parse(XML).getroot().find('actuator')];assert len(names)==42
 indices={leg:[i for i,n in enumerate(names) if n.startswith('joint_'+leg)] for leg in LEGS};assert all(len(v)==7 for v in indices.values())
 def summary(trace):
  a=5800;b=15400;load=trace['loads'][a:b];speed=trace['speed'][a:b];time=trace['time'][a:b];com=trace['com'][a:b];xy=float(np.linalg.norm(com[-1,:2]-com[0,:2])/(time[-1]-time[0]));den=load.sum(0);num=(load*speed).sum(0);ratio=num/den/xy;segments=[]
  for k in range(3):
   lo=k*3200;hi=(k+1)*3200;l=load[lo:hi];s=speed[lo:hi];n=(l*s).sum(0);d=l.sum(0);localxy=float(np.linalg.norm(com[hi-1,:2]-com[lo,:2])/(time[hi-1]-time[lo]));segments.append(dict(ANN_seconds=[3.2*k,3.2*(k+1)],global_slip_contribution=dict(zip(LEGS,(n/den/xy).tolist())),own_window_slip_diagnostic=dict(zip(LEGS,(n/d/localxy).tolist())),own_window_xy_speed=localxy,load_denominator_native=dict(zip(LEGS,d.tolist()))))
  perleg={}
  for j,leg in enumerate(LEGS):
   bins={}
   for threshold in (.1,.2):
    carrying=load[:,j]>=threshold*W;d=float(load[carrying,j].sum());n=float((load[carrying,j]*speed[carrying,j]).sum())
    bins[str(threshold)+'W']=dict(sample_fraction=float(carrying.mean()),load_fraction=d/float(den[j]),slip_numerator_fraction=n/float(num[j]),load_weighted_foot_speed=n/d if d>0 else None,ratio_using_fullwindow_xy_speed=n/d/xy if d>0 else None)
   carrying=load[:,j]>=.1*W;edges=np.diff(np.r_[False,carrying,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1);intervals=[]
   for lo,hi in zip(starts,ends):
    n=float((load[lo:hi,j]*speed[lo:hi,j]).sum());d=float(load[lo:hi,j].sum());intervals.append(dict(ANN_start_s=float(time[lo]-5.8),ANN_end_s=float(time[hi-1]-5.8),samples=int(hi-lo),global_slip_contribution=n/float(den[j])/xy,mean_load_W=float(load[lo:hi,j].mean()/W),load_weighted_foot_speed=n/d))
   perleg[leg]=dict(fullwindow_slip=float(ratio[j]),load_bins=bins,carrying_interval_count=len(intervals),top_carrying_intervals=sorted(intervals,key=lambda r:r['global_slip_contribution'],reverse=True)[:10])
  return dict(xy_speed=xy,segments=segments,per_leg=perleg,additive_segment_check_max_error=float(np.max(np.abs(np.sum([[s['global_slip_contribution'][leg] for leg in LEGS] for s in segments],axis=0)-ratio))))
 fits=[]
 for k in range(3):
  lo=960+k*640;hi=lo+640
  fits.append(dict(ANN_seconds=[3.2*k,3.2*(k+1)],per_leg={leg:dict(action_MSE=float(np.mean((pred[lo:hi,ids]-data['Y42'][lo:hi,ids])**2)),residual_latent_MSE=float(np.mean(latent_error[lo:hi,ids]**2)),residual_latent_abs_max=float(np.abs(latent_error[lo:hi,ids]).max())) for leg,ids in indices.items()}))
 errors=np.mean(latent_error[960:]**2,axis=0);worst=np.argsort(errors)[::-1][:8]
 r=dict(scope='only completed existing arrays/checkpoint; no physics, no training. Segment-local slip is diagnostic and never replaces official fullwindow gate. Carrying>=.1W/.2W are added diagnostic bins. Same-state residual error=(network_mean-base_mean)-(atanh(teacheraction)-base_mean), so base cancels exactly; full continuous teacher-history neural state, not reset per row. Descriptive, not single-cause proof.',actual=summary(t),teacher=summary(teacher_t),same_teacher_state_fit=dict(action_MSE=float(np.mean((pred[960:]-data['Y42'][960:])**2)),residual_latent_MSE=float(np.mean(latent_error[960:]**2)),residual_latent_abs_max=float(np.abs(latent_error[960:]).max()),three_segments=fits,worst_joint_latent_MSE=[dict(joint=names[i],index=int(i),MSE=float(errors[i])) for i in worst]),physics_steps=0,optimizer_updates=0,source_hashes={str(v):sha256(v) for v in (Path(__file__),XML,DATA,ck,root/'trace_1ms.npz',HERE.parent/'attempt02/right_yaw_m060/TRACE_1MS.npz')})
 (HERE/'SLIP_DIAGNOSTICS.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(actual_slip={leg:v['fullwindow_slip'] for leg,v in r['actual']['per_leg'].items()},actual_three_segment_contribution=[s['global_slip_contribution'] for s in r['actual']['segments']],teacher_state_action_MSE=r['same_teacher_state_fit']['action_MSE'],teacher_state_latent_MSE=r['same_teacher_state_fit']['residual_latent_MSE'],teacher_state_latent_max=r['same_teacher_state_fit']['residual_latent_abs_max'])),flush=True)
if __name__=='__main__':main()
