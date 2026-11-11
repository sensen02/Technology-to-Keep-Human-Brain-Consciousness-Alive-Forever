"""Read-only diagnostics of completed actual rollouts; no integration/training."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from model import HERE,DATA,sha256,load_checked

def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def main():
 torch.set_num_threads(1)
 checkpoint=HERE/'TRAIN/checkpoint_002000.pt';m,p=load_checked(checkpoint,sha256(checkpoint));teacher=arrays(DATA);fit=p['replay']['means'][:,0].tanh().numpy()
 mask=m.base.physical_mask.bool().numpy();reports={}
 for condition in ('neutral','right'):
  root=HERE/('EVAL_'+condition);r=json.loads((root/'RESULT.json').read_text());d=arrays(root/'neural_decisions.npz');t=arrays(root/'trace_1ms.npz');g=arrays(root/'geometry_1ms.npz')
  duration=float(t['time'][-1]-t['time'][5799]);delta=np.diff(t['com'][5799:,:2],axis=0);heading=g['thorax_yaw'][5799:-1];projection=float(np.sum(delta*np.column_stack([np.cos(heading),np.sin(heading)]))/duration)
  identity=json.loads((root/'IDENTITY.json').read_text());limit=np.asarray(identity['actuator']['torque_limit']);blocks=[]
  if condition=='right':
   for sec in range(10):
    start=960+sec*200;end=min(start+200,2880)
    if start>=end:continue
    blocks.append(dict(ANN_seconds=[sec,min(sec+1,9.6)],teacher_distribution_action_MSE=float(np.mean((fit[start:end]-teacher['Y42'][start:end])**2)),actual_vs_teacher_action_MSE_sameclock=float(np.mean((d['policy_action'][start:end]-teacher['Y42'][start:end])**2)),actual_vs_teacher_physical112_RMS_sameclock=float(np.sqrt(np.mean((d['X238'][start:end,mask]-teacher['X238'][start:end,mask])**2)))))
  state=m.base.initial_state(1);base=[]
  with torch.no_grad():
   for x in d['X238']:
    mean,state=m.base.step(torch.from_numpy(x[None]),state);base.append(mean[0].numpy())
  latent=d['mean']-np.asarray(base)
  reports[condition]=dict(engineering_pass=r['engineering_pass'],criteria=r['criteria'],actual_signed_yaw_deg=float(np.degrees(r['yaw_change_rad'])),original_worldX_speed=r['autonomous']['forward_signed_native_per_s'],body_heading_projected_speed_diagnostic=projection,body_heading_metric_does_not_replace_worldX=True,slip=r['autonomous']['slip_ratio_max'],per_leg_slip=r['autonomous']['per_leg_slip_ratio'],cycles=r['autonomous_cycles']['complete_per_leg'],native_nonfoot_peak=r['native']['post_nonfoot_peak_native'],native_COM_min=r['native']['post_COM_min'],postprime_action_abs_max=float(np.abs(d['policy_action'][960:]).max()),postprime_residual_latent_abs_max=float(np.abs(latent[960:]).max()),endpoint_ctrl_fraction_max=float(np.max(np.abs(t['actualctrl'][5800:])/limit)),sameclock_diagnostics=blocks,completed_trace_rows=len(t['time']),completed_decisions=len(d['time']))
 result=dict(scope='descriptive completed nominal rollouts only; temporal distribution difference is not isolated causal proof; no added physics or training',checkpoint=str(checkpoint),sha256=sha256(checkpoint),postprime_teacher_fit_MSE=float(np.mean((fit[960:]-teacher['Y42'][960:])**2)),conditions=reports,physics_steps=0,optimizer_updates=0,source_hashes={str(HERE/'analyze_results.py'):sha256(HERE/'analyze_results.py')})
 (HERE/'ANALYSIS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
