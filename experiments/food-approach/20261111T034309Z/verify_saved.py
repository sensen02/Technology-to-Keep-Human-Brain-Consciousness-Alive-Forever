"""Independent actual saved behavior audit. Training episodes only; no heldout evaluation."""
import argparse,json
import numpy as np,torch
from common import HERE,digest,save_json,guard
from learning import load_behavior,exact,production

def verify(path,sha):
 torch.set_num_threads(1);m,p=load_behavior(path,sha);step=p['curve'][-1]['step'];checks={}
 checks['saved_step_registered']=step in (500,1000);checks['curve0_to_global_contiguous']=[r['step'] for r in p['curve']]==list(range(step+1))
 checks['model_reload_exact']=exact(m.state_dict(),p['state_dict'])
 opt=torch.optim.Adam(m.parameters(),lr=.001);opt.load_state_dict(p['optimizer'])
 checks['actual_Adam_reload_exact']=exact(opt.state_dict(),p['optimizer'])
 checks['optimizer_covers_all_parameters_once']=len(opt.state)==len(list(m.parameters())) and len(opt.param_groups)==1 and len(opt.param_groups[0]['params'])==len(list(m.parameters()))
 checks['actual_Adam_counters']=all(float(v['step'])==step for v in opt.state.values())
 checks['moments_finite_nonnegative']=all(torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() and (v['exp_avg_sq']>=0).all() for v in opt.state.values())
 checks['hyperparameters']=opt.param_groups[0]['lr']==.001 and opt.param_groups[0]['betas']==(.9,.999) and opt.param_groups[0]['eps']==1e-8
 torch.set_rng_state(p['rng']['torch']);np.random.set_state(p['rng']['numpy'])
 checks['RNG_restore_exact']=exact(dict(torch=torch.get_rng_state(),numpy=np.random.get_state()),p['rng'])
 checks['full_batch_schedule20']=len(p['batch_schedule'])==step and all(len(v)==20 and len(set(v))==20 for v in p['batch_schedule'])
 with np.load(HERE/'train.npz',allow_pickle=False) as z:x=torch.tensor(z['receptors'][:3].transpose(1,0,2).copy())
 with torch.no_grad():a,h=production(m,x)
 clone,_=load_behavior(path,sha)
 with torch.no_grad():b,hb=production(clone,x);cmd,_=m.step(x[0],m.initial_state(3))
 checks['actual240_inference_clone_exact']=torch.equal(a,b) and torch.equal(h,hb)
 checks['speed_always_structuralzero']=torch.equal(cmd[:,0],torch.zeros_like(cmd[:,0]))
 checks['sensor_scope_not_deploy']=p['scope_tag']=='SENSOR_PRETRAIN_ONLY' and not p['deployment_qualified']
 checks['checkpoint_unchanged']=digest(path)==sha
 guard();r=dict(all_passed=all(checks.values()),checks=checks,step=step,checkpoint=str(path),sha256=sha,physical_steps=0,optimizer_updates=0)
 if not r['all_passed']:raise RuntimeError('saved verification failed '+json.dumps(r))
 return r
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);a.add_argument('--sha256',required=True);a.add_argument('--out',required=True);p=a.parse_args();r=verify(p.checkpoint,p.sha256);save_json(p.out,r);print(json.dumps(dict(all_passed=True,step=r['step'],checks=len(r['checks']))))
