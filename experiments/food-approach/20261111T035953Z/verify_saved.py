"""Independent balanced sensory checkpoint audit; never evaluate fresh test here."""
import argparse,json
import numpy as np,torch
from common import HERE,PREVIOUS,PARENT,PARENT_SHA,load_behavior,legacy_learning,exact,production,digest,save_json,guard

def verify(path,sha):
 torch.set_num_threads(1);m,p=load_behavior(path,sha,require_final=False);_,parent=legacy_learning.load_behavior(PARENT,PARENT_SHA);step=p['curve'][-1]['step'];checks={}
 checks['full_curve0_global_contiguous']=[r['step'] for r in p['curve']]==list(range(step+1));checks['actual_parent1001_curve_prefix_exact']=exact(p['curve'][:1001],parent['curve'])
 checks['actual_saved_model_exact']=exact(m.state_dict(),p['state_dict'])
 opt=torch.optim.Adam(m.parameters(),lr=.0003);opt.load_state_dict(p['optimizer']);checks['actual_optimizer_reload_exact']=exact(opt.state_dict(),p['optimizer'])
 checks['all_Adam_states_coverage']=len(opt.state)==len(list(m.parameters())) and len(opt.param_groups)==1 and len(opt.param_groups[0]['params'])==len(list(m.parameters()))
 checks['all_counters_global']=all(float(v['step'])==step for v in opt.state.values());checks['moment_shapes_finite_nonnegative']=all(v['exp_avg'].shape==parameter.shape and v['exp_avg_sq'].shape==parameter.shape and torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() and (v['exp_avg_sq']>=0).all() for parameter,v in opt.state.items())
 checks['lr_unchanged_balanced_stage']=opt.param_groups[0]['lr']==.0003
 torch.set_rng_state(p['rng']['torch']);np.random.set_state(p['rng']['numpy']);checks['rng_roundtrip_exact']=exact(dict(torch=torch.get_rng_state(),numpy=np.random.get_state()),p['rng'])
 schedule=p['batch_schedule'];meta=json.loads((PREVIOUS/'DATASET.json').read_text())['splits']['train']['episodes_config']
 checks['full_batch_schedule']=len(schedule)==step and exact(schedule[:1000],parent['batch_schedule'])
 checks['every_new_batch_exact10food10nofood']=all(len(row)==20 and sum(meta[int(i)]['no_food'] for i in row)==10 for row in schedule[1000:])
 resume=torch.load(HERE/'TRAIN/RESUME_STATE.pt',map_location='cpu',weights_only=False)
 checks['resume_model_exact']=exact(resume['state_dict'],parent['state_dict']);checks['resume_optimizer_only_lr']=exact(resume['optimizer']['state'],parent['optimizer']['state']) and resume['optimizer']['param_groups'][0]['lr']==.0003
 checks['resume_rng_curve_exact']=exact(resume['rng'],parent['rng']) and exact(resume['curve'],parent['curve'])
 with np.load(PREVIOUS/'train.npz',allow_pickle=False) as z:x=torch.tensor(z['receptors'][:3].transpose(1,0,2).copy())
 with torch.no_grad():a,h=production(m,x)
 clone,_=load_behavior(path,sha,require_final=False)
 with torch.no_grad():b,hb=production(clone,x);cmd,_=m.step(x[0],m.initial_state(3))
 checks['production_inference_reload_exact']=torch.equal(a,b) and torch.equal(h,hb);checks['forward_structuralzero']=torch.equal(cmd[:,0],torch.zeros_like(cmd[:,0]));checks['scope_not_deployment']=p['scope_tag']=='SENSOR_PRETRAIN_ONLY' and p['deployment_qualified'] is False
 checks['checkpoint_unchanged']=digest(path)==sha;guard()
 r=dict(all_passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},step=step,checkpoint=str(path),sha256=sha,physical_steps=0,optimizer_updates=0)
 if not r['all_passed']:raise ValueError('saved balanced verification failed '+json.dumps(r))
 return r
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);a.add_argument('--sha256',required=True);a.add_argument('--out',required=True);args=a.parse_args();r=verify(args.checkpoint,args.sha256);save_json(args.out,r);print(json.dumps(dict(all_passed=True,step=r['step'],checks=len(r['checks']))))
