"""Exact parent500->1500 original-two-only budget-matched control; no physical run."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
from model_io import HERE,PARENT,PARENT_SHA,load_parent,load_checked,save,sha256,verify_hashes,exact
from data import prepare,padded,neutral
from engine import make

def main(args):
 torch.set_num_threads(1)
 pre=json.loads((HERE/'PREFLIGHT.json').read_text())
 if not pre['all_passed']:raise ValueError('complete neural/gradient preflight required')
 verify_hashes(pre['source_hashes']);verify_hashes(pre['data_hashes'])
 model,parent=load_parent();opt=torch.optim.Adam(model.adapter_parameters(),lr=.001);opt.load_state_dict(parent['optimizer'])
 if not exact(model.state_dict(),parent['state_dict']) or not exact(opt.state_dict(),parent['optimizer']):raise ValueError('actual parent model/optimizer resume differs')
 if len(opt.state)!=8 or not all(float(s['step'])==500 for s in opt.state.values()):raise ValueError('eight actual Adam500 states required')
 episodes,data=prepare(model);p=padded(episodes);engine=make(model)
 sources=dict(parent['config']['source_hashes']);sources.update({str(HERE/n):sha256(HERE/n) for n in ('PREREG.json','model_io.py','data.py','engine.py','check_train.py','PREFLIGHT.json','continue_train.py','verify_saved.py')})
 amendment=HERE.parent/'TWO_TRAJECTORY_CONTROL_AMENDMENT.json';sources[str(amendment)]=sha256(amendment)
 # Parent source/data identities are original-two-only; retain ancestry but never add corrective paths.
 data.update(parent['config']['data_hashes']);verify_hashes(data)
 if any('/recovery_data/' in path or 'COLLECT_' in path for path in data):raise ValueError('corrective artifact entered original-two-only identity')
 out=Path(args.out).resolve()
 if not out.is_relative_to(HERE):raise ValueError('new output strictly within control scope required')
 out.mkdir(parents=True,exist_ok=False)
 curve=deepcopy(parent['config']['curve']);old=deepcopy(opt.state_dict())
 if old['param_groups'][0]['lr']!=.001:raise ValueError('parent lr mismatch')
 old['param_groups'][0]['lr']=.0003;opt.param_groups[0]['lr']=.0003
 if not exact(opt.state_dict(),old):raise ValueError('more than declared lr changed')
 torch.set_rng_state(parent['config']['rng']['torch']);np.random.set_state(parent['config']['rng']['numpy'])
 rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state())
 if not exact(rng,parent['config']['rng']):raise ValueError('actual RNG restore failed')
 config=dict(kind='original_two_only_budget_matched_control',parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,start_step=500,end_step=1500,physical_eval_only_step=1500,lr=.0003,source_hashes=sources,data_hashes=data,prereg=json.loads((HERE/'PREREG.json').read_text()),episode_names=[e['name'] for e in episodes],episode_lengths=[2880,2880],loss_episode_count=2,corrective_loss_episodes=0,loss='two equal contiguous actual original teacher postprime boundedaction mean losses',corrective_data_read=False)
 torch.save(dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),curve=curve,rng=rng,config=config),out/'RESUME_STATE.pt')
 with torch.no_grad():initial,_=engine(p['features'],p['gate'],p['means'],p['valid'])
 initial_metrics={e['name']:float((initial[960:,j].tanh()-e['y'][960:]).square().mean()) for j,e in enumerate(episodes)}
 handover=dict(model_exact=exact(model.state_dict(),parent['state_dict']),optimizer_only_lr_changed=exact(opt.state_dict(),old),rng_exact=exact(rng,parent['config']['rng']),curve_prefix_exact=exact(curve,parent['config']['curve']),old_lr=.001,new_lr=.0003,adam_states=8,all_counters=[float(s['step']) for s in opt.state.values()],initial_two_objective_metrics=initial_metrics,loss_episode_count=2,corrective_loss_episodes=0,resume_state_sha256=sha256(out/'RESUME_STATE.pt'))
 (out/'RESUME_CHECKS.json').write_text(json.dumps(handover,indent=2)+'\n');(out/'IDENTITY.json').write_text(json.dumps(config,indent=2)+'\n')
 started=time.monotonic();print(json.dumps(dict(step=500,initial_metrics=initial_metrics,preflight_batch_forward_backward_s=pre['batch_forward_backward_s'])),flush=True)
 with (out/'TRAINING_ADDITIONAL.jsonl').open('x') as stream:
  for step in range(501,1501):
   if time.monotonic()-started>14400:raise TimeoutError('fixed4h wall cap exceeded')
   opt.zero_grad(set_to_none=True);pred,_=engine(p['features'],p['gate'],p['means'],p['valid'])
   loss=((pred.tanh()-p['targets']).square()*p['weights']).sum();loss.backward()
   grad=torch.nn.utils.clip_grad_norm_(model.adapter_parameters(),1.)
   if not torch.isfinite(grad):raise ValueError('nonfinite adapter gradient')
   if any(q.requires_grad or q.grad is not None for q in model.base.parameters()):raise ValueError('frozen base changed')
   opt.step();row=dict(step=step,loss_before_update=float(loss.detach()),gradient_norm=float(grad),wall_s=time.monotonic()-started,stage='original_two_only_matched_control')
   if step%100==0 or step in (501,1500):
    with torch.no_grad():after,_=engine(p['features'],p['gate'],p['means'],p['valid'])
    row['metrics_after_update']={e['name']:float((after[960:,j].tanh()-e['y'][960:]).square().mean()) for j,e in enumerate(episodes)}
    print(json.dumps(row),flush=True)
   if step in (1000,1500):row['neutral']=neutral(model)
   curve.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
   if step in (1000,1500):
    verify_hashes(sources);verify_hashes(data);rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state())
    ck=out/f'checkpoint_{step:06d}.pt';digest=save(ck,model,opt,config,curve,rng)
    restored,payload=load_checked(ck,digest,require_final=False)
    if not exact(model.state_dict(),restored.state_dict()) or not exact(opt.state_dict(),payload['optimizer']) or not exact(curve,payload['curve']) or not exact(rng,payload['rng']):raise ValueError('live to saved state mismatch')
    torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy'])
    subprocess.run([sys.executable,'-B',str(HERE/'verify_saved.py'),'--checkpoint',str(ck),'--sha256',digest,'--out',str(out/f'VERIFY_{step:06d}.json')],check=True)
    print(json.dumps(dict(step=step,checkpoint=str(ck),sha256=digest,saved_independent_pass=True)),flush=True)
 result=dict(status='TRAINED_NOT_PHYSICALLY_EVALUATED',steps=1500,additional_updates=1000,checkpoint=str(ck),sha256=digest,wall_s=time.monotonic()-started,final=curve[-1],physical_steps=0,loss_episode_count=2,corrective_loss_episodes=0,source_hashes=sources,data_hashes=data)
 (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--out',required=True);main(a.parse_args())
