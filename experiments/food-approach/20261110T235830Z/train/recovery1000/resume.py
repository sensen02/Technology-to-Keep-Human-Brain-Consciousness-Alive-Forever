"""Recover interrupted500->1500 trial ONLY from last validated1000 weights."""
from pathlib import Path
import sys,json,time,subprocess
from copy import deepcopy
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;sys.path.insert(0,str(ROOT))
from model_io import load_checked,save,sha256,verify_hashes,exact
from data import prepare,padded,neutral
from engine import make
P1000=ROOT/'TRAIN/checkpoint_001000.pt';SHA1000='47a84f5058654f76624f897fa1d90580b2733739df64e99c234a25468260cd59'

def main():
 torch.set_num_threads(1)
 if (HERE/'RESULT.json').exists() or (HERE/'RESUME_STATE.pt').exists() or (ROOT/'TRAIN/checkpoint_001500.pt').exists():raise FileExistsError('fresh recovery required')
 model,parent=load_checked(P1000,SHA1000,require_final=False)
 if parent['curve'][-1]['step']!=1000 or len(parent['curve'])!=1001:raise ValueError('validated1000 required')
 opt=torch.optim.Adam(model.adapter_parameters(),lr=.0003);opt.load_state_dict(parent['optimizer'])
 if len(opt.state)!=8 or not all(float(v['step'])==1000 for v in opt.state.values()) or opt.param_groups[0]['lr']!=.0003:raise ValueError('eight Adam1000 unchanged lr required')
 episodes,data=prepare(model);batch=padded(episodes);engine=make(model)
 for filename,digest in data.items():
  if parent['config']['data_hashes'].get(filename)!=digest:raise ValueError('fourdata identity mismatch '+filename)
 config=deepcopy(parent['config']);sources=deepcopy(config['source_hashes'])
 sources.update({str(HERE/n):sha256(HERE/n) for n in ('AMENDMENT.json','resume.py','verify_recovery.py')})
 oldlog=ROOT/'TRAIN/TRAINING_ADDITIONAL.jsonl';rows=[json.loads(s) for s in oldlog.read_text().splitlines()]
 replay={r['step']:r for r in rows if r['step']>1000}
 if len(replay)!=99 or max(replay)!=1099:raise ValueError('interrupted log scope changed')
 data=deepcopy(config['data_hashes']);data[str(oldlog)]=sha256(oldlog);data[str(P1000)]=SHA1000
 config.update(source_hashes=sources,data_hashes=data,recovery_checkpoint=str(P1000),recovery_sha256=SHA1000,recovery_amendment=str(HERE/'AMENDMENT.json'),original_job_status='unknown236/no survivingprocess',remaining_updates=500,lost_unsaved_original_updates=99,replayed_original_steps=[1001,1099])
 curve=deepcopy(parent['curve']);torch.set_rng_state(parent['rng']['torch']);np.random.set_state(parent['rng']['numpy']);rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state())
 torch.save(dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),rng=rng,curve=curve,config=config),HERE/'RESUME_STATE.pt')
 with torch.no_grad():initial,_=engine(batch['features'],batch['gate'],batch['means'],batch['valid'])
 initial_metrics={e['name']:float((initial[960:e['length'],j].tanh()-e['y'][960:]).square().mean()) for j,e in enumerate(episodes)}
 checks=dict(model_exact=exact(model.state_dict(),parent['state_dict']),optimizer_exact=exact(opt.state_dict(),parent['optimizer']),rng_exact=exact(rng,parent['rng']),curve_exact=exact(curve,parent['curve']),same_lr=.0003,counters=[float(v['step']) for v in opt.state.values()],initial_metrics=initial_metrics,original_log_sha256=sha256(oldlog),lost_unsaved_rows=len(replay),resume_state_sha256=sha256(HERE/'RESUME_STATE.pt'))
 (HERE/'RESUME_CHECKS.json').write_text(json.dumps(checks,indent=2)+'\n');(HERE/'IDENTITY.json').write_text(json.dumps(config,indent=2)+'\n')
 if not all(checks[k] for k in ('model_exact','optimizer_exact','rng_exact','curve_exact')):raise ValueError('resume not exact')
 started=time.monotonic();diffs=[]
 print(json.dumps(dict(step=1000,resume_exact=True,lost_unsaved_rows=99,fixed_remaining_updates=500,initial_metrics=initial_metrics)),flush=True)
 with (HERE/'TRAINING_RECOVERY.jsonl').open('x') as stream:
  for step in range(1001,1501):
   if time.monotonic()-started>14400:raise TimeoutError('recovery wall cap4h exceeded')
   opt.zero_grad(set_to_none=True);pred,_=engine(batch['features'],batch['gate'],batch['means'],batch['valid']);loss=((pred.tanh()-batch['targets']).square()*batch['weights']).sum();loss.backward()
   grad=torch.nn.utils.clip_grad_norm_(model.adapter_parameters(),1.)
   if not torch.isfinite(grad) or any(p.requires_grad or p.grad is not None for p in model.base.parameters()):raise ValueError('gradient/freeze violated')
   opt.step();row=dict(step=step,loss_before_update=float(loss.detach()),gradient_norm=float(grad),wall_s=time.monotonic()-started,stage='corrective_slow_residual',recovery=True)
   if step in replay:
    error=max(abs(row[k]-replay[step][k]) for k in ('loss_before_update','gradient_norm'));diffs.append(error)
    if error>1e-12:raise ValueError('replayed validated-source loss/gradient differs '+str(error))
   if step%100==0 or step==1001:
    with torch.no_grad():after,_=engine(batch['features'],batch['gate'],batch['means'],batch['valid'])
    row['metrics_after_update']={e['name']:float((after[960:e['length'],j].tanh()-e['y'][960:]).square().mean()) for j,e in enumerate(episodes)}
    row['neutral']=neutral(model)
   curve.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
   if step%100==0 or step==1001:print(json.dumps(row),flush=True)
   if step%100==0:
    verify_hashes(sources);verify_hashes(data);rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state())
    ck=(ROOT/'TRAIN/checkpoint_001500.pt') if step==1500 else HERE/f'checkpoint_{step:06d}.pt';digest=save(ck,model,opt,config,curve,rng)
    pp=torch.load(ck,map_location='cpu',weights_only=False)
    if not exact(pp['state_dict'],model.state_dict()) or not exact(pp['optimizer'],opt.state_dict()) or not exact(pp['rng'],rng) or not exact(pp['curve'],curve):raise ValueError('live-to-saved differs')
    subprocess.run([sys.executable,'-B',str(HERE/'verify_recovery.py'),'--checkpoint',str(ck),'--sha256',digest,'--out',str(HERE/f'VERIFY_{step:06d}.json')],check=True)
    torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy'])
    print(json.dumps(dict(step=step,checkpoint=str(ck),sha256=digest,durability_only=step<1500)),flush=True)
 result=dict(status='RECOVERED_TRAINED_NOT_PHYSICALLY_EVALUATED',steps=1500,recovery_updates=500,prior_original_updates_logged=599,actual_production_optimizer_updates=1099,model_lineage_additional_updates=1000,replayed_unsaved_updates=99,replay_loss_gradient_max_error=max(diffs),checkpoint=str(ck),sha256=digest,wall_s=time.monotonic()-started,final=curve[-1],physical_steps=0,source_hashes=sources,data_hashes=data)
 (HERE/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 with (ROOT/'TRAIN/RESULT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')

if __name__=='__main__':main()
