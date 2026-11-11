"""Fixed1000 sensory-response BC. Heldout data is NOT evaluated here."""
import json,time,subprocess,sys
from pathlib import Path
import numpy as np,torch
from common import HERE,SteeringBehavior,BEHAVIOR,FIELD,GEOMETRY,digest,verify_hashes,save_json,guard
from learning import make_scan,load_behavior,exact


def train():
 torch.set_num_threads(1);torch.manual_seed(20261111);np.random.seed(20261111);guard()
 pre=json.loads((HERE/'PREFLIGHT.json').read_text())
 if not pre['all_passed']:raise ValueError('preflight required')
 verify_hashes(pre['source_hashes']);verify_hashes(pre['data_hashes'])
 names=['common.py','generate_data.py','learning.py','check_preflight.py','verify_saved.py','train.py','PREREG.json','PATH_RESOLUTION_AMENDMENT.json','CLOCK_CHECK_CLARIFICATION.json','PREFLIGHT.json']
 sources={str(HERE/n):digest(HERE/n) for n in names};sources.update({str(p):digest(p) for p in (BEHAVIOR,FIELD,GEOMETRY)})
 data={str(HERE/n):digest(HERE/n) for n in ('DATASET.json','train.npz','heldout.npz')}
 config=dict(kind='SENSOR_PRETRAIN_ONLY',steps=1000,lr=.001,batch_size=20,seed=20261111,source_hashes=sources,data_hashes=data,
             actor_input='Only receptor2, explicit state32; no bearing/time/position/sourceID',training='synthetic static audited-head engineering labels; NO PHYSICS',
             immutable_class_identity_note='identity.training is original frozen-source declaration, not actual training status; scope_tag/steps and deployment_qualified govern this artifact')
 with np.load(HERE/'train.npz',allow_pickle=False) as z:x=torch.tensor(z['receptors'].transpose(1,0,2).copy());y=torch.tensor(z['teacher_turn'].T.copy())
 model=SteeringBehavior();scan=make_scan(model);opt=torch.optim.Adam(model.parameters(),lr=.001)
 out=HERE/'TRAIN';out.mkdir(exist_ok=False);save_json(out/'IDENTITY.json',config)
 started=time.monotonic();curve=[];schedule=[]
 with torch.no_grad():pred,_=scan(x);initial=float((pred-y).square().mean())
 curve.append(dict(step=0,train_MSE=initial,wall_s=0.));print(json.dumps(curve[-1]),flush=True)
 with (out/'TRAINING.jsonl').open('x') as stream:
  stream.write(json.dumps(curve[-1])+'\n')
  for step in range(1,1001):
   idx=np.random.choice(x.shape[1],20,replace=False);schedule.append(idx.tolist())
   opt.zero_grad(set_to_none=True);pred,_=scan(x[:,idx]);loss=(pred-y[:,idx]).square().mean();loss.backward()
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
   if not torch.isfinite(norm):raise ValueError('nonfinite gradient')
   opt.step();row=dict(step=step,batch_MSE_before_update=float(loss.detach()),gradient_norm=float(norm),wall_s=time.monotonic()-started)
   curve.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
   if step%100==0:
    with torch.no_grad():p,_=scan(x);row['train_MSE_after_update']=float((p-y).square().mean())
    print(json.dumps(row),flush=True)
   if step in (500,1000):
    verify_hashes(sources);verify_hashes(data);rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state())
    ck=out/f'checkpoint_{step:06d}.pt'
    payload=dict(format_version=1,kind='food_steering_behavior',scope_tag='SENSOR_PRETRAIN_ONLY',deployment_qualified=False,identity=model.identity(),state_dict=model.state_dict(),optimizer=opt.state_dict(),rng=rng,curve=curve,batch_schedule=schedule,config=config)
    torch.save(payload,ck);sha=digest(ck);new,p=load_behavior(ck,sha)
    if not exact(model.state_dict(),new.state_dict()) or not exact(opt.state_dict(),p['optimizer']) or not exact(rng,p['rng']) or not exact(curve,p['curve']):raise ValueError('live saved state mismatch')
    torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy'])
    subprocess.run([sys.executable,'-B',str(HERE/'verify_saved.py'),'--checkpoint',str(ck),'--sha256',sha,'--out',str(out/f'VERIFY_{step:06d}.json')],check=True)
    print(json.dumps(dict(checkpoint=str(ck),sha256=sha,step=step,live_saved_state_exact=True)),flush=True)
 result=dict(status='SENSOR_PRETRAIN_ONLY_NOT_PHYSICAL_APPROACH',steps=1000,checkpoint=str(ck),sha256=sha,train_initial_MSE=initial,train_final_MSE=curve[-1]['train_MSE_after_update'],wall_s=time.monotonic()-started,physical_steps=0,Core_constructions=0,heldout_evaluation='NOT_RUN_BY_TRAINER',deployment_qualified=False)
 guard();save_json(out/'RESULT.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':train()
