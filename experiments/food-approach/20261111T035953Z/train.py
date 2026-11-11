"""Exact1000->3000 balanced sensory correction; fresh test evaluated only later."""
from copy import deepcopy
import json,time,sys,subprocess
import numpy as np,torch
from common import HERE,PREVIOUS,PARENT,PARENT_SHA,legacy_learning,make_scan,exact,digest,save_json,verify_hashes,guard,load_behavior

def main():
 torch.set_num_threads(1);guard();pre=json.loads((HERE/'PREFLIGHT.json').read_text())
 if not pre['all_passed']:raise ValueError('balanced preflight required')
 verify_hashes(pre['source_hashes']);verify_hashes(pre['data_hashes'])
 m,parent=legacy_learning.load_behavior(PARENT,PARENT_SHA);opt=torch.optim.Adam(m.parameters(),lr=.001);opt.load_state_dict(parent['optimizer'])
 if not exact(m.state_dict(),parent['state_dict']) or not exact(opt.state_dict(),parent['optimizer']):raise ValueError('parent resume mismatch')
 sources=dict(parent['config']['source_hashes']);sources.update({str(HERE/n):digest(HERE/n) for n in ('common.py','generate_test.py','check_resume.py','PREFLIGHT.json','verify_saved.py','train.py','PREREG.json','METRIC_CLARIFICATION_BEFORE_TRAINING.json')})
 data=dict(parent['config']['data_hashes']);data.update({str(f):digest(f) for f in (PARENT,HERE/'fresh_test.npz',HERE/'FRESH_TEST.json')})
 config=deepcopy(parent['config']);config.update(kind='balanced_sensor_pretrain3000',steps=3000,additional_updates=2000,lr=.0003,parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,source_hashes=sources,data_hashes=data,batch_composition='10 existing food episodes +10 existing no_food episodes, both sampled withreplacement; all20 shuffled byrecordedRNG',old_heldout_status='DEVELOPMENT_VALIDATION_HISTORY',fresh_test_status='FROZEN_BEFORE_UPDATES_NOT_EVALUATED_DURING_TRAINING',metric_clarification=json.loads((HERE/'METRIC_CLARIFICATION_BEFORE_TRAINING.json').read_text()))
 meta=json.loads((PREVIOUS/'DATASET.json').read_text())['splits']['train']['episodes_config'];food=np.asarray([i for i,c in enumerate(meta) if not c['no_food']]);zero=np.asarray([i for i,c in enumerate(meta) if c['no_food']])
 with np.load(PREVIOUS/'train.npz',allow_pickle=False) as z:x=torch.tensor(z['receptors'].transpose(1,0,2).copy());y=torch.tensor(z['teacher_turn'].T.copy())
 scan=make_scan(m);curve=deepcopy(parent['curve']);schedule=deepcopy(parent['batch_schedule']);oldopt=deepcopy(parent['optimizer']);opt.param_groups[0]['lr']=.0003;oldopt['param_groups'][0]['lr']=.0003
 if not exact(opt.state_dict(),oldopt):raise ValueError('more than lr changed')
 torch.set_rng_state(parent['rng']['torch']);np.random.set_state(parent['rng']['numpy']);rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state())
 if not exact(rng,parent['rng']):raise ValueError('RNG resume failed')
 out=HERE/'TRAIN';out.mkdir(exist_ok=False)
 torch.save(dict(state_dict=m.state_dict(),optimizer=opt.state_dict(),rng=rng,curve=curve,batch_schedule=schedule,config=config),out/'RESUME_STATE.pt')
 save_json(out/'RESUME_CHECKS.json',dict(model_exact=True,optimizer_moments_exact=True,only_lr_changed=True,rng_exact=True,curve_prefix_exact=exact(curve,parent['curve']),actual_counters=[float(v['step']) for v in opt.state.values()],old_lr=.001,new_lr=.0003,resume_state_sha256=digest(out/'RESUME_STATE.pt')))
 save_json(out/'IDENTITY.json',config);started=time.monotonic()
 with (out/'TRAINING_ADDITIONAL.jsonl').open('x') as stream:
  for step in range(1001,3001):
   idx=np.r_[np.random.choice(food,10,replace=True),np.random.choice(zero,10,replace=True)];idx=idx[np.random.permutation(20)];schedule.append(idx.tolist())
   opt.zero_grad(set_to_none=True);pred,_=scan(x[:,idx]);loss=(pred-y[:,idx]).square().mean();loss.backward();grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
   if not torch.isfinite(grad):raise ValueError('nonfinite gradient')
   opt.step();row=dict(step=step,batch_MSE_before_update=float(loss.detach()),gradient_norm=float(grad),wall_s=time.monotonic()-started,stage='balanced_nofood_correction');curve.append(row)
   if step%100==0:
    with torch.no_grad():pred,_=scan(x);row['oldtrain_uniform_MSE_after_update']=float((pred-y).square().mean());row['oldtrain_nofood_RMS_after_update']=float(pred[:,zero].square().mean().sqrt())
    print(json.dumps(row),flush=True)
   stream.write(json.dumps(row)+'\n');stream.flush()
   if step in (2000,3000):
    verify_hashes(sources);verify_hashes(data);rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state());ck=out/f'checkpoint_{step:06d}.pt'
    payload=dict(format_version=1,kind='food_steering_behavior',scope_tag='SENSOR_PRETRAIN_ONLY',deployment_qualified=False,identity=m.identity(),state_dict=m.state_dict(),optimizer=opt.state_dict(),rng=rng,curve=curve,batch_schedule=schedule,config=config)
    torch.save(payload,ck);sha=digest(ck);clone,p=load_behavior(ck,sha,require_final=False)
    if not exact(m.state_dict(),clone.state_dict()) or not exact(opt.state_dict(),p['optimizer']) or not exact(rng,p['rng']) or not exact(curve,p['curve']):raise ValueError('live saved exactness failed')
    torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy'])
    subprocess.run([sys.executable,'-B',str(HERE/'verify_saved.py'),'--checkpoint',str(ck),'--sha256',sha,'--out',str(out/f'VERIFY_{step:06d}.json')],check=True)
    print(json.dumps(dict(checkpoint=str(ck),sha256=sha,step=step,live_full_saved_state_exact=True)),flush=True)
 result=dict(status='BALANCED_SENSOR_PRETRAIN_ONLY_NOT_PHYSICAL_APPROACH',steps=3000,additional_updates=2000,checkpoint=str(ck),sha256=sha,wall_s=time.monotonic()-started,parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,physical_steps=0,Core_constructions=0,fresh_test_evaluation='NOT_RUN_BY_TRAINER',deployment_qualified=False)
 guard();save_json(out/'RESULT.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
