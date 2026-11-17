"""Parent-launched one candidate: behavior-only full1920 BPTT, fixed1000 updates."""
import time,subprocess
from common import *
from data import training_data
from engine import make,production,objective

def main():
 cpu1();guard();frozen=freeze_proof();pre=read(HERE/'PREFLIGHT.json')
 if not pre['all_passed']:raise ValueError('preflight required')
 out=HERE/'TRAIN'
 if out.exists():raise FileExistsError('No blindtrainingrestart/overwrite')
 torch.manual_seed(20261111);m,parent=initialize();m.train();opt=torch.optim.Adam(m.parameters(),lr=1e-4);generator=np.random.Generator(np.random.PCG64(20261111));data=training_data();scan=make(m);out.mkdir();started=time.monotonic();curve=[];schedule=[]
 sources=dict(frozen['source_hashes']);sources[str(HERE/'FROZEN.json')]=sha(HERE/'FROZEN.json');sources[str(HERE/'PREFLIGHT.json')]=sha(HERE/'PREFLIGHT.json')
 config=dict(kind=FORMAT,parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,optimizer='freshAdamALL8',lr=.0001,steps=1000,physical_eval_only_step=1000,source_hashes=sources,data_hashes={str(HERE/'train.npz'):sha(HERE/'train.npz'),str(HERE/'validation.npz'):sha(HERE/'validation.npz'),str(HERE/'DATASET.json'):sha(HERE/'DATASET.json')},batch_size=4,loss='equal on/neutral per episode actionMSE+.01latentMSE, full1920BPTT',alpha_unchanged=True,architecture_unchanged=True,no_runtime_gate_or_reset=True,motor_sha256=MOTOR_SHA,motor_model_loads=0,seed=20261111)
 with torch.no_grad():pred,latent,hidden=scan(data['receptors']);initial_loss=float(objective(pred,latent,data['targets'],data['weights']))
 curve.append(dict(step=0,actual_train_objective=initial_loss,wall_s=0.))
 def checkpoint(step):
  verify_hashes(sources);guard()
  with torch.no_grad():pred,latent,hidden=production(m,data['receptors'])
  rng=dict(torch=torch.get_rng_state(),numpy_generator=generator.bit_generator.state)
  payload=dict(format_version=FORMAT,kind='food_steering_behavior_long_history',scope_tag='BEHAVIOR_LONG_HISTORY_ONLY',deployment_qualified=False,parent_sha256=PARENT_SHA,identity=m.identity(),state_dict=m.state_dict(),optimizer=opt.state_dict(),rng=rng,curve=curve,batch_schedule=schedule,config=config,replay_train=dict(turn=pred,latent=latent,hidden=hidden))
  path=out/f'checkpoint_{step:06d}.pt';tmp=Path(str(path)+'.tmp');torch.save(payload,tmp);tmp.replace(path)
  restored,p=load_checked(path,sha(path),require_final=False)
  checks=dict(model=exact(m.state_dict(),restored.state_dict()),Adam=exact(opt.state_dict(),p['optimizer']),RNG=exact(rng,p['rng']),curve=exact(curve,p['curve']),schedule=exact(schedule,p['batch_schedule']))
  if not all(checks.values()):raise ValueError('live saved mismatch')
  torch.set_rng_state(rng['torch']);generator.bit_generator.state=rng['numpy_generator']
  save(out/f'LIVE_SAVED_{step:06d}.json',dict(all_exact=True,checks=checks,step=step,sha256=sha(path)))
  print(json.dumps(dict(saved=str(path),step=step,sha256=sha(path))),flush=True)
  del restored,p,payload,pred,latent,hidden
 checkpoint(0)
 with (out/'CURVE.jsonl').open('x') as stream:
  stream.write(json.dumps(curve[0])+'\n');stream.flush()
  for step in range(1,1001):
   if time.monotonic()-started>7200:raise TimeoutError('registered2hwallcap')
   indices=generator.choice(24,4,replace=False);schedule.append(indices.tolist());opt.zero_grad(set_to_none=True);pred,latent,hidden=scan(data['receptors'][:,indices]);loss=objective(pred,latent,data['targets'][:,indices],data['weights'][:,indices]);loss.backward();grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
   if not torch.isfinite(grad):raise ValueError('nonfinite gradient')
   opt.step();row=dict(step=step,batch_indices=indices.tolist(),batch_objective_before_update=float(loss.detach()),gradient_norm=float(grad),wall_s=time.monotonic()-started);curve.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
   if step%100==0 or step==1:print(json.dumps(row),flush=True)
   if step in (500,1000):checkpoint(step)
 # Independent audits run AFTER training exits under parent scheduling: no overlap model processes.
 save(out/'RESULT.json',dict(status='TRAINED_NOT_VALIDATED_OR_PHYSICALLY_EVALUATED',steps=1000,checkpoint=str(out/'checkpoint_001000.pt'),sha256=sha(out/'checkpoint_001000.pt'),initial_objective=initial_loss,wall_s=time.monotonic()-started,motor_model_loads=0,Core_constructions=0,optimizer_updates=1000,validation_status='NOT_RUN_BY_TRAINER',physical_status='NOT_RUN'))
 guard()
if __name__=='__main__':main()
