"""Parent finite250 eightNNparameter repair, two micro8 gradients per update; no runtimegate."""
from common import *
def main():
 torch,np=backend();f=frozen();out=HERE/'TRAIN';out.mkdir(exist_ok=False)
 from data import arrays,tensors
 from model import production,make_scan,objective,accumulated_backward
 import time,resource
 d=arrays();z,y,w=tensors(d);torch.manual_seed(20261113);m,parent=initialize();reference=anchors(m);scan=make_scan(m);opt=torch.optim.Adam(m.parameters(),lr=1e-4);curve=[];start=time.monotonic();j=11
 def store(step):
  with torch.no_grad():replay=production(m,z[:,j:j+1]);prediction=scan(z);loss,terms=objective(prediction,y,w,m,reference)
  if len(curve)==step:curve.append(dict(step=step,objective=float(loss),**{k:float(v) for k,v in terms.items()}))
  path=out/('checkpoint_'+str(step).zfill(6)+'.pt');payload=dict(format_version=FORMAT,step=step,parent_sha256=PARENT_SHA,state_dict=m.state_dict(),identity=m.identity(),repair_lineage=dict(kind='learnedneuraloffsetrepair',parent_sha256=PARENT_SHA,source_class_version=m.VERSION,source_ALPHA=float(m.ALPHA),no_hidden_reset=True,no_runtime_receptor_gate=True),optimizer=opt.state_dict(),curve=list(curve),source_hashes=f['source_hashes'],data_hashes=f['data_hashes'],torch_rng_state=torch.get_rng_state(),replay_index=j,replay=[v.clone() for v in replay],train_prediction=prediction[0].clone(),parameter_names=[k for k,p in m.named_parameters()],effective_batch=16,microbatch=8,accumulation=2,history_steps=1920,seed=20261113,lr=1e-4,no_input_noise_RNG=True,food_arrival_qualified=False,new_body_verified=False,deployment_qualified=False)
  with path.open('xb') as fp:torch.save(payload,fp)
  print(json.dumps(dict(step=step,saved=str(path),sha256=sha(path))),flush=True)
 store(0)
 for step in range(1,251):
  opt.zero_grad(set_to_none=True);loss,terms=accumulated_backward(scan,m,z,y,w,reference)
  if not __import__('math').isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in m.parameters()):raise ValueError('Finiteeightparametergradients required')
  grad=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.));opt.step();curve.append(dict(step=step,objective_before_update=loss,gradient_norm=grad,microbatches=2,optimizer_steps_this_update=1,**terms));rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
  if time.monotonic()-start>1800 or rss>1800:raise RuntimeError('Fixed250wall/RSS exceeded; noextension/nobatchreduction')
  if step%25==0:print(json.dumps(dict(step=step,seconds=time.monotonic()-start,peak_RSS_MiB=rss)),flush=True)
  if step in (125,250):store(step)
 final=out/'checkpoint_000250.pt';verify(f['source_hashes']);verify(f['data_hashes']);save(out/'RESULT.json',dict(status='FIXED250_ODOR_NEURAL_MEMORY_REPAIR_TRAINED_NOT_BODY_VALIDATED',updates=250,checkpoint=str(final),sha256=sha(final),parent_sha256=PARENT_SHA,seconds=time.monotonic()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,effective_batch=16,microbatch=8,trainable_tensors=8,source_ALPHA=float(m.ALPHA),sourceidentity_unchanged=True,no_hidden_reset=True,new_body_physics_steps=0))
if __name__=='__main__':raise SystemExit('Use parent explicitflag run.py')
