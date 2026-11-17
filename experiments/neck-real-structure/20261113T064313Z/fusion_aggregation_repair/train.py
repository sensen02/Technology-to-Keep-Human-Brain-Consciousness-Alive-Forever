"""Parent fixed400 effective24/micro8x3, frozen sensory/feedback parameters, no body."""
from common import *
def main():
 torch,np=backend();f=frozen();out=HERE/'TRAIN';out.mkdir(exist_ok=False)
 from data import arrays,tensors
 from model import make_scan,python_rollout,objective,accumulated_backward
 import time,resource,math
 d=arrays('train');x,u,y,w=tensors(d);torch.manual_seed(20261113);m,parent=initialize();fixed=frozen_tensors(m);ref=anchors(m);scan=make_scan(m);params=[v for k,v in trainable(m)];opt=torch.optim.Adam(params,lr=3e-4);curve=[];start=time.monotonic();j=9;ob=odor_binding()
 def store(step):
  with torch.no_grad():replay=python_rollout(m,x[:,j:j+1],u[:,j:j+1]);prediction=scan(x,u);loss,terms=objective(prediction,y,w,m,ref)
  if len(curve)==step:curve.append(dict(step=step,objective=float(loss),**{k:float(v) for k,v in terms.items()}))
  path=out/('checkpoint_'+str(step).zfill(6)+'.pt');p=dict(format_version=FORMAT,step=step,parent_sha256=PARENT_SHA,visual_sha256=VISION_SHA,odor_sha256=ob['sha256'],source_binding_sha256=NECK_SHA,state_dict=m.state_dict(),identity=m.identity(),optimizer=opt.state_dict(),curve=list(curve),source_hashes=f['source_hashes'],data_hashes=f['data_hashes'],torch_rng_state=torch.get_rng_state(),replay_index=j,replay=[v.clone() for v in replay],train_prediction=prediction[0].clone(),trainable_names=[k for k,v in trainable(m)],frozen_tensor_hashes={k:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for k,v in fixed.items()},effective_batch=24,microbatch=8,accumulation=3,history_steps=1920,seed=20261113,lr=3e-4,full_neck_training_allowed=False,new_body_verified=False,runtime_fusion_rule=False,offline_jointteacher_bound=.78)
  with path.open('xb') as fp:torch.save(p,fp)
  print(json.dumps(dict(step=step,saved=str(path),sha256=sha(path))),flush=True)
 store(0)
 for step in range(1,401):
  opt.zero_grad(set_to_none=True);loss,terms=accumulated_backward(scan,m,x,u,y,w,ref)
  if not math.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):raise ValueError('Nonfinitefusionrepairgradient')
  if any(p.grad is not None for p in m.parameters() if not p.requires_grad):raise ValueError('Frozenparametergradient')
  grad=float(torch.nn.utils.clip_grad_norm_(params,1.));opt.step();check_frozen(m,fixed);curve.append(dict(step=step,objective_before_update=loss,gradient_norm=grad,microbatches=3,optimizer_steps_this_update=1,**terms));rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
  if time.monotonic()-start>1800 or rss>1800:raise RuntimeError('Fixed400wall/RSSexceeded;noextension/batchreduction')
  if step%25==0:print(json.dumps(dict(step=step,seconds=time.monotonic()-start,peak_RSS_MiB=rss)),flush=True)
  if step in (200,400):store(step)
 final=out/'checkpoint_000400.pt';verify(f['source_hashes']);verify(f['data_hashes']);save(out/'RESULT.json',dict(status='FIXED400_MULTISENSORY_AGGREGATION_REPAIRED_NOT_PHYSICAL_VALIDATED',updates=400,checkpoint=str(final),sha256=sha(final),parent_sha256=PARENT_SHA,visual_sha256=VISION_SHA,odor_sha256=ob['sha256'],seconds=time.monotonic()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,effective_batch=24,microbatch=8,trainable_tensors=10,frozen_feedback_recon_normalizer_exact=True,full_neck_training_allowed=False,new_body_physics_steps=0))
if __name__=='__main__':raise SystemExit('Use explicitflag run.py')
