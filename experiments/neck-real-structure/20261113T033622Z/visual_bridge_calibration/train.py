"""Parent fixed300/full1920/B9; train only ten bridge parameters, not frozen vision/feedback."""
from common import *
def main():
 torch,np=backend();f=frozen();out=HERE/'TRAIN';out.mkdir(exist_ok=False)
 from data import arrays,tensors
 from model import python_rollout,make_scan,objective
 import time,resource
 d=arrays('train');x,u,y,w=tensors(d);torch.manual_seed(20261113);m,parent=initialize();fixed=frozen_tensors(m);ref=anchors(parent);scan=make_scan(m);params=[v for k,v in trainable(m)];opt=torch.optim.Adam(params,lr=3e-4);curve=[];start=time.monotonic();j=2
 def scalar(v):return float(v.detach())
 def store(step):
  with torch.no_grad():replay=python_rollout(m,x[:,j:j+1],u[:,j:j+1]);prediction=scan(x,u);loss,terms=objective(prediction,y,w,m,ref)
  if len(curve)==step:curve.append(dict(step=step,objective=scalar(loss),**{k:scalar(v) for k,v in terms.items()}))
  path=out/('checkpoint_'+str(step).zfill(6)+'.pt');payload=dict(format_version=FORMAT,parent_sha256=PARENT_SHA,visual_sha256=VISION_SHA,source_binding_sha256=NECK_SHA,step=step,state_dict=m.state_dict(),identity=m.identity(),optimizer=opt.state_dict(),curve=list(curve),source_hashes=f['source_hashes'],data_hashes=f['data_hashes'],torch_rng_state=torch.get_rng_state(),replay_index=j,replay=[v.clone() for v in replay],train_prediction=prediction[0].clone(),trainable_names=[k for k,v in trainable(m)],frozen_tensor_hashes={k:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for k,v in fixed.items()},batch_size=9,history_steps=1920,seed=20261113,lr=3e-4,full_neck_training_allowed=False,new_body_verified=False,training_target='causal_delayed_frozen_actualRGB_visualrawturn',no_runtime_gate_or_signal_rescaling=True)
  with path.open('xb') as fp:torch.save(payload,fp)
  print(json.dumps(dict(step=step,saved=str(path),sha256=sha(path))),flush=True)
 store(0)
 for step in range(1,301):
  opt.zero_grad(set_to_none=True);prediction=scan(x,u);loss,terms=objective(prediction,y,w,m,ref);loss.backward()
  if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):raise ValueError('Nonfinite tenparametergradient')
  if any(p.grad is not None for p in m.parameters() if not p.requires_grad):raise ValueError('Frozenparametergradient')
  grad=float(torch.nn.utils.clip_grad_norm_(params,1.));opt.step();check_frozen(m,fixed);curve.append(dict(step=step,objective_before_update=scalar(loss),gradient_norm=grad,**{k:scalar(v) for k,v in terms.items()}));del prediction,loss,terms
  rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
  if time.monotonic()-start>1800 or rss>1800:raise RuntimeError('Fixed300 wall/RSS exceeded; no extension/batchreduction')
  if step%25==0:print(json.dumps(dict(step=step,seconds=time.monotonic()-start,peak_RSS_MiB=rss)),flush=True)
  if step in (150,300):store(step)
 final=out/'checkpoint_000300.pt';verify(f['source_hashes']);verify(f['data_hashes']);save(out/'RESULT.json',dict(status='FIXED300_VISUAL_SCALAR_CALIBRATED_NOT_NEW_BODY',updates=300,checkpoint=str(final),sha256=sha(final),parent_sha256=PARENT_SHA,visual_sha256=VISION_SHA,seconds=time.monotonic()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,trainable_tensors=10,frozen_feedback_reconstruction_normalizer_unchanged=True,full_neck_training_allowed=False,new_body_physics_steps=0))
if __name__=='__main__':raise SystemExit('Use explicitflag run.py')
