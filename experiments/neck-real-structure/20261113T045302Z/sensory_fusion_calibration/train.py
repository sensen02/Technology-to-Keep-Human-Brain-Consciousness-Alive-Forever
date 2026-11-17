"""Parent finite400, effective27 via3micro9 fullhistory; no runtime fusion controller."""
from common import *
def main():
 torch,np=backend();f=frozen();out=HERE/'TRAIN';out.mkdir(exist_ok=False)
 from data import arrays,tensors
 from model import make_scan,python_rollout,objective,accumulated_backward
 import time,resource
 d=arrays('train');x,u,y,w=tensors(d);torch.manual_seed(20261113);m,parent,old=initialize();fixed=frozen_tensors(m);ref=anchors(m);scan=make_scan(m);params=[v for k,v in trainable(m)];opt=torch.optim.Adam(params,lr=3e-4);curve=[];start=time.monotonic();j=20
 def store(step):
  with torch.no_grad():replay=python_rollout(m,x[:,j:j+1],u[:,j:j+1]);prediction=scan(x,u);loss,terms=objective(prediction,y,w,m,ref)
  if len(curve)==step:curve.append(dict(step=step,objective=float(loss),**{k:float(v) for k,v in terms.items()}))
  p=out/('checkpoint_'+str(step).zfill(6)+'.pt');payload=dict(format_version=FORMAT,step=step,parent_sha256=PARENT_SHA,visual_sha256=VISION_SHA,odor_sha256=ODOR_SHA,source_binding_sha256=NECK_SHA,state_dict=m.state_dict(),identity=m.identity(),optimizer=opt.state_dict(),curve=list(curve),source_hashes=f['source_hashes'],data_hashes=f['data_hashes'],torch_rng_state=torch.get_rng_state(),replay_index=j,replay=[v.clone() for v in replay],train_prediction=prediction[0].clone(),trainable_names=[k for k,v in trainable(m)],frozen_tensor_hashes={k:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for k,v in fixed.items()},effective_batch=27,microbatch=9,accumulation=3,history_steps=1920,seed=20261113,lr=3e-4,parent_weight_ih_column_mapping='first2exactcopy thirdzero atstep0',full_neck_training_allowed=False,new_body_verified=False,runtime_fusion_rule=False)
  with p.open('xb') as fp:torch.save(payload,fp)
  print(json.dumps(dict(step=step,saved=str(p),sha256=sha(p))),flush=True)
 store(0)
 for step in range(1,401):
  opt.zero_grad(set_to_none=True);loss,terms=accumulated_backward(scan,m,x,u,y,w,ref)
  if not __import__('math').isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):raise ValueError('Nonfinitefusiongradient')
  if any(p.grad is not None for p in m.parameters() if not p.requires_grad):raise ValueError('Frozenparametergradient')
  grad=float(torch.nn.utils.clip_grad_norm_(params,1.));opt.step();check_frozen(m,fixed);curve.append(dict(step=step,objective_before_update=loss,gradient_norm=grad,microbatches=3,optimizer_steps_this_update=1,**terms));rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
  if time.monotonic()-start>1800 or rss>1800:raise RuntimeError('Fixed400 wall/RSS exceeded; no extension or batch reduction')
  if step%25==0:print(json.dumps(dict(step=step,seconds=time.monotonic()-start,peak_RSS_MiB=rss)),flush=True)
  if step in (200,400):store(step)
 final=out/'checkpoint_000400.pt';verify(f['source_hashes']);verify(f['data_hashes']);save(out/'RESULT.json',dict(status='FIXED400_SENSORY_FUSION_CALIBRATED_NOT_PHYSICAL_VALIDATED',updates=400,checkpoint=str(final),sha256=sha(final),parent_sha256=PARENT_SHA,visual_sha256=VISION_SHA,odor_sha256=ODOR_SHA,seconds=time.monotonic()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,effective_batch=27,microbatch=9,trainable_tensors=10,frozen_feedback_reconstruction_normalizer_exact=True,full_neck_training_allowed=False,new_body_physics_steps=0))
if __name__=='__main__':raise SystemExit('Use explicitflag run.py')
