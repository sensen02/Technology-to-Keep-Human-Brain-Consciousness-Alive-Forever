"""Read-only gradients: autograd only, no optimizer construction/update/physics."""
import sys,json
from pathlib import Path
import torch,numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from fast_motor import load,sha256
from fast_data import prepare,cached

def main():
 torch.set_num_threads(1);m,p=load(ROOT/'TRAIN_FAST_1000/checkpoint_001000.pt')
 episodes,_=prepare(m);params=m.adapter_parameters();grads=[];details={}
 for e in episodes:
  mean=cached(m,e)[960:];y=e['y'][960:]
  loss=(mean.tanh()-y).square().mean()
  gs=torch.autograd.grad(loss,params);g=torch.cat([v.flatten() for v in gs]);grads.append(g)
  with torch.no_grad():
   base=e['mean'][960:];gate=e['gate'][960:];floor=(torch.maximum((base-gate).tanh()-y,torch.zeros_like(y))+torch.minimum((base+gate).tanh()-y,torch.zeros_like(y)))**2
   floorrows=floor.mean(1);top=torch.argsort(floorrows,descending=True)[:10]
   f=e['features'][960:];a=torch.tanh(m.encoder(f));b=torch.tanh(m.hidden(a));h=m.head(b);res=h.tanh()
   details[e['name']]=dict(loss=float(loss),gradient_norm=float(g.norm()),encoder_saturated_abs_ge_p99=float((a.abs()>=.99).double().mean()),hidden_saturated_abs_ge_p99=float((b.abs()>=.99).double().mean()),residual_output_saturated_abs_ge_p99=float((res.abs()>=.99).double().mean()),floor_first1s=float(floor[:200].mean()),floor_after1s=float(floor[200:].mean()),worst_floor_times=[dict(post_elapsed=float(int(i)*.005),MSE=float(floorrows[i])) for i in top])
 names=[e['name'] for e in episodes];G=torch.stack(grads);cos=G@G.T/(G.norm(dim=1)[:,None]*G.norm(dim=1)[None,:])
 opt=p['optimizer'];ids=opt['param_groups'][0]['params']
 momentum=torch.cat([opt['state'][i]['exp_avg'].flatten() for i in ids]);variance=torch.cat([opt['state'][i]['exp_avg_sq'].flatten() for i in ids]);g=G.mean(0)
 # Stored Adam exp_avg and exp_avg_sq after update1000; use bias corrected current moments as a diagnostic direction only.
 direction=(momentum/(1-.9**1000))/(torch.sqrt(variance/(1-.999**1000))+1e-8)
 next_m=.9*momentum+.1*g;next_v=.999*variance+.001*g.square()
 next_direction=(next_m/(1-.9**1001))/(torch.sqrt(next_v/(1-.999**1001))+1e-8)
 next_math=dict(next_step=1001,cosine_with_current_gradient=float(torch.dot(next_direction,g)/(next_direction.norm()*g.norm())),parameter_delta_norm_at_lr_p001=float(.001*next_direction.norm()),parameter_delta_norm_at_lr_p0001=float(.0001*next_direction.norm()),linear_loss_change_at_lr_p001=float(-.001*torch.dot(g,next_direction)),linear_loss_change_at_lr_p0001=float(-.0001*torch.dot(g,next_direction)),weights_or_optimizer_mutated=False,interpretation='Exact next Adam moment/direction algebra using current fullbatch gradient; loss change is first-order only, no candidate weights evaluated.')
 report=dict(next_Adam_algebra=next_math,cases=details,gradient_cosines=dict(zip(names,cos.tolist())),current_mean_gradient_norm=float(g.norm()),stored_momentum_cos_current_gradient=float(torch.dot(momentum,g)/(momentum.norm()*g.norm())),stored_Adam_direction_cos_current_gradient=float(torch.dot(direction,g)/(direction.norm()*g.norm())),optimizer_updates=0,physical_steps=0,sources={str(HERE/'gradient_review.py'):sha256(__file__),str(ROOT/'TRAIN_FAST_1000/checkpoint_001000.pt'):sha256(ROOT/'TRAIN_FAST_1000/checkpoint_001000.pt')},interpretation='Current saved gradients and stored moments only, not causal proof of oscillation or future update outcome.')
 (HERE/'GRADIENT_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
