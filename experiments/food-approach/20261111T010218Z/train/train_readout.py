"""Fixed 2000-update zero-init positive-turn readout training; no physics."""
import argparse,json,time,subprocess,sys
from pathlib import Path
import numpy as np,torch
from model_io import HERE,load_parent,ReadoutMotor,save,load_checked,sha256,verify_hashes,exact
from data import prepare

def main(args):
 torch.set_num_threads(1);torch.manual_seed(20261111);np.random.seed(20261111)
 prereg=json.loads((HERE/'PREREG.json').read_text()); parent,parent_payload=load_parent(); model=ReadoutMotor(parent)
 e,data=prepare(model); checks=json.loads((HERE/'PREFLIGHT.json').read_text())
 if not checks['all_passed']:raise ValueError('preflight required')
 verify_hashes(checks['source_hashes']);verify_hashes(checks['data_hashes'])
 sources={str(HERE/n):sha256(HERE/n) for n in ('PREREG.json','model_io.py','data.py','check_train.py','train_readout.py')}
 data_hashes=dict(data)
 out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
 opt=torch.optim.Adam(model.adapter_parameters(),lr=.003)
 config=dict(kind='fixed_positive_neural_readout2000',parent_checkpoint=str(parent_payload.get('config',{}).get('parent_checkpoint', '/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/train_two_only_control/TRAIN/checkpoint_001500.pt')),parent_sha256=prereg['parent_sha256'],physical_eval_only_step=2000,source_hashes=sources,data_hashes=data_hashes,prereg=prereg,curve=[])
 (out/'IDENTITY.json').write_text(json.dumps(config,indent=2)+'\n')
 # restore parent-derived model and fresh optimizer by fixed design; neutral exact is checked each save.
 curve=[];started=time.monotonic()
 for step in range(1,2001):
  opt.zero_grad(set_to_none=True)
  # left full actual sequence only; causal cache from data.prepare
  mean= e['mean'] + model.correction(e['residual_hidden'],e['u'])
  loss=(mean[960:].tanh()-e['y'][960:]).square().mean();loss.backward()
  grad=torch.nn.utils.clip_grad_norm_(model.adapter_parameters(),1.)
  if not torch.isfinite(grad):raise ValueError('nonfinite gradient')
  if any(p.grad is not None or p.requires_grad for p in model.parent.parameters()):raise ValueError('parent freeze')
  opt.step(); row=dict(step=step,loss_before_update=float(loss.detach()),gradient_norm=float(grad),wall_s=time.monotonic()-started)
  curve.append(row)
  if step%1000==0:
   # Neutral parent invariance using preflight helper was already established; save independent state.
   rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state())
   config['curve']=curve
   ck=out/f'checkpoint_{step:06d}.pt';digest=save(ck,model,opt,config,curve,rng)
   fresh,p=load_checked(ck,digest,require_final=False)
   if not exact(model.state_dict(),fresh.state_dict()) or not exact(opt.state_dict(),p['optimizer']):raise ValueError('save reload mismatch')
   subprocess.run([sys.executable,'-B',str(HERE/'verify_readout.py'),'--checkpoint',str(ck),'--sha256',digest],check=True)
   (out/f'VERIFY_{step:06d}.json').write_text(json.dumps(dict(step=step,sha256=digest,save_reload_exact=True),indent=2)+'\n')
   print(json.dumps(dict(step=step,checkpoint=str(ck),sha256=digest,wall_s=time.monotonic()-started)),flush=True)
 config['curve']=curve
 (out/'RESULT.json').write_text(json.dumps(dict(status='TRAINED_NOT_PHYSICALLY_EVALUATED',steps=2000,checkpoint=str(ck),sha256=digest,wall_s=time.monotonic()-started),indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);main(p.parse_args())
