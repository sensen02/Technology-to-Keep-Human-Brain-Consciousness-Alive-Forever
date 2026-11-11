"""Validate sensory-only schedule, actual BPTT/optimizer and checkpoint roundtrip."""
import json,tempfile,time
from pathlib import Path
import numpy as np,torch
from common import HERE,SteeringBehavior,digest,save_json,verify_hashes,guard,frozen_sources
from learning import make_scan,production,exact,load_behavior
from generate_data import episode

def main():
 torch.set_num_threads(1);torch.manual_seed(20261111);np.random.seed(20261111);guard()
 d=json.loads((HERE/'DATASET.json').read_text());verify_hashes(d['source_hashes'])
 with np.load(HERE/'train.npz',allow_pickle=False) as z:x=torch.tensor(z['receptors'][:20].transpose(1,0,2).copy());y=torch.tensor(z['teacher_turn'][:20].T.copy());initial=z['receptors'].copy();c=z['environment_concentration'].copy()
 checks=dict(actor_only2channels=x.shape==(240,20,2),sequence240=y.shape==(240,20),initial_receptors_zero=bool(np.array_equal(initial[:,0],np.zeros((44,2)))))
 checks['heldout_angle_disjoint']=not set(r['angle'] for r in d['splits']['train']['episodes_config'])&set(r['angle'] for r in d['splits']['heldout']['episodes_config'] if not r['no_food'])
 checks['heldout_dose_disjoint']=not set(r['amplitude'] for r in d['splits']['train']['episodes_config'] if not r['no_food'])&set(r['amplitude'] for r in d['splits']['heldout']['episodes_config'] if not r['no_food'])
 checks['concentration_strictly_before_onset_exactzero']=all(np.max(c[i,np.arange(240)*.005<r['onset_s']-1e-12])==0 for i,r in enumerate(d['splits']['train']['episodes_config']))
 checks['nominal_onset_boundary_roundoff_below1e24']=all(np.max(c[i,np.abs(np.arange(240)*.005-r['onset_s'])<=1e-12])<=1e-24 for i,r in enumerate(d['splits']['train']['episodes_config']))
 meta=d['splits']['train']['episodes_config'][0];g=d['geometry'];rot=np.asarray(g['head_rotation']);forward=rot[:2,0];left=np.array([-forward[1],forward[0]])
 recreated=episode(meta['angle'],meta['amplitude'],meta['onset_s'],meta['seed'],np.asarray(g['sensor_xy']),np.asarray(g['sampling_midpoint_xy']),forward,left)
 checks['seeded_episode_recreation_exact']=np.array_equal(recreated[0],initial[0])
 model=SteeringBehavior();scan=make_scan(model)
 with torch.no_grad():p,h=production(model,x);q,s=scan(x)
 checks['initial_zero_turn']=torch.equal(p,torch.zeros_like(p));checks['initial_hidden_cache_tol']=float((h-s).abs().max())<=1e-12
 opt=torch.optim.Adam(model.parameters(),lr=.001);out,_=scan(x);loss=(out-y).square().mean();loss.backward()
 checks['actual_head_gradient_nonzero']=bool(model.head.weight.grad.abs().max()>0)
 opt.step();opt.zero_grad(set_to_none=True)
 p,h=production(model,x);loss=(p-y).square().mean();loss.backward();grads={k:v.grad.clone() for k,v in model.named_parameters()}
 model.zero_grad(set_to_none=True);q,s=scan(x);loss=(q-y).square().mean();loss.backward()
 error=max(float((v.grad-grads[k]).abs().max()) for k,v in model.named_parameters())
 checks['full240_outputs_tol1e12']=float((p-q).abs().max().detach())<=1e-12;checks['full240_states_tol1e12']=float((h-s).abs().max().detach())<=1e-12;checks['all_parameter_gradients_tol1e12']=error<=1e-12
 checks['all_parameters_finite_nonzero_gradient']=all(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().max()>0 for v in model.parameters())
 with torch.no_grad():commands,_=model.step(x[0],model.initial_state(20))
 checks['speed_structuralzero']=torch.equal(commands[:,0],torch.zeros_like(commands[:,0]))
 sources=frozen_sources();sources.update({str(HERE/n):digest(HERE/n) for n in ('common.py','generate_data.py','learning.py','check_preflight.py')})
 data={str(HERE/n):digest(HERE/n) for n in ('DATASET.json','train.npz','heldout.npz')}
 rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state());curve=[dict(step=0,loss=0.),dict(step=1,loss=float(loss.detach()))]
 with tempfile.TemporaryDirectory(prefix='sensor-only-unit-',dir=HERE) as td:
  path=Path(td)/'unit.pt';payload=dict(format_version=1,kind='food_steering_behavior',scope_tag='SENSOR_PRETRAIN_ONLY',deployment_qualified=False,identity=model.identity(),state_dict=model.state_dict(),optimizer=opt.state_dict(),rng=rng,curve=curve,config=dict(source_hashes=sources,data_hashes=data))
  torch.save(payload,path);new,payload2=load_behavior(path,digest(path));newopt=torch.optim.Adam(new.parameters(),lr=.001);newopt.load_state_dict(payload2['optimizer'])
  checks['saved_weights_exact']=exact(model.state_dict(),new.state_dict());checks['saved_actual_Adam_exact']=exact(opt.state_dict(),newopt.state_dict());checks['saved_RNG_curve_exact']=exact(rng,payload2['rng']) and exact(curve,payload2['curve'])
  with torch.no_grad():a,_=production(model,x);b,_=production(new,x)
  checks['saved_production_inference_exact']=torch.equal(a,b)
 report=dict(all_passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},gradient_max_error=error,source_hashes=sources,data_hashes=data,physical_steps=0,Core_constructions=0,temporary_optimizer_unit_steps=1,permanent_training_updates=0)
 guard();save_json(HERE/'PREFLIGHT.json',report);print(json.dumps(dict(all_passed=report['all_passed'],checks=len(checks),gradient_max_error=error)))
 if not report['all_passed']:raise RuntimeError('preflight failed '+json.dumps(report['checks']))
if __name__=='__main__':main()
