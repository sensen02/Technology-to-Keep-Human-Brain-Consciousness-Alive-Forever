"""Actual parent/optimizer/RNG resume and balanced full240 BPTT audit, no permanent update."""
import json
import numpy as np,torch
from common import HERE,PREVIOUS,PARENT,PARENT_SHA,legacy_learning,make_scan,production,exact,digest,save_json,guard

def main():
 torch.set_num_threads(1);guard();m,p=legacy_learning.load_behavior(PARENT,PARENT_SHA)
 opt=torch.optim.Adam(m.parameters(),lr=.001);opt.load_state_dict(p['optimizer']);checks={}
 checks['parent_model_exact']=exact(m.state_dict(),p['state_dict']);checks['all_actual_Adam_exact']=exact(opt.state_dict(),p['optimizer']);checks['all_counters1000']=all(float(v['step'])==1000 for v in opt.state.values())
 old=p['optimizer'];opt.param_groups[0]['lr']=.0003
 changed={**old,'param_groups':[dict(old['param_groups'][0],lr=.0003)]};checks['sole_hyperparameter_change_lr']=exact(opt.state_dict(),changed)
 torch.set_rng_state(p['rng']['torch']);np.random.set_state(p['rng']['numpy']);checks['actual_RNG_resume_exact']=exact(dict(torch=torch.get_rng_state(),numpy=np.random.get_state()),p['rng'])
 meta=json.loads((PREVIOUS/'DATASET.json').read_text());food=np.asarray([i for i,c in enumerate(meta['splits']['train']['episodes_config']) if not c['no_food']],dtype=int);zero=np.asarray([i for i,c in enumerate(meta['splits']['train']['episodes_config']) if c['no_food']],dtype=int)
 idx=np.r_[np.random.choice(food,10,replace=True),np.random.choice(zero,10,replace=True)];idx=idx[np.random.permutation(20)]
 checks['balanced20_exact10food10nofood']=sum(meta['splits']['train']['episodes_config'][int(i)]['no_food'] for i in idx)==10
 with np.load(PREVIOUS/'train.npz',allow_pickle=False) as z:x=torch.tensor(z['receptors'][idx].transpose(1,0,2).copy());y=torch.tensor(z['teacher_turn'][idx].T.copy())
 model_before={k:v.clone() for k,v in m.state_dict().items()};scan=make_scan(m)
 out,h=production(m,x);loss=(out-y).square().mean();loss.backward();grad={k:v.grad.clone() for k,v in m.named_parameters()}
 m.zero_grad(set_to_none=True);fast,hf=scan(x);newloss=(fast-y).square().mean();newloss.backward()
 error=max(float((v.grad-grad[k]).abs().max()) for k,v in m.named_parameters())
 checks['production_script_full240_output_tol']=float((out-fast).abs().max().detach())<=1e-12;checks['production_script_full240_state_tol']=float((h-hf).abs().max().detach())<=1e-12;checks['all_gradient_tol']=error<=1e-12
 checks['all_gradients_finite_nonzero']=all(torch.isfinite(v.grad).all() and v.grad.abs().max()>0 for v in m.parameters())
 opt.step();checks['actual_optimizer_update_changes_weights']=not exact(m.state_dict(),model_before)
 with torch.no_grad():cmd,_=m.step(x[0],m.initial_state(20))
 checks['forward_structuralzero']=torch.equal(cmd[:,0],torch.zeros_like(cmd[:,0]))
 fresh=json.loads((HERE/'FRESH_TEST.json').read_text());checks['fresh_dataset_frozen']=digest(HERE/'fresh_test.npz')==fresh['sha256'] and fresh['episodes']==44
 checks['parent_complete_curve0_1000']=[r['step'] for r in p['curve']]==list(range(1001))
 r=dict(all_passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},gradient_max_error=error,physical_steps=0,Core_constructions=0,temporary_optimizer_unit_steps=1,parent_sha256=PARENT_SHA,source_hashes={str(HERE/n):digest(HERE/n) for n in ('common.py','generate_test.py','check_resume.py','PREREG.json','METRIC_CLARIFICATION_BEFORE_TRAINING.json')},data_hashes={str(f):digest(f) for f in (PARENT,PREVIOUS/'train.npz',PREVIOUS/'DATASET.json',HERE/'fresh_test.npz',HERE/'FRESH_TEST.json')})
 save_json(HERE/'PREFLIGHT.json',r);guard();print(json.dumps(dict(all_passed=r['all_passed'],checks=len(checks),gradient_max_error=error)))
 if not r['all_passed']:raise RuntimeError('balanced preflight failed')
if __name__=='__main__':main()
