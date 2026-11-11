"""Independent saved final-state audit with explicit metadata limitations."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from model_io import HERE,ROOT,PARENT,PARENT_SHA,load_checked,load_parent,exact,sha256,verify_hashes
from data import invariance,prepare,cached


def main(a):
 torch.set_num_threads(1);m,p=load_checked(a.checkpoint,a.sha256)
 checks={};parent,pp=load_parent();step=p['curve'][-1]['step']
 checks['final2000']=step==2000 and p['config']['physical_eval_only_step']==2000
 checks['identity_actual_parent_path_sha']=p['identity']['parent_checkpoint']==str(PARENT) and p['identity']['parent_sha256']==PARENT_SHA
 checks['all194878_parent_weights_exact_to_actual1500']=exact(m.parent.state_dict(),parent.state_dict())
 checks['all_parent_frozen']=all(not q.requires_grad for q in m.parent.parameters())
 checks['all_new_weights_finite']=all(torch.isfinite(q).all() for q in m.adapter_parameters())
 checks['saved_model_reload_exact']=exact(m.state_dict(),p['state_dict'])
 checks['curve1_through2000_exact_contiguous']=len(p['curve'])==2000 and [r['step'] for r in p['curve']]==list(range(1,2001))
 opt=torch.optim.Adam(m.adapter_parameters(),lr=.003);opt.load_state_dict(p['optimizer'])
 checks['optimizer_saved_reload_exact']=exact(opt.state_dict(),p['optimizer'])
 checks['new_weight_bias_order_and_coverage']=len(opt.param_groups)==1 and [id(q) for q in opt.param_groups[0]['params']]==[id(m.readout.weight),id(m.readout.bias)] and len(opt.state)==2
 checks['actual_Adam_counters2000']=all(float(s['step'])==2000 for s in opt.state.values())
 checks['actual_Adam_moments_finite_nonnegative']=all(torch.isfinite(s['exp_avg']).all() and torch.isfinite(s['exp_avg_sq']).all() and (s['exp_avg_sq']>=0).all() for s in opt.state.values())
 checks['moments_match_parameter_shapes']=all(s['exp_avg'].shape==q.shape and s['exp_avg_sq'].shape==q.shape for q,s in opt.state.items())
 checks['fixed_hyperparameters']=opt.param_groups[0]['lr']==.003 and opt.param_groups[0]['betas']==(.9,.999) and opt.param_groups[0]['eps']==1e-8 and opt.param_groups[0]['weight_decay']==0
 torch.set_rng_state(p['rng']['torch']);np.random.set_state(p['rng']['numpy'])
 checks['saved_rng_restore_roundtrip']=torch.equal(torch.get_rng_state(),p['rng']['torch']) and exact(np.random.get_state(),p['rng']['numpy'])
 verify_hashes(p['config']['source_hashes']);verify_hashes(p['config']['data_hashes']);checks['all_saved_source_data_hashes_match']=True
 checks.update({'actual_'+k:v for k,v in invariance(m).items()})
 e,_=prepare(m)
 with torch.no_grad():
  mean,states=m.sequence(e['x'][:,None],e['u'][:,None]);vector=cached(m,e)
  clone,_=load_checked(a.checkpoint,a.sha256);cloned,_=clone.sequence(e['x'][:,None],e['u'][:,None])
 checks['clone_fullleft_inference_exact']=torch.equal(mean,cloned)
 checks['cached_fullleft_inference_tol1e12']=float((mean[:,0]-vector).abs().max())<=1e-12
 checks['left_dual_parent_states_unchanged']=torch.equal(states[0][:,0],e['base_hidden']) and torch.equal(states[1][:,0],e['residual_hidden'])
 checks['final_checkpoint_unchanged']=sha256(a.checkpoint)==a.sha256
 config_parent=p['config'].get('parent_checkpoint')
 findings=dict(ancestor_path_in_config_wrong=config_parent!=str(PARENT),recorded_config_parent=config_parent,recorded_config_parent_sha=p['config']['parent_sha256'],authoritative_identity_parent=str(PARENT),authoritative_identity_parent_sha=PARENT_SHA,
               step0_curve_record_missing=True,live_pre_save_fullAdam_RNG_curve_equality_not_recorded=True,constructor_load_at1000_consumes_RNG=True,
               updates_use_no_random_operations=True)
 report=dict(all_inference_and_saved_state_checks_passed=all(checks.values()),checks=checks,findings=findings,checkpoint=str(Path(a.checkpoint).resolve()),checkpoint_sha256=a.sha256,
             frozen_parent_identity_and_state_correct=checks['all194878_parent_weights_exact_to_actual1500'] and checks['identity_actual_parent_path_sha'],
             left_postprime_action_MSE=float((mean[960:,0].tanh()-e['y'][960:]).square().mean()),physical_steps=0,optimizer_updates=0,
             limitations=['Config parent_checkpoint metadata incorrectly names old500 ancestor while config parent_sha256 and model identity use actual1500; external correction manifest records this without rewriting original checkpoint.',
             'Recorded curve is1..2000; no step0 metric row was saved.','Saved-state reload consistency and exact frozen-parent/negative-command functions are established; no fabricated live-before-save Adam/RNG/curve assertion.',
             'RNG restore roundtrip is checked, but construction at checkpoint1000 consumed global RNG before later saves; training updates themselves deterministic without random draws. No stochastic continuation claim.',
             'No actual physical qualification performed by this audit.'],
             audit_source_sha256=sha256(Path(__file__)),root_prereg_sha256=sha256(ROOT/'PREREG.json'))
 with Path(a.out).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(dict(checks=len(checks),all_state_inference_checks_passed=all(checks.values()),findings=findings)))
 if not all(checks.values()):raise ValueError('actual state/inference audit failed')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--sha256',required=True);p.add_argument('--out',required=True);main(p.parse_args())
