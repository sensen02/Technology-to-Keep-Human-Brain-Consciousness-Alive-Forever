"""Future parent-released strict composition/save/three-stream exactness preflight.
CODE_ONLY users run check_codeonly.py instead: this module imports heavy dependencies.
"""
import argparse,json,os
import torch,numpy as np
from model_io import *
def dump_new(path,value):
 with Path(path).open('x') as f:json.dump(jsonable(value),f,indent=2,allow_nan=False);f.write('\n')
def arrays(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def split(command):
 t=command[:,:,1:2];positive=torch.relu(t);negative=torch.relu(-t);return torch.cat((torch.zeros_like(negative),-negative),-1),torch.cat((positive/torch.full_like(positive,3.),positive),-1)
def streams(bindings):
 neutral=arrays(REFERENCE);yield 'neutral',neutral,torch.zeros((2880,1,2),dtype=torch.float64),REFERENCE
 for condition,role in [('right','RIGHT'),('left','LEFT')]:
  path=Path(bindings[role]['evidence']['reference_stream']);data=arrays(path);yield condition,data,torch.from_numpy(data['command'])[:,None],path

def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-resource-released',action='store_true');p.add_argument('--left-sha256',required=True);p.add_argument('--right-option',choices=['fixed_ef768','qualified_original_feature_refine','visual_original_feature_refine'],default='visual_original_feature_refine');p.add_argument('--right-sha256');args=p.parse_args()
 if not args.parent_resource_released:raise ValueError('composition execution deferred until parent explicit resource/outcome release')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});torch.set_num_threads(1)
 if (HERE/'COMBINED.pt').exists() or (HERE/'COMPONENT_BINDINGS.json').exists():raise ValueError('no overwrite or blind composition restart')
 right,left,bindings,component_hashes=components(args.right_option,args.left_sha256,args.right_sha256);model=CombinedTurning(right,left,bindings).eval();checks={};replay={};stream_hashes={}
 for condition,data,u,path in streams(bindings):
  x=torch.from_numpy(data['X238'])[:,None];ur,ul=split(u)
  with torch.no_grad():means,h=model.sequence(x,u);rm,rh=right.sequence(x,ur);lm,lh=left.sequence(x,ul)
  checks[condition+'_full256_shape']=h[0].shape==(2880,1,128) and h[1].shape==h[2].shape==(2880,1,64)
  checks[condition+'_sharedbase_RIGHT_exact']=torch.equal(h[0],rh[0]);checks[condition+'_sharedbase_LEFT_exact']=torch.equal(h[0],lh[0]);checks[condition+'_RIGHThidden_exact']=torch.equal(h[1],rh[1]);checks[condition+'_LEFThidden_exact']=torch.equal(h[2],lh[1])
  if condition=='neutral':checks['neutral2880_originalmeans_exact']=np.array_equal(means[:,0].numpy(),data['mean']);checks['neutral2880_originalbasehidden_exact']=np.array_equal(h[0][:,0].numpy(),data['hidden_after']);checks['neutral_both_componentmeans_exact']=torch.equal(means,rm) and torch.equal(means,lm)
  else:
   active_mean=rm if condition=='right' else lm;active_h=h[1] if condition=='right' else h[2]
   checks[condition+'_active_componentmeans_exact']=torch.equal(means,active_mean);checks[condition+'_actual_componentmeans_exact']=np.array_equal(means[:,0].numpy(),data['mean']);checks[condition+'_actual_basehidden_exact']=np.array_equal(h[0][:,0].numpy(),data['hidden_after'][:,:128]);checks[condition+'_actual_activehidden_exact']=np.array_equal(active_h[:,0].numpy(),data['hidden_after'][:,128:])
  state=model.initial_state(1);manual=[];manual_h=[]
  with torch.no_grad():
   for a,b in zip(x,u):
    value,state=model.step(a,b,state);manual.append(value);manual_h.append(torch.cat(state,-1))
  checks[condition+'_step_sequence_means_exact']=torch.equal(torch.stack(manual),means);checks[condition+'_step_sequence_256state_exact']=torch.equal(torch.stack(manual_h),torch.cat(h,-1));stream_hashes[str(path)]=sha256(path);replay[condition]=dict(means=means,base_states=h[0],RIGHT_states=h[1],LEFT_states=h[2])
 checks['all_weights_frozen']=all(not q.requires_grad for q in model.parameters());checks['one_shared_base_module']=not any(n.startswith('right_branch.base') or n.startswith('left_branch.base') for n,_ in model.named_modules())
 if not all(checks.values()):dump_new(HERE/'PREFLIGHT_FAILURE.json',dict(checks=checks));raise ValueError('composed initial fixed-stream exactness failed')
 source_paths=[HERE/n for n in ('model.py','model_io.py','prepare.py','verify_saved.py','evaluate_motor.py','verify_actual.py','prepare_eval.py','check_codeonly.py','PREREG.json','AMENDMENT_01_LEFT_PENDING_FEATURE_REFINE.json','AMENDMENT_02_CONDITIONAL_RIGHT_FEATURE_REFINE.json','AMENDMENT_03_HONEST_DEMONSTRATION_COMPONENTS.json','AMENDMENT_04_POSTHOC_VISUAL_RIGHT_COMPONENT.json','COMPONENT_PLAN.json','EVAL_PREREG.json')]
 source_hashes={str(p):sha256(p) for p in source_paths};dump_new(HERE/'COMPONENT_BINDINGS.json',bindings);payload=dict(format_version=FORMAT,identity=model.identity(),state_dict=model.state_dict(),optimizer=None,training_performed=False,natural_joint_training=False,bindings_path=str(HERE/'COMPONENT_BINDINGS.json'),bindings_sha256=sha256(HERE/'COMPONENT_BINDINGS.json'),source_hashes=source_hashes,component_source_hashes=component_hashes,stream_hashes=stream_hashes,replay=replay)
 temporary=HERE/'COMBINED.pt.tmp';torch.save(payload,temporary);temporary.replace(HERE/'COMBINED.pt');digest=sha256(HERE/'COMBINED.pt');restored,saved=load_checked(HERE/'COMBINED.pt',digest);checks['live_saved_exact']=exact(restored.state_dict(),model.state_dict()) and saved['identity']==model.identity()
 dump_new(HERE/'PREPARATION.json',dict(all_passed=all(checks.values()),checks=checks,source_hashes=source_hashes,component_source_hashes=component_hashes,stream_hashes=stream_hashes,component_bindings_sha256=sha256(HERE/'COMPONENT_BINDINGS.json'),checkpoint=str(HERE/'COMBINED.pt'),checkpoint_sha256=digest,state_layout=[128,64,64],state_total=256,Core_constructions=0,physics_integration_steps=0,training_performed=False,component_qualification_at_assembly=bindings['qualification']))
 print(json.dumps(dict(all_passed=all(checks.values()),checks=len(checks),checkpoint_sha256=digest,composition_actual='NOT_EVALUATED')),flush=True)
if __name__=='__main__':main()
