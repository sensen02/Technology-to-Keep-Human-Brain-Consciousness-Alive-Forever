"""Read-only same-checkpoint three-episode conjunction, original strictflags preserved.
No torch, model/Core construction, optimization or integration.
"""
from pathlib import Path
import json,hashlib,math
import numpy as np
HERE=Path(__file__).resolve().parent
_cache={}
def sha256(path):
 path=Path(path).resolve()
 if path not in _cache:
  with path.open('rb') as f:_cache[path]=hashlib.file_digest(f,'sha256').hexdigest()
 return _cache[path]
def arrays(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def main():
 prep=json.loads((HERE/'PREPARATION.json').read_text());saved=json.loads((HERE/'VERIFY_SAVED.json').read_text());bindings=json.loads((HERE/'COMPONENT_BINDINGS.json').read_text());checks={};rows=[];parity={}
 def check(n,v):
  checks[n]=bool(v)
  if not checks[n]:raise ValueError('final composed audit failed '+n)
 digest=sha256(HERE/'COMBINED.pt');check('one_saved_checkpoint',digest==prep['checkpoint_sha256']==saved['sha256']);check('initial35exact_and_saved_replay',prep['all_passed'] and saved['all_passed'] and all(prep['checks'].values()) and all(saved['checks'].values()))
 for path,h in {**prep['source_hashes'],**prep['component_source_hashes'],**prep['stream_hashes'],**bindings['composition_scope_hashes']}.items():check('bound_input:'+path,sha256(path)==h)
 check('AMENDMENT04_posthoc_visual_bound',str(HERE/'AMENDMENT_04_POSTHOC_VISUAL_RIGHT_COMPONENT.json') in bindings['composition_scope_hashes'] and bindings['right_option']=='visual_original_feature_refine')
 for condition in ('neutral','right','left'):
  root=HERE/('EVAL_'+condition);r=json.loads((root/'RESULT.json').read_text());a=json.loads((HERE/('VERIFY_'+condition+'.json')).read_text());ident=json.loads((root/'IDENTITY.json').read_text());prefix=condition+'_'
  check(prefix+'all_independent_checks',a['integrity_all_pass'] and all(a['checks'].values()));check(prefix+'same_checkpoint',r['checkpoint_sha256']==a['checkpoint_sha256']==digest)
  for name in ('full2880_threehidden_decisions','full256_hidden_chain','actual2880_checkpoint_threehidden_exact'):check(prefix+name,a['checks'][name])
  canonical={k:v for k,v in ident.items() if k!='identity_sha256'};h=hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest();check(prefix+'identity_digest',h==ident['identity_sha256']==r['identity_sha256'])
  for path,info in ident['files'].items():check(prefix+'identity_file:'+path,sha256(path)==info['sha256'])
  for key,name in [('trace','trace_1ms.npz'),('neural_decisions','neural_decisions.npz'),('geometry','geometry_1ms.npz'),('runtime_geometry','runtime_geometry.npz')]:check(prefix+name,sha256(root/name)==r[key]['sha256'])
  t=arrays(root/'trace_1ms.npz');d=arrays(root/'neural_decisions.npz');check(prefix+'256layout',d['hidden_before'].shape==d['hidden_after'].shape==(2880,256) and r['state_layout']==[128,64,64] and r['state_total']==256);check(prefix+'fullchain_noreset',np.array_equal(d['hidden_before'][0],np.zeros(256)) and np.array_equal(d['hidden_before'][1:],d['hidden_after'][:-1]));check(prefix+'full15400_body',t['qpos'].shape==(15400,49) and t['qvel'].shape==(15400,48))
  yaw=math.degrees(r['yaw_change_rad']);signed=yaw if condition=='left' else -yaw;auto=r['autonomous'];native=r['native'];cycles=r['autonomous_cycles'];flags=dict(original_five=bool(auto['all_gates']),all_six=bool(cycles['all_legs_completed']),strict_nonfoot_zero=bool(auto['nonfoot_force_max_native']==0.),turn_sign_magnitude=bool(condition=='neutral' or signed>=30.),strict_native_nonfoot_zero=bool(native['post_nonfoot_peak_native']==0. and native['post_native_steps']==768000 and native['actual_native_steps']==1232000 and native['extra_force_max']==0.));flags['eligible']=all(flags.values());check(prefix+'originalstrict_scoring_unchanged',flags==r['criteria'] and r['engineering_pass']==r['engineering_qualified']==flags['eligible'])
  demo=all(auto['gates'][k] for k in ('support','nonfoot','com','forward')) and cycles['all_legs_completed'] and native['post_nonfoot_peak_native']==0. and (condition=='neutral' or signed>0.);check(prefix+'demo_not_qualification',bool(demo)==r['demonstration_stable_motion'])
  component_parity=None
  if condition!='neutral':
   role='RIGHT' if condition=='right' else 'LEFT';reference=Path(bindings[role]['evidence']['reference_stream']);old=arrays(reference);component_parity={k:bool(np.array_equal(d[k],old[k])) for k in ('X238','mean','policy_action','actually_applied_action','control_source','command')};active_slice=slice(128,192) if role=='RIGHT' else slice(192,256);component_parity['basehidden']=np.array_equal(d['hidden_after'][:,:128],old['hidden_after'][:,:128]);component_parity['activebranchhidden']=np.array_equal(d['hidden_after'][:,active_slice],old['hidden_after'][:,128:]);oldtrace=arrays(reference.parent/'trace_1ms.npz');component_parity['all_shared_physical_arrays']=all(np.array_equal(t[k],oldtrace[k]) for k in t if k in oldtrace);parity[condition]={k:bool(v) for k,v in component_parity.items()}
  rows.append(dict(condition=condition,checkpoint_sha256=digest,identity_sha256=r['identity_sha256'],result=str(root/'RESULT.json'),result_sha256=sha256(root/'RESULT.json'),audit=str(HERE/('VERIFY_'+condition+'.json')),audit_sha256=sha256(HERE/('VERIFY_'+condition+'.json')),integrity_checks=len(a['checks']),demonstration_stable_motion=r['demonstration_stable_motion'],engineering_qualified=r['engineering_qualified'],yaw_deg=yaw,slip=auto['slip_ratio_max'],per_leg_slip=auto['per_leg_slip_ratio'],worldX_signed_speed=auto['forward_signed_native_per_s'],support=auto['foot_support_mean'],COMmin=auto['com_height_min'],cycles=cycles['complete_per_leg'],nativeNF=native['post_nonfoot_peak_native'],criteria=r['criteria']))
 result=dict(all_integrity_passed=all(checks.values()),checks=checks,checkpoint=str(HERE/'COMBINED.pt'),checkpoint_sha256=digest,component_bindings_sha256=sha256(HERE/'COMPONENT_BINDINGS.json'),same_model_three_actual_conditions=True,state_layout=[128,64,64],state_total=256,results=rows,all_three_demonstration_stable=all(r['demonstration_stable_motion'] for r in rows),all_three_engineering_qualified=all(r['engineering_qualified'] for r in rows),component_same_actual_array_parity_diagnostic=parity,qualification_not_relaxed=True,goal_completion_not_claimed=True,Core_constructions=0,physics_integration_steps=0,model_replays=0,training_performed=False,source_sha256=sha256(Path(__file__)))
 out=HERE/'FINAL_AUDIT.json'
 if out.exists():raise ValueError('no overwrite final composed audit')
 out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(all_integrity_passed=result['all_integrity_passed'],checks=len(checks),all_three_demonstration_stable=result['all_three_demonstration_stable'],all_three_engineering_qualified=result['all_three_engineering_qualified'],same_component_actual_parity=parity)),flush=True)
if __name__=='__main__':main()
