"""Independent saved snapshot/prefix/labels audit; no Core/model construction or integration."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'ORACLE';ACTUAL=ROOT.parent/'left_hind_stance_refit/EVAL_right'
def sha256(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def main():
 r=json.loads((OUT/'RESULT.json').read_text());prefix=json.loads((OUT/'PREFIX_AUDIT.json').read_text());ident=json.loads((OUT/'IDENTITY.json').read_text());checks={}
 def check(n,v):
  checks[n]=bool(v)
  if not checks[n]:raise ValueError('oracle saved audit failed '+n)
 check('one_private_Core_zerointegration',r['Core_constructions']==1 and r['mj_step_calls']==0 and r['physics_integration_steps']==0);check('prefix_gate_first',prefix['all_passed'] and prefix['warm_actions']==960 and prefix['full_prefix_endpoints']==5800 and prefix['max_warm_action_error']<=1e-10 and prefix['max_geometry_error']<=1e-10)
 for p,h in {**r['input_source_hashes'],**r['artifacts']}.items():check('hash:'+p,sha256(p)==h)
 canonical={k:v for k,v in ident.items() if k!='identity_sha256'};digest=hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest();check('identity_digest',digest==ident['identity_sha256']==r['identity_sha256'])
 for p,info in ident['files'].items():check('identity_file:'+p,sha256(p)==info['sha256'])
 t=arrays(ACTUAL/'trace_1ms.npz');d=arrays(ACTUAL/'neural_decisions.npz');g=arrays(ACTUAL/'geometry_1ms.npz');z=arrays(OUT/'CORRECTIVE_EXAMPLES.npz');replay=arrays(OUT/'ORACLE_REPLAY.npz');initial=arrays(OUT/'TEACHER_INITIAL.npz');fg=arrays(OUT/'PREFIX_GEOMETRY.npz');indices=999+5*np.arange(2880)
 check('shape2880',z['X238'].shape==(2880,238) and z['Y42'].shape==(2880,42) and z['command2'].shape==(2880,2));check('all_numeric_finite',all(np.isfinite(a).all() for group in (z,replay,initial,fg) for a in group.values()));check('actualX_unchanged',np.array_equal(z['X238'],d['X238']));check('actual_command_unchanged',np.array_equal(z['command2'],d['command']));check('actual_times',np.array_equal(z['actual_time'],d['time']) and np.max(np.abs(t['time'][indices]-d['time']))<1e-8);check('prime_excluded',np.array_equal(z['post_prime'],np.arange(2880)>=960));check('labels_bounded',np.max(np.abs(z['Y42']))<=1.+1e-12)
 warm=float(np.max(np.abs(z['Y42'][:960]-d['actually_applied_action'][:960])));check('warm_actual_actions_tolerance',warm<=1e-10 and warm==r['max_warm_action_error']);check('private_qdes_actual_initial',np.array_equal(initial['q_des'],t['qpos'][999]) and np.array_equal(initial['initial_qpos'],t['qpos'][999]));qa=initial['qa'];root=np.setdiff1d(np.arange(49),qa);check('all42joint_root7_mapping',len(qa)==42 and len(root)==7);check('private_root_never_actualbody_written',np.array_equal(replay['private_q_des'][:,root],np.repeat(initial['q_des'][None,root],2880,axis=0)))
 for name,key in [('qpos','qpos'),('qvel','qvel'),('ctrl','actualctrl'),('target','target')]:check('actual_restored_'+name,np.array_equal(replay['restored_'+name],t[key][indices]))
 errors={}
 for name,expected in [('foot',t['relativefoot'][:5800]),('padz',t['padz'][:5800]),('com',t['com'][:5800]),('head_xyz',g['head_xyz'][:5800]),('head_rotation',g['head_rotation'][:5800])]:errors[name]=float(np.max(np.abs(fg[name]-expected)));check('all5800_FK_'+name,errors[name]<=1e-10)
 errors['yaw']=float(np.max(np.abs(np.angle(np.exp(1j*(fg['thorax_yaw']-g['thorax_yaw'][:5800]))))));check('all5800_FK_yaw',errors['yaw']<=1e-10);check('geometry_error_summary',max(errors.values())==r['max_prefix_geometry_error'])
 counts=dict(unsupported=0,unbacked=0,hypothetical=0,rateclip=0);rows=0
 with (OUT/'POST_DIAGNOSTICS.jsonl').open() as f:
  for line in f:
   diag=json.loads(line);k=diag['k'];check('diag_clock_'+str(k),k==960+rows and diag['label_kind']=='CORRECTIVE_EXAMPLES');check('diag_qdes_'+str(k),np.array_equal(diag['q_des_act'],replay['private_q_des'][k,qa]));raw=(replay['private_q_des'][k,qa]-replay['private_q_des'][k-1,qa])/.005;check('diag_private_history_'+str(k),np.max(np.abs(raw-np.asarray(diag['qdot_des_unclipped'])))<=1e-10);check('diag_label_'+str(k),np.max(np.abs(np.asarray(diag['rate_rad_s'])/2.-z['Y42'][k]))<=1e-12)
   counts['unsupported']+=int(diag['unsupported']);counts['unbacked']+=int(bool(diag['unbacked_support_indices']));counts['hypothetical']+=int(diag['support_is_hypothetical']);counts['rateclip']+=int(diag['rate_clip_count']>0);rows+=1
 check('post1920',rows==r['post_labels']==1920);check('diagnostic_counts',counts==r['post_diagnostic_counts']);check('diagnostic_rates',{k:v/1920 for k,v in counts.items()}==r['post_diagnostic_rates'])
 report=dict(all_passed=all(checks.values()),checks=checks,max_warm_action_error=warm,max_geometry_errors=errors,post_diagnostic_rates=r['post_diagnostic_rates'],Core_constructions=0,physics_integration_steps=0,model_replays=0,source_hashes={str(ROOT/'verify_oracle.py'):sha256(ROOT/'verify_oracle.py'),str(OUT/'RESULT.json'):sha256(OUT/'RESULT.json'),str(OUT/'CORRECTIVE_EXAMPLES.npz'):sha256(OUT/'CORRECTIVE_EXAMPLES.npz')},scope='saved zero-integration snapshot query audit; postlabels are unexecuted CORRECTIVE_EXAMPLES, not qualifiedteacherphysicalPASS; original privatechain has no direct saved groundtruth')
 path=OUT/'VERIFY.json'
 if path.exists():raise ValueError('do not overwrite oracle independent audit')
 path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(all_passed=report['all_passed'],checks=len(checks),max_warm_action_error=warm,max_geometry_errors=errors,post_diagnostic_rates=r['post_diagnostic_rates'])),flush=True)
if __name__=='__main__':main()
