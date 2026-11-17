"""Stdlib preparation check only; does not execute model or interventions."""
from pathlib import Path
import ast,json,hashlib,sys,importlib.util,inspect
HERE=Path(__file__).resolve().parent

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 pr=json.loads((HERE/'PREREG.json').read_text());tree=ast.parse((HERE/'run_interventions.py').read_text());checks={};checks['syntax']=True;checks['30matched_replays']=pr['trial_count']==30 and len(pr['conditions'])==6 and len(pr['modes'])==5;checks['no_training_or_physics']=pr['new_training_updates']==pr['new_physics_steps']==0
 checks['fixed_final200bytes']=sha(pr['checkpoint'])==pr['checkpoint_sha256'];checks['actual_sourcebindingbytes']=sha(pr['source_binding'])==pr['source_binding_sha256'];checks['public_main_and_rollout']=all(any(isinstance(n,ast.FunctionDef) and n.name==name for n in tree.body) for name in ('main','rollout','summarize'))
 spec=importlib.util.spec_from_file_location('_offline_intervention_codeonly_contract',HERE/'run_interventions.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 checks['actual_modes_exact']=list(m.MODES)==pr['modes'];checks['actual_flag_not_automatic']=any(isinstance(n,ast.Constant) and n.value=='--execute-parent-interventions' for n in ast.walk(tree));checks['top_level_stdlib_only']=not any(k in sys.modules for k in ('numpy','torch','mujoco','physics','model','student_wave'))
 s=(HERE/'run_interventions.py').read_text();checks['actual_sourcequeue_used']='transport_module.SourceTransport(binding_path,binding_sha)' in s;checks['one_ms_advance_5times_per_decision']='for ms in range(5):' in s and 'for packet in source.advance()' in s;checks['cut3000_not_brain_reset']='source.step==3000' in s and 'source.set_cut(cut_direction,True)' in s and 'cut_hbefore=h.clone()' in s;checks['baseline_prior_eval_required']='baseline_matches_prior_final200' in s and "z['reconstruction']" in s;checks['decoder_zero_report_not_output_clamp']='decoder_delivered_DN_zero_output=decoder_zero_output' in s and 'turn=model.decoder(held[\'DN\'])' in s;checks['queue_release_and_consume_separate']='queue_release_step' in s and 'neural_consume_step' in s;checks['parameters_frozen_and_hashchecked']='model.requires_grad_(False)' in s and 'tensor_hashes()==model_before' in s;checks['conditional_archived_body_scope']=pr['observations'].startswith('Each replay uses same archived actual111')
 if not all(checks.values()):raise ValueError(str({k:v for k,v in checks.items() if not v}))
 with (HERE/'CODEONLY_CHECK.json').open('x') as f:json.dump(dict(all_passed=True,checks=checks,source_hashes={str(HERE/n):sha(HERE/n) for n in ('PREREG.json','run_interventions.py','codecheck.py')},model_imports=0,array_reads=0,actual_interventions=0,training_updates=0,physics_steps=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=False,actual_interventions=0)))
if __name__=='__main__':main()
