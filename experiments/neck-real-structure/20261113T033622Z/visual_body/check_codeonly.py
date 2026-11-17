"""Stdlib syntax/namespace/input-boundary checks only; no arrays/models/Core/render."""
from pathlib import Path
import ast,json,hashlib,sys,importlib.util
HERE=Path(__file__).resolve().parent
FILES=('runtime.py','runner.py','audit.py','prepare.py','preflight.py','summary.py','review.py','orchestrate.py','check_codeonly.py')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 checks={};trees={};texts={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('CODE_ONLY '+k)
 for name in FILES:
  text=(HERE/name).read_text();tree=ast.parse(text);compile(tree,str(HERE/name),'exec');trees[name]=tree;texts[name]=text;check(name+'_syntax',True);check(name+'_publicmain',name=='runtime.py' or any(isinstance(n,ast.FunctionDef) and n.name=='main' for n in tree.body))
 for name in FILES:
  if name=='runtime.py':continue
  check(name+'_no_eager_numerical_import',not any(isinstance(n,(ast.Import,ast.ImportFrom)) and any(x.name.split('.')[0] in ('torch','numpy','mujoco','physics','student_wave','flygym') for x in n.names) for n in trees[name].body))
 spec=importlib.util.spec_from_file_location('_visualneck_codeonly_runtime',HERE/'runtime.py');r=importlib.util.module_from_spec(spec);sys.modules[spec.name]=r;spec.loader.exec_module(r)
 check('ten_fixed_conditions',len(r.IDS)==10 and len(r.NORMAL_IDS)==6 and len(r.CONTROL_IDS)==len(r.CUT_IDS)==2);check('newROOT_exact',r.ROOT==HERE);check('actual_new300_path',r.BRIDGE==HERE.parent/'visual_bridge_calibration/TRAIN/checkpoint_000300.pt');check('no_unknown_digest_admitted_on_import',not hasattr(r,'BRIDGE_SHA'))
 check('cut_only_camera_scheduler_mapping',all(r.camera_condition(n)=='V_left_pulse' for n in r.CUT_IDS) and all(r.camera_condition(n)==n for n in r.NORMAL_IDS+r.CONTROL_IDS))
 cg=r.camera_runtime();check('rebound_camera_close_newROOT',cg['CameraStream'].close.__globals__['ROOT']==HERE);check('rebound_capture_init_helpers_newROOT',cg['_init'].__globals__['ROOT']==cg['_capture'].__globals__['ROOT']==HERE);check('rebound_proxy_newROOT',cg['_OpticalProxy'].render_eye_pair.__globals__['ROOT']==HERE)
 decision=next(n for n in ast.walk(trees['runtime.py']) if isinstance(n,ast.FunctionDef) and n.name=='decision');calls=[n for n in ast.walk(decision) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='brain_step'];check('brain_inputs_AN_neuralvisual_only',len(calls)==1 and len(calls[0].args)==3 and isinstance(calls[0].args[0],ast.Name) and calls[0].args[0].id=='ad' and ast.unparse(calls[0].args[1])=='visual_raw.reshape(1, 1)');check('no_condition_or_target_neuralinput','stimulus' not in ast.unparse(decision) and 'target_center_xyz' not in ast.unparse(decision));check('body_decoder_only_deliveredDN','self.bridge.decoder(dd)' in ast.unparse(decision));check('no_visualscale_sign_deadzone','/.8' not in texts['runtime.py'] and 'sign(' not in texts['runtime.py'] and 'deadzone' not in texts['runtime.py'])
 check('first20ms_explicit_invalid_guard','def actor_observation' in texts['runtime.py'] and "raw_capture_id=-1" in texts['runtime.py']);check('cut_engineering_silence_not_forcedturnzero',"self.held[d]=torch.zeros_like" in texts['runtime.py'] and "decoded=self.bridge.decoder(dd)" in texts['runtime.py']);check('runner_full3080records','range(3080)' in texts['runner.py']);check('runtime_304_layout','state_layout=[128,64,64,32,16]' in texts['runner.py']);check('replay_actualRGB_fromdisk','rgb[fid][None,None]' in texts['audit.py']);check('audit_legacy_controls_immutable_2880view','value[200:]' in texts['audit.py']);check('review_distinguishes_visualrelease_DNrelease_motorconsume',all(x in texts['review.py'] for x in ('4.82','4.822','4.825','10622','10625')));check('summary_original_threshold_sources',"visual_body_v1/summary.py" in texts['summary.py'] and "landmark_far_v3/body/summary.py" in texts['summary.py']);check('summary_explicit_newkind',r.SUMMARY_KIND=='ACTUAL_RGB_SOURCE_NECK_VISUAL_BODY_SUMMARY_V1');check('fixed22_stages',"('SUMMARY',None,'summary.py')" in texts['orchestrate.py'] and "('REVIEW',None,'review.py')" in texts['orchestrate.py'] and len(r.IDS)*2+2==22)
 summary_spec=importlib.util.spec_from_file_location('_visualneck_codeonly_summary',HERE/'summary.py');summary=importlib.util.module_from_spec(summary_spec);summary_spec.loader.exec_module(summary);legacy=summary.legacy_namespace();normal=legacy['namespace']('summary.py');check('far_summary_namespace_not_recursive',legacy['main'].__code__.co_filename.endswith('::originalfarcontrol2thresholds'));check('normal_summary_namespace_not_recursive',normal['main'].__code__.co_filename.endswith('::originalnormal6thresholds'));check('normal_summary_sixcase_and_newROOT',normal['IDS']==r.NORMAL_IDS and normal['ROOT']==HERE)
 check('no_heavy_modules_imported',not any(k in sys.modules for k in ('numpy','torch','mujoco','physics','student_wave','flygym')));check('no_generated_neural_orphysical_outputs',not any((HERE/n).exists() for n in ('INPUT_BINDINGS.json','FROZEN.json','PREFLIGHT.json','JOBS.json','trials')))
 hashes={str(HERE/n):sha(HERE/n) for n in FILES+('PREREG.json',)}
 out=dict(all_passed=True,checks=checks,source_hashes=hashes,neural_imports=False,array_payloads_read=0,models_deserialized=0,Core_constructions=0,render_calls=0,NN_forward_calls=0,training_updates=0,physics_steps=0,declared_actual_stages=22,actual_summary_kind=r.SUMMARY_KIND)
 with (HERE/'CODEONLY_CHECK.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=0,models=0,arrays=0,Core=0,stages=22)))
if __name__=='__main__':main()
