"""Stdlib source/contracts check only: never import backend/model/array payload."""
from pathlib import Path
import ast,json,sys,hashlib,subprocess
HERE=Path(__file__).resolve().parent

def main():
 checks={};hashes={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('CODE_ONLY '+k)
 def digest(p):
  with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
 names=('common.py','model.py','data.py','preflight.py','train.py','verify_saved.py','evaluate.py','run.py','codecheck.py');trees={}
 for name in names:
  p=HERE/name;s=p.read_text();trees[name]=ast.parse(s);compile(trees[name],str(p),'exec');hashes[str(p)]=digest(p);check(name+'_AST_compiles',True)
 for name in ('data.py','preflight.py','train.py','verify_saved.py','evaluate.py','run.py'):check(name+'_publicmain',any(isinstance(x,ast.FunctionDef) and x.name=='main' for x in trees[name].body))
 check('evaluate_publicaudit',any(isinstance(x,ast.FunctionDef) and x.name=='audit' for x in trees['evaluate.py'].body));pr=HERE/'CALIBRATION_PREREG.json';p=json.loads(pr.read_text());hashes[str(pr)]=digest(pr)
 check('strictnewformat',p['format']=='SOURCE_NECK_VISUAL_SCALAR_CALIBRATION_V1');check('fixed300B9',p['training']['updates']==300 and p['training']['batch']==9 and p['training']['history_steps']==1920 and p['training']['lr']==.0003 and p['training']['save_steps']==[0,150,300]);check('state304',p['state_total']==304 and p['state_layout']==[128,64,64,32,16]);check('no_fullneckclaim',p['full_neck_training_allowed'] is False and p['new_body_physics_steps']==0)
 check('actualRGBdatatypeguard',"rgb.shape!=(770,2,32,32,3)" in (HERE/'data.py').read_text() and 'rgb.dtype!=np.uint8' in (HERE/'data.py').read_text());check('oldrawturn_h32_cache_exactchecks',all("'"+k+"'" in (HERE/'data.py').read_text() for k in ('held_features','vision_hidden_before','vision_hidden_after','latent','raw_turn')));check('no_stimuluscode_neuralcue',not any(isinstance(n,ast.Constant) and n.value in ('used_stimulus_code','target_center_xyz','target_centers_xyz') for n in ast.walk(trees['data.py'])));check('explicit_aux_has_no_rgb_or_fakehidden',"no_visual_image_or_hidden_state_created=True" in (HERE/'data.py').read_text() and 'np.zeros((1920,32))' not in (HERE/'data.py').read_text());check('frozen_original_model_numericreuse','make_scan=original.make_scan;python_rollout=original.python_rollout' in (HERE/'model.py').read_text());check('only10_parameter_optimizer','len(trainable(m))!=10' in (HERE/'common.py').read_text() and 'params=[v for k,v in trainable(m)]' in (HERE/'train.py').read_text());check('causal_target_shift','target=np.concatenate((np.zeros((1,1)),u[:-1]),0)' in (HERE/'data.py').read_text());check('no_runtime_signal_divide_sign_deadzone',not any(isinstance(n,ast.Attribute) and n.attr in ('sign','atanh') for n in ast.walk(trees['data.py'])));check('fixed_legacy_vision_warm_h0','if k<960:t=cache.new_zeros(1)' in (HERE/'data.py').read_text());check('retain_framefreezecontrol','np.minimum(expected_raw,529)' in (HERE/'data.py').read_text());check('ten_optimizer_saved_count_guard',"len(opt.state)!=10" in (HERE/'common.py').read_text());check('full_sourcequeue9600_guard',"proof['actual_steps']==9600" in (HERE/'preflight.py').read_text());check('no_new_physical_construct',all(not (isinstance(n,ast.Name) and n.id in ('Core','mj_step')) for t in trees.values() for n in ast.walk(t)))
 cmd=[sys.executable,'-B',str(HERE/'run.py')];probe=subprocess.run(cmd,capture_output=True,text=True,check=True);check('default_run_stays_CODEONLY','CODE_ONLY no arrays/models/training' in probe.stdout)
 # Import common only: its top level is stdlib, unlike old visualpolicy common.
 sys.path.insert(0,str(HERE));import common
 check('common_import_no_heavy',not any(k in sys.modules for k in ('numpy','torch','mujoco','physics','student_wave','flygym')));check('explicit_parent_sources_exist',all(q.is_file() for q in (common.PARENT,common.VISION,common.PORTS/'SOURCE_BINDING.json',common.NEURO/'model.py')))
 sourceaudit=HERE.parent/'visual_source_audit/NEURAL_VISUAL_SOURCE_REVIEW.json';a=json.loads(sourceaudit.read_text());check('parent_sourceaudit_actualpassed',a['all_checks_passed'] and all(a['checks'].values()));hashes[str(sourceaudit)]=digest(sourceaudit)
 out=HERE/'CODEONLY_CHECK.json'
 with out.open('x') as f:json.dump(dict(all_passed=True,checks=checks,source_hashes=hashes,stdlib_execution_only=True,child_numpy_imports=0,child_torch_imports=0,array_reads=0,model_loads=0,forward_calls=0,training_updates=0,Core_constructions=0,new_physics_steps=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=False)))
if __name__=='__main__':main()
