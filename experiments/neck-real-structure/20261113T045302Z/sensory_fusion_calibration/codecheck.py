"""Stdlib-only preparation contracts. Never import numerical modules or read arraypayload."""
from pathlib import Path
import ast,json,hashlib,sys,importlib.util
HERE=Path(__file__).resolve().parent

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 if (HERE/'CODEONLY_CHECK.json').exists():raise FileExistsError('Inspectbeforeoverwritingcheck')
 names=('common.py','model.py','data.py','preflight.py','train.py','verify_saved.py','evaluate.py','run.py','codecheck.py');checks={};trees={}
 for name in names:
  path=HERE/name;tree=ast.parse(path.read_text());compile(tree,str(path),'exec');trees[name]=tree;checks['syntax_'+name]=True
 for name,required in [('data.py',('main','arrays','tensors','visual_replay','odor_replay','zero_receptor_neural')),('preflight.py',('main',)),('train.py',('main',)),('verify_saved.py',('main',)),('evaluate.py',('main','audit')),('run.py',('main',))]:
  funcs={n.name for n in trees[name].body if isinstance(n,ast.FunctionDef)};checks['entrycontract_'+name]=all(k in funcs for k in required)
 spec=importlib.util.spec_from_file_location('_fusion_codecheck_common',HERE/'common.py');c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)
 checks['default_common_no_numericalimports']=not any(k in sys.modules for k in ('numpy','torch','mujoco','physics','student_wave','flygym'));checks['exact_format']=c.FORMAT=='SOURCE_NECK_VISUAL_OLFACTORY_FUSION_CALIBRATION_V1';checks['27_train8dev_named']=len(c.TRAIN)==27 and len(set(c.TRAIN))==27 and len(c.DEV)==8;checks['111mask_no_targetspeed']=len(c.MASK)==111 and 237 not in c.MASK and c.MASK==list(range(42))+list(range(84,126))+list(range(168,195));checks['exact_frozen_sensory_sources']=c.ODOR_SHA=='66f423df4f00d801b2f0a194f539baa63ce84973765694e3d3111fbeb62aa034' and c.VISION_SHA=='996980806ac107137d6d5f4743de207fb3350e84a8a145dc5f52b8807398bb87'
 pr=json.loads((HERE/'PREREG.json').read_text());cfg=pr['training'];checks['fixed400_27via9x3']=cfg['updates']==400 and cfg['effective_batch']==27 and cfg['microbatch']==9 and cfg['accumulation']==3 and cfg['history_steps']==1920 and cfg['save_steps']==[0,200,400];checks['limits_no_autofallback']=cfg['RSS_cap_MiB']==cfg['wallcap_s']==1800 and cfg['no_extension'] and cfg['no_automatic_batch_reduction'];checks['only_brain_read_two_modalities']=next(n for n in trees['model.py'].body if isinstance(n,ast.ClassDef) and n.name=='NeuralBridge').body[1].name=='brain_step'
 method=next(n for n in ast.walk(trees['model.py']) if isinstance(n,ast.FunctionDef) and n.name=='brain_step');checks['body_API_packed_modalities']= [p.arg for p in method.args.args]==['self','delivered_an','modalities','h'];checks['no_runtime_teacherfusion_model']=not any(isinstance(n,ast.Constant) and n.value==1e-8 for n in ast.walk(trees['model.py']))
 func=next(n for n in trees['model.py'].body if isinstance(n,ast.FunctionDef) and n.name=='accumulated_backward');checks['accumulation_loop3groups']=any(isinstance(n,ast.Tuple) and [getattr(x,'value',None) for x in n.elts]==[0,9,18] for n in ast.walk(func));checks['accumulation_doesnot_optimizerstep']=not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('step','zero_grad') for n in ast.walk(func));checks['single_train_optimizerstep_syntax']=sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='step' for n in ast.walk(trees['train.py']))==1
 summary=c.read(c.VISBODY/'NEURAL_VISUAL_BODY_SUMMARY.json');review=c.read(c.VISBODY/'PARENT_READY_REVIEW.json');jobs=c.read(c.VISBODY/'JOBS.json');checks['actual_visual_independent_gate']=summary['normal6_all_functional_pass'] and summary['control2_all_pass'] and summary['all10_integrity_verified'] and review['integrity_all_pass'] and all(review['checks'].values()) and review['summary_sha256']==sha(c.VISBODY/'NEURAL_VISUAL_BODY_SUMMARY.json') and len(jobs)==22 and all(row['exit_code']==0 for row in jobs)
 design=c.read(c.SCOPE/'olfactory_design/OLFACTORY_INTEGRATION_DESIGN_REVIEW.json');checks['actual12same66f_metadata']=design['all_checks_passed'] and len(design['same66f_actual_archives'])==12 and all(design['checks'].values());checks['noD_training']=not any(n.startswith(('normal_','cut_','swap_','D_')) for n in c.TRAIN)
 for name in names:
  if name in ('model.py',):continue
  for n in trees[name].body:
   if isinstance(n,(ast.Import,ast.ImportFrom)):
    modules=[a.name for a in n.names] if isinstance(n,ast.Import) else [n.module or ''];checks['topimports_stdlib_'+name]=all(x not in ('numpy','torch','mujoco','physics','student_wave','flygym') for x in modules)
 checks['no_actual_heavy_modules']=not any(k in sys.modules for k in ('numpy','torch','mujoco','physics','student_wave','flygym'));checks['source_only_no_prior_outputs']=not any((HERE/n).exists() for n in ('FROZEN.json','DATASET.json','PREFLIGHT.json','TRAIN','JOBS.json'))
 if not all(checks.values()):raise ValueError('CODEONLYcontracts '+str({k:v for k,v in checks.items() if not v}))
 paths=[HERE/n for n in names]+[HERE/'PREREG.json'];result=dict(all_passed=True,checks=checks,source_hashes={str(p):sha(p) for p in paths},numerical_imports=False,array_payload_reads=0,model_loads=0,forward_calls=0,optimizer_steps=0,Core_constructions=0,physics_steps=0,olfaction_connected=False)
 with (HERE/'CODEONLY_CHECK.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),numerical_imports=False,model_loads=0,physics_steps=0)))
if __name__=='__main__':main()
