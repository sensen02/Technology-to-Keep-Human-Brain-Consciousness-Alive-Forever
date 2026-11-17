"""Preparation-only stdlib checks of import/flag/schema contracts, never execute numerical code."""
from pathlib import Path
import ast,json,sys,hashlib,importlib.util,inspect
HERE=Path(__file__).resolve().parent

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 checks={};trees={};files=('common.py','model.py','data.py','preflight.py','train.py','verify_saved.py','evaluate.py','run.py','codecheck.py')
 for name in files:trees[name]=ast.parse((HERE/name).read_text());checks['syntax_'+name]=True
 p=json.loads((HERE/'PREREG.json').read_text());checks['actual_N2M1_ports']=p['DN_order']==['720575941433155799','720575941566493282'] and p['AN_order']==['720575941351919408'];checks['fixed200_only_final']=p['training']['updates']==200 and p['training']['save_steps']==[0,100,200] and p['training']['evaluate_only_step']==200;checks['same1920_fullbatch4']=p['training']['batch']==4 and p['training']['BPTT']==1920
 checks['not_fullneck_or_newbody']=p['full_structural_model_training_allowed'] is False and p['new_body_physics_steps']==0
 for name in ('data.py','preflight.py','train.py','evaluate.py','verify_saved.py'):
  checks['public_main_'+name]=any(isinstance(n,ast.FunctionDef) and n.name=='main' for n in trees[name].body)
 checks['public_eval_audit']=any(isinstance(n,ast.FunctionDef) and n.name=='audit' for n in trees['evaluate.py'].body)
 for name in ('run.py','common.py'):
  topimports=[n for n in trees[name].body if isinstance(n,(ast.Import,ast.ImportFrom))];names=[a.name.split('.')[0] for n in topimports for a in n.names] if name=='run.py' else [a.name.split('.')[0] for n in topimports for a in n.names]
  checks['entry_stdlib_only_'+name]=not any(x in names for x in ('torch','numpy','mujoco','physics','model'))
 spec=importlib.util.spec_from_file_location('_calibration_stdlib_common',HERE/'common.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
 checks['mask111_excludes_hiddenactiontargets']=len(c.MASK)==111 and c.MASK==list(range(0,42))+list(range(84,126))+list(range(168,195));checks['schema_parent_binding_exists']=Path(p['source_binding']).is_file() and sha(p['source_binding'])==p['actual_parent_source_binding_sha256'];checks['source_files_exist']=all(x.is_file() for x in c.sources())
 before=set(sys.modules);r=importlib.util.spec_from_file_location('_calibration_flagged_entry',HERE/'run.py');module=importlib.util.module_from_spec(r);r.loader.exec_module(module);checks['entry_import_has_main']=callable(module.main);checks['no_heavy_import']=not any(n in sys.modules for n in ('numpy','torch','mujoco','physics','student_wave','model'))
 checks['model_brain_step_onlyANcueh']=next(n for n in ast.walk(trees['model.py']) if isinstance(n,ast.FunctionDef) and n.name=='brain_step').args.args[1].arg=='delivered_an';checks['body111_normalization_trainonly']='normalization_from_train_only=True' in (HERE/'data.py').read_text();checks['real_sourcequeue_used']='SourceTransport(binding_path,binding_digest)' in (HERE/'common.py').read_text();checks['queue_release_and_consume_separate']='queue_release_step' in (HERE/'common.py').read_text() and 'neural_consume_step' in (HERE/'common.py').read_text()
 if not all(checks.values()):raise ValueError(str({k:v for k,v in checks.items() if not v}))
 report=dict(all_passed=True,checks=checks,source_hashes={str(HERE/n):sha(HERE/n) for n in files+('PREREG.json',)},numerical_imports=False,arrays_read=False,model_constructions=0,training_updates=0,physics_steps=0,kind='CODE_ONLY_IMPORT_CONTRACT_CHECK_NOT_NUMERICAL_PREFLIGHT')
 with (HERE/'CODEONLY_CHECK.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=False)))
if __name__=='__main__':main()
