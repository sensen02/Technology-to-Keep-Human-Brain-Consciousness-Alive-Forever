"""Standard-library-only preparation checks, no data/model/physical imports."""
from pathlib import Path
import ast,json,hashlib,sys
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 names=('common.py','data.py','engine.py','generate_data.py','preflight.py','train.py','verify_saved.py','evaluate.py','audit_validation.py','run_candidate.py');trees={n:ast.parse((HERE/n).read_text()) for n in names};pr=json.loads((HERE/'PREREG.json').read_text());checks={}
 checks['all10_AST_compile']=True
 for n in names:compile((HERE/n).read_text(),str(HERE/n),'exec')
 checks['prereg24train12validation']=len(pr['data']['motifs'])*len(pr['data']['train_variants'])==24 and len(pr['data']['motifs'])*len(pr['data']['validation_variants'])==12
 checks['fixed1000freshAdam1e4']=pr['training']['updates']==1000 and pr['training']['lr']==.0001 and pr['training']['save_steps']==[500,1000] and pr['training']['validation_only_step']==1000
 checks['no_runtime_model_redefinition']=not any(isinstance(n,ast.ClassDef) for n in trees['common.py'].body)
 api_calls=[n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else '' for t in trees.values() for n in ast.walk(t) if isinstance(n,ast.Call)]
 checks['no_physics_or_posewrite_APIs']=not any(x in ('Core','mj_step','mj_step1','mj_step2','reset_from','tick','tick_policy') for x in api_calls)
 checks['no_physical_imports']=not any(isinstance(n,(ast.Import,ast.ImportFrom)) and any(x in ast.unparse(n).split() for x in ('mujoco','physics','student_wave','flygym')) for t in trees.values() for n in ast.walk(t))
 checks['runtime_only_two_receptors_original_model']='m=SteeringBehavior' not in (HERE/'common.py').read_text() and 'm,p=io.load_behavior(PARENT,PARENT_SHA)' in (HERE/'common.py').read_text()
 checks['long1920fullBPTT_no_detach']='range(x.shape[0])' in (HERE/'engine.py').read_text() and '.detach(' not in (HERE/'engine.py').read_text()
 checks['validation_off_gate_preserved']='rms<=.05' in (HERE/'evaluate.py').read_text() and 'cfg[\'off_s\']+.5' in (HERE/'evaluate.py').read_text()
 checks['no_heavy_import_or_execution_now']=not any(k in sys.modules for k in ('torch','numpy','mujoco','physics','scipy'))
 checks['datasets_and_training_not_yet_created']=not any((HERE/n).exists() for n in ('train.npz','validation.npz','DATASET.json','FROZEN.json','TRAIN','VALIDATION','JOBS.json'))
 assert all(checks.values()),checks
 report=dict(status='CODEONLY_READY_NO_HEAVY_EXECUTION',all_passed=True,checks=checks,source_hashes={str(HERE/n):sha(HERE/n) for n in names+('PREREG.json','check_codeonly.py')},model_loads=0,Core_constructions=0,forward_calls=0,optimizer_updates=0,physics_steps=0,datasets_generated=0,not_actual_gradient_preflight=True)
 with (HERE/'CODEONLY_CHECK.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,ensure_ascii=False);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=False,physics_steps=0,forward_calls=0)))
if __name__=='__main__':main()
