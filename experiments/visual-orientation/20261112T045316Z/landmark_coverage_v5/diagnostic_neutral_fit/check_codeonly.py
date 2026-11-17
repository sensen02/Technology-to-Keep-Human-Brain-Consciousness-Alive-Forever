"""Stdlib checks only; no checkpoint/dataset/model/array load."""
from pathlib import Path
import ast,json,sys,hashlib
HERE=Path(__file__).resolve().parent

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 path=HERE/'evaluate_train_neutral.py';src=path.read_text();tree=ast.parse(src);pr=json.loads((HERE/'PREREG.json').read_text());checks={}
 checks['syntax_compiles']=bool(compile(tree,str(path),'exec'));checks['fixedfinal812f']=pr['checkpoint_sha256']=='812f0bf409f39550eed6e47d2dbafb0287ce894e725a632c0ffc48afc009f02b'
 checks['exacttrainmotifpopulation']="r['motif'] in ('empty','center','occluder_only')" in src and 'len(indices)!=240' in src
 checks['originaltrainframebatch']="data.batch(rgb,train,[j])" in src and 'original.production(model,x)' in src;checks['nograd_eval_weightsfrozen']='with torch.no_grad():' in src and 'model.requires_grad_(False);model.eval()' in src
 checks['no_optimizer_or_backward']=not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('backward','step','zero_grad') for n in ast.walk(tree))
 checks['no_physics_construction']='Core(' not in src and 'mj_step' not in src;checks['beforeafterparam_andinputs']='if before!=after' in src and 'input sourcechanged during diagnostic' in src
 checks['independent_saved_and_validation_identity']='actualfailed-validation and savedreplay identity required' in src;checks['whole_and_active_RMS_distinct']='full9_6s_RMS' in src and 'active_scene_RMS' in src
 checks['paircoverage_descriptive_only']='pair_times_selected' in src and 'no补训/nothresholdchange' in src;checks['aliases_finally_restored']='finally:' in src and "sys.modules.pop(name,None)" in src
 heavy={k:k in sys.modules for k in ('numpy','torch','mujoco','physics','flygym')};checks['heavy_absent']=not any(heavy.values())
 if not all(checks.values()):raise ValueError(checks)
 with (HERE/'CODEONLY_CHECK.json').open('x') as f:json.dump(dict(all_passed=True,checks=checks,source_hashes={str(p):sha(p) for p in (HERE/'PREREG.json',path,HERE/'check_codeonly.py')},heavy_imports=heavy,arrays_read=0,models_loaded=0,forward_calls=0,training_updates=0,Core_constructions=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=heavy)))
if __name__=='__main__':main()
