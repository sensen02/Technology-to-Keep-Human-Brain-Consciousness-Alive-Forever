"""Stdlib checks for fixed four-cell diagnostic plan, no source arrays or model execution."""
from pathlib import Path
import ast,json,hashlib,importlib.util,sys
HERE=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 source=(HERE/'diagnose.py').read_text();tree=ast.parse(source);spec=importlib.util.spec_from_file_location('_codeonly_factorial_diagnostic',HERE/'diagnose.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);pr=json.loads((HERE/'PREREG.json').read_text());checks={}
 checks['public_main_and_flag']=callable(m.main) and '--execute-parent' in source
 checks['default_no_heavy_imports']=not any(n in sys.modules for n in ('numpy','torch','mujoco','physics','model'))
 matrix=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and [x.id for x in n.target.elts if isinstance(x,ast.Name)]==['label','body','initial'])
 checks['exact_four_condition_labels']=[x.elts[0].value for x in matrix.iter.elts]==['archived_cold','archived_warm','actual_cold','actual_warm']
 checks['whole1920_not_shortwindow']=len(m.window_indices(next(w for w in pr['windows'] if w['label']=='wholeANN')))==1920
 checks['warmvalue_fromrecord1159_and1160']=all(s in source for s in ("actual['AN_encoded'][1159]","actual['AN_held'][1160]","actual['AN_capture_step'][1160]!=5795","actual['AN_release_step'][1160]!=5797"))
 checks['no_model_or_physics_training']=all(s not in source for s in ('torch.optim','loss.backward','mj_step','Core(','requires_grad_(True)'))
 checks['model_parameters_frozen']= 'model.requires_grad_(False)' in source and 'with torch.no_grad()' in source
 checks['ANinitialonly_followingstreamunchanged']= 'held_an=an;held_dn=dn' in source and 'held_dn=torch.zeros' in source
 checks['source_alias_restore']= "finally:" in source and "sys.modules['model']=previous" in source
 checks['actual_and_archived_reproduction_present']=all(s in source for s in ('archived_cold_reproduces_original_','actual_warm_reproduces_saved_ANN_'))
 checks['normalization_stats_and_saturation_descriptive']=all(s in source for s in ('floor1e_6_count','outside_train_minmax_fraction','fraction_abs_gt_0_95'))
 checks['tensor_hash_and_source_beforeafter']=all(s in source for s in ('parameter_tensor_SHA_before','parameter_tensor_SHA_after','source_unchanged:'))
 checks['not_physical_counterfactual_claim']= 'physical_counterfactual_claim=False' in source and 'bistability_proven=False' in source
 if not all(checks.values()):raise ValueError('Diagnostic CODEONLY '+str({k:v for k,v in checks.items() if not v}))
 with (HERE/'CODEONLY_CHECK.json').open('x') as f:json.dump(dict(all_passed=True,checks=checks,source_hashes={str(HERE/n):sha(HERE/n) for n in ('diagnose.py','check_codeonly.py','PREREG.json')},heavy_imports=False,arrays_loaded=0,forward_calls=0,training_updates=0,Core_constructions=0,physics_steps=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=False)))
if __name__=='__main__':main()
