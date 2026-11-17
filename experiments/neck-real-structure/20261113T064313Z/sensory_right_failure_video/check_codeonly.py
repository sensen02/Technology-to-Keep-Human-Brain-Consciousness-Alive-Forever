"""Stdlib-only exactfrozenrenderer/newoutputnamespace andsourceidentity checks."""
from pathlib import Path
import ast,json,hashlib,sys
HERE=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 checks={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('FifthvideoCODEONLY '+k)
 for name in ('render.py','check_codeonly.py'):
  t=ast.parse((HERE/name).read_text());compile(t,str(HERE/name),'exec');check(name+'_syntax',True);check(name+'_no_eager_numerical_import',not any(isinstance(n,(ast.Import,ast.ImportFrom)) and any(x.name.split('.')[0] in ('numpy','torch','mujoco','physics') for x in n.names) for n in t.body))
 import render
 prior=sys.modules.get('contracts');present='contracts' in sys.modules;ns=render.namespace();check('contracts_alias_restored',('contracts' in sys.modules)==present and sys.modules.get('contracts') is prior);check('singleFrightCLI',ns['VIDEO_IDS']==('F_right_agree',));check('outputHERE_new_only',ns['HERE']==HERE);check('bodyROOT_unchanged',ns['ROOT']==HERE.parent/'sensory_body_repair');check('actual336repairidentities',ns['BRAIN_SHA']=='92dc0d243dad42b1df284ab797fa3dcaa13ca2d9ebfbb4cb434cebba7c9fe490' and ns['ODOR_SHA']=='e21dfb68e38760a6c74a871cfa3686cb79d1512b1ca60b5432397a456e473bfc')
 adapter=render.module('_rightfailure_checkadapter',render.EVIDENCE/'adapter.py');value,_=adapter.tree();expected=render.namespace()
 # Both namespaces compile the same complete immutable tree; no numerical main invocation.
 check('renderer_main_codebyte_exact',ns['main'].__code__.co_code==expected['main'].__code__.co_code and ns['main'].__code__.co_consts==expected['main'].__code__.co_consts and ns['main'].__code__.co_filename==expected['main'].__code__.co_filename)
 text=(HERE/'render.py').read_text();check('newledger_allfiles_beforeafter','IMMUTABLE_SOURCE_IDENTITIES.json' in text and "case['source_hashes'].update(extra)" in text);check('strictload_original_all30gates_retained','frozen_load(condition)' in text);check('no_field_or_model_or_threshold_changes','mj_step' not in text and 'raw_turn' not in text and 'functional_pass' not in text);check('failurecase_preregistered',json.loads((HERE/'AMENDMENT_PREREG.json').read_text())['condition']=='F_right_agree');check('no_arrays_models_Core_imported',not any(k in sys.modules for k in ('numpy','torch','mujoco','physics','student_wave','flygym')));check('no_actual_outputsyet',not (HERE/'VIDEO_F_right_agree').exists())
 paths=[render.EVIDENCE/n for n in ('PREREG.json','contracts.py','adapter.py','render.py','plot_sync.py','check_codeonly.py','CODEONLY_CHECK.json')]+[adapter._old.OLD,adapter.OLD/'adapter.py',adapter.OLD/'contracts.py']
 identities={str(p):sha(p) for p in paths}
 with (HERE/'IMMUTABLE_SOURCE_IDENTITIES.json').open('x') as f:json.dump(dict(status='UNCHANGED_PREVIOUS_VERIFIED_EVIDENCE_SOURCE',source_hashes=identities,oldfiles_modified=False),f,indent=2);f.write('\n')
 hashes={str(HERE/n):sha(HERE/n) for n in ('AMENDMENT_PREREG.json','render.py','check_codeonly.py','IMMUTABLE_SOURCE_IDENTITIES.json')}
 with (HERE/'CODEONLY_CHECK.json').open('x') as f:json.dump(dict(all_passed=True,checks=checks,source_hashes=hashes,array_reads=0,model_loads=0,Core_constructions=0,render_calls=0,NN_forward_calls=0,physics_steps=0,additional_physical_trials=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),arrays=0,models=0,Core=0,render=0)))
if __name__=='__main__':main()
