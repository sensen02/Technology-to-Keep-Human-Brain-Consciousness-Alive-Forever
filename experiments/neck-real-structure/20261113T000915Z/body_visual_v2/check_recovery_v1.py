"""Stdlib-only independent schema recovery check; preserves failedplot and prior29 report."""
from pathlib import Path
import ast,json,sys,hashlib
HERE=Path(__file__).resolve().parent

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 checks={};p=json.loads((HERE/'SCHEMA_RECOVERY_PREREG.json').read_text());e=json.loads((HERE/'SCHEMA_RECOVERY_EVIDENCE.json').read_text());old=json.loads((HERE/'CODEONLY_CHECK_BEFORE_SCHEMA_RECOVERY.json').read_text());current=HERE/'CODEONLY_CHECK.json';checks['old29report_bytes_preserved']=sha(current)==sha(HERE/'CODEONLY_CHECK_BEFORE_SCHEMA_RECOVERY.json')==p['initial29check_sha256'];checks['failed109log_bytes_preserved']=sha(HERE/'PLOT.log')==sha(HERE/'PLOT_FAILED_109.log')==p['failed_log_sha256']
 checks['all_priorcheck_metadata_true']=old['all_passed'] and all(old['checks'].values());checks['no_visual_artifact_before_recovery']=e['actual_render_output_artifacts_found']==[] and not any(x.suffix.lower() in ('.png','.jpg','.jpeg','.mp4','.npz') or x.name in ('VIDEO.json','PLOT_REVIEW.json','FRAME_MAPPING.json') for x in HERE.rglob('*') if x.is_file())
 for path,h in e['evidence_hashes'].items():checks['failureevidence_unchanged:'+path]=sha(path)==h
 files=('contracts.py','adapter.py','render.py','plot_sync.py','codecheck.py','check_recovery_v1.py')
 for name in files:ast.parse((HERE/name).read_text());checks['syntax_'+name]=True
 import contracts as c
 import adapter as a
 checks['one_kind_one_checkpath_change']=c._fix.kind_changes==c._fix.check_changes==1;checks['actualnewkind_in_admission']=p['actual_new_summary_kind'] in c._base_admission.__code__.co_consts;checks['oldkind_not_in_newadmission']=p['old_summary_kind'] not in c._base_admission.__code__.co_consts;checks['new_recoverycheck_path_active']='CODEONLY_CHECK_RECOVERY_V1.json' in c._base_admission.__code__.co_consts;checks['oldv1comparison_kind_preserved']=p['old_summary_kind'] in c.load_v1_summary.__code__.co_consts
 oldtree=next(n for n in ast.parse((c.OLD_VISUAL/'contracts.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='admission');newtree=c._new_admission_tree
 # Reverse exactly the two documented substitutions to prove every integrity/weight gate unchanged.
 class Reverse(ast.NodeTransformer):
  def visit_Constant(self,node):
   replacements={p['actual_new_summary_kind']:p['old_summary_kind'],'CODEONLY_CHECK_RECOVERY_V1.json':'CODEONLY_CHECK.json'}
   return ast.copy_location(ast.Constant(replacements[node.value]),node) if isinstance(node.value,str) and node.value in replacements else node
 reverse=Reverse().visit(ast.parse(ast.unparse(newtree)).body[0]);checks['all_old_admission_gates_AST_exact_except_two_strings']=ast.dump(reverse,include_attributes=False)==ast.dump(oldtree,include_attributes=False)
 summary=c.read(c.ROOT/'BODY_SUMMARY.json');oldsummary=c.read(c.OLD_BODY/'BODY_SUMMARY.json');checks['actualnewsummary_schema_matches']=summary['kind']==p['actual_new_summary_kind'] and summary['unique_physical_trials']==6 and summary['all6_integrity_verified'] is True;checks['actualoldsummary_schema_matches']=oldsummary['kind']==p['old_summary_kind'] and oldsummary['all6_integrity_verified'] is True;checks['actual6newsourceidentity']=summary['brain_sha256']==c.BRAIN_SHA and summary['neck_binding_sha256']==c.NECK_SHA and summary['motor_sha256']==c.MOTOR_SHA
 checks['complete13stages_preserved']=len(c.read(c.ROOT/'JOBS.json'))==13 and all(x['exit_code']==0 for x in c.read(c.ROOT/'JOBS.json'))
 for name in ('render.py','plot_sync.py'):
  ns=a.namespace(name);checks['newnamespace_'+name]=ns['ROOT']==c.ROOT and ns['HERE']==HERE and ns['load'] is c.load;nums=lambda t:[n.value for n in ast.walk(t) if isinstance(n,ast.Constant) and type(n.value) in (int,float)];checks['oldrender_numericconstants_exact_'+name]=nums(a.source(name))==nums(ast.parse((c.OLD_VISUAL/name).read_text()))
 checks['only_contract_of_previous_sources_changed']=all(sha(path)==h for path,h in old['source_hashes'].items() if Path(path)!=HERE/'contracts.py')
 checks['no_heavy_imports_or_arrays']=not any(k in sys.modules for k in ('numpy','torch','mujoco','physics','student_wave','model'))
 if not all(checks.values()):raise ValueError(str({k:v for k,v in checks.items() if not v}))
 paths=[HERE/n for n in files+('PREREG.json','AMENDMENT_PREREG.json','AMENDMENT_INITIAL_IDENTITY.json','CODEONLY_CHECK_INITIAL.json','SCHEMA_RECOVERY_PREREG.json','SCHEMA_RECOVERY_EVIDENCE.json','PLOT.log','PLOT_FAILED_109.log','CODEONLY_CHECK.json','CODEONLY_CHECK_BEFORE_SCHEMA_RECOVERY.json')]+[c.OLD_VISUAL/n for n in ('contracts.py','render.py','plot_sync.py')]
 with (HERE/'CODEONLY_CHECK_RECOVERY_V1.json').open('x') as f:json.dump(dict(all_passed=True,checks=checks,source_hashes={str(x):sha(x) for x in paths},actualnewsummarykind=p['actual_new_summary_kind'],oldcomparisonkind=p['old_summary_kind'],prior29report_preserved=True,failed109log_preserved=True,numerical_imports=False,array_deserialization=False,model_forward_calls=0,Core_constructions=0,new_training_updates=0,new_physics_steps=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=False,array_deserialization=False,new_summary_kind=p['actual_new_summary_kind'])))
if __name__=='__main__':main()
