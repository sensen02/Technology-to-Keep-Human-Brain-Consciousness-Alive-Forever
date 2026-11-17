"""Stdlib-only diagnostic contracts and actual NPZ header field validation; no payload imports."""
from pathlib import Path
import ast,json,hashlib,zipfile,struct,sys
HERE=Path(__file__).resolve().parent;BODY=HERE.parent/'sensory_body'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 if (HERE/'CODEONLY_CHECK.json').exists():raise FileExistsError('Inspectbeforeoverwrite')
 tree=ast.parse((HERE/'diagnose.py').read_text());compile(tree,str(HERE/'diagnose.py'),'exec');checks={'diagnostic_AST_compiles':True};pr=json.loads((HERE/'PREREG.json').read_text());checks['exact6cases']=pr['cases']==['O_left','O_left_odorcut','O_noodor','F_left_agree','F_right_agree','V_empty'];checks['fixedwindows']=pr['windows']==[dict(name='pre',start_s=.82,end_s=1.6),dict(name='early_cut',start_s=2.1,end_s=2.6),dict(name='late',start_s=3.,end_s=3.5),dict(name='last',start_s=8.6,end_s=9.6)];checks['fixedactualdb25']=pr['fixed_checkpoint_sha256']=='db2510d46900644c7f8c7b556b4b892dad2a88e3e37ce13792811916a928d4c6'
 imports=[a.name for n in tree.body if isinstance(n,ast.Import) for a in n.names];checks['default_imports_only_stdlib']=not any(k in imports for k in ('numpy','torch','mujoco','physics','student_wave'))
 mains=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main'];checks['parent_explicit_main']=len(mains)==1 and '--execute-parent' in (HERE/'diagnose.py').read_text();checks['nooptimizerorphysicalcalls']=not any(isinstance(n,ast.Attribute) and n.attr in ('Adam','backward','mj_step','mj_step1','mj_step2') for n in ast.walk(tree));checks['onlySourceTransportqueue']=any(isinstance(n,ast.Attribute) and n.attr=='SourceTransport' for n in ast.walk(tree))
 tuples=[[n.value for n in node.elts] for node in ast.walk(tree) if isinstance(node,ast.Tuple) and all(isinstance(n,ast.Constant) for n in node.elts)];checks['fourbrainmodes_actual']= ['baseline','odor_neural_cut','AN_held_cut','both_cuts'] in tuples;checks['fourodorprobes_present']=all(k in (HERE/'diagnose.py').read_text() for k in ('cold_zero_receptors','actual_receptorcut','normal_Olefthistory_then_zero','Flefthistory_then_zero'));checks['baselineallportsandbrainexact']=all(k in (HERE/'diagnose.py').read_text() for k in ('baseline_exact_','DN_encoded','AN_encoded','DN_held','AN_held','brain_hidden_after'));checks['scope_notphysicalcounterfactual']='physical_counterfactual_claim=False' in (HERE/'diagnose.py').read_text()
 wanted=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='wanted' for t in n.targets));fields=ast.literal_eval(wanted);sources={str(HERE/n):sha(HERE/n) for n in ('PREREG.json','diagnose.py','check_codeonly.py')}
 jobs=json.loads((BODY/'JOBS.json').read_text());checks['actual30stagescompleted']=len(jobs)==30 and all(v['exit_code']==0 for v in jobs)
 for case in pr['cases']:
  folder=BODY/'trials'/case;p=folder/'neural_decisions.npz';r=json.loads((folder/'RESULT.json').read_text());a=json.loads((folder/'VERIFY.json').read_text());checks[case+'_actualindependentaudit']=a['integrity_all_pass'] and all(a['checks'].values()) and a['result_sha256']==sha(folder/'RESULT.json');headers={}
  with zipfile.ZipFile(p) as z:
   for name in fields:
    with z.open(name+'.npy') as f:
     if f.read(6)!=b'\x93NUMPY':raise ValueError('NPYmagic')
     v=tuple(f.read(2));size=struct.unpack('<H' if v[0]==1 else '<I',f.read(2 if v[0]==1 else 4))[0]
     if size>65536:raise ValueError('Headercap')
     headers[name]=ast.literal_eval(f.read(size).decode('latin1' if v[0]<3 else 'utf8').strip())
  checks[case+'_actualfields_shapes']=headers['X238']['shape']==(3080,238) and headers['odor_hidden_after']['shape']==(3080,32) and headers['brain_hidden_after']['shape']==(3080,16) and headers['actor_receptors']['shape']==(3080,2) and all(h['shape'][0]==3080 for h in headers.values());checks[case+'_fixedcheckpoint']=r['brain_checkpoint_sha256']==pr['fixed_checkpoint_sha256']
  for q in (folder/'RESULT.json',folder/'VERIFY.json'):sources[str(q)]=sha(q)
 checks['noheavyimports']=not any(k in sys.modules for k in ('numpy','torch','mujoco','physics','student_wave','flygym'));checks['noexecutionoutputs']=not any((HERE/n).exists() for n in ('INPUTS_BEFORE.json','RESULT.json','OUTPUTS.npz'))
 if not all(checks.values()):raise ValueError('DiagnosticCODEONLY '+str({k:v for k,v in checks.items() if not v}))
 with (HERE/'CODEONLY_CHECK.json').open('x') as f:json.dump(dict(all_passed=True,checks=checks,source_hashes=sources,NPZ_headers_only=True,array_payload_reads=0,numerical_imports=False,model_loads=0,training_updates=0,physics_steps=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),NPZ_headers_only=True,numerical_imports=False,model_loads=0,physics_steps=0)))
if __name__=='__main__':main()
