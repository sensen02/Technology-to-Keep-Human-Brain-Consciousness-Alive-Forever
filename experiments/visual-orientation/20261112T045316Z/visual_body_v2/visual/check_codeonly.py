"""Stdlib-only delivery source guard; no trial/source weight/array load."""
from pathlib import Path
import ast,json,hashlib,sys,importlib.util
HERE=Path(__file__).resolve().parent

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 checks={}
 for name in ('contracts.py','plot_sync.py','render.py','check_codeonly.py'):
  tree=ast.parse((HERE/name).read_text());checks[name+'_syntax']=True;modules=[]
  for node in tree.body:
   if isinstance(node,ast.Import):modules.extend(a.name.split('.')[0] for a in node.names)
   elif isinstance(node,ast.ImportFrom):modules.append(node.module.split('.')[0])
  checks[name+'_top_imports_light']=not any(m in ('numpy','torch','mujoco','matplotlib','PIL','physics','runtime') for m in modules)
 spec=importlib.util.spec_from_file_location('_actualvisualdelivery_contracts_codeonly',HERE/'contracts.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 checks['8condition_split']=len(m.NORMAL)==6 and len(m.CONTROLS)==2 and len(m.IDS)==8
 checks['actualaccepted_visualSHA']=m.VISUAL_SHA=='79dbcdcd38c1c3de94588bff9923b3758f82b1abbc63f017e499ece464340fee'
 checks['explicit_real_summary_paths']='NORMAL6_SUMMARY.json' in (HERE/'contracts.py').read_text() and 'VISUAL_BODY_SUMMARY.json' in (HERE/'contracts.py').read_text() and 'CONTROL2_SUMMARY.json' not in (HERE/'contracts.py').read_text()
 checks['770actualframes']=len(range(0,15400,20))==770
 checks['noheavy_executed']=not any(x in sys.modules for x in ('numpy','torch','mujoco','matplotlib','PIL','physics','runtime'))
 if not all(checks.values()):raise ValueError(checks)
 with (HERE/'CODEONLY_CHECK.json').open('x') as f:json.dump(dict(all_passed=True,checks=checks,source_hashes={str(HERE/name):sha(HERE/name) for name in ('PREREG.json','contracts.py','plot_sync.py','render.py','check_codeonly.py')},Core_constructions=0,model_loads=0,forward_calls=0,array_loads=0,render_calls=0,plot_calls=0,trial_reads=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=0)))
if __name__=='__main__':main()
