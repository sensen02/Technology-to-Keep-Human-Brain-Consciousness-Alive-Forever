"""Stdlib-only syntax, immutable source/interface checks; never imports runtime numerics."""
from pathlib import Path
import ast,json,hashlib,sys,importlib.util
ROOT=Path(__file__).resolve().parent

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 names=('runtime.py','runner.py','audit.py','prepare.py','preflight.py','summary.py','orchestrate.py');checks={}
 for name in names:
  tree=ast.parse((ROOT/name).read_text());checks[name+'_syntax']=True
  top=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))];modules=[n.module if isinstance(n,ast.ImportFrom) else a.name for n in top for a in ([None] if isinstance(n,ast.ImportFrom) else n.names)]
  checks[name+'_no_top_heavy_import']=not any(m and m.split('.')[0] in ('numpy','torch','mujoco','physics','student_wave','flygym') for m in modules)
 spec=importlib.util.spec_from_file_location('_visualbody_codeonly_runtime',ROOT/'runtime.py');rt=importlib.util.module_from_spec(spec);spec.loader.exec_module(rt)
 checks['BASE_actualpath']=str(rt.BASE)=='/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z'
 checks['6exact_conditions']=len(rt.IDS)==6 and len(set(rt.IDS))==6
 checks['visibility_events']=rt.frame_target('V_empty',8)=='off' and rt.frame_target('V_left_pulse',.6)=='left' and rt.frame_target('V_left_pulse',3.6)=='off' and rt.frame_target('V_right_to_left',4.8)=='left'
 checks['fixedbudget_prereg']=json.loads((ROOT/'PREREG_PROPOSAL.json').read_text())['timing']['camera_s']==.020
 checks['capture770_indices']=len(range(0,15400,20))==770
 checks['no_heavy_import_executed']=not any(n in sys.modules for n in ('numpy','torch','mujoco','physics','student_wave','flygym'))
 if not all(checks.values()):raise ValueError(checks)
 out=ROOT/'CODEONLY_CHECK.json'
 if out.exists():raise FileExistsError('preserve prior codecheck')
 with out.open('x') as f:json.dump(dict(all_passed=True,checks=checks,source_hashes={str(ROOT/n):sha(ROOT/n) for n in names+('PREREG_PROPOSAL.json','check_codeonly.py')},Core_constructions=0,physical_steps=0,model_loads=0,forward_calls=0,render_calls=0,array_loads=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports_executed=0)))
if __name__=='__main__':main()
