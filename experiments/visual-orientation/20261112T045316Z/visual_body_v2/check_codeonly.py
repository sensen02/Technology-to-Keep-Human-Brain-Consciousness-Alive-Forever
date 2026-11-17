"""Stdlib-onlybody2namespace/AST/interface check; no model or numerical imports."""
from pathlib import Path
import ast,json,hashlib,sys,importlib.util
ROOT=Path(__file__).resolve().parent

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 checks={};names=('runtime.py','adapter.py','runner.py','audit.py','prepare.py','preflight.py','summary.py','orchestrate.py')
 for name in names:ast.parse((ROOT/name).read_text());checks[name+'_syntax']=True
 spec=importlib.util.spec_from_file_location('_body2_ast_only_adapter',ROOT/'adapter.py');ad=importlib.util.module_from_spec(spec);spec.loader.exec_module(ad)
 for name in ('runner.py','audit.py','prepare.py','preflight.py','summary.py','orchestrate.py'):compile(ad.source(name),name+'::codeonly','exec');checks[name+'_exactASTadapt_compile']=True
 spec=importlib.util.spec_from_file_location('_body2_stdlib_runtime',ROOT/'runtime.py');rt=importlib.util.module_from_spec(spec);spec.loader.exec_module(rt)
 checks['8unique_conditions']=len(rt.IDS)==len(set(rt.IDS))==8 and len(rt.NORMAL_IDS)==6 and len(rt.CONTROL_IDS)==2
 checks['actualv2loader_path']=str(rt.POLICY)==str(ROOT.parent/'visual_policy_v2') and rt.ROOT==ROOT
 checks['worldevents_newcontrols']=rt.frame_target('V_left_occluded',8)=='left' and rt.frame_target('V_flip_fixedframe',4.8)=='right'
 checks['oldbody1_clock_left_untouched']="cap['capture_time_s']==i*.020" in (rt.BODY1/'audit.py').read_text()
 checks['newclock_tolerance_AST']=ad.Adapt('audit.py').clock_changes==0 and '1e-10' in ast.unparse(ad.source('audit.py'))
 checks['noheavy_import_executed']=not any(n in sys.modules for n in ('numpy','torch','mujoco','physics','student_wave','flygym','PIL'))
 if not all(checks.values()):raise ValueError(checks)
 with (ROOT/'CODEONLY_CHECK.json').open('x') as f:json.dump(dict(all_passed=True,checks=checks,source_hashes={str(ROOT/n):sha(ROOT/n) for n in names+('PREREG_PROPOSAL.json','check_codeonly.py')},Core_constructions=0,physical_steps=0,model_loads=0,forward_calls=0,render_calls=0,array_loads=0,body1_inputs_unchanged=True),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavyimports=0,uniquephysicalbudget=8)))
if __name__=='__main__':main()
