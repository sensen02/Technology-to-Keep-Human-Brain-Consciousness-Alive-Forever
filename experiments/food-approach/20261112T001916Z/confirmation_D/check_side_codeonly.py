"""Stdlib-only amendment ordering/sourcecontract checks, no old code/proposal mutation."""
from pathlib import Path
import ast,json,hashlib,importlib.util,sys
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 prior=json.loads((ROOT/'CODEONLY_CHECK.json').read_text());checks={};checks['original_CODEONLY_sources_unchanged']=all(sha(Path(p))==h for p,h in prior['source_hashes'].items())
 for n in ('AMENDMENT_SIDE_ONLY.json','side_admission.py','prepare_v2.py'):
  if n.endswith('.py'):compile((ROOT/n).read_text(),str(ROOT/n),'exec');ast.parse((ROOT/n).read_text())
 checks['newside_ASTcompile']=True;spec=importlib.util.spec_from_file_location('_D_side_stdcheck',ROOT/'prepare_v2.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);s=m.build_source();compile(s,'Dsideprepared','exec')
 checks['sideprepare_exact_compile']=True;checks['oldallCgate_not_edited']="s['all_integrity_pass'] and s['all_stable_body']" in (ROOT/'contracts.py').read_text();checks['newsideproof_reports_oldgate_andfront']="original_full_C6_gate_satisfied=bool(summary['all_stable_body'])" in (ROOT/'side_admission.py').read_text() and 'front_evidence' in (ROOT/'side_admission.py').read_text()
 checks['requiredfourSide_andnoodor_all6']=all(x in (ROOT/'side_admission.py').read_text() for x in ('C_left_near','C_right_near','C_left_far','C_right_far','C_noodor','stable and all6 and nf and other4'))
 checks['amendmentsourcebound']='side_amendment_sha256' in s and 'AMENDMENT_SIDE_ONLY.json' in s
 checks['noheavyimports']=not any(n in sys.modules for n in ('numpy','torch','mujoco','physics','scipy'));checks['noDexecutionfreezeortrials']=not any((ROOT/n).exists() for n in ('FROZEN.json','INPUT_BINDINGS_D.json','trials','JOBS_D.json'))
 assert all(checks.values()),checks
 with (ROOT/'CODEONLY_SIDE_CHECK.json').open('x') as f:json.dump(dict(status='D_SIDE_ONLY_CODEONLY_READY_ADMISSION_PENDING_PARENTSCREEN',all_passed=True,checks=checks,source_hashes={str(ROOT/n):sha(ROOT/n) for n in ('PREREG_PROPOSAL.json','AMENDMENT_SIDE_ONLY.json','side_admission.py','prepare_v2.py','check_side_codeonly.py')},Core_constructions=0,model_loads=0,physics_steps=0,execution_frozen=False),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=False,execution_frozen=False)))
if __name__=='__main__':main()
