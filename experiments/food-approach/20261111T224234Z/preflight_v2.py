"""Stdlib-only additive identity and exact-adaptation checks; no repeated model preflight."""
from pathlib import Path
import ast,hashlib,json,importlib.util
ROOT=Path(__file__).resolve().parent
MOTOR_DIR=ROOT.parent/'20261111T161007Z/combined_turning'
MOTOR_SHA='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def new(p,obj):
 with p.open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,ensure_ascii=False);f.write('\n')
def main():
 frozen=read(ROOT/'FROZEN.json');first=read(ROOT/'PREFLIGHT.json');checks={}
 checks['original20_model_checks_preserved']=first['all_passed'] and all(first['checks'].values()) and first['Core_constructions']==0
 checks['original_frozen_sources_unchanged']=all(sha(Path(p))==h for p,h in frozen['source_hashes'].items())
 checks['no_physical_trials_started']=not (ROOT/'trials').exists()
 r=read(MOTOR_DIR/'EVAL_neutral/RESULT.json');a=read(MOTOR_DIR/'VERIFY_neutral.json')
 checks['baseline_same_SHA']=r['checkpoint_sha256']==a['checkpoint_sha256']==MOTOR_SHA
 checks['baseline_result_audit_bound']=a['result_sha256']==sha(MOTOR_DIR/'EVAL_neutral/RESULT.json') and a['integrity_all_pass'] and all(a['checks'].values())
 checks['baseline_original_walking_exact']=all(v for k,v in a['checks'].items() if 'neutral_original' in k) and r['engineering_pass']
 for name,key in [('trace_1ms.npz','trace'),('neural_decisions.npz','neural_decisions'),('geometry_1ms.npz','geometry')]:checks['baseline_'+key+'_actual_hash']=sha(MOTOR_DIR/'EVAL_neutral'/name)==r[key]['sha256']
 names=('AMENDMENT_01_BEFORE_ANY_PHYSICS.json','bindings_v2.py','runner_v2.py','audit_v2.py','orchestrate_v2.py','summarize_B_v2.py','preflight_v2.py')
 for name in names:
  if name.endswith('.py'):ast.parse((ROOT/name).read_text())
 checks['all_new_AST']=True
 for name in ('orchestrate_v2','summarize_B_v2'):
  spec=importlib.util.spec_from_file_location('_check_'+name,ROOT/(name+'.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);compile(module.build_source(),name+'::exact_old_adapted','exec')
 checks['exact_source_adaptations_compile']=True
 checks['no_heavy_imports']=not any(n in __import__('sys').modules for n in ('torch','numpy','mujoco','physics'))
 assert all(checks.values()),checks
 report=dict(all_passed=True,checks=checks,Core_constructions=0,physics_steps=0,model_loads=0,original_zeroCore_preflight=str(ROOT/'PREFLIGHT.json'),scope='Only additive baseline proof and planned effective-input latency clarification; model and physical executables inherited unchanged')
 new(ROOT/'PREFLIGHT_V2.json',report)
 paths=[ROOT/n for n in names]+[ROOT/'PREFLIGHT_V2.json',ROOT/'FROZEN.json',ROOT/'PREFLIGHT.json',MOTOR_DIR/'VERIFY_neutral.json']+[MOTOR_DIR/'EVAL_neutral'/n for n in ('IDENTITY.json','RESULT.json','trace_1ms.npz','neural_decisions.npz','geometry_1ms.npz')]
 new(ROOT/'FROZEN_SUPPLEMENT.json',dict(kind='V2_ADDITIVE_FROZEN_BEFORE_ANY_PHYSICS',source_hashes={str(p):sha(p) for p in paths},old_frozen_sha256=sha(ROOT/'FROZEN.json'),amendment_sha256=sha(ROOT/'AMENDMENT_01_BEFORE_ANY_PHYSICS.json'),no_physics_or_training_started=True))
 print(json.dumps(dict(all_passed=True,checks=len(checks),supplement_sha256=sha(ROOT/'FROZEN_SUPPLEMENT.json'),Core_constructions=0,heavy_imports=False)))
if __name__=='__main__':main()
