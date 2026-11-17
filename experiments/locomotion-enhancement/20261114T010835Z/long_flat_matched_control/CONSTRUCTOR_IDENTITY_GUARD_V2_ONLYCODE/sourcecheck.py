"""Pure stdlib source audit/mocks and first READY freeze; no scientific execution."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import constructor_policy as policy
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
PREVIOUS=BASE/'OPAQUE_HASH_GUARD_APPENDIX'
sys.dont_write_bytecode=True

def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path): return json.loads(Path(path).read_text())
def save(path,value):
    with Path(path).open('x') as f: json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def counters(): return dict(base_Core_constructors=0,forbidden_step_attempts=0,payload_attempts=0,neural_attempts=0)
def mocks():
    checks={};evidence={}
    with tempfile.TemporaryDirectory(prefix='constructor-identity-stdlib-') as folder:
        paths=[Path(folder)/name for name in ('physics.py','repair_core.py','foreign.py')]
        texts=['class Core(BaseCore):\n def __init__(self):\n  super().__init__()\n','class Core:\n def __init__(self):\n  pass\n','class Core:\n def __init__(self):\n  pass\n']
        for p,t in zip(paths,texts): p.write_text(t)
        repair=dict(__name__='_mock_repair');exec(compile(texts[1],str(paths[1])+'::ASTlabel','exec'),repair)
        physics=dict(__name__='_mock_physics',BaseCore=repair['Core']);exec(compile(texts[0],str(paths[0]),'exec'),physics)
        foreign=dict(__name__='_mock_foreign');exec(compile(texts[2],str(paths[2]),'exec'),foreign)
        expected={str(p.resolve()):sha(p) for p in paths[:2]}
        c=counters();audit=policy.ConstructorAudit(expected,c,expected_paths=list(expected))
        def stub(*a,**k): return None
        mj=types.SimpleNamespace(mj_step=stub,mj_step1=stub,mj_step2=stub)
        np=types.SimpleNamespace(load=stub,loadtxt=stub,genfromtxt=stub,fromfile=stub)
        with policy.prohibit_execution(mj,np,c,audit):
            obj=physics['Core']();other=foreign['Core']()
            checks['same_object_super_unique_one']=c['base_Core_constructors']==1
            checks['foreign_same_named_Core_excluded']=len(audit.physical_objects)==1 and audit.events[-1]['classification']=='FOREIGN_CONSTRUCTOR'
            checks['strip_AST_filename_label']=any(e['constructor_source']['co_filename'].endswith('::ASTlabel') and e['classification']=='PHYSICAL_CONSTRUCTOR' for e in audit.events)
            try: physics['Core']()
            except RuntimeError as error: checks['second_real_physical_object_rejected']='source-qualified physical Core' in str(error)
            else: checks['second_real_physical_object_rejected']=False
        checks['all_candidates_have_identity_and_MRO']=all(e['constructor_source']['source_sha256'] and e['MRO'] and e['self_id'] for e in audit.events)
        checks['mock_callbacks_restored']=mj.mj_step is stub and np.load is stub
        evidence['classification_sequence']=[e['classification'] for e in audit.events]
        evidence['unique_physical_objects_on_second_attempt']=len(audit.physical_objects)
        for case,call in (('step',lambda mj,np:mj.mj_step()),('payload',lambda mj,np:np.load('unused')),('neural',lambda mj,np:None)):
            cc=counters();aa=policy.ConstructorAudit(expected,cc,expected_paths=list(expected))
            if case=='neural':
                neural={};exec('def forward():\n return None\n',neural);call=lambda mj,np:neural['forward']()
            try:
                with policy.prohibit_execution(mj,np,cc,aa): call(mj,np)
            except RuntimeError: checks[case+'_still_blocked']=True
            else: checks[case+'_still_blocked']=False
        cc=counters();wrong={p:'0'*64 for p in expected};aa=policy.ConstructorAudit(wrong,cc,expected_paths=list(expected))
        try:
            with policy.prohibit_execution(mj,np,cc,aa): physics['Core']()
        except RuntimeError as error: checks['source_hash_mismatch_rejected']='SHA mismatch' in str(error)
        else: checks['source_hash_mismatch_rejected']=False
    spec=importlib.util.spec_from_file_location('_mock_pipeline_locator',HERE/'orchestrate.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.sha=lambda path:'digest';previous=list(sys.argv);sys.argv=['mock','--parent-resource-released']
    try:
        rejected=types.SimpleNamespace(read=lambda path:dict(all_passed=False,checks={'failed':False}))
        try: m.configure_runtime(rejected)
        except ValueError: checks['false_or_old_proof_rejected']=True
        else: checks['false_or_old_proof_rejected']=False
        proof=dict(all_passed=True,checks={'qualified':True},source_qualified_constructor_policy=True,base_Core_constructors=1,Core_constructions=1,physics_steps=0,old_Core=0,old_xml_compiles=1,forbidden_step_attempts=0,payload_attempts=0,neural_attempts=0,constructor_identity_audit={'unique_physical_object_ids':[123]},freeze_sha256='digest',guard_appendix_freeze_sha256='digest',preserved_125_failure_sha256='digest')
        qualified=types.SimpleNamespace(read=lambda path:proof if Path(path)==m.NEW_PROOF else 'untouched')
        m.configure_runtime(qualified)
        checks['qualified_new_metadata_locator_only']=qualified.read(BASE/'PARENT_ZERO_STEP_GUARD.json') is proof and qualified.read(BASE/'other.json')=='untouched'
    finally: sys.argv=previous
    return checks,evidence

def main():
    if (HERE/'READY.json').exists():
        for p,d in read(HERE/'FROZEN.json')['source_hashes'].items():
            if sha(p)!=d: raise ValueError('V2 frozen drift '+p)
        print(HERE/'READY.json');return
    for root in (BASE,PREVIOUS):
        for p,d in read(root/'FROZEN.json')['source_hashes'].items():
            if sha(p)!=d: raise ValueError('Inherited freeze drift '+p)
    compiled={str(p):bool(compile(p.read_text(),str(p),'exec')) for p in HERE.glob('*.py')}
    checks,evidence=mocks();checks['all_sources_compile']=all(compiled.values())
    checks['zero_scientific_imports']=not any(n.split('.')[0] in {'numpy','torch','mujoco','PIL','physics','flygym','student_wave','scipy'} for n in sys.modules)
    failure=read(BASE/'PARENT_ZERO_STEP_GUARD.json')
    checks['125_failure_preserved_false']=failure['all_passed'] is False and failure['physics_steps']==0 and failure['base_Core_constructors']==2
    constructors=[]
    for path in (policy.PHYSICS,policy.REPAIR):
        tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Core');init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        constructors.append(dict(path=str(path),sha256=sha(path),qualname='Core.__init__',line=init.lineno,source=ast.get_source_segment(path.read_text(),init)))
    # Validate metadata-only AST injection counts without importing delegated modules.
    previous_guard=ast.parse((PREVIOUS/'parent_zero_step_guard.py').read_text())
    checks['one_previous_guard_dispatch']=sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='guard' and n.func.attr=='main' for n in ast.walk(previous_guard))==1
    root_orch=ast.parse((BASE/'orchestrate.py').read_text())
    checks['two_root_proof_read_digest_locators']=sum(isinstance(n,ast.BinOp) and isinstance(n.op,ast.Div) and isinstance(n.right,ast.Constant) and n.right.value=='PARENT_ZERO_STEP_GUARD.json' for n in ast.walk(root_orch))==2
    if not all(checks.values()): raise ValueError(checks)
    save(HERE/'CONSTRUCTOR_SOURCE_AUDIT.json',dict(expected_physical_constructors=constructors,normalized_filename_rule='strip first :: suffix then Path.resolve; exact path+frozen SHA and Core.__init__',additional_runtime_code_object_match=True,old125_second_constructor_identity='UNKNOWN; no new actual trace',all_triggering_candidates_saved_in_future_report=True,physical_count_limit=1,foreign_same_name_excluded=True))
    save(HERE/'CODECHECK.json',dict(all_passed=True,checks=checks,mock_evidence=evidence,compiled=compiled,Core_constructions=0,NN_forwards=0,array_payload_loads=0,physics_steps=0,scientific_imports=0,mock_objects_are_plain_python=True,temp_fixtures_removed=True,actual_new_guard='NOT_RUN'))
    hashes={str(p):sha(p) for p in HERE.iterdir() if p.is_file() and p.name not in {'FROZEN.json','READY.json'}}
    for p in (BASE/'FROZEN.json',PREVIOUS/'FROZEN.json',BASE/'PARENT_ZERO_STEP_GUARD.json',PREVIOUS/'GUARD_EXECUTION.log',policy.PHYSICS,policy.REPAIR):
        if p.exists(): hashes[str(p)]=sha(p)
    hashes.update({str(p):sha(p) for p in (PREVIOUS/'parent_zero_step_guard.py',PREVIOUS/'orchestrate.py',BASE/'parent_zero_step_guard.py',BASE/'orchestrate.py')})
    save(HERE/'FROZEN.json',dict(source_hashes=hashes,base_freeze_sha256=sha(BASE/'FROZEN.json'),previous_appendix_freeze_sha256=sha(PREVIOUS/'FROZEN.json'),preserved125_failure_sha256=sha(BASE/'PARENT_ZERO_STEP_GUARD.json')))
    save(HERE/'READY.json',dict(status='READY_ONLYCODE_SOURCE_QUALIFIED_CONSTRUCTOR_GUARD_NOT_RUN',freeze_sha256=sha(HERE/'FROZEN.json'),new_proof=str(HERE/'PARENT_ZERO_STEP_GUARD_V2.json'),old125_failure_immutable=True,actual_guard='NOT_RUN',actual_trial='NOT_RUN',Core_constructions=0,NN_forwards=0,array_payload_loads=0,physics_steps=0))
    print(HERE/'READY.json')
if __name__=='__main__': main()
