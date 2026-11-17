"""Parent future one-Core guard with source-qualified constructor audit; stdlib on import."""
from pathlib import Path
import ast
import hashlib
import json
import sys
import constructor_policy as policy
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
PREVIOUS=BASE/'OPAQUE_HASH_GUARD_APPENDIX'
NEW_PROOF=HERE/'PARENT_ZERO_STEP_GUARD_V2.json'

def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
def verify():
    frozen=json.loads((HERE/'FROZEN.json').read_text())
    for p,d in frozen['source_hashes'].items():
        if sha(p)!=d: raise ValueError('V2 source/old failure freeze drift '+p)
    return frozen

def install_and_run(guard):
    state={}
    expected={str(p.resolve()):sha(p) for p in (policy.PHYSICS,policy.REPAIR)}
    def prohibit(mj,np,counters):
        audit=policy.ConstructorAudit(expected,counters)
        state['audit']=audit
        return policy.prohibit_execution(mj,np,counters,audit)
    guard._prohibit_execution=prohibit
    previous_load=guard._load_runtime
    def load_runtime():
        r=previous_load();old_save=r.save
        def save(path,value):
            if Path(path)==BASE/'PARENT_ZERO_STEP_GUARD.json':
                audit=state.get('audit')
                value.update(kind='PARENT_FLAT_MATCHED_ZERO_STEP_GUARD_CONSTRUCTOR_IDENTITY_V2',constructor_identity_audit=audit.report() if audit else None,guard_appendix=str(HERE),guard_appendix_freeze_sha256=sha(HERE/'FROZEN.json'),previous_guard_appendix_freeze_sha256=sha(PREVIOUS/'FROZEN.json'),preserved_125_failure_sha256=sha(BASE/'PARENT_ZERO_STEP_GUARD.json'),source_qualified_constructor_policy=True,actual_identity_trace_pending=False)
                if value['all_passed'] and (audit is None or len(audit.physical_objects)!=1):
                    value['all_passed']=False;value['checks']['exact_source_qualified_physical_object']=False
                else:
                    value['checks']['exact_source_qualified_physical_object']=audit is not None and len(audit.physical_objects)==1
                return old_save(NEW_PROOF,value)
            return old_save(path,value)
        r.save=save
        return r
    guard._load_runtime=load_runtime
    guard.main()

def main():
    if '--parent-resource-released' not in sys.argv:
        raise SystemExit('ONLYCODE: explicit parent-resource-released required')
    verify()
    # Reuse frozen opaque-SHA and identical reference preprocessing wrappers.
    # Only the final dispatch is replaced; no frozen file is written.
    path=PREVIOUS/'parent_zero_step_guard.py';tree=ast.parse(path.read_text())
    class Dispatch(ast.NodeTransformer):
        count=0
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='guard' and node.value.func.attr=='main':
                self.count+=1
                return ast.copy_location(ast.Expr(ast.Call(ast.Name('_install_and_run',ast.Load()),[ast.Name('guard',ast.Load())],[])),node)
            return self.generic_visit(node)
    transformer=Dispatch();adapted=transformer.visit(tree)
    if transformer.count!=1: raise ValueError('Exact previous dispatch required')
    ast.fix_missing_locations(adapted)
    ns=dict(__name__='_frozen_opaque_guard_source',__file__=str(path),_install_and_run=install_and_run)
    exec(compile(adapted,str(path)+'::constructor_identity_v2_dispatch','exec'),ns)
    ns['main']()
if __name__=='__main__': main()
