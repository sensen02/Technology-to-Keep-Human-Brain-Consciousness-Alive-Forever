"""Parent entry: qualify NEW proof, then delegate unchanged single-trial pipeline."""
from pathlib import Path
import ast
import hashlib
import json
import sys
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
PREVIOUS=BASE/'OPAQUE_HASH_GUARD_APPENDIX'
NEW_PROOF=HERE/'PARENT_ZERO_STEP_GUARD_V2.json'

def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
def configure_runtime(r):
    if '--parent-resource-released' not in sys.argv: return
    # Read and qualify the new actual report before installing a locator proxy.
    proof=r.read(NEW_PROOF)
    audit=proof.get('constructor_identity_audit') or {}
    conditions=(proof.get('all_passed') is True,all(proof.get('checks',{}).values()),proof.get('source_qualified_constructor_policy') is True,proof.get('base_Core_constructors')==1,proof.get('Core_constructions')==1,proof.get('physics_steps')==0,proof.get('old_Core')==0,proof.get('old_xml_compiles')==1,proof.get('forbidden_step_attempts')==0,proof.get('payload_attempts')==0,proof.get('neural_attempts')==0,len(audit.get('unique_physical_object_ids',[]))==1,proof.get('freeze_sha256')==sha(BASE/'FROZEN.json'),proof.get('guard_appendix_freeze_sha256')==sha(HERE/'FROZEN.json'),proof.get('preserved_125_failure_sha256')==sha(BASE/'PARENT_ZERO_STEP_GUARD.json'))
    if not all(conditions): raise ValueError('NEW source-qualified actual guard proof required; old125 FALSE is never accepted')
    old_read=r.read
    def read(path):
        if Path(path)==BASE/'PARENT_ZERO_STEP_GUARD.json': return proof
        return old_read(path)
    r.read=read

def main():
    for p,d in json.loads((HERE/'FROZEN.json').read_text())['source_hashes'].items():
        if sha(p)!=d: raise ValueError('V2 freeze drift '+p)
    path=PREVIOUS/'orchestrate.py';tree=ast.parse(path.read_text())
    class Configure(ast.NodeTransformer):
        count=0
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='r' and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='load':
                self.count+=1
                return [node,ast.copy_location(ast.Expr(ast.Call(ast.Name('_configure_runtime',ast.Load()),[ast.Name('r',ast.Load())],[])),node)]
            return self.generic_visit(node)
    transformer=Configure();tree=transformer.visit(tree)
    if transformer.count!=1: raise ValueError('Exactly one runtime locator injection required')
    ast.fix_missing_locations(tree)
    # __file__ intentionally locates THIS new appendix freeze and same parent scope.
    ns=dict(__name__='_frozen_flat_pipeline_constructor_identity_v2',__file__=str(HERE/'orchestrate.py'),_configure_runtime=configure_runtime)
    exec(compile(tree,str(path)+'::new_qualified_guard_locator','exec'),ns)
    previous_load=ns['load']
    def load(name,path):
        if Path(path)==BASE/'orchestrate.py':
            import types
            original=ast.parse(Path(path).read_text())
            class ProofLocator(ast.NodeTransformer):
                count=0
                def visit_BinOp(self,node):
                    if isinstance(node.op,ast.Div) and isinstance(node.right,ast.Constant) and node.right.value=='PARENT_ZERO_STEP_GUARD.json' and isinstance(node.left,ast.Attribute) and isinstance(node.left.value,ast.Name) and node.left.value.id=='r' and node.left.attr=='HERE':
                        self.count+=1
                        return ast.copy_location(ast.Constant(str(NEW_PROOF)),node)
                    return self.generic_visit(node)
            locator=ProofLocator();adapted=locator.visit(original)
            if locator.count!=2: raise ValueError('Exactly guard read and summary proof SHA locators required')
            ast.fix_missing_locations(adapted)
            module=types.ModuleType(name);module.__file__=str(path);sys.modules[name]=module
            exec(compile(adapted,str(path)+'::new_proof_read_and_digest_locator','exec'),module.__dict__)
            return module
        return previous_load(name,path)
    ns['load']=load
    ns['main']()
if __name__=='__main__': main()
