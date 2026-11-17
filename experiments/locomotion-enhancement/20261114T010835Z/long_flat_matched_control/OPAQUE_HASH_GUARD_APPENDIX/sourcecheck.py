"""Additive standard-library verification/freeze; never executes scientific parent entry."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.dont_write_bytecode=True

def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
def save(path,value):
    with Path(path).open('x') as f: json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def main():
    if (HERE/'READY.json').exists():
        for p,d in json.loads((HERE/'FROZEN.json').read_text())['source_hashes'].items():
            if sha(p)!=d: raise ValueError('Appendix frozen drift '+p)
        print(HERE/'READY.json');return
    for p,d in json.loads((BASE/'FROZEN.json').read_text())['source_hashes'].items():
        if sha(p)!=d: raise ValueError('Base freeze drift '+p)
    files=list(HERE.glob('*.py'))
    for p in files: compile(p.read_text(),str(p),'exec')
    tree=ast.parse((HERE/'parent_zero_step_guard.py').read_text())
    mainnode=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    auditnode=next(n for n in mainnode.body if isinstance(n,ast.FunctionDef) and n.name=='payload_audit')
    namespace=dict(Path=Path,sys=sys)
    exec(compile(ast.Module(body=[auditnode],type_ignores=[]),'::audit_only','exec'),namespace)
    counters={'payload_attempts':0};active,audit=namespace['payload_audit'](counters)
    checks={}
    with tempfile.TemporaryDirectory(prefix='flat-opaque-guard-') as temporary:
        p=Path(temporary)/'opaque_dummy.npz';p.write_bytes(b'opaque dummy; not an array archive')
        sys.addaudithook(audit)
        try:
            checks['SHA_opaque_rb_allowed']=len(sha(p))==64
            ns=dict(Path=Path,hashlib=hashlib)
            exec("def sha256(path):\n return hashlib.sha256(Path(path).read_bytes()).hexdigest()\n",ns)
            checks['SHA_opaque_readbytes_allowed']=len(ns['sha256'](p))==64
            try:
                p.open('rb')
            except RuntimeError:
                checks['direct_payload_access_rejected']=True
            else:
                checks['direct_payload_access_rejected']=False
        finally: active[0]=False
    checks['all_base_frozen_hashes_unchanged']=True
    checks['zero_scientific_imports']=not any(n.split('.')[0] in {'numpy','torch','mujoco','PIL','physics','student_wave','scipy','flygym'} for n in sys.modules)
    if not all(checks.values()): raise ValueError(checks)
    save(HERE/'CODECHECK.json',dict(all_passed=True,checks=checks,Core_constructions=0,NN_forwards=0,array_payload_loads=0,physics_steps=0,render_calls=0,parent_compiled_parity='NOT_RUN',dummy_bytes='non-array temporary fixture removed',known_frozen_guard_repairs=['opaque SHA allowed while payload decoding blocked','reference model gets identical frozen mu1.5 and mesh/material alpha1 preprocessing'],late_window_diagnostic_only=True))
    hashes={str(p):sha(p) for p in HERE.iterdir() if p.is_file() and p.name not in {'FROZEN.json','READY.json'}}
    hashes[str(BASE/'FROZEN.json')]=sha(BASE/'FROZEN.json')
    save(HERE/'FROZEN.json',dict(source_hashes=hashes,base_freeze_sha256=sha(BASE/'FROZEN.json')))
    save(HERE/'READY.json',dict(status='READY_ADDITIVE_SOURCE_ONLY_PARENT_GUARD_NOT_RUN',freeze_sha256=sha(HERE/'FROZEN.json'),base_freeze_sha256=sha(BASE/'FROZEN.json'),scientific_execution=False))
    print(HERE/'READY.json')
if __name__=='__main__': main()
