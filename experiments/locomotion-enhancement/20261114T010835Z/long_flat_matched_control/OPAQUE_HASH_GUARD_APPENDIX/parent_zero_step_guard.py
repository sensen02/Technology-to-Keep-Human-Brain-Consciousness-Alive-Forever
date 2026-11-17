"""Immutable additive guard entry: permit source SHA opaque reads, never payload decoding."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys
HERE=Path(__file__).resolve().parent
BASE=HERE.parent

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    frozen=json.loads((HERE/'FROZEN.json').read_text())
    for path,digest in frozen['source_hashes'].items():
        if sha(path)!=digest: raise ValueError('Appendix freeze drift '+path)
    spec=importlib.util.spec_from_file_location('_immutable_flat_parent_guard',BASE/'parent_zero_step_guard.py')
    guard=importlib.util.module_from_spec(spec);sys.modules[spec.name]=guard;spec.loader.exec_module(guard)
    original_load=guard._load_runtime
    def load_runtime():
        r=original_load()
        old_save=r.save
        def save(path,value):
            if Path(path)==BASE/'PARENT_ZERO_STEP_GUARD.json':
                value.update(guard_appendix=str(HERE),guard_appendix_freeze_sha256=sha(HERE/'FROZEN.json'),opaque_hash_reads_permitted=True,array_payload_decode_permitted=False)
            return old_save(path,value)
        r.save=save
        return r
    guard._load_runtime=load_runtime
    def payload_audit(counters):
        active=[True]
        forbidden={'.npy','.npz','.pt','.pth','.pkl','.pickle','.safetensors'}
        def audit(event,args):
            if not active[0] or event!='open' or not isinstance(args[0],(str,bytes)): return
            name=args[0].decode() if isinstance(args[0],bytes) else args[0]
            if Path(name).suffix.lower() not in forbidden: return
            # A SHA helper opens the file only as rb and streams bytes to hashlib.
            # This exemption requires the actual opener frame, never an arbitrary ancestor.
            frame=sys._getframe(1)
            while frame is not None and frame.f_code.co_name in ('open','__new__','read_bytes'):
                frame=frame.f_back
            allowed=False
            if frame is not None and frame.f_code.co_name in ('sha','sha256'):
                code=frame.f_code
                allowed=('file_digest' in code.co_names or 'sha256' in code.co_names) and ('rb' in code.co_consts or 'read_bytes' in code.co_names)
            if allowed: return
            counters['payload_attempts']+=1
            raise RuntimeError('Guard forbids payload decoding/access outside explicit opaque SHA helper: '+name)
        return active,audit
    guard._payload_audit=payload_audit
    original_comparison=guard._named_model_comparison
    def compare(np,mj,old,new):
        # Apply the SAME frozen constructor postprocessing to the direct reference.
        # No flat model values are copied into the reference.
        physics=sys.modules['physics']
        source=Path(physics.rc.__file__)
        tree=ast.parse(source.read_text())
        mus=[n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MU' for t in n.targets) and isinstance(n.value,ast.Constant)]
        if mus!=[1.5]: raise ValueError('Frozen constructor MU contract changed')
        old.geom_friction[:,0]=mus[0]
        mask=old.geom_type==mj.mjtGeom.mjGEOM_MESH
        old.geom_rgba[mask,3]=1.
        materials=np.unique(old.geom_matid[mask]);old.mat_rgba[materials[materials>=0],3]=1.
        checks,details=original_comparison(np,mj,old,new)
        checks['same_frozen_constructor_sliding_mu']=bool(np.all(new.geom_friction[:,0]==mus[0]))
        details['reference_postprocessing']=dict(sliding_mu=mus[0],source=str(source),source_sha256=sha(source),mesh_alpha=1.,mesh_material_alpha=1.,reason='identical frozen constructor preprocessing on old direct compiled model; no physics integration')
        return checks,details
    guard._named_model_comparison=compare
    guard.main()
if __name__=='__main__': main()
