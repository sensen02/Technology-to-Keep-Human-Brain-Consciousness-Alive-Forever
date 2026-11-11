"""Explicit captured metadata and scalar-serialization hook; frozen physical code unchanged."""
from pathlib import Path
from types import SimpleNamespace
import argparse,importlib.util,json
import numpy as np
HERE=Path(__file__).resolve().parent
OLD=HERE.parent.parent/'20261110T191449Z'

def native(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:native(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [native(v) for v in value]
    return value

def prepare():
    source=OLD/'residual_profiles/evaluate_residual_profile.py'
    spec=importlib.util.spec_from_file_location('range_frozen_residual_profile',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=module.prepare();prior=original.pin_imports
    def pin():
        checked,cm,sw,identity=prior()
        def capture(core,config,extra_paths=()):
            config={**config,'range_diagnostic_amendment':json.loads((HERE/'PREREG.json').read_text()),'food_or_deployment_qualification':False,'new_hook':'metadata and numpy-scalar JSON conversion only'}
            return identity.capture(core,config,extra_paths=[*extra_paths,HERE/'PREREG.json',HERE/'run_one.py',HERE/'pipeline.py'])
        def atomic_json(path,value):return identity.atomic_json(path,native(value))
        return checked,cm,sw,SimpleNamespace(capture=capture,atomic_json=atomic_json,verify=identity.verify)
    original.pin_imports=pin
    return original

def main():
    p=argparse.ArgumentParser();p.add_argument('--profile',required=True);a=p.parse_args()
    reg=json.loads((HERE/'PREREG.json').read_text())
    if a.profile not in reg['profiles']:raise ValueError('Only four preregistered profiles')
    args=SimpleNamespace(checkpoint=reg['checkpoint'],sha256=reg['checkpoint_sha256'],profile=a.profile,out=str(HERE/a.profile),seed=20261110)
    return prepare().evaluate(args)
if __name__=='__main__':main()
