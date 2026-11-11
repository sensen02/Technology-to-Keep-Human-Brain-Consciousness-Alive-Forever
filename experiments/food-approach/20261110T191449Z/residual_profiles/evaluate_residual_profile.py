"""Explicit residual-loader/state adapter for unchanged fixed-profile evaluator.

No physical execution at import. The runtime pin_imports hook substitution below
is recorded in identity.capture config and adds every adapter source to hashes.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import importlib.util
import json
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
ORIGINAL=ROOT/'command_profiles/evaluate_profile.py'


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def prepare():
    original=load_module('food_original_fixed_profiles',ORIGINAL)
    original_pin=original.pin_imports
    def residual_pin():
        _,cm,sw,identity=original_pin() # Preserve actual baseline module pinning.
        sys.path.insert(0,str(ROOT/'residual_candidate'))
        candidate=load_module('food_residual_strict_profile_loader',ROOT/'residual_candidate/evaluate_candidate.py')
        flat=load_module('food_flat_residual_adapter',HERE/'flat_motor.py')
        def checked_load(checkpoint,expected_sha):
            model,payload=candidate.load_checked(checkpoint,expected_sha)
            return flat.FlatResidualMotor(model).eval(),payload
        def capture(core,config,extra_paths=()):
            config={**config,'explicit_residual_adapter':dict(
                loader='residual_candidate/evaluate_candidate.py:load_checked',state_layout='base128,residual64 flattened192',
                original_evaluator=str(ORIGINAL),new_prereg=json.loads((HERE/'PREREG.json').read_text()),
                hook='only checked_loader return wrapper and capture metadata; unmodified original scoring/physics')}
            added=[Path(__file__),HERE/'flat_motor.py',HERE/'PREREG.json',ROOT/'command_profiles/PREREG.json',ROOT/'REVIEW.md',ROOT/'residual_candidate/evaluate_candidate.py']
            return identity.capture(core,config,extra_paths=[*extra_paths,*added])
        identity_proxy=SimpleNamespace(capture=capture,verify=identity.verify,atomic_json=identity.atomic_json)
        return SimpleNamespace(load_checked_motor=checked_load),cm,sw,identity_proxy
    original.pin_imports=residual_pin # Explicit runtime hook only; no file mutation.
    return original


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--profile',choices=['neutral','halfleft','halfright','left_to_right','right_to_left'],required=True)
    p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=20261110)
    args=p.parse_args()
    if not Path(args.checkpoint).is_absolute() or not Path(args.out).is_absolute():raise ValueError('absolute checkpoint and freshout required')
    if len(args.sha256)!=64 or any(c not in '0123456789abcdef' for c in args.sha256):raise ValueError('exact lowercase SHA256 required')
    prereg=json.loads((HERE/'PREREG.json').read_text());original=prepare()
    if prereg['profiles']!=list(original.PROFILES) or prereg['turns_first_second_4p8s']!={k:list(v) for k,v in original.PROFILES.items()} or prereg['forward']!=0:
        raise ValueError('adapter profile prereg mismatch')
    return original.evaluate(args)

if __name__=='__main__':main()
