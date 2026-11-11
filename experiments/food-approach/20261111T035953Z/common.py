"""Balanced sensory-only continuation uses immutable original classes and equations."""
from pathlib import Path
import importlib.util,sys
HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'20261111T034309Z'
spec=importlib.util.spec_from_file_location('_balanced_sensor_original_common',PREVIOUS/'common.py');original=importlib.util.module_from_spec(spec);sys.modules[spec.name]=original;spec.loader.exec_module(original)
SteeringBehavior=original.SteeringBehavior
BEHAVIOR=original.BEHAVIOR;FIELD=original.FIELD;GEOMETRY=original.GEOMETRY
field_module=original.field_module;behavior_module=original.behavior_module
digest=original.digest;verify_hashes=original.verify_hashes;save_json=original.save_json;guard=original.guard

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
legacy_learning=module('_balanced_sensor_original_learning',PREVIOUS/'learning.py')
make_scan=legacy_learning.make_scan;production=legacy_learning.production;exact=legacy_learning.exact
PARENT=PREVIOUS/'TRAIN/checkpoint_001000.pt'
PARENT_SHA='76a4cbb48c7857aaa3a09e568c1a7b50877e0fa266028562a33d451e3c48aeb6'

def load_behavior(path,sha,require_final=True):
 m,p=legacy_learning.load_behavior(path,sha)
 if p['config'].get('kind')!='balanced_sensor_pretrain3000' or p['config']['steps']!=3000:raise ValueError('balanced sensory artifact identity mismatch')
 if p['curve'][-1]['step'] not in (2000,3000) or require_final and p['curve'][-1]['step']!=3000:raise ValueError('only final3000 for fresh sensory evaluation')
 return m,p
