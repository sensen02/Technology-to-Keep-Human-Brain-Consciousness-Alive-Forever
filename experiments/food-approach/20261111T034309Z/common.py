"""Sensor-only identities and immutable imports. No physical backend allowed."""
from pathlib import Path
import hashlib,importlib.util,sys,json
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'20261110T191449Z'
BEHAVIOR=OLD/'steering_behavior.py'
FIELD=OLD/'odor/food_field.py'
GEOMETRY=OLD/'odor/HEAD_GEOMETRY_AUDIT.json'

def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
behavior_module=module('_sensor_pretrain_frozen_behavior',BEHAVIOR)
field_module=module('_sensor_pretrain_frozen_field',FIELD)
SteeringBehavior=behavior_module.SteeringBehavior

def guard():
 bad=[n for n in sys.modules if n in ('mujoco','physics','student_wave','flygym') or n.startswith('mujoco.')]
 if bad:raise RuntimeError('Forbidden physical backend imported: '+str(bad))
def verify_hashes(hashes):
 for p,h in hashes.items():
  if not Path(p).is_absolute() or digest(p)!=h:raise ValueError('Source/data drift '+p)
def save_json(path,value):
 Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def frozen_sources():
 return {str(p):digest(p) for p in (BEHAVIOR,FIELD,GEOMETRY,HERE/'PREREG.json',HERE/'PATH_RESOLUTION_AMENDMENT.json')}
