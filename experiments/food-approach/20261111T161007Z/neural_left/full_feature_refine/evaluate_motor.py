"""New feature-refinement final2000 binding to original complete LEFT rollout."""
from pathlib import Path
import argparse,sys,json,types
import numpy as np
import model_io as io
HERE=io.HERE
PARENT_EVALUATOR=io.PARENT_DIR/'evaluate_motor.py'

def replace_once(source,old,new):
 if source.count(old)!=1:raise ValueError('refinement adaptation drift '+old)
 return source.replace(old,new)

def adapted_parent_source():
 source=PARENT_EVALUATOR.read_text()
 for old,new in [("checkpoint_004000.pt","checkpoint_002000.pt"),("prereg['checkpoint_step']!=4000","prereg['checkpoint_step']!=2000"),("strict_LEFT_HEAD_V1 final4000 coupled positive turn bound6 actual physical evaluation","LEFT_FEATURE_REFINE_V1 final2000 coupled positive turn bound6 actual physical evaluation")]:source=replace_once(source,old,new)
 return source

_namespace={'__file__':str(PARENT_EVALUATOR),'__name__':'_left_feature_complete_evaluator_adapter'}
exec(compile(adapted_parent_source(),str(PARENT_EVALUATOR)+'::FEATURE_FINAL2000','exec'),_namespace)
for _name in ('BASE','PRIMARY','HEAD_AUDIT','BASE_CHECKPOINT','ORIGINAL','CHECKPOINT','COMMANDS','jsonable','command_at','flatten','score','pin','checked_load','require_left_pass','sha256'):
 globals()[_name]=_namespace[_name]

def source_paths(include_readiness=True):
 names=['model_io.py','train.py','verify_saved.py','evaluate_motor.py','verify_actual.py','prepare_eval.py','PREREG.json','EVAL_PREREG.json']
 if include_readiness:names+=['READINESS.json']
 paths=[HERE/n for n in names]+[PARENT_EVALUATOR,io.PARENT_DIR/'verify_actual.py',PRIMARY,io.ROOT/'20261110T235830Z/evaluation/verify_actual.py',HEAD_AUDIT,HERE/'TRAIN/VERIFY_001000.json',HERE/'TRAIN/VERIFY_002000.json']
 paths+=list(BASE.glob('*.py'))
 sw,_,_=pin();paths += [Path(p) for p in sw.teacher_preflight()['files']]+[BASE/'STUDENT_CHECKS.json']
 import torch
 payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
 paths += [Path(p) for p in payload['config']['source_hashes']]+[Path(p) for p in payload['config']['data_hashes']]
 paths=list(dict.fromkeys(paths))
 for path in paths:
  if not path.is_absolute() or not path.is_file():raise FileNotFoundError(path)
 return paths
_namespace['source_paths']=source_paths

def build():return _namespace['build']()

def main():
 p=argparse.ArgumentParser();p.add_argument('--condition',choices=list(COMMANDS),required=True);p.add_argument('--parent-resource-released',action='store_true');args=p.parse_args()
 if not args.parent_resource_released:raise ValueError('parent exclusiveheavy release required')
 import os,torch
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});torch.set_num_threads(1)
 if args.condition=='neutral':require_left_pass()
 run=argparse.Namespace(condition=args.condition,checkpoint=str(CHECKPOINT),sha256=sha256(CHECKPOINT),out=str(HERE/('EVAL_'+args.condition)),training_complete=True,native=False);build()(run)
if __name__=='__main__':main()
