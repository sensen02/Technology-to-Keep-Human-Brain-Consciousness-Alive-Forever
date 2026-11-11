"""Separate fixed4000 head-refit physical binding to immutable original rollout."""
from pathlib import Path
import argparse,sys,types,json
import numpy as np
import model_io as io
HERE=io.HERE
# Parent physical adapter expects module name model; bind to exact original right source.
if 'model' in sys.modules and Path(sys.modules['model'].__file__).resolve()!=io.PARENT_DIR/'model.py':raise ValueError('conflicting original model alias')
sys.modules['model']=io.old
parent=io.old.module('_head_refit_parent_evaluator',io.PARENT_DIR/'evaluate_motor.py')
BASE=parent.BASE;ORIGINAL=parent.ORIGINAL;COMMANDS=parent.COMMANDS;HEAD_AUDIT=parent.HEAD_AUDIT;BASE_CHECKPOINT=parent.BASE_CHECKPOINT;PRIMARY=parent.PRIMARY
sha256=io.sha256;jsonable=parent.jsonable;command_at=parent.command_at;score=parent.score;flatten=parent.flatten;replace_once=parent.replace_once
CHECKPOINT=HERE/'TRAIN/checkpoint_004000.pt'
def pin():
 sw,identity,_=parent.pin();return sw,identity,io
def checked_load(module,path,digest):
 if Path(path).resolve()!=CHECKPOINT:raise ValueError('only fixed head-refit final4000 for physics')
 return io.load_checked(path,digest,require_final=True)
def source_paths():
 paths=parent.source_paths()+[HERE/n for n in ('model_io.py','train.py','verify_saved.py','evaluate_motor.py','verify_actual.py','prepare_eval.py','PREREG.json','EVAL_PREREG.json','READINESS.json')]+[HERE/'TRAIN/VERIFY_002000.json',HERE/'TRAIN/VERIFY_004000.json']
 import torch
 p=torch.load(CHECKPOINT,map_location='cpu',weights_only=False);paths += [Path(v) for v in p['config']['source_hashes']]+[Path(v) for v in p['config']['data_hashes']]
 for path in paths:
  if not path.is_file():raise FileNotFoundError(path)
 return list(dict.fromkeys(paths))
def build():
 # The parent's exact-count adapted rollout supplies native recorder and immutable dependencies.
 original=parent.build();src=PRIMARY.read_text();src=replace_once(src,"prereg['checkpoint_step']!=1500","prereg['checkpoint_step']!=4000")
 src=replace_once(src,"recorder=sw.NativeRecorder(out/'native') if args.native else None","recorder=PeakRecorder()")
 src=replace_once(src,"criteria=score(args.condition,auto,cycles,delta)","criteria=score(args.condition,auto,cycles,delta,recorder.close())")
 src=replace_once(src,"kind='round2 fixed1500 slow residual actual physical evaluation'","kind='RIGHT_ONLY uniform head-refit fixed4000 yaw006 bound3 actual physical evaluation'")
 src=replace_once(src,"native_audit='NOT_PERFORMED'","native_audit='LIGHTWEIGHT_EVERY_NATIVE_STEP_NONFOOT_PEAK_ONLY'")
 namespace={'__file__':str(PRIMARY),'__name__':'_head_refit_frozen_rollout'};exec(compile(src,str(PRIMARY)+'::HEAD_REFIT4000_explicit_adaptations','exec'),namespace)
 fn=namespace['evaluate'];g=dict(original.__globals__);g.update(HERE=HERE,ROOT=HERE,TRAIN=HERE,CHECKPOINT=CHECKPOINT,pin=pin,checked_load=checked_load,source_paths=source_paths,COMMANDS=COMMANDS,command_at=command_at,score=score,flatten=flatten)
 return types.FunctionType(fn.__code__,g,fn.__name__,fn.__defaults__,fn.__closure__)
def main():
 a=argparse.ArgumentParser();a.add_argument('--condition',choices=list(COMMANDS),required=True);a.add_argument('--parent-resource-released',action='store_true');args=a.parse_args()
 if not args.parent_resource_released:raise ValueError('parent must release shared heavy resource first')
 import os,torch
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});torch.set_num_threads(1)
 run=argparse.Namespace(condition=args.condition,checkpoint=str(CHECKPOINT),sha256=sha256(CHECKPOINT),out=str(HERE/('EVAL_'+args.condition)),training_complete=True,native=False);build()(run)
if __name__=='__main__':main()
