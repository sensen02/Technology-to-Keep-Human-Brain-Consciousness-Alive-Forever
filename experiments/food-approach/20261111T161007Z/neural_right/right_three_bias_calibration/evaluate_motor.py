"""Case-bound new actual3bias candidate evaluator, immutable original physics."""
from pathlib import Path
import argparse,sys,types,json
import numpy as np
import model_io as io
ROOT=HERE=io.ROOT
right=io.legacy.old.old
if 'model' in sys.modules and Path(sys.modules['model'].__file__).resolve()!=ROOT.parent/'model.py':raise ValueError('conflicting original RIGHT alias')
sys.modules['model']=right
parent=right.module('_large_RIGHT3bias_root_evaluator',ROOT.parent/'evaluate_motor.py')
BASE=parent.BASE;ORIGINAL=parent.ORIGINAL;COMMANDS=parent.COMMANDS;HEAD_AUDIT=parent.HEAD_AUDIT;BASE_CHECKPOINT=parent.BASE_CHECKPOINT;PRIMARY=parent.PRIMARY
sha256=io.sha256;jsonable=parent.jsonable;command_at=parent.command_at;score=parent.score;flatten=parent.flatten;replace_once=parent.replace_once

def pin():
 sw,identity,_=parent.pin();return sw,identity,io
def checked_load(module,path,digest):return io.load_checked(path,digest,require_frozen=True)
def source_paths(case=None):
 paths=parent.source_paths()+[ROOT/n for n in ('PREREG.json','EVAL_PREREG.json','model_io.py','prepare.py','evaluate_motor.py','verify_actual.py','pipeline.py','PROPOSAL.json','PREPARATION.json','FROZEN.json')]+[io.PARENT_DIR/'model_io.py',io.PARENT]
 if case is not None:paths += [Path(case)/n for n in ('candidate.pt','EVAL_PREREG.json','READINESS.json')]
 for p in paths:
  if not p.is_file():raise FileNotFoundError(p)
 return list(dict.fromkeys(paths))
def build(case):
 case=Path(case).resolve()
 if case.parent!=ROOT or case.name not in ('c00','c01','c02','c03'):raise ValueError('only four frozen candidate cases')
 original=parent.build();src=PRIMARY.read_text();src=replace_once(src,"prereg['checkpoint_step']!=1500","prereg['candidate_kind']!='RIGHT_THREE_BIAS_ACTUAL_V1'")
 src=replace_once(src,"recorder=sw.NativeRecorder(out/'native') if args.native else None","recorder=PeakRecorder()")
 src=replace_once(src,"criteria=score(args.condition,auto,cycles,delta)","criteria=score(args.condition,auto,cycles,delta,recorder.close())")
 src=replace_once(src,"kind='round2 fixed1500 slow residual actual physical evaluation'","kind='new large-angle RIGHT neural3bias fixed4 actual calibration candidate'")
 src=replace_once(src,"native_audit='NOT_PERFORMED'","native_audit='LIGHTWEIGHT_EVERY_NATIVE_STEP_NONFOOT_PEAK_ONLY'")
 namespace={'__file__':str(PRIMARY),'__name__':'_RIGHT3bias_frozen_rollout'};exec(compile(src,str(PRIMARY)+'::RIGHT3bias_exact_adaptations','exec'),namespace);fn=namespace['evaluate'];g=dict(original.__globals__)
 g.update(HERE=case,ROOT=ROOT,TRAIN=ROOT,CHECKPOINT=case/'candidate.pt',pin=pin,checked_load=checked_load,source_paths=lambda:source_paths(case),COMMANDS=COMMANDS,command_at=command_at,score=score,flatten=flatten)
 return types.FunctionType(fn.__code__,g,fn.__name__,fn.__defaults__,fn.__closure__)
def main():
 a=argparse.ArgumentParser();a.add_argument('--candidate',choices=['c00','c01','c02','c03'],required=True);a.add_argument('--condition',choices=list(COMMANDS),required=True);a.add_argument('--parent-resource-released',action='store_true');args=a.parse_args()
 if not args.parent_resource_released:raise ValueError('parent resource release required')
 import os,torch
 if args.condition=='neutral':
  proof=json.loads((ROOT/'SELECTED.json').read_text())
  if proof['selected'] is None or proof['selected']['id']!=args.candidate or len(proof['all4_results'])!=4:raise ValueError('neutral only same selected candidate after all4 RIGHT audits')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});torch.set_num_threads(1);case=ROOT/args.candidate;ck=case/'candidate.pt';run=argparse.Namespace(condition=args.condition,checkpoint=str(ck),sha256=sha256(ck),out=str(case/('EVAL_'+args.condition)),training_complete=True,native=False);build(case)(run)
if __name__=='__main__':main()
