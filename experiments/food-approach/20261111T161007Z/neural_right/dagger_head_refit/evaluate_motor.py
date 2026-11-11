"""New dense corrective-head final4000 actual deployment; RIGHT first, onlyPASS neutral."""
from pathlib import Path
import argparse,sys,types,json
import numpy as np
import model_io as io
HERE=io.HERE;right=io.legacy.old.old
if 'model' in sys.modules and Path(sys.modules['model'].__file__).resolve()!=HERE.parent/'model.py':raise ValueError('conflicting original RIGHT alias')
sys.modules['model']=right;parent=right.module('_dagger_original_physical_evaluator',HERE.parent/'evaluate_motor.py')
BASE=parent.BASE;ORIGINAL=parent.ORIGINAL;COMMANDS=parent.COMMANDS;HEAD_AUDIT=parent.HEAD_AUDIT;BASE_CHECKPOINT=parent.BASE_CHECKPOINT;PRIMARY=parent.PRIMARY
sha256=io.sha256;jsonable=parent.jsonable;command_at=parent.command_at;score=parent.score;flatten=parent.flatten;replace_once=parent.replace_once;CHECKPOINT=HERE/'TRAIN/checkpoint_004000.pt'
def pin():
 sw,identity,_=parent.pin();return sw,identity,io
def checked_load(module,path,digest):
 if Path(path).resolve()!=CHECKPOINT:raise ValueError('only fixed final4000 densecorrective model for physics')
 return io.load_checked(path,digest,require_final=True)
def source_paths():
 paths=parent.source_paths()+[HERE/n for n in ('model_io.py','check_train.py','train.py','verify_saved.py','evaluate_motor.py','verify_actual.py','prepare_eval.py','PREREG.json','EVAL_PREREG.json','READINESS.json','oracle.py','verify_oracle.py')]+[HERE/'ORACLE/RESULT.json',HERE/'ORACLE/VERIFY.json',HERE/'TRAIN/VERIFY_002000.json',HERE/'TRAIN/VERIFY_004000.json']
 import torch
 payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=False);paths += [Path(p) for p in payload['config']['source_hashes']]+[Path(p) for p in payload['config']['data_hashes']]
 for proof in (HERE/'VERIFY_right.json',HERE/'EVAL_right/RESULT.json'):
  if proof.is_file():paths.append(proof)
 for path in paths:
  if not path.is_file():raise FileNotFoundError(path)
 return list(dict.fromkeys(paths))
def build():
 original=parent.build();src=PRIMARY.read_text();src=replace_once(src,"prereg['checkpoint_step']!=1500","prereg['checkpoint_step']!=4000");src=replace_once(src,"recorder=sw.NativeRecorder(out/'native') if args.native else None","recorder=PeakRecorder()");src=replace_once(src,"criteria=score(args.condition,auto,cycles,delta)","criteria=score(args.condition,auto,cycles,delta,recorder.close())");src=replace_once(src,"kind='round2 fixed1500 slow residual actual physical evaluation'","kind='RIGHT_ONLY equal qualified CORRECTIVE densehead2730 fixed4000 actual physical evaluation'");src=replace_once(src,"native_audit='NOT_PERFORMED'","native_audit='LIGHTWEIGHT_EVERY_NATIVE_STEP_NONFOOT_PEAK_ONLY'")
 ns={'__file__':str(PRIMARY),'__name__':'_dagger_dense_frozen_rollout'};exec(compile(src,str(PRIMARY)+'::DAGGER_DENSE4000_exact_adaptations','exec'),ns);fn=ns['evaluate'];g=dict(original.__globals__);g.update(HERE=HERE,ROOT=HERE,TRAIN=HERE,CHECKPOINT=CHECKPOINT,pin=pin,checked_load=checked_load,source_paths=source_paths,COMMANDS=COMMANDS,command_at=command_at,score=score,flatten=flatten);return types.FunctionType(fn.__code__,g,fn.__name__,fn.__defaults__,fn.__closure__)
def main():
 a=argparse.ArgumentParser();a.add_argument('--condition',choices=list(COMMANDS),required=True);a.add_argument('--parent-resource-released',action='store_true');args=a.parse_args()
 if not args.parent_resource_released:raise ValueError('parent sharedresource release required')
 if args.condition=='neutral':
  proof=json.loads((HERE/'VERIFY_right.json').read_text())
  if not proof['integrity_all_pass'] or not proof['engineering_pass'] or proof['checkpoint_sha256']!=sha256(CHECKPOINT):raise ValueError('neutral only after sameSHA completeRIGHT PASS+audit')
 import os,torch
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});torch.set_num_threads(1);run=argparse.Namespace(condition=args.condition,checkpoint=str(CHECKPOINT),sha256=sha256(CHECKPOINT),out=str(HERE/('EVAL_'+args.condition)),training_complete=True,native=False);build()(run)
if __name__=='__main__':main()
