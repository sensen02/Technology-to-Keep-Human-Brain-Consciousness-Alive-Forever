"""Check teacher qualification, immutable neck and the original loader bridge."""
from pathlib import Path
import json
import numpy as np
from context import HERE,NECK,OLD_WEIGHT,first,hash_file,phase
from tasks import TRAIN,DEVELOP


def main():
 bridge={}
 for name in ('body_held_cruise_0','ground_bluebell__held0__open'):
  a=HERE/'BRIDGE'/name/'TRACE.npz'
  b=NECK/'FORMAL'/name/'TRACE.npz'
  with np.load(a,allow_pickle=False) as left,np.load(b,allow_pickle=False) as right:
   exact={key:bool(np.array_equal(left[key],right[key])) for key in left.files}
  assert all(exact.values()),(name,exact)
  bridge[name]=exact
 passed={}
 paths=[]
 for group,cases in (('TEACHER',TRAIN),('TEACHER_DEV',DEVELOP)):
  passed[group]={}
  for case in cases:
   folder=HERE/group/case.name
   result=json.loads((folder/'RESULT.json').read_text())
   assert result['source_sha256']==hash_file(HERE/'collect.py')
   assert result['parameter_sha256']==hash_file(NECK/'PARAMETERS.json')
   assert result['readout_sha256']==hash_file(NECK/'READOUT.json')
   assert result['trace_sha256']==hash_file(folder/'TRACE.npz')
   passed[group][case.name]=result['pass_']
   paths += [folder/'RESULT.json',folder/'TRACE.npz']
 for name in ('PROTOCOL.zh-CN.md','context.py','tasks.py','TASKS.json','collect.py','train.py','evaluate.py','batch.py','regress.sh'):
  paths.append(HERE/name)
 paths += [NECK/'PARAMETERS.json',NECK/'READOUT.json',NECK/'fixed_neck.py',NECK/'adapters.py',OLD_WEIGHT,
           NECK.parents[2]/'data/banc888/banc_888_meta.feather',first.MODEL_FILE,Path(phase.__file__)]
 old=json.loads((NECK/'IDENTITY.json').read_text())['sha256']
 assert all(hash_file(Path(p))==d for p,d in old.items())
 identity={str(p.resolve()):hash_file(p) for p in paths}
 identity.update(old)
 first.write(HERE/'TRAINING_INPUTS.json',dict(sha256=identity,teacher_pass=passed,bridge=bridge,
              baseline_weight_sha256=hash_file(OLD_WEIGHT)))
 print('bridge exact',len(bridge),'teacher', {group:sum(items.values()) for group,items in passed.items()},
       'input identity',len(identity))


if __name__=='__main__':main()
