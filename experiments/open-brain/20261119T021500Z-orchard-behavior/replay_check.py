"""Compare independent matching histories through the contact boundary."""
import argparse
import json
from pathlib import Path

import numpy as np


def compare(a,b,matched_inventory):
    result={}
    for name in ('INITIAL','TRACE','MEAL','QPOS'):
        result[name]=np.array_equal(np.load(a/f'{name}.npy'),np.load(b/f'{name}.npy'))
    with np.load(a/'NEURAL.npz') as first,np.load(b/'NEURAL.npz') as second:
        assert first.files==second.files
        result.update({f'NEURAL/{name}': np.array_equal(first[name],second[name])
                       for name in first.files})
    files = ['EYES.jsonl','eye_start_left.png','eye_start_right.png']
    if not matched_inventory:
        files.append('RESULT.json')
    for name in files:
        result[name]=(a/name).read_bytes()==(b/name).read_bytes()
    print(json.dumps(result,indent=2))
    assert all(result.values())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference',type=Path)
    p.add_argument('replay',type=Path)
    p.add_argument('--matched-inventory',action='store_true')
    args=p.parse_args()
    compare(args.reference,args.replay,args.matched_inventory)
