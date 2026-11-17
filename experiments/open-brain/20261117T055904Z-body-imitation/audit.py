"""Audit source-to-neck timing, physical metrics and restored action inference."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from body_run import HERE, PARENT, TICKS, MODEL_FILE, score, sha, write
from cases import CASES
from model import action, load_policy


def load_json(path):
    return json.loads(path.read_text())


def audit(folder, replay, checkpoint):
    data = Path(folder)
    tested = {}
    catalog=load_json(PARENT/'CATALOG.json')['ports']
    model=load_policy(checkpoint)
    with np.load(MODEL_FILE,allow_pickle=False) as saved:
        neutral=saved['neutral']
    for path in sorted(data.glob('*/RESULT.json')):
        entry=load_json(path)
        case=CASES[entry['case']['name']]
        assert (data/'log'/f'{case.name}.exit').read_text().strip()=='0'
        assert entry['weight_sha256']==sha(checkpoint)
        assert entry['body_snapshot_sha256']==sha(MODEL_FILE) and entry['catalog_sha256']==sha(PARENT/'CATALOG.json')
        trace_path=path.parent/'TRACE.npz'
        assert sha(trace_path)==entry['trace_sha256']
        with np.load(trace_path,allow_pickle=False) as z:
            trace=z['trace']; emitted=z['emitted']; arrived=z['arrived']; x=z['x']; ctrl=z['ctrl']; qpos=z['qpos']
        assert len(trace)==TICKS and len(x)==len(ctrl)==len(qpos)==TICKS//10
        assert x.shape[1]==entry['observed_dim']
        expected=np.zeros_like(arrived)
        expected[20:]=emitted[:-20]
        if case.cut_end:
            dn=np.array([catalog[root]['direction']=='DN' for root in entry['arrival_roots']])
            eligible=(np.arange(TICKS)>=case.cut_start)&(np.arange(TICKS)<case.cut_end+20)
            expected[np.ix_(eligible,dn)]=0
        assert np.array_equal(arrived,expected),case.name
        assert score(case,trace)['gates']==entry['gates']
        exact=sum(np.array_equal(action(model,row)+neutral,applied) for row,applied in zip(x,ctrl,strict=True))
        assert exact==len(x),case.name
        independent=None
        if replay and (replay/case.name/'TRACE.npz').exists():
            independent=all(np.array_equal(np.load(replay/case.name/'TRACE.npz',allow_pickle=False)[key],value)
                            for key,value in (('trace',trace),('arrived',arrived),('emitted',emitted),('ctrl',ctrl),('qpos',qpos)))
            assert independent
        tested[case.name]=dict(pass_=entry['pass_'], native_steps=len(trace), native_arrival_exact=True,
                               action_exact=exact, independent_physical_replay=independent,
                               arrival_count=int(arrived.sum()), gates=entry['gates'])
    assert tested
    write(data/'INDEPENDENT.json',{'cases':tested,'checkpoint_sha256':sha(checkpoint),'qualified':sum(r['pass_'] for r in tested.values())})
    print(tested)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--replay',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    audit(args.folder,args.replay,args.checkpoint)
