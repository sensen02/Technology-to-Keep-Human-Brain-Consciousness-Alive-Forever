"""Audit the secondary development search and select frozen weights without held-out metrics."""
import json
import shutil
import numpy as np
import train as t


def read(p):
    return json.loads(p.read_text())


def checkpoint(key):
    path=t.HERE/'checkpoints'/(key+'.npz')
    assert t.sha(path)==key
    return path


def verify_proposal(previous, field, ids, groups, x, expected):
    with np.load(checkpoint(previous)) as f:
        t.ANCHOR={k:f[k] for k in f.files}
    t.FIELD,t.IDS,t.GROUP=field,ids,groups
    values=t.decode(np.asarray(x))
    with np.load(checkpoint(expected)) as f:
        for key in f.files:
            np.testing.assert_array_equal(values[key],f[key])
        assert abs(f['exc_fraction']).max()<=.08 and abs(f['inh_fraction']).max()<=.08


def main():
    branch=t.HERE/'explore'
    status=read(branch/'progress.json')
    start=read(branch/'seed.json')
    with np.load(t.HERE/f"block_{start['source_block']:04d}.npz") as b:
        verify_proposal(start['parent'],str(b['field']),b['ids'],b['groups'],start['x'],start['seed_weight'])
    prev=start['seed_weight']
    best=''
    nc=0
    for k in range(status['blocks']):
        with np.load(branch/f'block_{k:04d}.npz') as b:
            assert prev==str(b['anchor'])
            for x,key in [('selected','incumbent'),('winner','accepted')]:
                if len(b[x]):
                    verify_proposal(prev,str(b['field']),b['ids'],b['groups'],b[x],str(b[key]))
                else:
                    assert str(b[key])==(prev if key=='incumbent' else best)
            prev=str(b['incumbent'])
            best=str(b['accepted'])
        nc+=sum(len(g['x']) for g in read(branch/f'generation_{k:04d}.json'))
    assert (best or None)==status['accepted'] and nc==status['candidates']
    main_state=read(t.HERE/'progress.json')
    assert main_state['training_seconds']>=t.TARGET_SECONDS
    candidates=[('primary',main_state['best'],main_state['metrics'])]
    if best:
        candidates.append(('exploration',best,status['eligible_metrics']))
    for _,_,scores in candidates:
        assert all(s[0]>=r for s,r in zip(scores,[141.1,42.2,142.4]))
    name,digest,metrics=max(candidates,key=lambda a:t.rank(a[2]))
    selected=t.HERE/'SELECTED.npz'
    shutil.copyfile(checkpoint(digest),selected)
    t.dump(t.HERE/'FROZEN.json',dict(winner=name,sha256=t.sha(selected),metrics=metrics,
           development_candidates=candidates,parallel_training=status))
    print(name,digest,[m[0] for m in metrics])


if __name__=='__main__':
    main()
