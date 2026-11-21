"""Select by development behavior and verify the exported actor independently."""
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from train_long import HERE, PREVIOUS, rating, sha
from balanced import MirrorActor
from policy_runtime import actor_mean


def run():
    torch.set_num_threads(1)
    groups=('TRAIN','GATED')
    data={k:json.loads((HERE/k/'HELDOUT.json').read_text()) for k in groups}
    assert all(d['last']['step']==180000 and d['last']['updates']==89251 for d in data.values())
    best=max(groups,key=lambda k:rating(data[k]['selected']['stats']))
    old=torch.load(PREVIOUS,map_location='cpu',weights_only=True)
    row=data[best]['selected']
    place=Path(row['checkpoint'])
    assert sha(place)==row['sha256']
    payload=torch.load(place,map_location='cpu',weights_only=True)
    assert any(not torch.equal(p,old['actor'][k]) for k,p in payload['actor'].items())
    model=MirrorActor()
    model.load_state_dict(payload['actor'])
    inputs=np.random.default_rng(437202).uniform(-1.,1.,(1000,14)).astype(np.float32)
    source=HERE/best/'POLICY.npz'
    with np.load(source,allow_pickle=False) as saved,torch.inference_mode():
        match=float(np.max(np.abs(actor_mean(dict(saved),inputs)-model(torch.from_numpy(inputs),False)[0].numpy())))
        assert match<8e-6
        assert json.loads(saved['identity'].item())['brain_sha256']==old['identity']['brain_sha256']
    target=HERE/'POLICY.npz'
    shutil.copyfile(source,target)
    result={
        'branch':best,'checkpoint':row,'sha256':sha(target),'numpy_parity_max':match,
        'selection_used_development_only':True,
        'heads':{k:{tag:val[tag] for tag in ('holdout_before','holdout_after','selected','last')}
                 for k,val in data.items()}}
    (HERE/'SELECTION.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(best,row['step'],data[best]['holdout_after'],flush=True)


if __name__=='__main__':
    run()
