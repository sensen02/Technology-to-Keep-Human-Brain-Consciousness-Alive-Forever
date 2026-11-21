"""Independently replay saved SAC updates and compare restored deployment tensors."""
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import torch
import numpy as np
import learn_stage
import terminal_replay

HERE=Path(__file__).resolve().parent
original=learn_stage.SAC.update
hashes=[]

def record(agent,batch):
    losses=original(agent,batch)
    h=hashlib.sha256()
    for module in (agent.actor,agent.q1,agent.q2,agent.t1,agent.t2):
        for value in module.parameters():
            h.update(value.detach().numpy().tobytes())
    hashes.append(h.hexdigest())
    return losses

def main():
    learn_stage.SAC.update=record
    learn_stage.analyze=terminal_replay.read_rows
    for name in ('S1','S2','S3','S3_REPLAY'):
        info=json.loads((HERE/name/'TRAIN.json').read_text())
        phase=info['phase']
        parent=(learn_stage.ODOR/'TRAIN_DETOUR/0110000.pt' if phase==1 else HERE/f'S{phase-1}/checkpoint.pt')
        inputs=[(HERE/e if (HERE/e/'CONFIG.json').exists() else learn_stage.ODOR/e) for e in info['episodes']]
        hashes.clear()
        with TemporaryDirectory() as tmp:
            dest=Path(tmp)/name
            sys.argv=['learn_stage',str(phase),str(parent),str(dest),*[str(p) for p in inputs]]
            learn_stage.main()
            repeat=json.loads((dest/'TRAIN.json').read_text())
            assert info['log']==repeat['log']
            with np.load(HERE/name/'POLICY.npz') as a,np.load(dest/'POLICY.npz') as b:
                assert all(np.array_equal(a[k],b[k]) for k in a.files)
            before=torch.load(HERE/name/'checkpoint.pt',weights_only=True)
            after=torch.load(dest/'checkpoint.pt',weights_only=True)
            for key in ('actor','q1','q2','t1','t2'):
                assert all(torch.equal(v,after[key][k]) for k,v in before[key].items())
            assert all(torch.equal(before[k],after[k]) for k in ('log_alpha','rng'))
        (HERE/name/'REPLAY.json').write_text(json.dumps(dict(losses_exact=True,deploy_arrays_exact=True,
            actor_and_critic_exact=True,weights_after_updates=hashes),indent=2)+'\n')

if __name__=='__main__':main()
