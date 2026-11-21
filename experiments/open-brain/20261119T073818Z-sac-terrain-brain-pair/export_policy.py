"""Export the deterministic SAC mean and verify it against the saved Torch actor."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from forage import OBS
from learn import Actor


def inference(parameters, observation):
    y=np.asarray(observation,dtype=np.float32)
    for layer in (0,2,4):
        y=y@parameters[f'w{layer}'].T+parameters[f'b{layer}']
        if layer!=4:
            y=np.maximum(y,0.)
    return np.tanh(y[...,:2])


def main(source,out):
    torch.set_num_threads(1)
    data=torch.load(source,map_location='cpu',weights_only=True)
    actor=Actor()
    actor.load_state_dict(data['actor'])
    parameters={}
    for layer in (0,2,4):
        parameters[f'w{layer}']=data['actor'][f'net.{layer}.weight'].numpy()
        parameters[f'b{layer}']=data['actor'][f'net.{layer}.bias'].numpy()
    probes=np.random.default_rng(2389).uniform(-1.,1.,(300,OBS)).astype(np.float32)
    with torch.inference_mode():
        expected=actor(torch.from_numpy(probes),sample=False)[0].numpy()
    assert np.allclose(expected,inference(parameters,probes),atol=3e-6,rtol=0.)
    assert int(data['updates'])>0
    np.savez_compressed(out,**parameters,identity=json.dumps(data['identity']),source=str(source))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('out',type=Path)
    args=p.parse_args()
    main(args.source,args.out)
