"""Verify and export a symmetric policy and its physical-intent translation grid."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from balanced import MIRROR, MirrorActor, BalancedMotor
from policy_runtime import actor_mean
from forage import OBS


def run(checkpoint,cal_path,out):
    torch.set_num_threads(1)
    data=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=MirrorActor()
    model.load_state_dict(data['actor'])
    par={}
    for layer in (0,2,4):
        par[f'w{layer}']=data['actor'][f'net.{layer}.weight'].numpy()
        par[f'b{layer}']=data['actor'][f'net.{layer}.bias'].numpy()
    par["mirror"]=np.array(MIRROR)
    rng=np.random.default_rng(4627)
    obs=rng.uniform(-1.,1.,(500,OBS)).astype(np.float32)
    with torch.inference_mode():
        expected=model(torch.from_numpy(obs),sample=False)[0].numpy()
    assert np.allclose(actor_mean(par,obs),expected,atol=8e-6,rtol=0.)
    cal=json.loads(cal_path.read_text())
    motor=BalancedMotor(cal['trials'][str(data['identity']['gain'])]['rows'])
    assert np.array_equal(motor.intent,np.load(checkpoint.parent/'INTENT_GRID.npy',allow_pickle=False))
    np.savez_compressed(out,**par,identity=json.dumps(data['identity']),source=str(checkpoint),
                        intent_grid=motor.intent)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path)
    p.add_argument('cal',type=Path)
    p.add_argument('out',type=Path)
    a=p.parse_args()
    run(a.checkpoint,a.cal,a.out)
