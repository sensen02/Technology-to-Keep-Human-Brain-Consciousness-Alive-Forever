"""Hold PFL3 stimulation and random input fixed while changing registered front AN drive."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from brain_probe import OPEN, case, read_weights, patchable, MotorAdapter, Neck, NeuralInputs


def run(calibration, output):
    torch.set_num_threads(1)
    cal=json.loads(calibration.read_text())
    idx,graph=read_weights()
    inputs=NeuralInputs(idx)
    ref=Neck(idx,inputs,tactile=True)
    positions,_=patchable(idx,graph,inputs,ref)
    w=graph.data[positions].copy()
    adapter=MotorAdapter(json.loads((OPEN/'TRANSLATED_MOTOR.json').read_text()))
    rows={}
    for arm,gain in [('frozen',cal['frozen_gain']),('tuned',cal['tuned_gain'])]:
        graph.data[positions]=w*gain
        tests=[]
        for n,(advance,turn) in enumerate(((.55,0.),(.55,-.15),(.55,.15))):
            options=[]
            for touch in ((0.,0.),(.08,.08)):
                def create_neck():
                    neck=Neck(idx,inputs,tactile=True)
                    update=neck.update
                    def with_touch(tick,dn,_front):
                        return update(tick,dn,np.asarray(touch))
                    neck.update=with_touch
                    return neck
                obs=case(graph,(advance,turn),create_neck,inputs,adapter.rates(advance,turn,0.),73113+n)
                obs['front']=list(touch)
                options.append(obs)
            delta=np.subtract(options[1]['drive'],options[0]['drive'])
            tests.append(dict(without=options[0],with_touch=options[1],
                              yaw_shift_rad_s=float(3.25*(delta[0]-delta[1]))))
        rows[arm]=tests
        print(arm,[round(row['yaw_shift_rad_s'],3) for row in tests],flush=True)
    (output).write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('calibration',type=Path)
    cli.add_argument('output',type=Path)
    args=cli.parse_args()
    run(args.calibration,args.output)
