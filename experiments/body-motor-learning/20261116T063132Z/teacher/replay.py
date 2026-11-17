"""Independent physical replay of a qualified observation/action trace."""
import argparse
from pathlib import Path

import numpy as np

from physical import read,run,stan


class Replay:
    def __init__(self, file):
        self.z={k:v.copy() for k,v in np.load(file,allow_pickle=False).items()}
        self.i=0

    def __call__(self,x,intent,local):
        i=self.i
        self.i+=1
        np.testing.assert_array_equal(x,self.z['x'][i])
        np.testing.assert_array_equal(local,self.z['local'][i])
        assert intent==self.z['intent'][i]
        return self.z['action'][i].copy()


def main(source,output):
    summary=read(source/'RESULT.json')
    manifest=read(source/'TEACHER_ACCEPTANCE.json')
    assert manifest['passed']
    assert stan.sha(source/'DECISIONS.npz')==manifest['decision_sha256']
    sample=Replay(source/'DECISIONS.npz')
    recorded=run(output,'replay',summary['scenario_source'],policy=sample,native=True,
                 policy_source=source/'DECISIONS.npz')
    assert sample.i==len(sample.z['ms'])
    keys=('TRACE.npz','NATIVE.npz','DECISIONS.npz')
    identical={}
    for k in keys:
        digest=stan.sha(output/k)
        identical[k]=digest==stan.sha(source/k)
        if not identical[k]:
            raise ValueError(f'Independent replay output departed in {k}')
    stan.put(output/'REPLAY_AUDIT.json',dict(source=str(source),output=str(output),
        native_integrations=recorded['physical_native_iterations'],decisions=sample.i,
        bitwise_by_file=identical,passed=recorded['native_joint_pass'] and all(identical.values())))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    main(a.source.resolve(),a.output.resolve())
