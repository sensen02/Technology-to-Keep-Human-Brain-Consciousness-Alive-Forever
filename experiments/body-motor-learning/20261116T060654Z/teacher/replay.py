"""Independent physical reconstruction with archived teacher actions."""
import argparse
from pathlib import Path

import numpy as np

from physical import read, run, stan


def replay(parent, destination):
    accepted=read(parent/'TEACHER_ACCEPTANCE.json')
    assert accepted['passed']
    assert stan.sha(parent/'DECISIONS.npz')==accepted['decision_sha256']
    assert stan.sha(parent/'RESULT.json')==accepted['result_sha256']
    result=read(parent/'RESULT.json')
    with np.load(parent/'DECISIONS.npz',allow_pickle=False) as z:
        x, local, intent, action=tuple(z[k].copy() for k in ('x','local','intent','action'))
    i=0
    def saved(observation, active, current):
        nonlocal i
        np.testing.assert_array_equal(observation,x[i])
        np.testing.assert_array_equal(current,local[i])
        assert active==intent[i]
        y=action[i].copy()
        i+=1
        return y
    new=run(destination,'student',Path(result['scenario_source']),native=True,policy=saved,
            policy_source=parent/'DECISIONS.npz')
    assert i==len(action)
    assert new['trace_sha256']==result['trace_sha256']
    assert new['decision_sha256']==result['decision_sha256']
    old_raw=read(parent/'NATIVE_AUDIT.json')
    new_raw=read(destination/'NATIVE_AUDIT.json')
    assert old_raw['native_sha256']==new_raw['native_sha256']
    stan.put(destination/'REPLAY_AUDIT.json',
             dict(physical_trace_equal=True, decisions_equal=True, native_contact_trace_equal=True,
                  qualification_sha256=stan.sha(parent/'TEACHER_ACCEPTANCE.json'),
                  native_integrations=new_raw['native_integrations'], rows=i,
                  replay_of=str(parent), policy_source='teacher actions checksum binding, not candidate weight'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('parent',type=Path)
    p.add_argument('destination',type=Path)
    args=p.parse_args()
    replay(args.parent.resolve(),args.destination.resolve())
