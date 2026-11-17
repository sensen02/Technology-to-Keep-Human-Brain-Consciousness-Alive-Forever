"""Verify fixed initial state, delivered cause, and independent native replay."""
import json
from pathlib import Path

import numpy as np

from audit_source import sha

HERE=Path(__file__).resolve().parent
active,repeat,cut = (HERE/name for name in ('SPEED_CONNECTED','SPEED_CONNECTED_REPEAT','SPEED_CUT'))


def read(path):
    return json.loads(path.read_text())


def compare():
    conditions=[]
    for folder in (active,repeat,cut):
        evidence=read(folder/'DNG97_ACCEPTANCE.json')
        assert evidence['passed'] and evidence['native_steps']==904000
        assert evidence['link']['source_files_unchanged']
        assert sha(folder/'RESULT.json')==evidence['source_physical_sha256']
        assert sha(folder/'DNG97_DELIVERY.npz')==evidence['delivery_sha256']
        result=read(folder/'RESULT.json')
        for key,name in (('trace_sha256','TRACE.npz'),('decision_sha256','DECISIONS.npz'),('native_sha256','NATIVE.npz')):
            assert sha(folder/name)==result[key]
        conditions.append(evidence)
    files=('TRACE.npz','DECISIONS.npz','NATIVE.npz','DNG97_DELIVERY.npz')
    exact=dict((key,sha(active/key)==sha(repeat/key)) for key in files)
    assert all(exact.values())
    with np.load(active/'DECISIONS.npz') as a, np.load(cut/'DECISIONS.npz') as b, \
         np.load(active/'DNG97_DELIVERY.npz') as link_a, np.load(cut/'DNG97_DELIVERY.npz') as link_b:
        for key in ('command','ms'):
            np.testing.assert_array_equal(a[key],b[key])
        np.testing.assert_array_equal(link_a['decision'][:,1],link_b['decision'][:,1])
        assert not np.any(link_b['decision'][:,2])
        positive=link_a['delivery'][link_a['delivery'][:,2]>0]
        capture, arrival = [int(x) for x in positive[0,:2]]
        idx=np.flatnonzero(np.any(a['action'] != b['action'],axis=1))[0]
        first=int(a['ms'][idx])
        np.testing.assert_array_equal(a['x'][:idx+1],b['x'][:idx+1])
        np.testing.assert_array_equal(a['action'][:idx],b['action'][:idx])
        assert capture < arrival <= first and first-arrival < 5
    with np.load(active/'TRACE.npz') as a, np.load(cut/'TRACE.npz') as b:
        for field in ('qpos','qvel','com','loads','nonfoot'):
            np.testing.assert_array_equal(a[field][:first+1],b[field][:first+1])
    pair=dict(passed=True,physical_hashes_exact=exact,first_nonzero_capture_ms=capture,
              first_nonzero_delivery_ms=arrival,first_different_motor_action_ms=first,
              matched_physical_prefix_through_ms=first,enabled_and_cut=conditions[::2],
              scope='Matched educational DNg97 engineering waveform vs full DN cut, plus independent enabled replay.')
    (HERE/'DNG97_MATCHED_ACCEPTANCE.json').write_text(json.dumps(pair,indent=2)+'\n')
    print(capture, arrival, first, exact)


if __name__=='__main__':
    compare()
