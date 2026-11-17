"""Check original brain replay, transducer calibration inputs and source identity."""
from pathlib import Path

import numpy as np

from overlap import BIDIR, HERE, digest, read, write


def audit():
    manifest=read(HERE/'IDENTITY.json')
    files=manifest['sha256']
    assert all(digest(Path(name))==value for name,value in files.items())
    inputs=read(HERE/'CALIBRATION.json')['source']
    assert all(digest(Path(name))==value for name,value in inputs.items())
    exact={}
    for name in ('cruise','a01_left','a02_right'):
        original=HERE/'BRAIN'/f'271_{name}'
        replay=HERE/'BRAIN_REPLAY'/name
        first_input=read(original/'INPUT.json')
        replay_input=read(replay/'INPUT.json')
        assert first_input==replay_input
        with np.load(original/'BRAIN.npz',allow_pickle=False) as a, np.load(replay/'BRAIN.npz',allow_pickle=False) as b:
            exact[name]={key:bool(np.array_equal(a[key],b[key])) for key in a.files}
        assert all(exact[name].values()),exact[name]
    catalog=read(BIDIR/'CATALOG.json')['ports']
    diagnostics={}
    for side in ('left','right'):
        raw=read(HERE/'BRAIN'/f'271_pfl_{side}'/'RESULT.json')
        result=read(HERE/'PFL_DIAGNOSTIC'/f'pfl_{side}'/'RESULT.json')
        counts={f'{catalog[root]["cell_type"]}/{catalog[root]["side"]}':n
                for root,n in raw['counts'].items() if catalog[root]['cell_type'] in ('DNa01','DNa02')}
        selected=counts['DNa02/'+side]>counts['DNa02/'+('right' if side=='left' else 'left')]
        diagnostics[side]=dict(Eon_counts=counts,dna02_preferred_side=selected,body=result)
    write(HERE/'SOURCE_AUDIT.json',dict(identity_count=len(files),calibration_input_count=len(inputs),
          independent_brain_runs=exact,pfl_diagnostic=diagnostics))
    print('identity',len(files),'calibration',len(inputs),'independent full brains',exact)
    for side,entry in diagnostics.items():
        print('PFL3',side,entry['Eon_counts'],entry['body']['yaw_degrees'],entry['body']['pass_'])


if __name__=='__main__':
    audit()
