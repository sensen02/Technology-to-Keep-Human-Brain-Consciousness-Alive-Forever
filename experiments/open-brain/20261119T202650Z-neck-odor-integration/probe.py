"""Compare one left DNa02 pulse on the live ground receiver with OmniSim's replay."""
import json
from pathlib import Path
import sys
import numpy as np
from neck_profile import CATALOG, configure

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'20261118T031132Z-flat-neck-assay'))
from assay_timing import FrontTouchNeck


def probe(mode, fixture):
    receipt=json.loads((HERE/'live_intact_left/APPLIED.json').read_text())
    catalog=json.loads(CATALOG.read_text())['ports']
    ground={row['root']:catalog[row['root']] for row in receipt['applied']}
    binding=HERE.parent/'20261117T235930Z-ascending-reference-fit/BINDING.json'
    an=[row['banc_root'] for row in json.loads(binding.read_text())['records']]
    model=FrontTouchNeck({'ports':ground}, an, matched_lanes=(0,), feedback_lanes=(0,), batch=1)
    _, applied=configure(model,fixture,mode)
    root=applied['target']
    column=model.port_index[root]
    recipients=[model.index[t] for t,_ in catalog[root]['outgoing']]
    drive=np.zeros((1,len(model.roots)),dtype=np.float32)
    eon=np.zeros((1,len(model.port_roots)),dtype=bool)
    other_cuts=np.zeros(len(model.port_roots),dtype=bool)
    emitted,arrivals=[],[]
    peak=0.
    for tick in range(int(model.neck_delay.max())+30):
        eon[0,column]=tick==0
        source,received,_,_=model.step(drive,other_cuts,eon_output=eon)
        if source[0,column]:
            emitted.append(tick)
        if received[0,column]:
            arrivals.append(tick)
        peak=max(peak,float(model.last_conductance[:,:,recipients].max()))
    expected=json.loads(fixture.read_text())['selected']['frozen_runtime']['intact' if mode=='intact' else 'injured']
    assert emitted==expected['launch_ticks'] and arrivals==expected['arrival_ticks']
    assert np.isclose(peak,expected['receiver_peak_conductance_nS'],rtol=1e-12,atol=1e-15)
    return dict(source_ticks=emitted,arrival_ticks=arrivals,peak_nS=peak)


if __name__=='__main__':
    results={}
    for arm,mode,name in [('healthy','intact','blocked'),('selective_block','target_only','blocked'),
                          ('selective_partial','target_only','partial'),('whole_shaft','shaft','blocked')]:
        results[arm]=probe(mode,HERE/'profiles'/f'{name}.json')
    print(json.dumps(results,indent=2))
