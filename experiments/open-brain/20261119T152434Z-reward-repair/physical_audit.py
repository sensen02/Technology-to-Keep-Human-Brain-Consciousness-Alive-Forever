"""Read saved live runs, verify neck timing and compare the matched lip approaches."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'20261119T084744Z-closed-loop-forage'
SPEC=importlib.util.spec_from_file_location('p1_previous_audit',PRIOR/'assess.py')
older=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(older)

CASES=('REPEAT_L','REPEAT_R','HOLD_L','HOLD_R','FLAT_CONTROL')


def inspect(partial=False):
    candidates={}
    for name in CASES:
        if partial and not (HERE/name/'RESULT.json').is_file():
            continue
        row=older.inspect(HERE/name)
        with np.load(HERE/name/'BODY.npz',allow_pickle=False) as z:
            commands=z['commands']
            assert np.isfinite(commands).all()
            trace=z['trace']
        side=row['geometry']['side']
        indices=np.flatnonzero(trace[:,1]>=row['geometry']['near']-.045) if side else []
        row['lip_lookback']=None
        if len(indices):
            i=indices[0]
            j=max(0,i-32)
            near=trace[i]
            delta_t=near[0]-trace[j,0]
            nearby=commands[(commands[:,0]>=near[0]-.40)&(commands[:,0]<near[0])]
            intent=-side*nearby[:,2]
            output=side*(nearby[:,3]-nearby[:,4])
            row['lip_lookback']=dict(
                mean_forward_speed_mm_s=float(10*(near[1]-trace[j,1])/delta_t) if delta_t else None,
                command_packets=len(nearby),
                emitted_intent_toward_safe=int(np.count_nonzero(intent>.01)),
                readout_toward_safe=int(np.count_nonzero(output>.01)))
        candidates[name]=row
    previous=json.loads((PRIOR/'FULL_LOOP.json').read_text())['cases']
    paired={}
    for key,n in zip(('REPEAT_L','REPEAT_R'),('WIDE_L','WIDE_R'),strict=True):
        if key not in candidates:
            continue
        ref=previous[n]
        cur=candidates[key]
        assert cur['geometry']==ref['geometry'] and cur['neural_seed']==ref['neural_seed']
        old,now=ref['lip_margin_mm'],cur['lip_margin_mm']
        paired[key]=dict(previous_margin=old,current_margin=now,
                         delta=None if now is None or old is None else now-old,
                         previous_meal=ref['meal_pass'],current_meal=cur['meal_pass'])
    wide=[v for k,v in candidates.items() if k!='FLAT_CONTROL']
    reduced=dict(n=len(wide),sustained_meal=sum(r['meal_pass'] for r in wide),
                 support_cross=sum(r['supported_side_cross'] for r in wide),
                 fall=sum(r['first_fall_s'] is not None for r in wide),
                 ingestion=sum(r['intake_target_pass'] for r in wide))
    report=dict(cases=candidates,wide_summary=reduced,matched=paired,
                native_steps=sum(v['native_steps'] for v in candidates.values()),
                neural_steps=sum(v['neural_steps'] for v in candidates.values()),
                complete=len(candidates)==len(CASES))
    (HERE/'PHYSICAL_AUDIT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('wide_summary','matched','native_steps','neural_steps')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial',action='store_true')
    x=parser.parse_args()
    inspect(x.partial)
