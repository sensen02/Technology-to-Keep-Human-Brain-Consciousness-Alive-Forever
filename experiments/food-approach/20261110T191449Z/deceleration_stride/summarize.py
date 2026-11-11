"""Concise descriptive gate/cycle/slip comparison from saved results only."""
from pathlib import Path
import json
import numpy as np
from api import HERE,CONDITIONS
import physics

def longest(mask):
    best=run=0
    for v in mask:run=run+1 if v else 0;best=max(best,run)
    return best

def main():
    cases={}
    for cfg in CONDITIONS:
        folder=HERE/cfg['name'];r=json.loads((folder/'RESULT.json').read_text())
        with np.load(folder/'TRACE_1MS.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
        post=a['control_source']==5;front={}
        for leg,i in (('LF',0),('RF',3)):
            load=a['loads'][post,i]/physics.W;z=a['padz'][post,i];x=a['relativefoot'][post,i,0]
            events=[e for e in r['cycles']['events'] if e['leg']==leg]
            front[leg]=dict(load_min_W=float(load.min()),load_max_W=float(load.max()),padz_max=float(z.max()),padz_min=float(z.min()),
                relativefoot_x_range=float(np.ptp(x)),longest_loaded_grounded_ms=longest((load>=.1)&(z<=.004)),
                complete_count=r['cycles']['complete_per_leg'][leg],abort_reasons={reason:sum(e['type']=='abort' and e.get('reason')==reason for e in events) for reason in ('timeout','recontact','terminal')},events=events)
        cases[cfg['name']]=dict(original_all5=r['original_all5_gates'],original_gates=r['gates'],stride_gates=r['stride_gates'],all_stride_gates=r['all_stride_gates'],
            forward=r['forward_signed_native_per_s'],forward_threshold=cfg['forward_threshold'],slip_max=r['slip_ratio_max'],per_leg_slip=r['per_leg_slip_ratio'],
            slip_load_denominator_native=r['slip_load_denominator_native'],support_mean=r['foot_support_mean'],com_min=r['com_height_min'],nonfoot_max=r['nonfoot_force_max_native'],
            cycles=r['cycles']['complete_per_leg'],front_predicates=front)
    result=dict(cases=cases,interpretation='Descriptive saved-evidence comparison only. All gates must pass in the samecase. No pooling fullstride/halfstride best columns or changing slip denominator/cycletimeout.')
    (HERE/'CASE_SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({c:{k:v for k,v in r.items() if k!='front_predicates'} for c,r in cases.items()},indent=2))
if __name__=='__main__':main()
