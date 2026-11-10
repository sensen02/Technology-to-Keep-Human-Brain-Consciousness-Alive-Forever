"""Record the phase-dependent-dimension results to JSON (the runs printed earlier were ad hoc)."""
import json
from pathlib import Path
import gait_v3 as V
HERE=Path(__file__).resolve().parent
rows=[]
for stride,tau in [(0.03,0.05),(-0.03,0.05),(0.03,0.02),(-0.03,0.02)]:
    s,_=V.run(period=60,stride=stride,lift=0.08,ticks=800,tau=tau)
    rows.append(dict(stride=stride,tau=tau,**{k:v for k,v in s.items() if k!='per_leg_slip_ratio'}))
    print('stride %+.3f tau %.2f | fwd %+.4f lat %.4f supmin %.3f nf %5.2f com %.3f slip %6.2f'%(
        stride,tau,s['forward_signed_native_per_s'],s['lateral_speed_native_per_s'],
        s['foot_support_min'],s['nonfoot_force_max_W'],s['com_height_min'],s['slip_ratio_max']),flush=True)
(HERE/'GAIT_PHASE_DIM.json').write_text(json.dumps(dict(
    label='phase-dependent task dimension tripod gait; algorithm-assisted warm-up; NOT walking acceptance',
    gates=V.GATES, rows=rows),indent=2)+'\n')
