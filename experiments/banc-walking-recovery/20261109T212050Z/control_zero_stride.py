import json
from pathlib import Path
import gait_teacher as G
HERE=Path(__file__).resolve().parent
rows=[]
for tag,kw in [('stride0_lift0.08',dict(stride=0.0,lift=0.08)),
               ('stride0_lift0',dict(stride=0.0,lift=0.0)),
               ('stride0.010_lift0',dict(stride=0.010,lift=0.0))]:
    s,_=G.run(period=120,ticks=2400,padz_gain=6.0,height_gain=6.0,**kw)
    rows.append(dict(tag=tag,**{k:v for k,v in s.items() if k!='per_leg_slip_ratio'}))
    print('%-18s fwd %.4f supmean %.3f supmin %.3f nf %5.2f com %.3f slip %5.2f'%(
        tag,s['forward_speed_native_per_s'],s['foot_support_mean'],s['foot_support_min'],
        s['nonfoot_force_max_W'],s['com_height_min'],s['slip_ratio_max']),flush=True)
(HERE/'GAIT_CONTROLS.json').write_text(json.dumps(dict(label='zero-stride controls: does the body move without a gait?',rows=rows),indent=2)+'\n')
