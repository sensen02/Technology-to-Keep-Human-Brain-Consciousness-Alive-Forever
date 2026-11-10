import json
from pathlib import Path
import gait_teacher as G
HERE=Path(__file__).resolve().parent
rows=[]
for tag,kw in [('rest_only_pg0',dict(stride=0.0,lift=0.0,padz_gain=0.0,height_gain=0.0)),
               ('pg0_hg2_stride0',dict(stride=0.0,lift=0.0,padz_gain=0.0,height_gain=2.0)),
               ('pg0_hg2_stride0.02',dict(stride=0.02,lift=0.08,padz_gain=0.0,height_gain=2.0)),
               ('pg2_hg2_stride0.02',dict(stride=0.02,lift=0.08,padz_gain=2.0,height_gain=2.0))]:
    s,_=G.run(period=120,ticks=2400,**kw)
    rows.append(dict(tag=tag,**{k:v for k,v in s.items() if k!='per_leg_slip_ratio'}))
    print('%-22s fwdsigned %+.4f lat %.4f sup %.3f supmin %.3f nf %5.2f com %.3f slip %5.2f'%(
        tag,s['forward_signed_native_per_s'],s['lateral_speed_native_per_s'],s['foot_support_mean'],
        s['foot_support_min'],s['nonfoot_force_max_W'],s['com_height_min'],s['slip_ratio_max']),flush=True)
(HERE/'GAIT_DRIFT_CONTROLS.json').write_text(json.dumps(dict(label='drift controls with signed forward metric',rows=rows),indent=2)+'\n')
