"""Does the body-frame ramp actually propel the body forward (+x)? Signed measurement."""
import json, itertools
from pathlib import Path
import gait_teacher as G
HERE=Path(__file__).resolve().parent
rows=[]
for period,stride in itertools.product([60,120],[0.02,0.035,0.06]):
    s,_=G.run(period=period,stride=stride,lift=0.08,z_mode='mixed',padz_gain=4.0,height_gain=3.0,ticks=1600)
    rows.append(dict(period=period,stride=stride,**{k:v for k,v in s.items() if k!='per_leg_slip_ratio'}))
    print('p%3d s%.3f | fwdsigned %+.4f lat %.4f sup %.3f supmin %.3f nf %5.2f com %.3f slip %5.2f'%(
        period,stride,s['forward_signed_native_per_s'],s['lateral_speed_native_per_s'],
        s['foot_support_mean'],s['foot_support_min'],s['nonfoot_force_max_W'],
        s['com_height_min'],s['slip_ratio_max']),flush=True)
    (HERE/'GAIT_MIXED_SIGNED.json').write_text(json.dumps(dict(label='body-frame ramp with signed forward',rows=rows),indent=2)+'\n')
