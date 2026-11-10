import json, itertools
from pathlib import Path
import gait_teacher as G
HERE=Path(__file__).resolve().parent
rows=[]
# Stride is chosen so the commanded stance speed matches the measured walking speed (~0.15/s):
# stride = v * stance_time. Too large and the planted feet must skid.
for period,stride,lift,pg,hg in itertools.product([120,240],[0.006,0.010,0.016],[0.08],[6.0],[6.0]):
    s,_=G.run(period=period,stride=stride,lift=lift,padz_gain=pg,height_gain=hg,ticks=2400)
    rows.append(dict(period=period,stride=stride,lift=lift,padz_gain=pg,height_gain=hg,
                     **{k:v for k,v in s.items() if k!='per_leg_slip_ratio'}))
    print('p%3d s%.3f | fwd %.4f supmean %.3f supmin %.3f nf %5.2f com %.3f slip %5.2f %s'%(
        period,stride,s['forward_speed_native_per_s'],s['foot_support_mean'],s['foot_support_min'],
        s['nonfoot_force_max_W'],s['com_height_min'],s['slip_ratio_max'],
        ''.join(k[0].upper() for k,v in s['gates'].items() if v)),flush=True)
    (HERE/'GAIT_SWEEP3.json').write_text(json.dumps(dict(label='stride matched to measured speed',gates=G.GATES,rows=rows),indent=2)+'\n')
print('passing all gates:',[r for r in rows if r['all_gates']])
