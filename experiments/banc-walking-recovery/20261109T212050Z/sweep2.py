import json, itertools
from pathlib import Path
import gait_teacher as G
HERE=Path(__file__).resolve().parent
rows=[]
for period,stride,lift in itertools.product([60,120],[0.02,0.035,0.05],[0.06,0.10]):
    s,_=G.run(period=period,stride=stride,lift=lift,ticks=1600)
    rows.append(dict(period=period,stride=stride,lift=lift,**{k:v for k,v in s.items() if k!='per_leg_slip_ratio'}))
    print('p%3d s%.3f l%.2f | fwd %.4f supmean %.3f supmin %.3f nf %5.2f com %.3f slip %6.2f %s'%(
        period,stride,lift,s['forward_speed_native_per_s'],s['foot_support_mean'],s['foot_support_min'],
        s['nonfoot_force_max_W'],s['com_height_min'],s['slip_ratio_max'],
        ''.join(k[0].upper() for k,v in s['gates'].items() if v)),flush=True)
    (HERE/'GAIT_SWEEP2.json').write_text(json.dumps(dict(label='mixed z-channel tripod sweep',gates=G.GATES,rows=rows),indent=2)+'\n')
print('passing all gates:',[r for r in rows if r['all_gates']])
