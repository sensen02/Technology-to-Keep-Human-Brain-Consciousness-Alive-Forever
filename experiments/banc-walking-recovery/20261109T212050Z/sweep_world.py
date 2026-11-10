import json, itertools
from pathlib import Path
import gait_world as W
HERE=Path(__file__).resolve().parent
rows=[]
for period,stride,lg in itertools.product([120,200],[0.015,0.03],[0.0,0.5,1.0]):
    s,_=W.run(period=period,stride=stride,lift=0.08,settle=300,ticks=1600,lateral_gain=lg)
    rows.append(dict(period=period,stride=stride,lateral_gain=lg,
                     **{k:v for k,v in s.items() if k!='per_leg_slip_ratio'}))
    print('p%3d s%.3f lg%.1f | fwdsigned %+.4f lat %.4f sup %.3f supmin %.3f nf %5.2f com %.3f slip %5.2f %s'%(
        period,stride,lg,s['forward_signed_native_per_s'],s['lateral_speed_native_per_s'],
        s['foot_support_mean'],s['foot_support_min'],s['nonfoot_force_max_W'],
        s['com_height_min'],s['slip_ratio_max'],
        ''.join(k[0].upper() for k,v in s['gates'].items() if v)),flush=True)
    (HERE/'GAIT_WORLD_ANCHOR.json').write_text(json.dumps(dict(label='world-anchored tripod gait, lateral anchor correction, algorithm-assisted',gates=W.GATES,rows=rows),indent=2)+'\n')
print('passing all gates:',[r for r in rows if r['all_gates']])
