"""Recheck the accepted straight walking and the original fixed stopping-lift example."""
from scope import *
from rollout_repair import run
checkpoint=HERE/'FIT/epoch040.pt';rows=[]
for h,seed in ((0.,1151),(.15,1723)):
 pair=[]
 for mode in ('teacher','student'):
  pair.append(run(h,seed,mode,HERE/'PARKING_REGRESSION'/f'{h}_{seed}_{mode}',checkpoint))
 t,s=pair
 g=dict(advance=s['dx']>=.75*t['dx'],stand=s['stand_rms']<=max(.30,t['stand_rms']+.10),
  height=s['min_height']>=t['min_height']-.10,upright=s['min_upright']>=min(.8,t['min_upright']-.05),
  joint=s['joint_excess']<=max(.02,t['joint_excess']+.01),nonfoot=s['nonfoot_peak_bodyweight']<=max(.2,t['nonfoot_peak_bodyweight']+.05),target=s['gates']['target_limits'])
 rows.append(dict(height=h,seed=seed,relative_pass=all(g.values()),relative_gates=g,teacher=t,student=s))
write(HERE/'PARKING_REGRESSION_RESULT.json',rows)
