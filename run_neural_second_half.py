"""Continuous-gait assisted neural closed loop (A-path, longer, no settle hold).

Same declarations as the previous assisted closed loop, but the trial never stops
walking: the algorithm keeps driving the gait for the whole run while the network
receives real feedback and drives the muscles. Still explicitly NOT autonomous:
the algorithm is holding the body up.

Also records per-leg foot lift so drag/lift asymmetry is measurable rather than
judged from a screenshot.
"""
from pathlib import Path
import json,hashlib
import run_physical_walk_prewarm as w
P=Path(__file__).parent
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-second-half/20261028T107000Z')
def main():
 pre=O.parent/(O.name+'_PREDECLARED.json');pre.parent.mkdir(parents=True,exist_ok=True)
 assert not pre.exists() and not O.exists()
 sources=[Path(__file__),P/'run_physical_walk_prewarm.py',P/'engine/embodied/continuous_handover.py',P/'engine/neural_cond.py',P/'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json']
 pre.write_text(json.dumps(dict(
  scope='User-requested split: first half = algorithm holds and walks the body while the network is already in the loop; second half = algorithm WITHDRAWN, neural closed loop only. The second half is judged as-is, including failure.',
  declaration=dict(duration_s=1.3,warmup_settle_s=0.,warmup_hold_s=0.,neural_prewarm_s=1.1,neural_only_s=.7,
                   neural_starts_s=.2,total_s=2.0,keep_assistance_after_handover=False,
                   change_vs_previous='no settle/hold phase: the fly keeps walking, so the whole clip shows gait instead of a standing hold',swing_height=.45,swing_height_reason='the transplanted reference gait left the hind feet almost never lifting (measured hind lift 0.063 vs front 0.511); every leg now gets the same explicit lift window over its own swing phase, x/y unchanged',
                   not_claimed='nothing yet: the neural-only second half is expected to fail on this plant and will be reported as measured',handover_rule='algorithm assistance is zero for the whole second half; any auxiliary or external force there is an error'),
  source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2)+'\n')
 w.run(O,1.3,100.,50.,.0000125,neural_duration=.4,neural_command_dt_s=.001,
       warmup_settle_s=0.,warmup_hold_s=0.,neural_prewarm_s=1.1,keep_assistance_after_handover=False,swing_height=.45)
if __name__=='__main__':main()
