"""A-path deliverable: algorithm-assisted support WITH a live neural closed loop.

This is explicitly NOT autonomous weight bearing. The algorithm keeps holding the
body up (leg-joint assistance is applied throughout), while the 12000-cell network
receives real sensory feedback and drives the muscles through the previous-completed
window. Nothing about it counts as neural self-support.
"""
from pathlib import Path
import json,hashlib
import run_physical_walk_prewarm as w
P=Path(__file__).parent
O=Path('/home/sensen/Desktop/engine修复验收/s5-assisted-neural-closedloop/20261028T104000Z')
def main():
 pre=O.parent/(O.name+'_PREDECLARED.json');pre.parent.mkdir(parents=True,exist_ok=True)
 assert not pre.exists() and not O.exists()
 sources=[Path(__file__),P/'run_physical_walk_prewarm.py',P/'engine/embodied/continuous_handover.py',P/'engine/neural_cond.py',P/'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json']
 pre.write_text(json.dumps(dict(
  scope='Support-assisted neural closed loop only. Algorithmic leg assistance is KEPT for the whole trial, so this is a hybrid and NOT autonomous weight bearing.',
  declaration=dict(neural_prewarm_s=1.0,keep_assistance_after_handover=True,neural_only_s=.25,
                   neural_starts_s=.2,handover_s=1.2,total_s=1.45,
                   assistance='leg-joint PD with bias/passive compensation, root always zero force',
                   muscles='real Hill-muscle actuators driven only by the network during the neural windows',
                   not_claimed='autonomous standing/walking; the body is still being held up by the algorithm'),
  source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2)+'\n')
 w.run(O,1.,100.,50.,.0000125,neural_duration=.25,neural_command_dt_s=.001,
       warmup_settle_s=.15,warmup_hold_s=.05,neural_prewarm_s=1.0,keep_assistance_after_handover=True)
if __name__=='__main__':main()
