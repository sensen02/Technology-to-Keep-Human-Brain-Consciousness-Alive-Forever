"""Declared sensory-polarity comparison, motor side measured by physics.

Why: at the handover instant the network's muscle force had cosine -0.51 with the
torque that was holding the body up, i.e. the loop was pulling the wrong way. One
concrete, checkable cause is the SIGN of the proprioceptive current injected into
the motor system: the encoder injects a signed tibia deviation with no evidence
for the per-cell preferred direction, so the sign may be inverted.

This runs the SAME walking / handover / neural-only protocol twice, changing only
the sign of the strain and tibia channel weights. The criterion is physical, not
cosmetic: cosine between muscle generalized force and the anti-gravity demand, and
the time to first non-foot contact. No controller is added; only the declared input
polarity is flipped, and the negative result is reported just as plainly.
"""
from pathlib import Path
import json,hashlib
import run_physical_walk_prewarm as w
P=Path(__file__).parent
O=Path('/home/sensen/Desktop/engine修复验收/s5-sensory-polarity/20261028T108000Z')
WEIGHTS={'claw_position':-1.0,'hook_direction':-1.0,'hairplate_angle':-1.0}
def main():
 pre=O.parent/(O.name+'_PREDECLARED.json');pre.parent.mkdir(parents=True,exist_ok=True)
 assert not pre.exists() and not O.exists()
 sources=[Path(__file__),P/'run_physical_walk_prewarm.py',P/'engine/embodied/continuous_handover.py',
          P/'engine/embodied/sensory_channels.py',P/'engine/neural_cond.py',
          P/'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json',
          Path('/home/sensen/Desktop/engine修复验收/s5-neural-second-half/20261028T107000Z/BODY.npz')]
 pre.write_text(json.dumps(dict(
  scope='Declared input-polarity comparison on the same protocol. Baseline is 107000Z (weights +1). This run inverts the sign of the three proprioceptive channels that carry leg geometry (claw position, hook direction, hair-plate angle); strain/touch stay at +1 so the change is specific. No controller, no anatomy change, no model parameter change.',
  weight_override=WEIGHTS,allow_signed_sensory_weights=True,
  same_protocol=dict(duration_s=1.3,swing_height=.45,neural_prewarm_s=1.1,neural_only_s=.7,handover_s=1.3,keep_assistance_after_handover=False),
  change_of_only='sign of the three proprioceptive channel weights (claw_position, hook_direction, hairplate_angle)',
  criteria=dict(primary='cosine between muscle generalized force and (-bias-passive) at the handover instant > 0',
                secondary='time to first non-foot contact after withdrawal > 100 ms',
                reporting='both numbers reported whatever they are; failure is a valid outcome'),
  not_claimed='autonomous weight bearing unless both criteria are met and movement is coordinated',
  source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2)+'\n')
 w.run(O,1.3,100.,50.,.0000125,neural_duration=.7,neural_command_dt_s=.001,
       warmup_settle_s=0.,warmup_hold_s=0.,neural_prewarm_s=1.1,
       neural_sensory_channel_weights=WEIGHTS,allow_signed_sensory_weights=True,
       keep_assistance_after_handover=False,swing_height=.45)
if __name__=='__main__':main()
