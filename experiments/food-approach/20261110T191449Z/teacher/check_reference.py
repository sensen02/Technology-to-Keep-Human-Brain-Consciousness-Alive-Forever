"""Read-only mathematical checks, no physical success claims."""
import json
from pathlib import Path
import numpy as np
from turning_teacher import pin_baseline, ConditionalTwistReference
pin_baseline()
from wave_reference import WaveReference

rest = np.array([[.8,.5,-1],[-.8,.5,-1],[0,.7,-1],[.8,-.5,-1],[-.8,-.5,-1],[0,-.7,-1.]])
base = WaveReference(rest, stride_native=.36,step_s=.8,lift_native=.12,cycles=100)
checks = {}
neutral = ConditionalTwistReference(base)
checks['neutral_bitexact_1001times'] = all(np.array_equal(neutral.sample(t)['target_xy'],base.sample(t)['target_xy']) for t in np.linspace(0,20,1001))
left, right = (ConditionalTwistReference(base, .5, s*.004) for s in [1,-1])
checks['mirrored_turn_antisymmetric_offsets'] = all(np.allclose(left.sample(t)['target_xy']+right.sample(t)['target_xy'],2*base.sample(left.command(t)[0])['target_xy'],atol=1e-14,rtol=0) for t in np.linspace(0,20,1001))
checks['prime_command_neutral'] = all(left.command(t)[1:]==(1.,0.) for t in np.linspace(0,4.8,101))
checks['half_speed_phase_after_ramp'] = bool(np.allclose(np.diff([left.command(t)[0] for t in [6.,7.,8.]]),.5,atol=1e-14,rtol=0))
# Avoid measured shape knots when comparing analytic velocities to finite differences.
errors=[]
for t in [4.83321,5.12341,5.71231,6.12341,7.23151]:
    eps=1e-7
    fd=(left.sample(t+eps)['target_xy']-left.sample(t-eps)['target_xy'])/(2*eps)
    errors.append(float(np.max(np.abs(fd-left.sample(t)['velocity_xy']))))
checks['velocity_derivative_error_below_1e-6']=max(errors)<1e-6
assert all(checks.values()), checks
result=dict(all_passed=True,checks=checks,max_fd_velocity_error=max(errors),physics_success_claim=False)
Path(__file__).with_name('REFERENCE_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
