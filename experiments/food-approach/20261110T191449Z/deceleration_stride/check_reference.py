"""No-physics numerical reference contract checks before actual two-case run."""
import json
import numpy as np
from api import HERE,FixedCadenceStrideReference,CONDITIONS
from wave_reference import WaveReference

def main():
    rest=np.array([[.8,.5,0],[-.8,.5,0],[0,.5,0],[.8,-.5,0],[-.8,-.5,0],[0,-.5,0]])
    base=WaveReference(rest,stride_native=.36,lift_native=.12,step_s=.8,cycles=100)
    checks={}
    for cfg in CONDITIONS:
        ref=FixedCadenceStrideReference(base,cfg['stride_scale']);prefix=True;profiles=True;targets=True;derivative=True;command=True
        for t in np.r_[np.arange(960)*.005,5.137,5.389,6.137,7.389,10.137]:
            b=base.sample(float(t));s=ref.sample(float(t));scale,rate=ref.scale(float(t))
            profiles=profiles and all(np.array_equal(b[k],s[k]) for k in ('target_z','velocity_z','stance')) and np.array_equal(b['target_xy'][:,1],s['target_xy'][:,1]) and np.array_equal(b['velocity_xy'][:,1],s['velocity_xy'][:,1])
            targets=targets and np.allclose(s['target_xy'][:,0],rest[:,0]+scale*(b['target_xy'][:,0]-rest[:,0]),atol=1e-15,rtol=0)
            derivative=derivative and np.allclose(s['velocity_xy'][:,0],scale*b['velocity_xy'][:,0]+rate*(b['target_xy'][:,0]-rest[:,0]),atol=1e-14,rtol=0)
            if t<4.8:prefix=prefix and all(np.array_equal(b[k],s[k]) for k in ('target_xy','target_z','velocity_xy','velocity_z','stance'))
        for t in (5.137,5.389,6.137,7.389,10.137):
            eps=1e-7
            # Check scale derivative independently; base velocities carry inherited
            # measured interpolation diagnostics, so do not replace their semantics.
            fd=(ref.scale(t+eps)[0]-ref.scale(t-eps)[0])/(2*eps)
            command=command and abs(fd-ref.scale(t)[1])<1e-8
        checks[cfg['name']+'_neutral960_bitexact']=prefix
        checks[cfg['name']+'_lateral_z_stance_unchanged']=profiles
        checks[cfg['name']+'_sagittal_scale_exact']=targets
        checks[cfg['name']+'_full_velocity_product_rule']=derivative
        checks[cfg['name']+'_smoothscale_derivative_finite_difference']=command
        checks[cfg['name']+'_asymptotic_scale']=ref.scale(5.8)==(cfg['stride_scale'],0.)
    result=dict(all_passed=all(checks.values()),checks=checks,actual_physics_steps=0)
    (HERE/'REFERENCE_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));assert result['all_passed']
if __name__=='__main__':main()
