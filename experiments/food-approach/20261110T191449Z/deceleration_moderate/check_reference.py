"""Zero-physics wholephase reference/command contract with frozen result hashes."""
import json,hashlib
import numpy as np
from api import HERE,TEACHER,CONDITIONS,previous
from wave_reference import WaveReference

def main():
    original=HERE.parent/'teacher'/'left_full'/'TRACE_1MS.npz'
    with np.load(original,allow_pickle=False) as z:rest=z['relativefoot'][999].copy()
    base=WaveReference(rest,stride_native=.36,lift_native=.12,step_s=.8,cycles=100)
    checks={}
    for cfg in CONDITIONS:
        ref=previous.ConditionalTwistReference(base,speed_scale=cfg['speed_scale'],yaw_rate=0.)
        cmd_ok=profile_ok=prefix_ok=True
        for decision in range(2880):
            t=decision*.005;dt=max(0,t-4.8);u=min(dt,1);blend=u*u*(3-2*u)
            phase=t+(cfg['speed_scale']-1)*(u**3-.5*u**4+max(0,dt-1));scale=1+(cfg['speed_scale']-1)*blend
            p,s,k=ref.command(t);actual=ref.sample(t);b=base.sample(phase)
            cmd_ok=cmd_ok and abs(p-phase)<1e-14 and abs(s-scale)<1e-15 and k==0
            profile_ok=profile_ok and np.array_equal(actual['target_xy'],b['target_xy']) and np.array_equal(actual['target_z'],b['target_z']) and np.array_equal(actual['stance'],b['stance']) and np.allclose(actual['velocity_xy'],b['velocity_xy']*scale,atol=0,rtol=0) and np.allclose(actual['velocity_z'],b['velocity_z']*scale,atol=0,rtol=0)
            if decision<960:
                originalsample=base.sample(t)
                prefix_ok=prefix_ok and all(np.array_equal(actual[k],originalsample[k]) for k in ('target_xy','target_z','stance','velocity_xy','velocity_z'))
        checks[cfg['name']+'_independent_phase_and_command_all2880']=cmd_ok
        checks[cfg['name']+'_fullxyz_profile_only_phase_slowed']=profile_ok
        checks[cfg['name']+'_neutral960_bitexact']=prefix_ok
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    result=dict(all_passed=all(checks.values()),checks=checks,actual_physics_steps=0,source_hashes={str(p):sha(p) for p in (HERE/'PREREG.json',HERE/'AMENDMENT_BEFORE_EXECUTION.json',HERE/'api.py',HERE/'check_reference.py',TEACHER,original)})
    (HERE/'REFERENCE_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));assert result['all_passed']
if __name__=='__main__':main()
