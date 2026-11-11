"""CPU1 pure NumPy geometry checks. Reads saved rest; never imports physics/Core."""
from pathlib import Path
import json
import os
import sys
import numpy as np
from source_identity import HERE, verify_sources, sha256
from turning_teacher import FiniteTwistReference, finite_xy_velocity, BASELINE


def main():
    allowed = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {1 if 1 in allowed else min(allowed)})
    verify_sources()
    sys.path.insert(0, str(BASELINE))
    from wave_reference import WaveReference
    trace = HERE.parent/'attempt02'/'left_yaw_p060'/'TRACE_1MS.npz'
    with np.load(trace, allow_pickle=False) as z:
        assert abs(float(z['time'][999])-1.) < 1e-8
        rest = z['relativefoot'][999].copy()
    base = WaveReference(rest, stride_native=.36, lift_native=.12, step_s=.8, cycles=100)
    left = FiniteTwistReference(base)
    zero = FiniteTwistReference(base, yaw_rate=0.)
    checks, errors = {}, {}
    def check(name, value):
        checks[name] = bool(value)
    def maxerr(name, a, b, tol):
        error = float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
        errors[name] = error
        check(name, error <= tol)
    def same(a, b):
        return set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a)
    times = np.arange(2880)*.005
    check('zero_yaw_all_fields_exact', all(same(zero.sample(t), base.sample(t)) for t in times))
    check('preprime_all_fields_exact', all(same(left.sample(t), base.sample(t)) for t in times[times <= 4.8]))
    unchanged = ('target_z','velocity_z','stance','phase','source_time_s')
    check('nonhorizontal_fields_exact_full2880', all(all(np.array_equal(left.sample(t)[k], base.sample(t)[k]) for k in unchanged) for t in times))
    check('command_identical_inherited', all(left.command(t) == (t, 1., .8*(lambda u:u*u*(3-2*u))(float(np.clip((t-4.8),0,1)))) or np.allclose(left.command(t), (t,1.,.8*(lambda u:u*u*(3-2*u))(float(np.clip(t-4.8,0,1)))), atol=1e-15, rtol=0) for t in times))
    p, v = finite_xy_velocity(rest[:,:2], rest[:,:2]+[0.,.013], np.zeros((6,2)), .8, .3)
    check('s_zero_target_exact', np.array_equal(p, rest[:,:2]+[0.,.013]))
    check('s_zero_stationary_velocity_exact', np.array_equal(v, np.zeros((6,2))))
    rng = np.random.default_rng(20261111)
    r = rng.normal(size=(128,2)); s = rng.uniform(-.4,.4,128); ell = rng.uniform(-.08,.08,128)
    vel = rng.normal(size=(128,2))*.2
    target = r+np.column_stack([s,ell]); sign=np.array([1.,-1.])
    for k in (0.,1e-12,1e-5,.8,-.8):
        p,v=finite_xy_velocity(r,target,vel,k,.4)
        pm,vm=finite_xy_velocity(r*sign,target*sign,vel*sign,-k,-.4)
        maxerr('mirror_position_'+str(k),pm,p*sign,2e-14)
        maxerr('mirror_velocity_'+str(k),vm,v*sign,2e-14)
        h=1e-6
        plus=finite_xy_velocity(r,target+h*vel,vel,k+h*.4,.4)[0]
        minus=finite_xy_velocity(r,target-h*vel,vel,k-h*.4,.4)[0]
        maxerr('analytic_chain_fd_'+str(k),v,(plus-minus)/(2*h),2e-8)
    # F(s+d)=F(d)F(s): stance feet may have different completed-step s_i.
    # Distances are constant in time for shared ds at constant k, ell=0.
    p0=finite_xy_velocity(r,r+np.column_stack([s,np.zeros(128)]),vel,.8,0.)[0]
    p1=finite_xy_velocity(r,r+np.column_stack([s+.017,np.zeros(128)]),vel,.8,0.)[0]
    dist=lambda p:np.linalg.norm(p[:,None,:]-p[None,:,:],axis=-1)
    maxerr('common_stance_step_spacing',dist(p0),dist(p1),3e-14)
    same_s=np.full(128,.17)
    p2=finite_xy_velocity(r,r+np.column_stack([same_s,np.zeros(128)]),vel,.8,0.)[0]
    maxerr('shared_s_rest_spacing',dist(p2),dist(r),3e-14)
    a,b=left.sample(6.123),left.sample(6.133)
    mask=a['stance'] & b['stance']
    check('actual_reference_five_shared_stance', int(mask.sum())==5)
    maxerr('actual_reference_stance_spacing',dist(a['target_xy'][mask]),dist(b['target_xy'][mask]),2e-14)
    # Physical reference has piecewise measured interpolation knots. Exclude a
    # point only if its baseline analytic derivative disagrees with centered FD.
    tested=0; excluded=0; worst=0.; h=1e-6
    for t in rng.uniform(4.81,14.39,240):
        bm,bp=base.sample(t-h),base.sample(t+h)
        bv=(bp['target_xy']-bm['target_xy'])/(2*h)
        if np.max(np.abs(bv-base.sample(t)['velocity_xy']))>2e-7:
            excluded+=1; continue
        fd=(left.sample(t+h)['target_xy']-left.sample(t-h)['target_xy'])/(2*h)
        worst=max(worst,float(np.max(np.abs(fd-left.sample(t)['velocity_xy']))));tested+=1
    errors['actual_reference_velocity_fd']=worst
    check('actual_reference_velocity_fd',worst<=3e-6 and tested>=200)
    check('no_physics_or_heavy_imports',not any(m in sys.modules for m in ('physics','mujoco','torch','static_teacher','student_wave')))
    report=dict(status='NUMPY_PREFLIGHT_PASS' if all(checks.values()) else 'NUMPY_PREFLIGHT_FAIL',
        all_pass=all(checks.values()),checks=checks,max_abs_errors=errors,sourcehash_sha256=sha256(HERE/'SOURCEHASH.json'),
        rest_provenance=dict(path=str(trace),sha256=sha256(trace),field='relativefoot[999]',time_s=1.,
            meaning='Actual saved settled foot from same measure used by StaticTeacher immediately after settle; no Core/FK'),
        rest_body_feet=rest.tolist(),reference_metadata=base.metadata['cfg'],
        finite_difference=dict(h_s=h,synthetic_chain_atol=2e-8,actual_reference_atol=3e-6,baseline_knot_screen_atol=2e-7,tested=tested,excluded_near_knots=excluded),
        cpu_affinity=list(os.sched_getaffinity(0)),python=sys.executable,
        physical_execution=False,training=False,rendering=False,actual_new_trajectory_audit='NOT_RUN')
    (HERE/'PREFLIGHT_NUMPY.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],checks=len(checks),failed=[k for k,v in checks.items() if not v],velocity_error=worst,tested=tested,excluded=excluded)))
    if not report['all_pass']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
