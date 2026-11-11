"""CPU1 pure NumPy exact-prefix/continuity/velocity/geometry checks; no Core."""
import json
import os
import sys
import numpy as np
from source_identity_wider import HERE,FREEZE,verify_sources,sha256
from turning_teacher import WiderRadiusReference,finite_xy_velocity,BASELINE


def main():
    allowed=os.sched_getaffinity(0);os.sched_setaffinity(0,{1 if 1 in allowed else min(allowed)})
    verify_sources()
    sys.path.insert(0,str(BASELINE))
    from wave_reference import WaveReference
    old=HERE.parent/'left_finite_twist'/'left_yaw_p060'
    refpath=old/'REFERENCE.json';rest=np.asarray(json.loads(refpath.read_text())['rest_body_feet'])
    with np.load(old/'TRACE_1MS.npz',allow_pickle=False) as z:
        assert np.array_equal(rest,z['relativefoot'][999])
    base=WaveReference(rest,stride_native=.36,lift_native=.12,step_s=.8,cycles=100)
    wide=WiderRadiusReference(base)
    base48=WaveReference(rest,stride_native=.48,lift_native=.12,step_s=.8,cycles=100)
    checks={};errors={};details={}
    def check(name,val):checks[name]=bool(val)
    def err(name,a,b,tol):
        value=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
        errors[name]=value;check(name,value<=tol)
    def same(a,b):return set(a)==set(b) and all(np.array_equal(a[k],b[k]) for k in a)
    times=np.arange(2880)*.005
    check('entire_original_prefix_exact',all(same(wide.sample(float(t)),base.sample(float(t))) for t in np.r_[[-.1],times[times<=4.8],[4.8]]))
    fields=('target_z','velocity_z','stance','phase','source_time_s')
    check('all_nonhorizontal_fields_exact',all(all(np.array_equal(wide.sample(t)[key],base.sample(t)[key]) for key in fields) for t in times))
    commands=np.array([wide.command(t) for t in times]);u=np.clip(times-4.8,0,1);b=u*u*(3-2*u)
    v=.075+.025*b;omega=.06*b
    err('no_phase_warp',commands[:,0],times,0.)
    err('physical_nominal_speed',.075*commands[:,1],v,3e-17)
    err('physical_nominal_yaw',.075*commands[:,1]*commands[:,2],omega,3e-17)
    err('curvature_quotient',commands[:,2],omega/v,1e-15)
    err('normalized_forward',commands[:,1]-1.,v/.075-1.,4e-16)
    check('final_forward_domain_nonzero',abs(commands[-1,1]-1.-1/3)<1e-15)
    check('final_k_v_yaw',np.allclose([wide.kinematics(6.)[k] for k in ('k','nominal_v','nominal_omega')],[.6,.1,.06],atol=1e-15,rtol=0))
    check('base_still_original_cfg',base.stride_native==.36 and base.step_s==.8 and base.lift_native==.12 and base.cycles==100)
    # Independent direct stride.48 WaveReference provides final reference check.
    maxp=maxv=0.
    for t in np.linspace(5.8,14.39,160):
        target=base48.sample(t)
        p,vv=finite_xy_velocity(rest[:,:2],target['target_xy'],target['velocity_xy'],.6,0.)
        actual=wide.sample(t)
        maxp=max(maxp,float(np.max(np.abs(p-actual['target_xy']))))
        maxv=max(maxv,float(np.max(np.abs(vv-actual['velocity_xy']))))
    errors['steady_stride48_position']=maxp;errors['steady_stride48_velocity']=maxv
    check('steady_stride48_equivalence',maxp<2e-14 and maxv<2e-13)
    rng=np.random.default_rng(20261111);h=1e-6;worst=0.;tested=excluded=0
    for t in np.r_[rng.uniform(4.801,5.799,180),rng.uniform(5.801,14.399,180)]:
        fd_base=(base.sample(t+h)['target_xy']-base.sample(t-h)['target_xy'])/(2*h)
        if np.max(np.abs(fd_base-base.sample(t)['velocity_xy']))>2e-7:
            excluded+=1;continue
        fd=(wide.sample(t+h)['target_xy']-wide.sample(t-h)['target_xy'])/(2*h)
        worst=max(worst,float(np.max(np.abs(fd-wide.sample(t)['velocity_xy']))));tested+=1
    errors['analytic_qdot_kdot_velocity_fd']=worst
    check('analytic_qdot_kdot_velocity_fd',worst<3e-6 and tested>=330)
    details['velocity_fd']=dict(h_s=h,atol=3e-6,baseline_knot_screen_atol=2e-7,tested=tested,excluded=excluded)
    # Explicit kdot finite difference of quotient, independent of foot geometry.
    kerr=0.
    for t in np.linspace(4.801,5.799,99):
        numeric=(wide.kinematics(t+h)['k']-wide.kinematics(t-h)['k'])/(2*h)
        kerr=max(kerr,abs(numeric-wide.kinematics(t)['kdot']))
    errors['quotient_kdot_fd']=kerr;check('quotient_kdot_fd',kerr<1e-8)
    # Test prefix onset, internal ramp step seam, ramp end, later step/cycle seams.
    seam_reports=[]
    for t in (4.8,5.6,5.8,6.4,7.2,8.,8.8,9.6,14.4):
        at=wide.sample(t);before=wide.sample(t-h);after=wide.sample(t+h)
        residual=max(float(np.max(np.abs(before['target_xy']+h*at['velocity_xy']-at['target_xy']))),
                     float(np.max(np.abs(after['target_xy']-h*at['velocity_xy']-at['target_xy']))))
        vdiff=float(np.max(np.abs(after['velocity_xy']-before['velocity_xy'])))
        seam_reports.append(dict(t=t,position_linear_residual=residual,velocity_side_difference=vdiff))
    check('no_horizontal_position_jump_seams',max(x['position_linear_residual'] for x in seam_reports)<1e-8)
    check('no_horizontal_velocity_jump_seams',max(x['velocity_side_difference'] for x in seam_reports)<1e-3)
    details['seams']=seam_reports
    # Dense ramp envelope: a 5ms target delta must be accounted for by continuous
    # analytic velocity over that interval; compare against 1ms envelope +1e-5.
    dense=np.arange(4.8,5.8000001,.001)
    samples=[wide.sample(t) for t in dense]
    xy=np.array([s['target_xy'] for s in samples]);vv=np.array([s['velocity_xy'] for s in samples])
    jump=np.linalg.norm(xy[5:]-xy[:-5],axis=2)
    bound=np.array([np.max(np.linalg.norm(vv[i:i+6],axis=2),axis=0)*.005 for i in range(len(samples)-5)])
    check('ramp_5ms_displacement_velocity_bound',bool(np.all(jump<=bound+1e-5)))
    details['ramp_5ms']=dict(max_target_displacement=float(jump.max()),max_velocity_bound=float(bound.max()),allowance=1e-5)
    # Constant-k stance spacing after stride ramp.
    a=wide.sample(6.123);bb=wide.sample(6.133);mask=a['stance']&bb['stance']
    dist=lambda p:np.linalg.norm(p[:,None,:]-p[None,:,:],axis=-1)
    check('five_stance_legs',int(mask.sum())==5)
    err('steady_stance_spacing',dist(a['target_xy'][mask]),dist(bb['target_xy'][mask]),2e-14)
    # LM first diagnostic swing occurs after ramp end; retain original X metric.
    ts=np.linspace(7.2,8.,801);p=np.array([wide.sample(t)['target_xy'][2] for t in ts]);z=np.array([wide.sample(t)['target_z'][2] for t in ts])
    dx=p[:,0]-p[0,0]
    check('LM_reference_original_X_margin',float(dx.max())>.10 and float(dx[z>.015].max())>.10)
    details['LM_reference_only']=dict(window_teacher_elapsed=[7.2,8.],max_positive_X=float(dx.max()),clear_max_positive_X=float(dx[z>.015].max()),old_gate=.03,physical_feasibility='NOT_RUN')
    from nonfoot_pairs import PairPeaks
    empty=PairPeaks().summary()
    check('zero_nonfoot_empty_attribution',empty['pairs']=={} and empty['positive_native_steps']==0 and empty['overflow_contact_observations']==0)
    pairs=PairPeaks()
    for j in range(1,66):
        pairs.observe(dict(geom_ids=[0,j],time_before_s=6.+j*.001,positive_world_z_native=float(j),geom_names=['plane','synthetic_record_'+str(j)]))
    pairs.observe(dict(geom_ids=[0,1],time_before_s=7.,positive_world_z_native=70.,geom_names=['plane','synthetic_record_1']))
    bounded=pairs.summary()
    check('nonfoot_attribution_bound64',len(bounded['pairs'])==64 and bounded['overflow_contact_observations']==1 and bounded['overflow_peak']['positive_world_z_native']==65.)
    first=bounded['pairs']['0:1']
    check('nonfoot_attribution_counts_peak_times',first['contact_observations']==2 and first['post_contact_observations']==2 and first['first_time_before_s']==6.001 and first['last_time_before_s']==7. and first['peak']['positive_world_z_native']==70. and first['post_peak']['positive_world_z_native']==70.)
    details['nonfoot_aggregation_test']='Synthetic records only: no Core/contact construction/MuJoCo import; zero/bound64/overflow/count/first-last/peak checked. Real accessor names are read only in future solved() runtime.'
    check('no_heavy_imports',not any(m in sys.modules for m in ('physics','mujoco','torch','static_teacher','student_wave')))
    verify_sources()
    report=dict(status='NUMPY_PREFLIGHT_PASS' if all(checks.values()) else 'NUMPY_PREFLIGHT_FAIL',all_pass=all(checks.values()),checks=checks,max_abs_errors=errors,details=details,
        sourcehash_sha256=sha256(FREEZE),rest_source=str(refpath),rest_source_sha256=sha256(refpath),rest_body_feet=rest.tolist(),
        cpu_affinity=list(os.sched_getaffinity(0)),python=sys.executable,physical_execution=False,training=False,rendering=False,new_trajectory_audited=False,
        note='Reference targets/derivatives only. Exact mathematical prefix does not claim physical prefix replay before actual execution; future saved audit requires bitexact physical prefix.')
    out=HERE/'PREFLIGHT_WIDER_NUMPY.json'
    if out.exists():raise RuntimeError('Preflight report exists; preserve it and use a new supplementary filename')
    with out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=report['status'],checks=len(checks),failed=[k for k,v in checks.items() if not v],max_velocity_fd=worst,LM_X=float(dx.max()),seams=seam_reports,ramp_5ms=details['ramp_5ms'])))
    if not report['all_pass']:raise SystemExit(1)


if __name__=='__main__':main()
