"""Numpy-only independent checks; writes only WAVE_REFERENCE_CHECKS.json."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from wave_reference import (WaveReference, DEFAULT_REFERENCE, REFERENCE_SHA256,
                            TARGET_LEGS, SOURCE_LEGS)

BASE = Path(__file__).resolve().parent
REPORT = BASE/'WAVE_REFERENCE_CHECKS.json'


def main():
    checks = []
    def check(name, condition, detail=None):
        checks.append(dict(name=name, passed=bool(condition), detail=detail))
    rest = np.array([[.6,.4,-1.],[-.5,.4,-1.],[0.,.5,-1.],
                     [.6,-.4,-1.],[-.5,-.4,-1.],[0.,-.5,-1.]])
    r = WaveReference(rest)
    check('verified_raw_npz_sha256', hashlib.sha256(DEFAULT_REFERENCE.read_bytes()).hexdigest() == REFERENCE_SHA256)
    raw = np.load(DEFAULT_REFERENCE, allow_pickle=False)
    check('retained_raw_arrays_exact', all(np.array_equal(raw[k], r.source_arrays[k]) for k in raw.files))
    check('source_and_target_leg_identity', tuple(raw['leg_order']) == SOURCE_LEGS and r.metadata['source_index_for_target'] == [5,3,4,0,2,1])
    coef = raw['inferred_plane_z_abc'].astype(float)
    world = raw['foot_world_cm'].astype(float)
    distance = (world[:,:,2]-(world[:,:,0]*coef[0]+world[:,:,1]*coef[1]+coef[2]))/np.sqrt(1+sum(coef[:2]**2))
    check('source_height_is_world_inferred_plane_distance', np.max(np.abs(distance-raw['height_above_inferred_plane_cm'])) < 1e-7)
    # Reconstruct with an independent convolution/interpolation implementation.
    def smoothed(a):
        return np.convolve(np.pad(a.astype(float),(2,2),mode='edge'),np.ones(5)/5,mode='valid')
    independent_peaks = {}
    reconstruction_errors = {}
    for j, leg in enumerate(TARGET_LEGS):
        source_i = SOURCE_LEGS.index(leg)
        xy = np.column_stack([smoothed(raw['root2site'][:,source_i,k]) for k in (0,1)])
        h = smoothed(raw['height_above_inferred_plane_cm'][:,source_i])
        candidates = []
        for lo in range(300):
            if not (h[lo] <= .005 < h[lo+1]):
                continue
            down = [hi for hi in range(lo+1,300) if h[hi] > .005 >= h[hi+1]]
            if not down:
                continue
            hi = down[0]
            start = lo+(.005-h[lo])/(h[lo+1]-h[lo])
            end = hi+(.005-h[hi])/(h[hi+1]-h[hi])
            peak = lo+1+int(np.argmax(h[lo+1:hi+1]))
            span = np.interp(end,np.arange(301),xy[:,0])-np.interp(start,np.arange(301),xy[:,0])
            if span >= .03 and h[peak] > .015 and h[peak] > h[peak-1] and h[peak] >= h[peak+1]:
                candidates.append((start,end,peak,span))
        start,end,peak,span = candidates[0]
        rec = r.segments[leg]
        check(leg+'_first_complete_positive_measured_segment',
              abs(start-rec['start_fractional_index']) < 1e-10 and abs(end-rec['end_fractional_index']) < 1e-10 and peak == rec['peak_native_index'] and abs(span-rec['advance_root_local_x_cm']) < 1e-12,
              dict(valid_complete_count=len(candidates), peak_native_index=peak, advance_cm=float(span)))
        idx = np.r_[start,np.arange(int(np.floor(start))+1,int(np.ceil(end))),end]
        u = (idx-start)/(end-start)
        points = np.column_stack([np.interp(idx,np.arange(301),xy[:,k]) for k in (0,1)])
        hh = np.interp(idx,np.arange(301),h)
        x = np.clip((points[:,0]-points[0,0])/span,0,1)
        y = (points[:,1]-((1-u)*points[0,1]+u*points[-1,1]))/span
        z = np.maximum(0,hh-((1-u)*hh[0]+u*hh[-1])); z /= z.max()
        grid = np.linspace(0,1,201)
        expected = np.column_stack([np.interp(grid,u,v) for v in (x,y,z)])
        expected[:,2] /= expected[:,2].max(); expected[0]=[0,0,0]; expected[-1]=[1,0,0]
        error = float(np.max(np.abs(expected-r.shapes[j])))
        reconstruction_errors[leg] = error
        check(leg+'_shape_reconstructed_from_raw', error < 1e-11, error)
        check(leg+'_native_time_provenance', np.allclose(rec['source_time_range_s'], .1+np.array([start,end])*.002,rtol=0,atol=1e-12) and rec['method']=='first_valid_complete_low_height_crossings')
        independent_peaks[leg] = float(raw['source_time_s'][peak])
    check('default_order_measured_peak_time', list(r.order)==sorted(TARGET_LEGS,key=independent_peaks.get),list(r.order))
    max_derivative_error = 0.
    max_airborne = 0
    visited = set()
    # Avoid shape knots (u multiples of 1/200), schedule seams, and end clamps.
    for step in range(12):
        for u in (.0173,.1237,.3373,.5017,.7873,.9817):
            t = (step+u)*r.step_s
            a = r.sample(t); eps=1e-7
            minus,plus = r.sample(t-eps),r.sample(t+eps)
            for pos,vel in [('target_xy','velocity_xy'),('target_z','velocity_z')]:
                err = np.max(np.abs((plus[pos]-minus[pos])/(2*eps)-a[vel]))
                max_derivative_error=max(max_derivative_error,float(err))
            air=np.flatnonzero(~a['stance'])
            max_airborne=max(max_airborne,len(air)); visited.update(TARGET_LEGS[k] for k in air)
            active = r.order_indices[step%6]
            others = np.arange(6)!=active
            check('step%d_u%.4f_stance_motion'%(step,u),
                  np.allclose(a['velocity_xy'][others,0],-r.stride_native/(6*r.step_s)) and np.all(a['velocity_xy'][others,1]==0) and np.all(a['target_z'][others]==0))
    check('analytic_velocity_finite_difference_away_knots',max_derivative_error<3e-8,max_derivative_error)
    dense = [r.sample(t) for t in np.linspace(0,r.duration_s,12001)]
    max_airborne=max(max_airborne,max(np.sum(~a['stance']) for a in dense))
    check('at_most_one_airborne_target_and_five_support_labels',max_airborne==1,int(max_airborne))
    check('all_six_legs_visited',visited==set(TARGET_LEGS),sorted(visited))
    seam_error=0.
    for step in range(13):
        t=step*r.step_s;eps=1e-9
        a,b,c=r.sample(t-eps),r.sample(t),r.sample(t+eps)
        seam_error=max(seam_error,*[float(np.max(np.abs(v[key]-b[key]))) for v in (a,c) for key in ('target_xy','target_z')])
    check('position_continuity_all_leg_and_cycle_and_clamp_seams',seam_error<1e-7,seam_error)
    check('cycle_offsets_return_zero',all(np.allclose(r.sample(k*6*r.step_s)['target_xy'],rest[:,:2],rtol=0,atol=1e-14) for k in (0,1,2)))
    check('clamped_end_zero_velocity',all(np.all(r.sample(t)[key]==0) for t in (r.duration_s,r.duration_s+100) for key in ('velocity_xy','velocity_z')))
    check('negative_time_rest_and_zero_velocity',np.array_equal(r.sample(-1)['target_xy'],rest[:,:2]) and np.all(r.sample(-1)['velocity_xy']==0))
    check('sample_api_matches_FootReference',set(r.sample(.3))=={'target_xy','target_z','velocity_xy','velocity_z','stance','source_time_s','phase'})
    check('sample_copies_do_not_mutate_reference',not np.shares_memory(r.sample(.3)['target_xy'],r.rest_body_feet))
    check('metadata_declares_adaptation_teacher_only_no_physics',all(term in json.dumps(r.metadata) for term in ('NOT exact original animal timing','never deployment NN phase input','no root writes','not contact/force')))
    check('duration_contract',r.duration_s==r.cycles*6*r.step_s)
    check('source_clock_matches_active_segment',all(abs(r.sample((i+.371)*r.step_s)['source_time_s']-(r.segments[leg]['source_time_range_s'][0]+.371*np.ptp(r.segments[leg]['source_time_range_s'])))<1e-12 for i,leg in enumerate(r.order)))
    override=tuple(reversed(TARGET_LEGS)); other=WaveReference(rest,order=override,cycles=1)
    check('explicit_order_override_logged',other.order==override and other.metadata['order_selection']=='explicit caller override')
    from wave_reference import _extract
    rejected=False
    try:
        _extract(np.zeros((301,2)),smoothed(raw['height_above_inferred_plane_cm'][:,0]),raw['time_s'],raw['source_time_s'],'RF')
    except ValueError as exc:
        rejected='No complete positive measured swing' in str(exc)
    check('missing_positive_segment_fails_no_synthetic_fallback',rejected)
    invalid_count=0
    for kwargs in ({'order':('LF',)*6},{'cycles':0},{'cycles':1.5},{'step_s':0},{'stride_native':-1},{'lift_native':float('nan')}):
        try: WaveReference(rest,**kwargs)
        except ValueError: invalid_count+=1
    for t in (float('nan'),float('inf')):
        try:r.sample(t)
        except ValueError:invalid_count+=1
    check('invalid_configuration_and_nonfinite_time_rejected',invalid_count==8)
    report=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',
                goal='S2/S5 measured-shape algorithm teacher reference only',
                command=[sys.executable,'-B',str(Path(__file__).resolve())],
                checks=checks,passed=sum(c['passed'] for c in checks),total=len(checks),
                input_identity={str(DEFAULT_REFERENCE):REFERENCE_SHA256},
                source_files={str(BASE/name):hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in ('wave_reference.py','check_wave_reference.py')},
                numpy_version=np.__version__,python_version=sys.version,
                metadata=r.metadata,shape_reconstruction_max_abs_error=reconstruction_errors,
                max_derivative_error_native_s=max_derivative_error,max_seam_error_native=seam_error,
                not_executed=['Physics simulation','Physical support/contact/load verification','Walking acceptance','Deployment neural-network integration'],
                interpretation='Reference checks only. Adapted teacher wave timing is not original animal timing or evidence of physical walking.')
    REPORT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],passed=report['passed'],total=len(checks),report=str(REPORT),order=list(r.order),max_derivative_error=max_derivative_error,max_seam_error=seam_error,failures=[c for c in checks if not c['passed']]),indent=2))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
