"""NumPy-only independent reference checks. No simulator or training imports."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from reference import FootReference, DEFAULT_REFERENCE, REFERENCE_SHA256, TARGET_LEGS

HERE = Path(__file__).resolve().parent


def run():
    checks = []
    def check(name, condition, **details):
        if not bool(condition):
            raise AssertionError(name + ': ' + repr(details))
        checks.append(dict(name=name, passed=True, **details))
    # Deliberately asymmetric synthetic target geometry; no simulator required.
    rest = np.array([[1.4,.9,-1.3],[-1.2,.8,-1.3],[.1,1.1,-1.3],
                     [1.5,-.85,-1.3],[-1.3,-.75,-1.3],[.2,-1.2,-1.3]])
    original_rest = rest.copy()
    ref = FootReference(rest)
    rest[:] = 999.
    check('rest copied', np.array_equal(ref.rest_body_feet, original_rest))
    with np.load(DEFAULT_REFERENCE, allow_pickle=False) as z:
        raw_xy = z['root2site'][:, [5,3,4,0,2,1], :2].astype(float)
        raw_h = z['height_above_inferred_plane_cm'][:, [5,3,4,0,2,1]].astype(float)
        times = z['time_s'].copy()
    check('reference SHA256 verified', hashlib.sha256(DEFAULT_REFERENCE.read_bytes()).hexdigest() == REFERENCE_SHA256)
    check('exact leg mapping', ref.metadata['source_index_for_target'] == [5,3,4,0,2,1] and list(TARGET_LEGS) == ['LF','LH','LM','RF','RH','RM'])
    check('source immutable and finite', all(not x.flags.writeable and (not np.issubdtype(x.dtype,np.number) or np.isfinite(x).all()) for x in ref.source_arrays.values()))
    check('helper arrays readonly', all(not x.flags.writeable for x in [ref.xy_cm, ref.height_cm, ref.time_s, ref.source_mean_xy, ref.rest_body_feet]))
    check('default duration', np.isclose(ref.duration_s,2.9), duration_s=ref.duration_s)
    # Independently implement smoothing using convolution per scalar trace.
    def smooth(a, weights):
        out = np.empty_like(a)
        for idx in np.ndindex(a.shape[1:]):
            out[(slice(None),)+idx] = np.convolve(np.pad(a[(slice(None),)+idx],len(weights)//2,mode='edge'), weights,mode='valid')
        return out
    smooth_xy = smooth(raw_xy,np.ones(5)/5)
    smooth_h = smooth(raw_h,np.ones(5)/5)
    check('symmetric five sample edge box', np.allclose(ref.xy_cm,smooth_xy,rtol=0,atol=1e-15) and np.allclose(ref.height_cm,smooth_h,rtol=0,atol=1e-15))
    check('full clip per-leg mean before smoothing', np.array_equal(ref.source_mean_xy,raw_xy.mean(axis=0)))
    initial = ref.sample(0.)
    check('blend starts at own rest and zero world height', np.array_equal(initial['target_xy'],original_rest[:,:2]) and np.count_nonzero(initial['target_z']) == 0 and initial['stance'].all() and initial['phase']=='blend')
    check('negative elapsed clamps rest', np.array_equal(ref.sample(-1.)['target_xy'],original_rest[:,:2]) and not ref.sample(-1.)['velocity_xy'].any())
    # Independent equations at many non-knot times check mapping and true phases.
    max_position_error = 0.
    for s in np.arange(.00073,.239, .00317):
        elapsed = .5+10*s
        sample = ref.sample(elapsed)
        xy = np.array([[np.interp(s,times,smooth_xy[:,i,j]) for j in range(2)] for i in range(6)])
        h = np.array([np.interp(s,times,smooth_h[:,i]) for i in range(6)])
        expected_xy = original_rest[:,:2]+10*(xy-raw_xy.mean(axis=0))
        expected_z = 2.5*np.maximum(0.,h-.005)
        max_position_error = max(max_position_error,float(np.max(np.abs(sample['target_xy']-expected_xy))),float(np.max(np.abs(sample['target_z']-expected_z))))
        check('independent target equation %.8f'%s,np.allclose(sample['target_xy'],expected_xy,rtol=0,atol=1e-13) and np.allclose(sample['target_z'],expected_z,rtol=0,atol=1e-13) and np.array_equal(sample['stance'],h<=.005) and abs(sample['source_time_s']-(.1+s))<1e-14)
    derivative_errors = []
    for blend_curve in ('linear','smoothstep'):
        for warp in (2.7,10.):
            test = FootReference(original_rest,time_warp=warp,blend_curve=blend_curve)
            # All source interval interiors, avoid height clipping crossings.
            elapsed_samples = [.5*u for u in (.13,.37,.71,.91)]
            for s in times[:120]+.00073:
                elapsed = .5+warp*s
                eps = 1e-7
                lo, hi = test.sample(elapsed-eps), test.sample(elapsed+eps)
                if np.array_equal(lo['stance'],hi['stance']):
                    elapsed_samples.append(elapsed)
            max_error = 0.
            for elapsed in elapsed_samples:
                eps = 1e-7
                lo, mid, hi = [test.sample(elapsed+x) for x in (-eps,0,eps)]
                for pos,vel in [('target_xy','velocity_xy'),('target_z','velocity_z')]:
                    error = float(np.max(np.abs((hi[pos]-lo[pos])/(2*eps)-mid[vel])))
                    max_error = max(max_error,error)
            check('finite difference '+blend_curve+' warp '+str(warp), max_error<1e-7,max_abs_error=max_error,samples=len(elapsed_samples))
            derivative_errors.append(max_error)
    eps = 1e-9
    l,m,r = [ref.sample(.5+x) for x in (-eps,0,eps)]
    jump = max(float(np.max(np.abs(x[k]-m[k]))) for x in (l,r) for k in ('target_xy','target_z'))
    check('position continuity at blend end',jump<1e-7,max_near_boundary_error=jump)
    check('phase0 held throughout blend', all(ref.sample(x)['source_time_s']==.1 for x in [0.,.1,.3,.49999]))
    sequence = [ref.sample(x) for x in np.linspace(0,20,1001)]
    source_clock = np.array([x['source_time_s'] for x in sequence])
    check('no source phase wrap',np.all(np.diff(source_clock)>=0) and np.isclose(source_clock[-1],.34,rtol=0,atol=1e-15) and np.isclose(source_clock.max(),.34,rtol=0,atol=1e-15))
    end = ref.sample(ref.duration_s)
    late = ref.sample(1000.)
    check('endpoint held and zero derivatives',end['phase']=='end' and late['phase']=='end' and all(np.array_equal(end[k],late[k]) for k in ('target_xy','target_z','velocity_xy','velocity_z','stance')) and not end['velocity_xy'].any() and not end['velocity_z'].any())
    check('endpoint position continuity',all(np.max(np.abs(ref.sample(ref.duration_s-eps)[k]-end[k]))<1e-7 for k in ('target_xy','target_z')))
    expected_keys = {'target_xy','target_z','velocity_xy','velocity_z','stance','source_time_s','phase'}
    check('sample API shapes and finiteness', all(set(s)==expected_keys and s['target_xy'].shape==(6,2) and s['target_z'].shape==(6,) and s['velocity_xy'].shape==(6,2) and s['velocity_z'].shape==(6,) and s['stance'].dtype==bool and all(np.isfinite(s[k]).all() for k in ('target_xy','target_z','velocity_xy','velocity_z')) for s in sequence))
    saved = ref.sample(.7)
    modified = ref.sample(.7)
    modified['target_xy'][:]=999.
    check('sample output independent copies',np.array_equal(saved['target_xy'],ref.sample(.7)['target_xy']))
    for n, weights in [(1,[1.]),(3,[1/3]*3),(5,[.2]*5)]:
        test = FootReference(original_rest,smoothing_samples=n)
        check('box kernel '+str(n),np.allclose(test.xy_cm,smooth(raw_xy,np.array(weights)),rtol=0,atol=1e-15))
    for n, weights in [(3,[.25,.5,.25]),(5,[1/9,2/9,3/9,2/9,1/9])]:
        test = FootReference(original_rest,smoothing_samples=n,smoothing_kernel='triangular')
        check('triangular kernel '+str(n),np.allclose(test.xy_cm,smooth(raw_xy,np.array(weights)),rtol=0,atol=1e-15))
    raw = FootReference(original_rest,smooth_xy=False,smooth_height=False,blend_s=0,source_duration_s=.6)
    check('smoothing disabled and full clip available',np.array_equal(raw.xy_cm,raw_xy) and np.array_equal(raw.height_cm,raw_h) and raw.sample(0)['phase']=='reference' and raw.sample(6)['phase']=='end')
    for kwargs in [dict(scale=0),dict(time_warp=0),dict(vertical_gain=-1),dict(source_duration_s=.601),dict(blend_s=-.1),dict(smoothing_samples=4),dict(smoothing_kernel='bad'),dict(blend_curve='bad'),dict(stance_height_cm=float('nan'))]:
        try:
            FootReference(original_rest,**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid argument accepted '+repr(kwargs))
    check('invalid configuration rejected',True)
    for elapsed in (float('nan'),float('inf')):
        try:
            ref.sample(elapsed)
        except ValueError:
            pass
        else:
            raise AssertionError('nonfinite elapsed accepted')
    check('invalid sample time rejected',True)
    return dict(status='PASS', scope='S2/S5 NumPy reference target checks only; no physics simulation/training',
                check_count=len(checks),checks=checks,max_position_error=max_position_error,
                check_development_note='Initial checker run exited 1 because .1+.24 was compared to .34 with exact float equality; changed only that assertion to absolute tolerance 1e-15 and reran.',
                max_finite_difference_error=max(derivative_errors),metadata=ref.metadata,
                geometry_note='Checks use explicitly synthetic asymmetric rest feet, not measured target-body geometry.',
                source_files_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'reference.py',HERE/'check_reference.py']},
                python=sys.executable,numpy=np.__version__,limitations=ref.metadata['limitations'])


if __name__ == '__main__':
    report=run()
    output=HERE/'REFERENCE_CHECKS.json'
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],check_count=report['check_count'],max_finite_difference_error=report['max_finite_difference_error'],output=str(output))))
