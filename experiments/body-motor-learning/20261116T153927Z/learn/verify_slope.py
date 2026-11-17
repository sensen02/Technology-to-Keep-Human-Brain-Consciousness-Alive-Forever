"""Compare frozen slope/control evidence and preserve the first raw failure."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import slope


def identify(folder):
    result=slope.bench.read(folder/'RESULT.json')
    for name,digest in result['files'].items():
        if slope.STAND.sha(folder/name)!=digest:
            raise ValueError(f'Slope evidence changed: {folder/name}')
    return result


def compare(control_path, slope_path, output):
    control,candidate=map(identify,(control_path,slope_path))
    names=('time','qpos','qvel')
    with np.load(control_path/'TRACE.npz',allow_pickle=False) as a, np.load(slope_path/'TRACE.npz',allow_pickle=False) as b:
        equal={name:bool(np.array_equal(a[name][0],b[name][0])) for name in names}
    expected=candidate['walking']['terrain_ascent']
    actual=candidate['walking']['actual_ascent']
    control_level=abs(control['walking']['actual_ascent'])<.012
    data=dict(matched_initial=equal,control=str(control_path),candidate=str(slope_path),
              same_walking_duration=candidate['walk_ms']==control['walk_ms'],
              actual_candidate_ascent=actual,expected_candidate_ground_rise=expected,
              actual_control_ascent=control['walking']['actual_ascent'],
              matched_control_level=control_level, slope_gate_passed=candidate['passed'],flat_gate_passed=control['passed'],
              same_policy=candidate.get('policy_kind')==control.get('policy_kind'),
              both_pass=bool(all(equal.values()) and control_level and candidate['passed'] and control['passed']))
    slope.STAND.put(output,data)


def archived_failure(folder, output):
    # The original run wrote the raw arrays and had already completed its runtime
    # verification before the final JSON encoder refused a numpy.boolean value.
    identity=slope.bench.read(folder/'RUN_IDENTITY.json')
    cfg=identity['config']
    old=slope.HERE/'slope_original_archive.py'
    expected=identity['files'][str(slope.HERE/'slope.py')]['sha256']
    if expected!=slope.STAND.sha(old):
        raise ValueError('Original source archive does not match the run')
    angle=np.deg2rad(cfg['slope_deg'])
    terrain=SimpleNamespace(angle=angle,normal=np.array([-np.sin(angle),0.,np.cos(angle)]),
                uphill=np.array([np.cos(angle),0.,np.sin(angle)]),point=np.array([0.,0.,-slope.PIVOT*np.tan(angle)]))
    with np.load(folder/'TRACE.npz',allow_pickle=False) as a, np.load(folder/'NATIVE.npz',allow_pickle=False) as b:
        trace={field:a[field] for field in a.files}
        poses=b['pose']
        count=len(poses)
        np.testing.assert_array_equal(poses[::slope.NATIVE_PER_MS,:3],trace['com'][:-1])
        recorder=SimpleNamespace(terrain=terrain,poses=poses,normal=b['surface_normal'],forces=b['forces'],
                    count=count,endpoints=trace['qpos'],cross_checks=1+(count-1)//4031)
    result=slope.audit(recorder,trace,slope.WARM+cfg['walk_ms'],cfg['horizon_ms'])
    assert not result['passed']
    mapped=dict(**result,angle_deg=cfg['slope_deg'],scope='offline audit of retained partial-JSON run',
                raw_files={name:slope.STAND.sha(folder/name) for name in ('TRACE.npz','DECISIONS.npz','NATIVE.npz')},
                original_source_sha256=slope.STAND.sha(old),checks_movement_before_failed_summary=True,
                native_com_endpoint_recheck=True,native_replay_performed=False)
    slope.STAND.put(output,mapped)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('output',type=Path)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--control',type=Path)
    group.add_argument('--retained-failure',action='store_true')
    a=p.parse_args()
    if a.control:
        compare(a.control.resolve(),a.run.resolve(),a.output.resolve())
    else:
        archived_failure(a.run.resolve(),a.output.resolve())
