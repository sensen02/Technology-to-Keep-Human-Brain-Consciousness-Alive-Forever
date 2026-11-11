"""One wider-radius LEFT physical adapter, CODEONLY unless --run-physics.

Only frozen parent one/PeakRecorder are reused. Its fixed .075 observation
feature is labelled as historical; nominal speed/yaw come from reference.command
and correctly ramp to .1/.06. Never call parent sweep or old audit writer.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import sys
import time
from source_identity_wider import HERE,FREEZE,verify_sources,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-physics',action='store_true')
    args=p.parse_args()
    frozen,prereg=verify_sources()
    if not args.run_physics:
        print(json.dumps(dict(status='CODEONLY_READY_NO_PHYSICS',root=str(HERE),conditions=prereg['conditions'],frozen_files=len(frozen['files']))))
        return
    for var in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(var)!='1':raise RuntimeError(f'{var}=1 required before numerical imports')
    allowed=os.sched_getaffinity(0);os.sched_setaffinity(0,{1 if 1 in allowed else min(allowed)})
    folder=HERE/'left_yaw_p060'
    if folder.exists():raise RuntimeError('Candidate output exists; never overwrite or silently rerun')
    import turning_teacher as wider
    if Path(wider.__file__).resolve()!=HERE/'turning_teacher.py':raise RuntimeError('Wrong teacher module')
    wider.pin_baseline()
    legacy_path=HERE.parent/'attempt02'/'diagnose.py'
    spec=importlib.util.spec_from_file_location('_wider_radius_parent_diagnostic',legacy_path)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    if mod.make_teacher is not wider.make_teacher:raise RuntimeError('Parent imported wrong factory')
    mod.HERE=HERE
    if mod.HERE.resolve()!=Path(prereg['root']).resolve():raise RuntimeError('Output ROOT mismatch')
    from nonfoot_pairs import recorder_class
    mod.PeakRecorder=recorder_class(mod.PeakRecorder)
    def factory(core,**kwargs):
        teacher=wider.make_teacher(core,**kwargs)
        mod.dump(folder/'REFERENCE_WIDER_RADIUS.json',dict(rest_body_feet=teacher.reference.base.rest_body_feet.tolist(),
                 original_reference=teacher.reference.base.metadata,wider_radius=teacher.reference.metadata))
        return teacher
    mod.make_teacher=factory
    mod.sw.seed_all(prereg['seed'])
    mod.dump(HERE/'RUNTIME_WIDER_RADIUS.json',dict(pid=os.getpid(),cpu_affinity=list(os.sched_getaffinity(0)),python=sys.executable,
        sourcehash_sha256=sha256(FREEZE),root=str(HERE),legacy_runner=str(legacy_path),
        environment={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        base_obs_target_speed_constant=.075,base_obs_note='Historical saved X238 feature, not actual desired nominal speed',
        nominal_speed_schedule='.075+.025*b',nominal_yaw_schedule='.06*b',
        start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    # Parent one stores commands [.075*scale, .075*scale*k]. New command
    # returns phase=t, scale=q=v/.075, k=omega/v, with no phase warping.
    mod.one(dict(prereg['conditions'][0]))
    verify_sources()
    result_path=folder/'RESULT.json';r=json.loads(result_path.read_text())
    import numpy as np
    with np.load(folder/'TRACE_1MS.npz',allow_pickle=False) as z:
        loads=z['loads'][5800:];speed=z['speed'][5800:]
    load_den=loads.sum(0);vt=(loads*speed).sum(0)/np.maximum(load_den,1e-12)
    eligible=bool(r['all_gates'] and r['cycles']['all_legs_completed'] and r['native']['post_nonfoot_peak_native']==0.
                  and r['actual_signed_yaw_rad']>=prereg['acceptance']['left_actual_yaw_rad_min'] and r['early_stop'] is None)
    r.update(teacher_eligible=eligible,sourcehash_sha256=sha256(FREEZE),mapping='finite SE2 with smooth wider-radius stride',
        nominal_speed_schedule='.075+.025*b',nominal_yaw_schedule='.06*b',curvature_schedule='(.06*b)/(.075+.025*b)',
        base_obs_target_speed_constant=.075,base_obs_note='Historical input feature only; nominal desired speed ramps to .1; never used as actual measured speed gate',
        normalized_forward_domain='v/.075-1 ramps0..1/3; diagnostic only, future ANN domain unqualified',
        slip_numerator_loadweighted_foot_speed=vt.tolist(),slip_COM_speed_denominator=r['forward_speed_native_per_s'],
        artifact_hashes={name:sha256(folder/name) for name in ('YAW_1MS.npz','REFERENCE_WIDER_RADIUS.json','NATIVE_MANIFEST.json','INITIAL_RUNTIME.npz','IDENTITY.json')},
        independent_saved_audit='PENDING; unique AUDIT_WIDER_RADIUS_*.json only')
    mod.dump(result_path,r)
    mod.dump(HERE/'CANDIDATE_WIDER_RADIUS.json',dict(condition=prereg['conditions'][0],teacher_eligible=eligible,
        result=str(result_path),result_sha256=sha256(result_path),actual_yaw_deg=r['actual_signed_yaw_deg'],slip=r['slip_ratio_max'],
        native_peak=r['native']['post_nonfoot_peak_native'],cycles=r['cycles']['complete_per_leg'],independent_saved_audit='PENDING'))
    print(json.dumps(dict(teacher_eligible=eligible,actual_yaw_deg=r['actual_signed_yaw_deg'],slip=r['slip_ratio_max'],independent_saved_audit='PENDING')))


if __name__=='__main__':main()
