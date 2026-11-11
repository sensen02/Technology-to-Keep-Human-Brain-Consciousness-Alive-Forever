"""Full single LEFT candidate; safe CODEONLY default. Parent --run-physics only."""
from pathlib import Path
import argparse
import json
import os
import time
from candidate_identity import HERE,FREEZE,verify_sources,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-physics',action='store_true');args=p.parse_args()
    m,prereg=verify_sources()
    if not args.run_physics:
        print(json.dumps(dict(status='CODEONLY_READY_NO_PHYSICS',root=str(HERE),frozen_files=len(m['files']))));return
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(name)!='1':raise RuntimeError(f'{name}=1 required before imports')
    allowed=os.sched_getaffinity(0);os.sched_setaffinity(0,{1 if 1 in allowed else min(allowed)})
    folder=HERE/'left_yaw_p060'
    if folder.exists():raise RuntimeError('Candidate output already exists; never overwrite/rerun')
    import turning_teacher as teacher_source
    if Path(teacher_source.__file__).resolve()!=HERE/'turning_teacher.py':raise RuntimeError('Wrong candidate factory import')
    teacher_source.pin_baseline()
    mod=teacher_source.load('_clearance_parent_diagnostic',HERE.parent/'attempt02'/'diagnose.py')
    if mod.make_teacher is not teacher_source.make_teacher:raise RuntimeError('Wrong parent factory')
    mod.HERE=HERE
    if mod.HERE!=Path(prereg['root']):raise RuntimeError('Output ROOT mismatch')
    pairs=teacher_source.load('_clearance_frozen_nonfoot_pairs',HERE.parent/'left_wider_radius'/'nonfoot_pairs.py')
    mod.PeakRecorder=pairs.recorder_class(mod.PeakRecorder)
    teachers=[]
    def factory(core,**kwargs):
        t=teacher_source.make_teacher(core,diagnostic_path=folder/'CLEARANCE_DIAGNOSTICS.jsonl',**kwargs);teachers.append(t)
        mod.dump(folder/'REFERENCE_CLEARANCE_TEACHER.json',dict(rest_body_feet=t.reference.base.rest_body_feet.tolist(),original_reference=t.reference.base.metadata,
            wider_radius=t.reference.metadata,RH_secondary=t.metadata['RH_secondary'],actuated_qpos_indices=t.qa.tolist(),
            initial_q_des_act=t.initial_qpos[t.qa].tolist(),RH_actuator_indices=t._rh.tolist()))
        return t
    mod.make_teacher=factory;mod.sw.seed_all(prereg['seed'])
    mod.dump(HERE/'RUNTIME_CLEARANCE_TEACHER.json',dict(pid=os.getpid(),cpu_affinity=list(os.sched_getaffinity(0)),
        sourcehash_sha256=sha256(FREEZE),root=str(HERE),start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        environment={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        nominal_speed_schedule='.075+.025*b',nominal_yaw_schedule='.06*b',base_obs_target_speed_constant=.075,
        base_obs_note='Historical X238 feature only; never called current nominal desired speed'))
    try:mod.one(dict(prereg['conditions'][0]))
    finally:
        for t in teachers:t.close_diagnostics()
    verify_sources()
    import numpy as np
    rpath=folder/'RESULT.json';r=json.loads(rpath.read_text())
    with np.load(folder/'TRACE_1MS.npz',allow_pickle=False) as z:loads=z['loads'][5800:];speed=z['speed'][5800:]
    vt=(loads*speed).sum(0)/np.maximum(loads.sum(0),1e-12)
    summary=dict(records=0,triggers=0,accepted=0,actual_unsafe_q_des_safe=0,status_counts={},max_secondary_delta=0.,max_pad_error=0.)
    with (folder/'CLEARANCE_DIAGNOSTICS.jsonl').open() as f:
        for line in f:
            d=json.loads(line);summary['records']+=1
            for key in ('trigger','accepted','actual_unsafe_q_des_safe'):
                target={'trigger':'triggers','accepted':'accepted','actual_unsafe_q_des_safe':'actual_unsafe_q_des_safe'}[key];summary[target]+=int(d[key])
            summary['status_counts'][d['status']]=summary['status_counts'].get(d['status'],0)+1
            summary['max_secondary_delta']=max(summary['max_secondary_delta'],d['max_abs_secondary_delta'])
            summary['max_pad_error']=max(summary['max_pad_error'],d['pad_error_norm'])
    eligible=bool(r['all_gates'] and r['cycles']['all_legs_completed'] and r['native']['post_nonfoot_peak_native']==0.
        and r['actual_signed_yaw_rad']>=prereg['acceptance']['left_actual_yaw_rad_min'] and r['early_stop'] is None)
    r.update(teacher_eligible=eligible,sourcehash_sha256=sha256(FREEZE),mapping='Unchanged wider-radius reference plus RH q_des-shadow secondary IK',
        nominal_speed_schedule='.075+.025*b',nominal_yaw_schedule='.06*b',base_obs_target_speed_constant=.075,
        base_obs_note='Historical constant X238 feature only; nominal speed ramps to .1',
        slip_numerator_loadweighted_foot_speed=vt.tolist(),slip_COM_speed_denominator=r['forward_speed_native_per_s'],
        clearance_summary=summary,independent_saved_audit='PENDING; unique AUDIT_CLEARANCE_TEACHER_*.json only',
        artifact_hashes={name:sha256(folder/name) for name in ('YAW_1MS.npz','REFERENCE_CLEARANCE_TEACHER.json','CLEARANCE_DIAGNOSTICS.jsonl','NATIVE_MANIFEST.json','INITIAL_RUNTIME.npz','IDENTITY.json')})
    mod.dump(rpath,r)
    mod.dump(HERE/'CANDIDATE_CLEARANCE_TEACHER.json',dict(teacher_eligible=eligible,result=str(rpath),result_sha256=sha256(rpath),actual_yaw_deg=r['actual_signed_yaw_deg'],
        slip=r['slip_ratio_max'],native_peak=r['native']['post_nonfoot_peak_native'],cycles=r['cycles']['complete_per_leg'],clearance_summary=summary,independent_saved_audit='PENDING'))
    print(json.dumps(dict(teacher_eligible=eligible,yaw_deg=r['actual_signed_yaw_deg'],slip=r['slip_ratio_max'],native_peak=r['native']['post_nonfoot_peak_native'],clearance_summary=summary)))


if __name__=='__main__':main()
