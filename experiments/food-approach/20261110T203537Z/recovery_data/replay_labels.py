"""Private saved-state teacher replay. Restores only private Core data; mj_step blocked."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import numpy as np
from collect import HERE,ROOT,BASE,TEACHER,digest,jsonable
sys.path.insert(0,str(BASE))


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def main(condition):
    import mujoco as mj
    import student_wave as sw
    from identity import verify,verify_runtime
    def blocked(*a,**kw):raise RuntimeError('No physical integration permitted in label replay')
    mj.mj_step=mj.mj_step1=mj.mj_step2=blocked
    run=HERE/('COLLECT_'+condition);ident=json.loads((run/'IDENTITY.json').read_text());verify(ident)
    t=arrays(run/'trace_1ms.npz');s=arrays(run/'SAMPLES_5MS.npz')
    core=sw.Core(adhesion=False);verify_runtime(ident,core)
    def restore(index):
        core.d.qpos[:]=t['qpos'][index];core.d.qvel[:]=t['qvel'][index]
        core.d.ctrl[:]=t['actualctrl'][index];core.d.time=float(t['time'][index])
        core.actuator.target_rad[:]=t['target'][index]
        mj.mj_forward(core.plant,core.d)
    restore(999)
    spec=importlib.util.spec_from_file_location('_independent_recovery_label_teacher',TEACHER)
    tm=importlib.util.module_from_spec(spec);spec.loader.exec_module(tm)
    teacher=tm.make_teacher(core,speed_scale=1.,yaw_rate=.008 if condition=='left' else -.008,prime_s=4.8)
    labels=[];root_residuals=[]
    for k in range(len(s['time'])):
        restore(999+k*5)
        label,diag=teacher.action(core,k*.005)
        labels.append(label);root_residuals.append(diag['root_balance_residual_norm_over_W'])
    labels=np.asarray(labels);root_residuals=np.asarray(root_residuals)
    report=dict(condition=condition,all_teacher_labels_bitexact=bool(np.array_equal(labels,s['teacher_action'])),
                max_label_error=float(np.max(np.abs(labels-s['teacher_action']))),
                root_balance_residuals_bitexact=bool(np.array_equal(root_residuals,s['root_balance_residual'])),
                root_balance_max=float(root_residuals.max()),root_balance_above_point02_count=int((root_residuals>.02).sum()),
                label_rows=len(labels),physical_integration_steps=0,optimizer_updates=0,
                scope='private qpos/qvel/target/ctrl replay; same1920ordered queries recreate private teacher IK history; no new physical trajectory',
                source_hashes={str(p):digest(p) for p in (HERE/'replay_labels.py',TEACHER,run/'SAMPLES_5MS.npz',run/'trace_1ms.npz')})
    verify(ident);verify_runtime(ident,core)
    with (run/'LABEL_REPLAY.json').open('x') as f:json.dump(jsonable(report),f,indent=2);f.write('\n')
    print(json.dumps(report))
    if not report['all_teacher_labels_bitexact']:raise ValueError('reconstructed teacher labels differ; do not infer exact teacher replay')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',choices=['left','right'],required=True);a=p.parse_args();main(a.condition)
