"""Exactly two preregistered teacher trajectories; write only this new directory."""
from pathlib import Path
import os
# One allowed actual CPU, in addition to numerical library thread limits.
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
import sys
import json
import time
import hashlib
import numpy as np
from api import HERE, BASELINE, TEACHER, CONDITIONS, make_teacher, command2, halfspeed_gates
import physics
import student_wave as sw
import static_teacher
import wave_reference
import models
import identity
from cycle_reward import CycleTracker, CycleConfig

def sha(p): return identity.sha256(p)
def dump(p, value): identity.atomic_json(p, value)
def yaw(c):
    r=c.d.xmat[c.th].reshape(3,3)
    return float(np.arctan2(r[1,0],r[0,0]))

def one(cfg):
    out=HERE/cfg['name']; out.mkdir(exist_ok=False)
    c=physics.Core(adhesion=False)
    manifest=identity.capture(c, cfg, extra_paths=[HERE/'PREREG.json', HERE/'api.py', HERE/'run.py', HERE/'verify.py', TEACHER])
    dump(out/'IDENTITY.json', manifest)
    tr=sw.Trace(full=True)
    sw.settle(c,tr)
    teacher=make_teacher(c,cfg['name'])
    previous=np.zeros(42)
    xs=[];ys=[];cmds=[];phys=[];phases=[];times=[];yaws=[]
    events=[]; observer=CycleTracker(CycleConfig(timeout=4.)).reset(start_time=5.8)
    start_com=None;start_yaw=None;stop=None;unsafe_ms=0
    begun=time.monotonic()
    for decision in range(4800):
        elapsed=decision*.005
        post=decision>=960
        if decision==960:
            start_com=c.d.subtree_com[c.th].copy(); start_yaw=yaw(c)
        x=c.obs238(previous,target_speed=.075)
        action,_=teacher.action(c,elapsed)
        normalized,physical,phase=command2(teacher.reference,elapsed)
        if not np.isfinite(x).all(): raise ValueError('nonfinite observation')
        xs.append(x.copy());ys.append(action.copy());cmds.append(normalized);phys.append(physical)
        phases.append(phase);times.append(float(c.d.time))
        for _ in range(5):
            m=tr.tick(c,action,sw.SOURCE['teacher_replay'] if post else sw.SOURCE['teacher'])
            yaws.append(yaw(c))
            if post:
                events.extend(observer.update(dict(time=float(c.d.time),loads=m['loads'],padz=m['padz'],foot=m['foot']))['events'])
            unsafe_ms=unsafe_ms+1 if m['com']<.8 or m['nonfoot']>.5*physics.W else 0
            if unsafe_ms>=100:
                stop=dict(time_s=float(c.d.time),reason='100ms COM<.8 or nonfoot>.5W');break
        previous=action.copy()
        if stop: break
    trace=tr.save(out/'TRACE_1MS.npz')
    np.savez_compressed(out/'SAMPLES_5MS.npz',X238=np.asarray(xs),Y42=np.asarray(ys),command2=np.asarray(cmds),
        command_speed_yaw=np.asarray(phys),source_phase=np.asarray(phases),actual_time=np.asarray(times),post_prime=np.arange(len(xs))>=960)
    np.savez_compressed(out/'YAW_1MS.npz',time=np.asarray([r[0] for r in tr.rec[1000:]]),yaw_world=np.asarray(yaws))
    report=tr.summary(c,sw.SOURCE['teacher_replay'])
    cycles=sw.cycle_report(observer,events)
    hg=halfspeed_gates(report,cycles)
    actual_yaw=float(np.unwrap(np.r_[start_yaw,np.asarray(yaws)[4800:]])[-1]-start_yaw)
    report.update(condition=cfg,early_stop=stop,cycles=cycles,original_all5_gates=report['all_gates'],
        halfspeed_gates=hg,all_halfspeed_gates=bool(all(hg.values())),strict_nonfoot_zero=bool(report['nonfoot_force_max_native']==0),
        actual_signed_yaw_rad=actual_yaw,actual_average_yaw_rate=actual_yaw/(float(c.d.time)-5.8),
        requested_sign_agreement=None if cfg['yaw_rate']==0 else bool(actual_yaw*cfg['yaw_rate']>0 and abs(actual_yaw)>.001),
        actual_com_displacement=(c.d.subtree_com[c.th]-start_com).tolist(),trace=trace,
        samples=dict(path=str(out/'SAMPLES_5MS.npz'),sha256=sha(out/'SAMPLES_5MS.npz'),decisions=len(xs),neutral_warmup=960,post_prime_decisions=len(xs)-960),
        artifacts={str(p):sha(p) for p in (out/'TRACE_1MS.npz',out/'SAMPLES_5MS.npz',out/'YAW_1MS.npz',out/'IDENTITY.json')},
        extra_forces='adhesion=False; applied body/generalized force checked zero before and after every actual1ms tick',
        physical=dict(native_dt_s=float(c.plant.opt.timestep),actual_native_steps=len(tr.rec)*80,nq=c.plant.nq,nv=c.plant.nv,actuators=c.plant.nu),
        scope='Algorithm teacher only; 1ms endpoints over real12.5us native physics; no neural autonomy, food behavior, training or native D1-D8 certification',
        wall_s=time.monotonic()-begun)
    identity.verify(manifest)
    dump(out/'RESULT.json',report)
    print(json.dumps(dict(name=cfg['name'],forward=report['forward_signed_native_per_s'],slip=report['slip_ratio_max'],original=report['gates'],halfspeed=hg,cycles=cycles['complete_per_leg'],yaw=actual_yaw,early_stop=stop,wall_s=report['wall_s'])),flush=True)
    return report

def main():
    prereg=json.loads((HERE/'PREREG.json').read_text())
    if prereg['conditions']!=list(CONDITIONS): raise ValueError('fixed two-case preregistration mismatch')
    modules={m.__name__:str(Path(m.__file__).resolve()) for m in (physics,sw,static_teacher,wave_reference,models,identity)}
    if any(Path(p).parent!=BASELINE for p in modules.values()): raise ValueError('wrong actual baseline imports')
    # Freeze all previous teacher evidence and all baseline source files, not merely imported filenames.
    frozen=list(TEACHER.parent.glob('*'))+list(BASELINE.glob('*.py'))+list((BASELINE/'sources').glob('*.py'))
    frozen=[p for p in frozen if p.is_file()]
    for folder in TEACHER.parent.iterdir():
        if folder.is_dir(): frozen.extend(p for p in folder.glob('*') if p.is_file())
    hashes={str(p.resolve()):sha(p) for p in frozen}
    hashes.update({str(p):sha(p) for p in (HERE/'PREREG.json',HERE/'api.py',HERE/'run.py',HERE/'verify.py')})
    dump(HERE/'INPUT_IDENTITY.json',dict(source_hashes=hashes,resolved_modules=modules,teacher_explicit_path=str(TEACHER),
        python=sys.executable,cpu_affinity=sorted(os.sched_getaffinity(0)),
        command='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 provided-python -B '+str(HERE/'run.py'),
        checkpoint_loaded=False,training=False))
    sw.seed_all(prereg['seed'])
    reports=[one(cfg) for cfg in prereg['conditions']]
    drift=[p for p,h in hashes.items() if sha(p)!=h]
    if drift: raise RuntimeError('input/evidence drift: '+str(drift))
    dump(HERE/'SWEEP_RESULT.json',dict(source_identity_unchanged=True,completed_fixed_conditions=len(reports),all_jobs_finished=True,conditions=reports))

if __name__=='__main__': main()
