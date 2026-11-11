"""Exactly two preregistered fixed-cadence shortened-stride actual trajectories."""
from pathlib import Path
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import sys,json,time,traceback
import numpy as np
from api import HERE,BASELINE,TEACHER,CONDITIONS,make_teacher,command2,stride_gates
import physics
import student_wave as sw
import static_teacher,wave_reference,models,identity
from cycle_reward import CycleTracker,CycleConfig

def dump(p,o):identity.atomic_json(p,o)
def sha(p):return identity.sha256(p)
def yaw(c):
    r=c.d.xmat[c.th].reshape(3,3);return float(np.arctan2(r[1,0],r[0,0]))

def one(cfg):
    out=HERE/cfg['name'];out.mkdir(exist_ok=False)
    c=physics.Core(adhesion=False)
    manifest=identity.capture(c,cfg,extra_paths=[HERE/'PREREG.json',HERE/'api.py',HERE/'run.py',HERE/'verify.py',HERE/'check_reference.py',TEACHER])
    dump(out/'IDENTITY.json',manifest)
    tr=sw.Trace(full=True);xs=[];ys=[];commands=[];physical=[];phases=[];times=[];yaws=[]
    previous=np.zeros(42);observer=CycleTracker(CycleConfig(timeout=4.)).reset(start_time=5.8);events=[]
    start_com=None;start_yaw=None;stop=None;exception=None;unsafe=0;begun=time.monotonic()
    try:
        sw.settle(c,tr)
        teacher=make_teacher(c,cfg['name'])
        for decision in range(2880):
            elapsed=decision*.005;post=decision>=960
            if decision==960:start_com=c.d.subtree_com[c.th].copy();start_yaw=yaw(c)
            x=c.obs238(previous,target_speed=.075);action,_=teacher.action(c,elapsed)
            if not np.isfinite(x).all():raise ValueError('nonfinite observation')
            cmd,phys,phase=command2(teacher.reference,elapsed)
            xs.append(x.copy());ys.append(action.copy());commands.append(cmd);physical.append(phys);phases.append(phase);times.append(float(c.d.time))
            for _ in range(5):
                m=tr.tick(c,action,sw.SOURCE['teacher_replay'] if post else sw.SOURCE['teacher'])
                yaws.append(yaw(c))
                if post:events.extend(observer.update(dict(time=float(c.d.time),loads=m['loads'],padz=m['padz'],foot=m['foot']))['events'])
                unsafe=unsafe+1 if m['com']<.8 or m['nonfoot']>.5*physics.W else 0
                if unsafe>=100:stop=dict(time_s=float(c.d.time),reason='100ms COM<.8 or nonfoot>.5W');break
            previous=action.copy()
            if stop:break
    except Exception as err:
        exception=dict(type=type(err).__name__,message=str(err),traceback=traceback.format_exc(),actual_time=float(c.d.time))
        print(json.dumps(dict(case=cfg['name'],exception=exception)),flush=True)
    finally:
        trace=tr.save(out/'TRACE_1MS.npz') if tr.rows else None
        np.savez_compressed(out/'SAMPLES_5MS.npz',X238=np.asarray(xs).reshape(-1,238),Y42=np.asarray(ys).reshape(-1,42),
            command2=np.asarray(commands).reshape(-1,2),command_speed_yaw=np.asarray(physical).reshape(-1,2),
            source_phase=np.asarray(phases),actual_time=np.asarray(times),post_prime=np.arange(len(xs))>=960)
        np.savez_compressed(out/'YAW_1MS.npz',time=np.asarray([r[0] for r in tr.rec[1000:]]),yaw_world=np.asarray(yaws))
    cycles=sw.cycle_report(observer,events)
    report=tr.summary(c,sw.SOURCE['teacher_replay'])
    valid=report.get('sample_count',0)>=2
    gates=stride_gates(report,cycles,cfg['forward_threshold']) if valid else dict(incomplete=False)
    actual_yaw=None if start_yaw is None or len(yaws)<=4800 else float(np.unwrap(np.r_[start_yaw,np.asarray(yaws)[4800:]])[-1]-start_yaw)
    report.update(condition=cfg,early_stop=stop,exception=exception,cycles=cycles,original_all5_gates=report.get('all_gates',False),
        stride_gates=gates,all_stride_gates=bool(valid and all(gates.values())),actual_signed_yaw_rad=actual_yaw,
        actual_com_displacement=None if start_com is None else (c.d.subtree_com[c.th]-start_com).tolist(),trace=trace,
        samples=dict(path=str(out/'SAMPLES_5MS.npz'),sha256=sha(out/'SAMPLES_5MS.npz'),decisions=len(xs),post_prime_decisions=max(0,len(xs)-960)),
        artifacts={str(p):sha(p) for p in (out/'TRACE_1MS.npz',out/'SAMPLES_5MS.npz',out/'YAW_1MS.npz',out/'IDENTITY.json') if p.exists()},
        physical=dict(native_dt_s=float(c.plant.opt.timestep),actual_native_steps=len(tr.rec)*80,nq=c.plant.nq,nv=c.plant.nv,actuators=c.plant.nu),
        extra_forces='adhesion=False; zero body/generalized applied force asserted before/after every1ms tick',
        scope='Algorithm teacher fixed-cadence stride shortening only; 1ms endpoints/native12.5us physics; no training/food/neural autonomy/fullnativeD1-D8',wall_s=time.monotonic()-begun)
    identity.verify(manifest);dump(out/'RESULT.json',report)
    print(json.dumps(dict(name=cfg['name'],forward=report.get('forward_signed_native_per_s'),slip=report.get('slip_ratio_max'),
        original=report.get('gates'),stride_gates=gates,cycles=cycles['complete_per_leg'],exception=exception,early_stop=stop,wall_s=report['wall_s'])),flush=True)
    return report

def main():
    prereg=json.loads((HERE/'PREREG.json').read_text())
    if prereg['conditions']!=list(CONDITIONS):raise ValueError('preregistered cases mismatch')
    modules={m.__name__:str(Path(m.__file__).resolve()) for m in (physics,sw,static_teacher,wave_reference,models,identity)}
    if any(Path(p).parent!=BASELINE for p in modules.values()):raise ValueError('actual baseline import mismatch')
    frozen=list(BASELINE.glob('*.py'))+list((BASELINE/'sources').glob('*.py'))
    for parent in (HERE.parent/'teacher',HERE.parent/'deceleration'):
        frozen.extend(p for p in parent.rglob('*') if p.is_file())
    frozen.extend(HERE/x for x in ('PREREG.json','api.py','run.py','verify.py','check_reference.py'))
    hashes={str(p.resolve()):sha(p) for p in frozen}
    dump(HERE/'INPUT_IDENTITY.json',dict(source_hashes=hashes,resolved_modules=modules,teacher_explicit_path=str(TEACHER),
        python=sys.executable,cpu_affinity=sorted(os.sched_getaffinity(0)),checkpoint_loaded=False,training=False))
    reference=json.loads((HERE/'REFERENCE_CHECKS.json').read_text())
    if not reference['all_passed']:raise ValueError('reference contract failed')
    sw.seed_all(prereg['seed'])
    reports=[one(cfg) for cfg in prereg['conditions']]
    drift=[p for p,h in hashes.items() if sha(p)!=h]
    if drift:raise RuntimeError('input/evidence drift '+str(drift))
    dump(HERE/'SWEEP_RESULT.json',dict(source_identity_unchanged=True,completed_fixed_conditions=2,all_jobs_finished=True,conditions=reports))
if __name__=='__main__':main()
