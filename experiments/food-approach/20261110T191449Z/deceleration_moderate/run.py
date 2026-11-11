"""Preregistered matched control + exactly two moderate deceleration candidates."""
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
from pathlib import Path
import json,time,sys,traceback
import numpy as np
from api import HERE,BASELINE,TEACHER,REFERENCE_REPORT,REFERENCE_FORWARD,CONDITIONS,make_teacher,command2,functional_gates
import physics,student_wave as sw,static_teacher,wave_reference,models,identity
from cycle_reward import CycleTracker,CycleConfig

def sha(p):return identity.sha256(p)
def dump(p,v):identity.atomic_json(p,v)
def yaw(c):
    r=c.d.xmat[c.th].reshape(3,3);return float(np.arctan2(r[1,0],r[0,0]))

def one(cfg,control=None):
    out=HERE/cfg['name'];out.mkdir(exist_ok=False);c=physics.Core(adhesion=False)
    manifest=identity.capture(c,cfg,extra_paths=[HERE/x for x in ('PREREG.json','AMENDMENT_BEFORE_EXECUTION.json','api.py','run.py','verify.py','check_reference.py')]+[TEACHER,REFERENCE_REPORT])
    dump(out/'IDENTITY.json',manifest);tr=sw.Trace(full=True)
    xs=[];ys=[];commands=[];phys=[];phases=[];times=[];yaws=[];previous=np.zeros(42)
    observer=CycleTracker(CycleConfig(timeout=4.)).reset(start_time=5.8);events=[]
    start_com=start_yaw=None;stop=exception=None;unsafe=0;begun=time.monotonic()
    try:
        sw.settle(c,tr);teacher=make_teacher(c,cfg['name'])
        for decision in range(2880):
            elapsed=decision*.005;post=decision>=960
            if decision==960:start_com=c.d.subtree_com[c.th].copy();start_yaw=yaw(c)
            x=c.obs238(previous,target_speed=.075);action,_=teacher.action(c,elapsed)
            if not np.isfinite(x).all():raise ValueError('nonfiniteobs')
            cmd,p,phase=command2(teacher.reference,elapsed)
            xs.append(x.copy());ys.append(action.copy());commands.append(cmd);phys.append(p);phases.append(phase);times.append(float(c.d.time))
            for _ in range(5):
                m=tr.tick(c,action,sw.SOURCE['teacher_replay'] if post else sw.SOURCE['teacher']);yaws.append(yaw(c))
                if post:events.extend(observer.update(dict(time=float(c.d.time),loads=m['loads'],padz=m['padz'],foot=m['foot']))['events'])
                unsafe=unsafe+1 if m['com']<.8 or m['nonfoot']>.5*physics.W else 0
                if unsafe>=100:stop=dict(time_s=float(c.d.time),reason='100ms COM<.8 or nonfoot>.5W');break
            previous=action.copy()
            if stop:break
    except Exception as err:exception=dict(type=type(err).__name__,message=str(err),traceback=traceback.format_exc(),actual_time=float(c.d.time))
    finally:
        trace=tr.save(out/'TRACE_1MS.npz') if tr.rows else None
        np.savez_compressed(out/'SAMPLES_5MS.npz',X238=np.asarray(xs).reshape(-1,238),Y42=np.asarray(ys).reshape(-1,42),command2=np.asarray(commands).reshape(-1,2),
            command_speed_yaw=np.asarray(phys).reshape(-1,2),source_phase=np.asarray(phases),actual_time=np.asarray(times),post_prime=np.arange(len(xs))>=960)
        np.savez_compressed(out/'YAW_1MS.npz',time=np.asarray([r[0] for r in tr.rec[1000:]]),yaw_world=np.asarray(yaws))
    r=tr.summary(c,sw.SOURCE['teacher_replay']);cycles=sw.cycle_report(observer,events);valid=r.get('sample_count',0)>=2
    cf=None if control is None else control.get('forward_signed_native_per_s')
    fg=functional_gates(r,cycles,cf,True if control is None else control['all_functional_gates']) if valid else dict(incomplete=False)
    dy=None if start_yaw is None or len(yaws)<=4800 else float(np.unwrap(np.r_[start_yaw,np.asarray(yaws)[4800:]])[-1]-start_yaw)
    r.update(condition=cfg,is_matched_control=control is None,early_stop=stop,exception=exception,cycles=cycles,
        original_all5_gates=r.get('all_gates',False),functional_gates=fg,all_functional_gates=bool(valid and all(fg.values())),
        matched_control_forward=cf,deceleration_maximum_forward=None if cf is None else .95*cf,
        achieved_forward_fraction_of_control=None if cf is None or not valid else r['forward_signed_native_per_s']/cf,
        actual_signed_yaw_rad=dy,actual_com_displacement=None if start_com is None else (c.d.subtree_com[c.th]-start_com).tolist(),trace=trace,
        samples=dict(path=str(out/'SAMPLES_5MS.npz'),sha256=sha(out/'SAMPLES_5MS.npz'),decisions=len(xs),post_prime_decisions=max(0,len(xs)-960)),
        artifacts={str(p):sha(p) for p in (out/'TRACE_1MS.npz',out/'SAMPLES_5MS.npz',out/'YAW_1MS.npz',out/'IDENTITY.json') if p.exists()},
        physical=dict(native_dt_s=float(c.plant.opt.timestep),actual_native_steps=len(tr.rec)*80,nq=c.plant.nq,nv=c.plant.nv,actuators=c.plant.nu),
        scope='Algorithmteacher only, matchedcontrolfunctionalmoderatedeceleration; no training/food/neuralautonomy/fullnativeD1D8',wall_s=time.monotonic()-begun)
    identity.verify(manifest);dump(out/'RESULT.json',r)
    print(json.dumps(dict(name=cfg['name'],forward=r.get('forward_signed_native_per_s'),slip=r.get('slip_ratio_max'),gates=fg,cycles=cycles['complete_per_leg'],
        fraction=r['achieved_forward_fraction_of_control'],exception=exception,early_stop=stop,wall_s=r['wall_s'])),flush=True)
    return r

def main():
    amendment=json.loads((HERE/'AMENDMENT_BEFORE_EXECUTION.json').read_text())
    if amendment['conditions']!=list(CONDITIONS):raise ValueError('amended3casecontractmismatch')
    modules={m.__name__:str(Path(m.__file__).resolve()) for m in (physics,sw,static_teacher,wave_reference,models,identity)}
    if any(Path(p).parent!=BASELINE for p in modules.values()):raise ValueError('actualbaselineimportmismatch')
    historical=json.loads(REFERENCE_REPORT.read_text())
    if historical['forward_signed_native_per_s']!=REFERENCE_FORWARD:raise ValueError('historicallineagechanged')
    paths=list(BASELINE.glob('*.py'))+list((BASELINE/'sources').glob('*.py'))+[REFERENCE_REPORT]
    for parent in (HERE.parent/'teacher',HERE.parent/'deceleration',HERE.parent/'deceleration_stride'):paths.extend(p for p in parent.rglob('*') if p.is_file())
    paths.extend(HERE/x for x in ('PREREG.json','AMENDMENT_BEFORE_EXECUTION.json','api.py','run.py','verify.py','check_reference.py','REFERENCE_CHECKS.json'))
    hashes={str(p.resolve()):sha(p) for p in paths}
    dump(HERE/'INPUT_IDENTITY.json',dict(source_hashes=hashes,resolved_modules=modules,teacher_explicit_path=str(TEACHER),python=sys.executable,
        cpu_affinity=sorted(os.sched_getaffinity(0)),checkpoint_loaded=False,training=False,reference_is_matched_control=True))
    checks=json.loads((HERE/'REFERENCE_CHECKS.json').read_text())
    if not checks['all_passed'] or not all(checks['checks'].values()):raise ValueError('referencechecksfailed')
    sw.seed_all(20261110)
    control=one(CONDITIONS[0]);reports=[control]+[one(cfg,control) for cfg in CONDITIONS[1:]]
    drift=[p for p,h in hashes.items() if sha(p)!=h]
    if drift:raise RuntimeError('input/evidencedrift '+str(drift))
    dump(HERE/'SWEEP_RESULT.json',dict(source_identity_unchanged=True,completed_fixed_conditions=3,matched_controls=1,fixed_candidates=2,all_jobs_finished=True,conditions=reports))
if __name__=='__main__':main()
