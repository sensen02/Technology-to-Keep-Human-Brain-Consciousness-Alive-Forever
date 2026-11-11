"""Bounded preregistered real-Core diagnostic; no training or food field."""
from pathlib import Path
import json
import hashlib
import time
import sys
import numpy as np
from turning_teacher import pin_baseline, make_teacher, BASELINE
pin_baseline()
import physics
import static_teacher
import student_wave as sw
import models
import wave_reference
from cycle_reward import CycleTracker, CycleConfig

HERE = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')

def yaw(c):
    r = c.d.xmat[c.th].reshape(3, 3)
    return float(np.arctan2(r[1, 0], r[0, 0]))

def one(cfg):
    out = HERE/cfg['name']
    out.mkdir(exist_ok=False)
    c = physics.Core(adhesion=False)
    tr = sw.Trace(full=True)
    sw.settle(c, tr)
    teacher = make_teacher(c, speed_scale=cfg['speed_scale'], yaw_rate=cfg['yaw_rate'])
    previous = np.zeros(42)
    xs, ys, commands, phases, source_times = [], [], [], [], []
    yaw_series = []
    events = []
    observer = CycleTracker(CycleConfig(timeout=4.)).reset(start_time=5.8)
    start_com = start_yaw = None
    stop = None
    unsafe_ms = 0
    begun = time.monotonic()
    for decision in range(2880):
        t = decision*.005
        post = decision >= 960
        if decision == 960:
            start_com = c.d.subtree_com[c.th].copy()
            start_yaw = yaw(c)
        x = c.obs238(previous, target_speed=.075)  # baseline actor observation identity preserved
        action, diagnostic = teacher.action(c, t)
        xs.append(x.copy()); ys.append(action.copy())
        phase, scale, curvature = teacher.reference.command(t)
        commands.append([.075*scale, .075*scale*curvature])
        phases.append(phase); source_times.append(float(c.d.time))
        for _ in range(5):
            measured = tr.tick(c, action, sw.SOURCE['teacher'] if not post else sw.SOURCE['teacher_replay'])
            yaw_series.append(yaw(c))
            if post:
                # Observer is read-only; use exact baseline completion semantics.
                events.extend(observer.update(dict(time=float(c.d.time), loads=measured['loads'], padz=measured['padz'], foot=measured['foot']))['events'])
            unsafe_ms = unsafe_ms+1 if measured['com'] < .8 or measured['nonfoot'] > .5*physics.W else 0
            if unsafe_ms >= 100:
                stop = dict(time_s=float(c.d.time), reason='100ms COM<.8 or nonfoot>.5W')
                break
        previous = action.copy()
        if stop:
            break
    trace_path = out/'TRACE_1MS.npz'
    trace_info = tr.save(trace_path)
    sample_path = out/'SAMPLES_5MS.npz'
    physical_commands = np.asarray(commands)
    command2 = np.column_stack([physical_commands[:, 0]/.075-1., physical_commands[:, 1]/.008])
    np.savez_compressed(sample_path, X238=np.asarray(xs), Y42=np.asarray(ys), command2=command2, command_speed_yaw=physical_commands,
                        source_phase=np.asarray(phases), actual_time=np.asarray(source_times),
                        post_prime=np.arange(len(xs))>=960)
    np.savez_compressed(out/'YAW_1MS.npz', time=np.asarray([r[0] for r in tr.rec[1000:]]), yaw_world=np.asarray(yaw_series))
    report = tr.summary(c, sw.SOURCE['teacher_replay'])
    cycles = sw.cycle_report(observer, events)
    y = np.unwrap(np.r_[start_yaw, np.asarray(yaw_series)[4800:]])
    actual_yaw = float(y[-1]-y[0])
    duration = float(c.d.time)-5.8
    report.update(condition=cfg, early_stop=stop, cycles=cycles, algorithm_teacher_only=True,
        actual_signed_yaw_rad=actual_yaw, actual_average_yaw_rate=actual_yaw/duration,
        requested_sign_agreement=bool(actual_yaw*cfg['yaw_rate']>0 and abs(actual_yaw)>.001),
        actual_com_displacement=(c.d.subtree_com[c.th]-start_com).tolist(),
        original_all5_gates=report['all_gates'], strict_nonfoot_zero=bool(report['nonfoot_force_max_native']==0),
        trace=trace_info, samples=dict(path=str(sample_path),sha256=digest(sample_path), decisions=len(xs),post_prime_decisions=max(0,len(xs)-960)),
        extra_forces='checked zero before and after all actual1ms ticks; adhesion=False',
        physical=dict(native_dt_s=float(c.plant.opt.timestep), actual_native_steps=len(tr.rec)*80,
                      nq=c.plant.nq,nv=c.plant.nv,actuators=c.plant.nu),
        scope='1ms-endpoint engineering diagnostic of native12.5us real physics; NOT full native D1-D8 audit, neural autonomy or biology',
        wall_s=time.monotonic()-begun)
    dump(out/'RESULT.json', report)
    print(json.dumps(dict(name=cfg['name'], yaw=actual_yaw, displacement=report['actual_com_displacement'],
                          slip=report['slip_ratio_max'],gates=report['gates'],early_stop=stop)),flush=True)
    return report


def main():
    prereg = json.loads((HERE/'PREREG.json').read_text())
    modules = {m.__name__:str(Path(m.__file__).resolve()) for m in (physics,static_teacher,sw,models,wave_reference)}
    for name, path in modules.items():
        if Path(path).parent != BASELINE:
            raise RuntimeError('Wrong lineage: '+name+' '+path)
    sources = [HERE/'PREREG.json',HERE/'turning_teacher.py',HERE/'diagnose.py']
    sources += list(BASELINE.glob('*.py')) + list((BASELINE/'sources').glob('*.py'))
    sources += [physics.PAD_XML, physics.POSE_FILE,wave_reference.DEFAULT_REFERENCE]
    hashes = {str(Path(p).resolve()):digest(p) for p in sources}
    dump(HERE/'IDENTITY.json',dict(resolved_modules=modules,source_hashes=hashes,python=sys.executable,
         verified_checkpoint_not_loaded='/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/BC_GRU_ROLE_3000/checkpoint_002500.pt'))
    sw.seed_all(20261110)
    reports = [one(cfg) for cfg in prereg['conditions']]
    drift = [p for p,h in hashes.items() if digest(p)!=h]
    if drift:
        raise RuntimeError('Input drift: '+str(drift))
    dump(HERE/'SWEEP_RESULT.json',dict(source_identity_unchanged=True,conditions=reports,
         completed_fixed_conditions=len(reports),all_jobs_finished=True))

if __name__=='__main__':
    main()
