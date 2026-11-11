"""Large-turn preregistered physical teacher; no ANN training or old output writes."""
from pathlib import Path
import json, time, sys, os, hashlib
import numpy as np
from turning_teacher import pin_baseline, make_teacher, BASELINE
pin_baseline()
import physics, static_teacher, student_wave as sw, models, wave_reference, identity
from cycle_reward import CycleTracker, CycleConfig
HERE = Path(__file__).resolve().parent

def dump(path,obj): identity.atomic_json(path,obj)
def yaw(c):
    r=c.d.xmat[c.th].reshape(3,3)
    return float(np.arctan2(r[1,0],r[0,0]))
def initial(c,path):
    np.savez_compressed(path,qpos=c.d.qpos.copy(),qvel=c.d.qvel.copy(),target=c.actuator.target_rad.copy(),buffer=c.buf.copy(),ctrl=c.d.ctrl.copy(),xfrc=c.d.xfrc_applied.copy(),qfrc=c.d.qfrc_applied.copy(),initial_heading=c.initial_heading.copy())

class PeakRecorder:
    """Native solved nonfoot peaks and state minima; no stored native arrays."""
    def __init__(self):
        self.count=0; self.post_count=0; self.peak=0.; self.post_peak=0.; self.com_min=float('inf'); self.extra_max=0.
    def before(self,c):
        self.extra_max=max(self.extra_max,float(np.abs(c.d.xfrc_applied).max()),float(np.abs(c.d.qfrc_applied).max()))
        assert not c.adhesion_on and self.extra_max==0
        return float(c.d.time)
    def solved(self,c,before):
        import mujoco as mj
        nonfoot=0.
        for ci in range(c.d.ncon):
            contact=c.d.contact[ci]; ga,gb=int(contact.geom1),int(contact.geom2)
            if not (c.planes[ga] or c.planes[gb]): continue
            geom=gb if c.planes[ga] else ga
            if int(c.geom_leg[geom])>=0: continue
            f=np.zeros(6); mj.mj_contactForce(c.plant,c.d,ci,f)
            world=contact.frame.reshape(3,3).T@f[:3]
            nonfoot+=max(0.,float(world[2])*(1 if c.planes[ga] else -1))
        return nonfoot
    def after(self,c,before,solved):
        self.count+=1; self.peak=max(self.peak,solved)
        if before>=5.8-1e-8:
            self.post_count+=1; self.post_peak=max(self.post_peak,solved)
            self.com_min=min(self.com_min,float(c.d.subtree_com[c.th][2]))
    def close(self):
        return dict(actual_native_steps=self.count,post_native_steps=self.post_count,nonfoot_peak_native=self.peak,post_nonfoot_peak_native=self.post_peak,post_COM_min=self.com_min,extra_force_max=self.extra_max,full_arrays_retained=False,scope='native peak summary, not full D1-D8')

def one(cfg):
    out=HERE/cfg['name']; out.mkdir(exist_ok=False)
    native=PeakRecorder()
    c=physics.Core(adhesion=False,recorder=native)
    manifest=identity.capture(c,dict(condition=cfg,prereg=str(HERE/'PREREG.json')),extra_paths=[HERE/'PREREG.json',HERE/'diagnose.py',HERE/'turning_teacher.py'])
    dump(out/'IDENTITY.json',manifest)
    initial(c,out/'INITIAL_RUNTIME.npz')
    tr=sw.Trace(full=True); sw.settle(c,tr)
    teacher=make_teacher(c,speed_scale=cfg['speed_scale'],yaw_rate=cfg['yaw_rate'])
    previous=np.zeros(42); xs=[]; ys=[]; commands=[]; phases=[]; source_times=[]; yaws=[]
    observer=CycleTracker(CycleConfig(timeout=4.)).reset(start_time=5.8)
    events=[]; start_com=start_yaw=None; stop=None; unsafe_ms=0; begun=time.monotonic()
    for decision in range(2880):
        t=decision*.005; post=decision>=960
        if decision==960:
            start_com=c.d.subtree_com[c.th].copy(); start_yaw=yaw(c)
        x=c.obs238(previous,target_speed=.075)
        action,diagnostic=teacher.action(c,t); action=sw.bounded_action(action)
        if x.shape!=(238,) or not np.isfinite(x).all(): raise ValueError('invalid observation')
        xs.append(x.copy()); ys.append(action.copy())
        phase,scale,curvature=teacher.reference.command(t)
        commands.append([.075*scale,.075*scale*curvature]); phases.append(phase); source_times.append(float(c.d.time))
        for _ in range(5):
            measured=tr.tick(c,action,sw.SOURCE['teacher_replay'] if post else sw.SOURCE['teacher'])
            yaws.append(yaw(c))
            if post: events.extend(observer.update(dict(time=float(c.d.time),loads=measured['loads'],padz=measured['padz'],foot=measured['foot']))['events'])
            unsafe_ms=unsafe_ms+1 if measured['com']<.8 or measured['nonfoot']>.5*physics.W else 0
            if unsafe_ms>=100:
                stop=dict(time_s=float(c.d.time),reason='100ms COM<.8 or nonfoot>.5W'); break
        previous=action.copy()
        if decision%160==0: print(json.dumps(dict(condition=cfg['name'],decision=decision,physical_s=float(c.d.time),wall_s=time.monotonic()-begun)),flush=True)
        if stop: break
    native_info=native.close(); dump(out/'NATIVE_MANIFEST.json',native_info)
    trace_info=tr.save(out/'TRACE_1MS.npz')
    physical=np.asarray(commands)
    samples=out/'SAMPLES_5MS.npz'
    np.savez_compressed(samples,X238=np.asarray(xs),Y42=np.asarray(ys),command2=np.column_stack([physical[:,0]/.075-1,physical[:,1]/.06]),command_speed_yaw=physical,source_phase=np.asarray(phases),actual_time=np.asarray(source_times),post_prime=np.arange(len(xs))>=960)
    np.savez_compressed(out/'YAW_1MS.npz',time=np.asarray([r[0] for r in tr.rec[1000:]]),yaw_world=np.asarray(yaws),boundary_yaw=np.asarray(start_yaw),boundary_com=np.asarray(start_com))
    report=tr.summary(c,sw.SOURCE['teacher_replay']); cycles=sw.cycle_report(observer,events)
    yy=np.unwrap(np.r_[start_yaw,np.asarray(yaws)[4800:]])
    actual=float(yy[-1]-yy[0]); duration=float(c.d.time)-5.8
    postrows=tr.rows[5800:]; com=np.asarray([r['com'] for r in postrows]); heading=np.asarray(yaws)[4800:]
    delta=np.diff(np.vstack([start_com,com]),axis=0)[:,:2]
    body_projection=float(np.sum(delta*np.column_stack([np.cos(np.r_[start_yaw,heading[:-1]]),np.sin(np.r_[start_yaw,heading[:-1]])]))/duration)
    identity.verify_runtime(manifest,c)
    report.update(condition=cfg,early_stop=stop,cycles=cycles,algorithm_teacher_only=True,actual_signed_yaw_rad=actual,actual_signed_yaw_deg=float(np.degrees(actual)),actual_average_yaw_rate=actual/duration,requested_sign_agreement=bool(actual*cfg['yaw_rate']>0),actual_com_displacement=(c.d.subtree_com[c.th]-start_com).tolist(),body_heading_projected_speed=body_projection,original_all5_gates=report['all_gates'],strict_nonfoot_zero_1ms=bool(report['nonfoot_force_max_native']==0),trace=trace_info,samples=dict(path=str(samples),sha256=identity.sha256(samples),decisions=len(xs),post_prime_decisions=max(0,len(xs)-960)),native=native_info,source_runtime_unchanged=True,scope='physical algorithm teacher, no ANN; 1ms engineering metrics plus native audit pending; no full D1-D8/biology certification',wall_s=time.monotonic()-begun)
    dump(out/'RESULT.json',report)
    print(json.dumps(dict(finished=cfg['name'],yaw_deg=report['actual_signed_yaw_deg'],gates=report['gates'],slip=report['slip_ratio_max'],cycles=cycles['complete_per_leg'],early_stop=stop)),flush=True)
    return {k:v for k,v in report.items() if k!='native'}

def main():
    allowed=os.sched_getaffinity(0)
    cpu=1 if 1 in allowed else min(allowed)
    os.sched_setaffinity(0,{cpu})
    prereg=json.loads((HERE/'PREREG.json').read_text())
    sw.seed_all(prereg['seed'])
    dump(HERE/'RUNTIME.json',dict(pid=os.getpid(),cpu_affinity=list(os.sched_getaffinity(0)),python=sys.executable,environment={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    reports=[one(cfg) for cfg in prereg['conditions']]
    dump(HERE/'SWEEP_RESULT.json',dict(conditions=reports,completed_fixed_conditions=len(reports),all_jobs_finished=True))
if __name__=='__main__': main()
