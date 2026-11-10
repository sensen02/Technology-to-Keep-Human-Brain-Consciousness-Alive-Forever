"""Bounded target-space controller experiment, same physical plant."""
from dataclasses import asdict
import json
import time
import numpy as np
from physics import Core,LEGS,W,summarise
from cycle_reward import CycleTracker,CycleConfig
from target_teacher import TargetTeacher,TargetConfig
from identity import HERE,capture,verify,atomic_json,sha256


def main():
    out=HERE/'target_diag';out.mkdir(exist_ok=False)
    conditions=[('current_frame',TargetConfig(vertical_gain=.5,shadow_root_mode='current',feedback_space='velocity')),
                ('fixed_frame',TargetConfig(vertical_gain=.5)),
                ('fixed_frame_support',TargetConfig(vertical_gain=.5,height_gain=1.,orientation_gain=1.))]
    results=[];start=time.monotonic()
    for name,cfg in conditions:
        c=Core(adhesion=False)
        identity=capture(c,dict(name=name,config=asdict(cfg),settle_s=1.,adhesion=False))
        atomic_json(out/(name+'_IDENTITY.json'),identity)
        for _ in range(1000):c.tick(np.zeros(42))
        teacher=TargetTeacher(c,cfg)
        before=(c.d.qpos.copy(),c.d.qvel.copy(),c.d.xfrc_applied.copy(),c.actuator.target_rad.copy(),c.d.ctrl.copy())
        a,info=teacher.action(c,0.)
        for current,old in zip((c.d.qpos,c.d.qvel,c.d.xfrc_applied,c.actuator.target_rad,c.d.ctrl),before):assert np.array_equal(current,old)
        if name=='hold':assert np.max(np.abs(a))<1e-12
        steps=1000 if name=='hold' else round(teacher.reference.duration_s*1000)
        cycles=CycleTracker(CycleConfig(timeout=4.)).reset(start_time=float(c.d.time))
        rec=[];rows={key:[] for key in ['time','qpos','qvel','com','loads','nonfoot','speed','padz','foot','action','target','raw_rate','error','shadow_pad','reference_z','torque_fraction']}
        counts={leg:0 for leg in LEGS};events=[];early=False
        for k in range(0,steps,5):
            action,info=teacher.action(c,k*.001)
            assert np.max(np.abs(action))<=1+1e-10
            for _ in range(min(5,steps-k)):
                m=c.tick(action);e=cycles.measure(c);events.extend(e['events'])
                for item in e['events']:
                    if item['type']=='complete':counts[item['leg']]+=1
                row=dict(time=float(c.d.time),qpos=c.d.qpos.copy(),qvel=c.d.qvel.copy(),com=c.d.subtree_com[c.th].copy(),
                    loads=m['loads'].copy(),nonfoot=m['nonfoot'],speed=m['speed'].copy(),padz=m['padz'].copy(),foot=m['foot'].copy(),
                    action=action.copy(),target=c.actuator.target_rad.copy(),raw_rate=info['raw_rate'].copy(),
                    error=info['tracking_error'].copy(),shadow_pad=info['shadow_pad'].copy(),reference_z=info['reference_z'].copy(),
                    torque_fraction=np.abs(c.d.ctrl/c.actuator.torque_limit))
                for key,value in row.items():rows[key].append(value)
                rec.append((row['time'],row['com'],row['loads'],row['nonfoot'],row['speed']))
                if row['com'][2]<1. or row['nonfoot']>.2*W:early=True;break
            if early:break
        events.extend(cycles.finalize()['events'])
        data={key:np.asarray(value) for key,value in rows.items()};path=out/(name+'.npz')
        np.savez_compressed(path,**data)
        with np.load(path,allow_pickle=False) as z:assert all(np.isfinite(z[k]).all() for k in z.files)
        report=summarise(rec,len(rec)*80,initial_heading=c.initial_heading)
        report.update(name=name,config=asdict(cfg),early_stop=early,actual_ticks=len(rec),requested_ticks=steps,
             complete_per_leg=counts,events=events,npz_sha256=sha256(path),teacher=teacher.metadata(),
             max_clearance=data['padz'].max(0).tolist(),target_total_shift=float(np.linalg.norm(data['target'][-1]-data['target'][0])),
             target_error_rms=float(np.sqrt(np.mean(data['error']**2))),
             max_requested_rate=float(2*np.max(np.abs(data['action']))))
        verify(identity);atomic_json(out/(name+'_RESULT.json'),report);results.append(report)
        atomic_json(out/'RESULTS.json',dict(status='running',results=results))
        print(json.dumps(dict(name=name,early=early,ticks=len(rec),forward=report['forward_signed_native_per_s'],com=report['com_height_min'],
              nonfoot=report['nonfoot_force_max_native'],slip=report['slip_ratio_max'],cycles=counts,maxclearance=report['max_clearance'],
              targetshift=report['target_total_shift'])),flush=True)
    atomic_json(out/'RESULTS.json',dict(status='completed',results=results,wall_s=time.monotonic()-start,
        note='private target geometry computation only; all physical movement via original bounded actuators'))

if __name__=='__main__':main()
