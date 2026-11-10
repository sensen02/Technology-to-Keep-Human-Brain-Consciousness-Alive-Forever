"""Four preregistered measured-reference physical teacher conditions, no ANN claims."""
from pathlib import Path
from dataclasses import asdict
import json
import time
import numpy as np
from physics import Core,summarise,W,LEGS
from cycle_reward import CycleConfig,CycleTracker
from measured_teacher import MeasuredTeacher,TeacherConfig
from identity import HERE,capture,verify,verify_runtime,atomic_json,sha256

CONDITIONS=[('A_w10_z025_h0',TeacherConfig(height_gain=0.)),
 ('B_w10_z025_h2',TeacherConfig()),
 ('C_w20_z025_h2',TeacherConfig(time_warp=20.)),
 ('D_w20_z050_h2',TeacherConfig(time_warp=20.,vertical_gain=.50))]


def main():
    checks=json.loads((HERE/'TEACHER_CHECKS.json').read_text());assert checks['all_passed']
    for path,digest in checks['source_hashes'].items():assert sha256(path)==digest
    out=HERE/'teacher_diag';out.mkdir(exist_ok=False)
    started=time.monotonic();allresults=[]
    for name,config in CONDITIONS:
        c=Core(adhesion=False)
        settings=dict(name=name,teacher=asdict(config),settle_ms=1000,adhesion=False,
                      control_source='algorithm-assisted measured-reference teacher, not autonomous ANN')
        identity=capture(c,settings,extra_paths=[HERE/'TEACHER_CHECKS.json'])
        atomic_json(out/(name+'_IDENTITY.json'),identity)
        for _ in range(1000):c.tick(np.zeros(42))
        controller=MeasuredTeacher(c,config)
        metadata=controller.metadata()
        cycles=CycleTracker(CycleConfig(timeout=4.)).reset(start_time=float(c.d.time))
        steps=int(round(controller.reference.duration_s/.001))
        arrays={k:[] for k in ['time','qpos','qvel','com','loads','nonfoot','speed','padz','foot',
             'action','target','tracking_error','raw_rate','bounds_hit','actual_rate','torque_fraction','reference_xy','reference_z','stance']}
        rec=[];events=[];counts={leg:0 for leg in LEGS};early=None
        for k in range(0,steps,5):
            action,info=controller.action(c,k*.001)
            for _ in range(min(5,steps-k)):
                m=c.tick(action)
                ev=cycles.measure(c)
                events.extend(ev['events'])
                for e in ev['events']:
                    if e['type']=='complete':counts[e['leg']]+=1
                state=dict(time=float(c.d.time),qpos=c.d.qpos.copy(),qvel=c.d.qvel.copy(),
                  com=c.d.subtree_com[c.th].copy(),loads=m['loads'].copy(),nonfoot=float(m['nonfoot']),
                  speed=m['speed'].copy(),padz=m['padz'].copy(),foot=m['foot'].copy(),action=action.copy(),
                  target=c.actuator.target_rad.copy(),tracking_error=info['tracking_error'].copy(),
                  raw_rate=info['unlimited_rate'].copy(),bounds_hit=info['bounds_hit'].copy(),
                  actual_rate=c.d.qvel[c.a_dof].copy(),
                  torque_fraction=np.abs(c.d.ctrl/c.actuator.torque_limit),
                  reference_xy=info['reference']['target_xy'].copy(),reference_z=info['reference']['target_z'].copy(),
                  stance=info['reference']['stance'].copy())
                for key,value in state.items():arrays[key].append(value)
                rec.append((state['time'],state['com'],state['loads'],state['nonfoot'],state['speed']))
                if state['com'][2]<1.0 or state['nonfoot']>.2*W:
                    early='COM<1.0 or nonfoot>.2W';break
            if early:break
        events.extend(cycles.finalize()['events'])
        data={k:np.asarray(v) for k,v in arrays.items()}
        path=out/(name+'.npz');np.savez_compressed(path,**data)
        report=summarise(rec,len(rec)*80,initial_heading=c.initial_heading)
        actualtime=np.arange(len(rec))*.001
        ix=actualtime>=config.blend_s
        report['reference_window']=None if ix.sum()<2 else summarise([row for row,select in zip(rec,ix) if select],int(ix.sum())*80,initial_heading=c.initial_heading)
        report.update(name=name,config=settings,teacher_metadata=metadata,complete_per_leg=counts,events=events,
          early_stop=early,requested_ticks=steps,actual_ticks=len(rec),npz_sha256=sha256(path),
          per_leg_max_clearance=data['padz'].max(0).tolist(),
          rate_unbounded_p95=np.quantile(np.abs(data['raw_rate']),.95,axis=0).tolist(),
          bounded_fraction=float(data['bounds_hit'].mean()),
          tracking_xy_rms=float(np.sqrt(np.mean(data['tracking_error'][:,:,:2]**2))),
          tracking_z_rms=float(np.sqrt(np.mean(data['tracking_error'][:,:,2]**2))),
          torque_saturation_fraction=float(np.mean(data['torque_fraction']>=.999)))
        verify_runtime(identity,Core(adhesion=False));verify(identity)
        with np.load(path,allow_pickle=False) as z:
            assert all(np.isfinite(z[key]).all() for key in z.files)
            assert len(z['time'])==len(rec)
            assert np.allclose(np.diff(z['time']),.001,rtol=0,atol=1e-10)
        atomic_json(out/(name+'_RESULT.json'),report)
        allresults.append(report)
        atomic_json(out/'RESULTS.json',dict(status='running',results=allresults))
        print(json.dumps(dict(name=name,actual_ticks=len(rec),early=early,forward=report['forward_signed_native_per_s'],
             com=report['com_height_min'],nonfoot=report['nonfoot_force_max_native'],slip=report['slip_ratio_max'],
             cycles=counts,maxclearance=report['per_leg_max_clearance'],bounds=report['bounded_fraction'])),flush=True)
    atomic_json(out/'RESULTS.json',dict(status='completed',results=allresults,wall_s=time.monotonic()-started,
                  scope='declared algorithm teacher only; failed conditions retained; not original walking certification'))

if __name__=='__main__':main()
