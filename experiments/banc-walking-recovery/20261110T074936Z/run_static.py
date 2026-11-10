"""Bounded inverse-statics teacher physical controls, model arrays unchanged."""
import json
import time
import numpy as np
from physics import Core,LEGS,W,summarise
from cycle_reward import CycleTracker,CycleConfig
from static_teacher import StaticTeacher
from identity import HERE,atomic_json,capture,verify,sha256


def main():
    check=json.loads((HERE/'STATIC_CHECKS.json').read_text())
    assert check['all_passed'] and check['passed']==check['total']
    for p,h in check['source_hashes'].items():assert sha256(p)==h
    out=HERE/'static_diag';out.mkdir(exist_ok=False)
    conditions=[('hold_inverse_statics',dict(mode='hold',gravity_ff=True),1000),
                ('wave_position_only',dict(mode='measuredwave',gravity_ff=False,cycles=1),None),
                ('wave_inverse_statics',dict(mode='measuredwave',gravity_ff=True,cycles=2),None)]
    results=[];start=time.monotonic()
    for name,cfg,steps in conditions:
        c=Core(adhesion=False)
        identity=capture(c,dict(name=name,config=cfg,settle_s=1.,landing_tail_s=.4,adhesion=False),
                         extra_paths=[HERE/'STATIC_CHECKS.json'])
        atomic_json(out/(name+'_IDENTITY.json'),identity)
        for _ in range(1000):c.tick(np.zeros(42))
        teacher=StaticTeacher(c,cfg)
        physical_before=[x.copy() for x in (c.d.qpos,c.d.qvel,c.d.ctrl,c.d.xfrc_applied,c.actuator.target_rad)]
        action,diagnostic=teacher.action(c,0.)
        for actual,old in zip((c.d.qpos,c.d.qvel,c.d.ctrl,c.d.xfrc_applied,c.actuator.target_rad),physical_before):assert np.array_equal(actual,old)
        # Start fresh controller history after the preflight query, without resetting physics.
        teacher=StaticTeacher(c,cfg)
        atomic_json(out/(name+'_METADATA.json'),teacher.metadata)
        if steps is None:steps=round((teacher.reference.duration_s+.4)*1000)
        observer=CycleTracker(CycleConfig(timeout=4.)).reset(start_time=float(c.d.time))
        counts={leg:0 for leg in LEGS};events=[];rec=[]
        keys=['time','qpos','qvel','com','loads','nonfoot','speed','padz','foot','action','target','torque_fraction',
              'q_des','qdot_des','tau_ff','desired_motor_target','predicted_fz','root_residual','ik_error','hypothetical','unsupported']
        rows={key:[] for key in keys};early=None
        for k in range(0,steps,5):
            action,diag=teacher.action(c,k*.001)
            if diag['root_balance_residual_norm_over_W']>.02:
                early='root force model residual>.02';break
            for _ in range(min(5,steps-k)):
                m=c.tick(action);event=observer.measure(c);events.extend(event['events'])
                for e in event['events']:
                    if e['type']=='complete':counts[e['leg']]+=1
                row=dict(time=float(c.d.time),qpos=c.d.qpos.copy(),qvel=c.d.qvel.copy(),com=c.d.subtree_com[c.th].copy(),
                  loads=m['loads'].copy(),nonfoot=float(m['nonfoot']),speed=m['speed'].copy(),padz=m['padz'].copy(),foot=m['foot'].copy(),
                  action=action.copy(),target=c.actuator.target_rad.copy(),torque_fraction=np.abs(c.d.ctrl/c.actuator.torque_limit),
                  q_des=diag['q_des_act'],qdot_des=diag['qdot_des'],tau_ff=diag['tau_ff'],desired_motor_target=diag['desired_motor_target'],
                  predicted_fz=diag['per_leg_fz_native'],root_residual=diag['root_balance_residual_over_W'],ik_error=diag['ik_error'],
                  hypothetical=diag['support_is_hypothetical'],unsupported=diag['unsupported'])
                for key,value in row.items():rows[key].append(value)
                rec.append((row['time'],row['com'],row['loads'],row['nonfoot'],row['speed']))
                if row['com'][2]<1. or row['nonfoot']>.2*W:early='physical COM<1 or nonfoot>.2W';break
            if early:break
        events.extend(observer.finalize()['events'])
        data={key:np.asarray(value) for key,value in rows.items()};path=out/(name+'.npz')
        np.savez_compressed(path,**data)
        with np.load(path,allow_pickle=False) as z:
            assert all(np.isfinite(z[key]).all() for key in z.files)
            if len(z['time'])>1:assert np.allclose(np.diff(z['time']),.001,rtol=0,atol=1e-10)
        report=summarise(rec,len(rec)*80,initial_heading=c.initial_heading) if len(rec)>1 else dict(all_gates=False)
        report.update(name=name,config=cfg,requested_ticks=steps,actual_ticks=len(rec),early_stop=early,
            complete_per_leg=counts,events=events,metadata_file=str(out/(name+'_METADATA.json')),npz_sha256=sha256(path),
            max_clearance=data['padz'].max(0).tolist() if len(rec) else None,
            model_root_residual_max=float(np.max(np.abs(data['root_residual']))) if len(rec) else None,
            hypothetical_support_fraction=float(data['hypothetical'].mean()) if len(rec) else None,
            model_unsupported_fraction=float(data['unsupported'].mean()) if len(rec) else None,
            ik_error_max=float(np.max(np.abs(data['ik_error']))) if len(rec) else None,
            actual_torque_saturation_fraction=float(np.mean(data['torque_fraction']>=.999)) if len(rec) else None)
        verify(identity);atomic_json(out/(name+'_RESULT.json'),report);results.append(report)
        atomic_json(out/'RESULTS.json',dict(status='running',results=results))
        print(json.dumps(dict(name=name,early=early,ticks=len(rec),forward=report.get('forward_signed_native_per_s'),
           com=report.get('com_height_min'),nonfoot=report.get('nonfoot_force_max_native'),slip=report.get('slip_ratio_max'),
           cycles=counts,maxclearance=report['max_clearance'],IK=report['ik_error_max'],unsupported=report['model_unsupported_fraction'])),flush=True)
    atomic_json(out/'RESULTS.json',dict(status='completed',results=results,wall_s=time.monotonic()-start,
       interpretation='all contact forces are predictions only; actual motion/contact from bounded finite-PD physical simulation'))

if __name__=='__main__':main()
