"""Independent teacher replay, optional native records, direct causal BC samples."""
import argparse
import json
import time
import numpy as np
from physics import Core,LEGS,W,NativeRecorder,summarise
from cycle_reward import CycleTracker,CycleConfig
from static_teacher import StaticTeacher
from identity import HERE,atomic_json,capture,verify,sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    p.add_argument('--native',action='store_true')
    p.add_argument('--cycles',type=int,default=2)
    p.add_argument('--pulse-seed',type=int)
    a=p.parse_args()
    out=__import__('pathlib').Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    config=dict(mode='measuredwave',gravity_ff=True,cycles=a.cycles)
    c=Core(adhesion=False)
    identity=capture(c,dict(scope='independent algorithm teacher replay',config=config,native=a.native,
                           settle_s=1.,landing_tail_s=.4,pulse_seed=a.pulse_seed),extra_paths=[HERE/'static_diag/wave_inverse_statics_RESULT.json'])
    atomic_json(out/'IDENTITY.json',identity)
    for _ in range(1000):c.tick(np.zeros(42))
    if a.pulse_seed is not None:
        rng=np.random.default_rng(a.pulse_seed)
        pulse=np.clip(rng.normal(0,.02,42),-1,1)
        for _ in range(20):c.tick(pulse)
    controller=StaticTeacher(c,config)
    previous=np.zeros(42) if a.pulse_seed is None else pulse.copy()
    observer=CycleTracker(CycleConfig(timeout=4.)).reset(start_time=float(c.d.time))
    recorder=NativeRecorder(out/'native') if a.native else None
    c.recorder=recorder
    ticks=round((controller.reference.duration_s+.4)*1000)
    samples={k:[] for k in ['obs','critic_obs','action','time','source_phase']}
    trace={k:[] for k in ['time','qpos','qvel','com','loads','nonfoot','speed','padz','foot','action','target']}
    rec=[];counts={leg:0 for leg in LEGS};events=[];start=time.monotonic()
    try:
        for k in range(0,ticks,5):
            obs=c.obs238(previous,.36/(6*.8))
            critic=np.r_[obs,c.critic_extras(),(ticks-k)*.001]
            action,diag=controller.action(c,k*.001)
            samples['obs'].append(obs);samples['critic_obs'].append(critic);samples['action'].append(action.copy())
            samples['time'].append(float(c.d.time));samples['source_phase'].append(k*.001)
            for _ in range(min(5,ticks-k)):
                m=c.tick(action);event=observer.measure(c);events.extend(event['events'])
                for e in event['events']:
                    if e['type']=='complete':counts[e['leg']]+=1
                row=dict(time=float(c.d.time),qpos=c.d.qpos.copy(),qvel=c.d.qvel.copy(),com=c.d.subtree_com[c.th].copy(),
                    loads=m['loads'].copy(),nonfoot=float(m['nonfoot']),speed=m['speed'].copy(),padz=m['padz'].copy(),
                    foot=m['foot'].copy(),action=action.copy(),target=c.actuator.target_rad.copy())
                for key,value in row.items():trace[key].append(value)
                rec.append((row['time'],row['com'],row['loads'],row['nonfoot'],row['speed']))
            previous=action.copy()
    finally:
        native=None if recorder is None else recorder.close()
    events.extend(observer.finalize()['events'])
    np.savez_compressed(out/'BC_SAMPLES.npz',**{k:np.asarray(v) for k,v in samples.items()})
    np.savez_compressed(out/'TRACE_1MS.npz',**{k:np.asarray(v) for k,v in trace.items()})
    result=summarise(rec,len(rec)*80,initial_heading=c.initial_heading,native_summary=native)
    result.update(complete_per_leg=counts,events=events,wall_s=time.monotonic()-start,
                  sample_file_sha256=sha256(out/'BC_SAMPLES.npz'),trace_sha256=sha256(out/'TRACE_1MS.npz'),
                  control_source='algorithm-assisted inverse-statics measured-shape wave teacher ONLY',
                  actor_features='238 current sensory + ownpreviousactualaction andconstantdesiredspeed; no source_phase usedasactorinput',
                  source_phase_field='teacher provenance only, excludedfromactorinputs')
    if a.cycles==2 and a.pulse_seed is None:
        old=json.loads((HERE/'static_diag/wave_inverse_statics_RESULT.json').read_text())
        keys=['forward_signed_native_per_s','com_height_min','foot_support_mean','nonfoot_force_max_native','slip_ratio_max']
        difference={key:result[key]-old[key] for key in keys}
        result['independent_replay_differences']=difference
        assert all(abs(value)<1e-8 for value in difference.values()),difference
        assert result['complete_per_leg']==old['complete_per_leg']
    verify(identity);atomic_json(out/'RESULT.json',result)
    print(json.dumps(dict(all_engineering_gates=result['all_gates'],forward=result['forward_signed_native_per_s'],
          com=result['com_height_min'],nonfoot=result['nonfoot_force_max_native'],slip=result['slip_ratio_max'],
          counts=counts,native_rows=0 if native is None else native['native_samples'],wall_s=result['wall_s'])),flush=True)

if __name__=='__main__':main()
