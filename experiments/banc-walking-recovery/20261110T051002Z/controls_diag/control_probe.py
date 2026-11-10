"""Bounded derived-body zero-rate controls; writes only controls_diag/."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import traceback
import numpy as np
import mujoco as mj
import physics
import identity

HERE = Path(__file__).resolve().parent
OUT = HERE / 'controls_diag'

def save(name, obj):
    identity.atomic_json(OUT / name, obj)

def stats(x):
    x = np.asarray(x)
    return dict(mean=float(x.mean()), p95=float(np.quantile(x, .95)), max=float(x.max()), rms=float(np.sqrt(np.mean(x*x))))

def main():
    OUT.mkdir(exist_ok=False)
    script = Path(__file__).read_bytes()
    (OUT / 'control_probe.py').write_bytes(script)
    config = dict(goal='S2/S5 diagnostic: does contact velocity exist without active policy?',
        duration_each_s=.5, ticks_each=500, sample_dt_s=.001, initial_sample=True,
        cases={'A_default_adhesion':True, 'B_adhesion_false':False}, action=[0.]*42,
        action_path='unchanged physics.Core.tick(np.zeros(42)); finite PD hold enabled',
        post_reset_writes='none by probe: no pose, force, ctrl or gain writes; only unchanged Core.tick',
        lowpass='causal trailing uniform boxcar COM, 20/60 samples at 1ms; full windows only; velocity backward difference of filtered COM',
        speed_contract='horizontal world XY magnitude; loads positive normal native force; Core.measure speed=load weighted contact velocity, raw_speed=unweighted contact mean (0 if none); pad center speeds measured separately regardless of contact',
        summary_window='tick ends 1..500ms; initial sample only for displacements/filter boundary',
        units={'position':'native length; absolute SI NOT_ESTABLISHED','speed':'native length/s','load':'native force; absolute SI NOT_ESTABLISHED'},
        exclusions='not walking/gait inference; not frozen acceptance; no native recorder; no video; no motor cut; no gain search',
        executable=sys.executable, command=f'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 {HERE}/env/bin/python -B {__file__}',
        environment={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},
        script_sha256=hashlib.sha256(script).hexdigest())
    save('config.json',config)
    start=time.monotonic()
    results={}
    initial=None
    manifests=[]
    for label, adhesion in config['cases'].items():
        core=physics.Core() if adhesion else physics.Core(adhesion=False)
        manifest=identity.capture(core,config,extra_paths=[__file__])
        save(label+'_identity.json',manifest)
        manifests.append(manifest)
        q=core.d.qpos.copy(); v=core.d.qvel.copy(); target=core.actuator.target_rad.copy()
        if initial is None:
            initial=(q.copy(),v.copy(),target.copy())
            np.savez_compressed(OUT/'initial_state.npz',qpos=q,qvel=v,target_rad=target)
        else:
            assert all(np.array_equal(a,b) for a,b in zip(initial,(q,v,target))), 'reset mismatch'
        rows=[]
        def record(m):
            centers=[]
            for leg in physics.LEGS:
                velocities=[]
                for geom in core.pads[leg]:
                    jac=np.zeros((3,core.plant.nv))
                    mj.mj_jac(core.plant,core.d,jac,None,core.d.geom_xpos[geom],int(core.plant.geom_bodyid[geom]))
                    velocities.append(jac @ core.d.qvel)
                centers.append(velocities)
            return dict(time_s=float(core.d.time),com=core.d.subtree_com[core.th].copy(),
                com_velocity=core.com_velocity().copy(),loads=m['loads'].copy(),
                weighted_contact_speed=m['speed'].copy(),raw_contact_speed=m['raw_speed'].copy(),
                contact_count=m['contact'].copy(),nonfoot=np.asarray(m['nonfoot']),
                pad_center_velocity=np.asarray(centers),ctrl=core.d.ctrl.copy())
        rows.append(record(core.measure()))
        for i in range(500):
            m=core.tick(np.zeros(42))
            rows.append(record(m))
            assert np.array_equal(core.actuator.target_rad,target), 'zero rate changed target'
        data={key:np.asarray([r[key] for r in rows]) for key in rows[0]}
        assert all(np.isfinite(a).all() for a in data.values())
        assert np.allclose(data['time_s'],np.arange(501)*.001,rtol=0,atol=1e-12)
        data['com_displacement']=data['com']-data['com'][0]
        filters={}
        for n in (20,60):
            cumsum=np.vstack([np.zeros((1,3)),np.cumsum(data['com'],axis=0)])
            filtered=(cumsum[n:]-cumsum[:-n])/n
            data[f'com_lp{n}ms']=filtered
            data[f'com_lp{n}ms_time_s']=data['time_s'][n-1:]
            speed=np.linalg.norm(np.diff(filtered,axis=0)[:,:2]/.001,axis=1)
            filters[f'{n}ms']=dict(first_full_window_time_s=float(data['time_s'][n-1]),xy_speed=stats(speed))
        loads=data['loads'][1:]; speed=data['weighted_contact_speed'][1:]
        total=loads.sum()
        perleg={}
        for j,leg in enumerate(physics.LEGS):
            active=data['contact_count'][1:,j]>0
            denominator=loads[:,j].sum()
            raw=data['raw_contact_speed'][1:,j][active]
            pad=np.linalg.norm(data['pad_center_velocity'][1:,j,:,:2],axis=-1)
            perleg[leg]=dict(load_mean_native=float(loads[:,j].mean()),contact_samples=int(active.sum()),
                load_sum_native=float(denominator),load_weighted_speed_native_per_s=float((loads[:,j]*speed[:,j]).sum()/denominator) if denominator>0 else None,
                weighted_contact_speed_peak_native_per_s=float(speed[:,j].max()),
                raw_contact_speed_native_per_s=stats(raw) if raw.size else None,
                pad_center_xy_speed_native_per_s=stats(pad))
        summary=dict(duration_s=float(data['time_s'][-1]),sample_count=501,ticks=500,native_steps=40000,
            com_start_native=data['com'][0].tolist(),com_end_native=data['com'][-1].tolist(),
            com_delta_native=data['com_displacement'][-1].tolist(),
            com_xy_displacement_native=float(np.linalg.norm(data['com_displacement'][-1,:2])),
            com_xy_speed_raw_native_per_s=stats(np.linalg.norm(data['com_velocity'][1:,:2],axis=1)),
            com_lowpass=filters,com_z_min_native=float(data['com'][:,2].min()),
            total_contact_load_mean_native=float(loads.sum(axis=1).mean()),
            nonfoot_peak_native=float(data['nonfoot'][1:].max()),
            contact_load_weighted_speed_native_per_s=float((loads*speed).sum()/total),
            per_leg=perleg,reset_matches=True,actuator_target_unchanged=True,mujoco_warning_count=core.d.warning.number.tolist())
        np.savez_compressed(OUT/(label+'_1ms.npz'),**data)
        save(label+'_summary.json',summary)
        results[label]=summary
        identity.verify_runtime(manifest,core)
        print(label,json.dumps(summary,ensure_ascii=False),flush=True)
    for manifest in manifests:
        identity.verify(manifest)
    save('summary.json',dict(config=config,results=results,wall_time_s=time.monotonic()-start,input_hashes_unchanged=True,
        conclusion_scope='500ms diagnostic only: no gait inference or frozen acceptance; native substep peaks not measured'))
    outputs={str(p):identity.sha256(p) for p in OUT.iterdir() if p.is_file()}
    save('output_sha256.json',outputs)

if __name__=='__main__':
    code=1
    try:
        main()
        code=0
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if OUT.exists():
            save('exit_status.json',dict(exit_code=code,finished_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))
