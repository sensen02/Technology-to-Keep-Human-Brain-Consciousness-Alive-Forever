"""One bounded reference-only counterfactual. No existing source or physics writes."""
from pathlib import Path
from dataclasses import asdict
import json
import time
import numpy as np
from physics import Core,W,LEGS
from measured_teacher import MeasuredTeacher,TeacherConfig
from identity import HERE,capture,verify,verify_runtime,atomic_json,sha256

OUT=HERE/'xy_counter_diag'
BASE=HERE/'teacher_diag'/'B_w10_z025_h2.npz'
BASE_REPORT=BASE.with_name('B_w10_z025_h2_RESULT.json')
BASE_ID=BASE.with_name('B_w10_z025_h2_IDENTITY.json')


def main():
    OUT.mkdir(exist_ok=False)
    started=time.monotonic()
    config=TeacherConfig(time_warp=10.,vertical_gain=.25,height_gain=2.)
    settings=dict(name='B_reference_XY_frozen_at_settled_rest',teacher=asdict(config),
        settle_ms=1000,diagnostic_ms=800,adhesion=False,
        sole_intervention='private reference.sample target_xy=controller.rest[:,:2] and velocity_xy=zeros(6,2)',
        preserved='source stance/swing mask and source Z; common body height/orientation feedback, bounded solver, original physical integration/PD/limits',
        early_stop='COM<1.0 or nonfoot>.2W or numerical/physics exception',
        baseline=str(BASE),comparison='same common prefix from physical settled start; full new window separately',
        scope='algorithm teacher controller counterfactual; not walking acceptance')
    atomic_json(OUT/'CONFIG.json',settings)
    baseline_report=json.loads(BASE_REPORT.read_text())
    assert sha256(BASE)==baseline_report['npz_sha256']
    baseline_id=json.loads(BASE_ID.read_text())
    verify_runtime(baseline_id,Core(adhesion=False))
    c=Core(adhesion=False)
    identity=capture(c,settings,extra_paths=[Path(__file__),BASE,BASE_REPORT,BASE_ID])
    atomic_json(OUT/'IDENTITY_BEFORE.json',identity)
    for _ in range(1000):c.tick(np.zeros(42))
    initial_qpos=c.d.qpos.copy();initial_qvel=c.d.qvel.copy();initial_target=c.actuator.target_rad.copy()
    initial_error=initial_target-initial_qpos[c.a_qpos]
    controller=MeasuredTeacher(c,config)
    original_sample=controller.reference.sample
    rest_xy=controller.rest[:,:2].copy()
    def sample_xy_frozen(elapsed):
        sample=original_sample(elapsed)
        sample['target_xy']=rest_xy.copy()
        sample['velocity_xy']=np.zeros((6,2))
        return sample
    controller.reference.sample=sample_xy_frozen
    teacher_start=float(c.d.time)
    keys=('time','elapsed','qpos','qvel','com','loads','nonfoot','speed','padz','foot',
          'action','target','tracking_error','raw_rate','bounds_hit','actual_rate',
          'torque_fraction','reference_xy','reference_z','stance','body_vz_command','orientation_error')
    arrays={k:[] for k in keys};early=None
    for k in range(0,800,5):
        action,info=controller.action(c,k*.001)
        assert np.array_equal(info['reference']['target_xy'],rest_xy)
        assert np.array_equal(info['reference']['velocity_xy'],np.zeros((6,2)))
        # Preserve the unmodified Z and stance reference at the same controller clock.
        expected=original_sample(k*.001)
        for field in ('target_z','velocity_z','stance'):
            assert np.array_equal(info['reference'][field],expected[field]),field
        for _ in range(5):
            m=c.tick(action)
            state=dict(time=float(c.d.time),elapsed=float(c.d.time)-teacher_start,
              qpos=c.d.qpos.copy(),qvel=c.d.qvel.copy(),com=c.d.subtree_com[c.th].copy(),
              loads=m['loads'].copy(),nonfoot=float(m['nonfoot']),speed=m['speed'].copy(),
              padz=m['padz'].copy(),foot=m['foot'].copy(),action=action.copy(),
              target=c.actuator.target_rad.copy(),tracking_error=info['tracking_error'].copy(),
              raw_rate=info['unlimited_rate'].copy(),bounds_hit=info['bounds_hit'].copy(),
              actual_rate=c.d.qvel[c.a_dof].copy(),torque_fraction=np.abs(c.d.ctrl/c.actuator.torque_limit),
              reference_xy=info['reference']['target_xy'].copy(),reference_z=info['reference']['target_z'].copy(),
              stance=info['reference']['stance'].copy(),body_vz_command=info['body_vertical_velocity_command'],
              orientation_error=info['orientation_error'].copy())
            for key,value in state.items():arrays[key].append(value)
            if state['com'][2]<1.0 or state['nonfoot']>.2*W:
                early='COM<1.0 or nonfoot>.2W';break
        if early:break
    data={k:np.asarray(v) for k,v in arrays.items()}
    path=OUT/'XY_FROZEN_1MS.npz'
    np.savez_compressed(path,**data,initial_qpos=initial_qpos,initial_qvel=initial_qvel,
                        initial_target=initial_target,initial_tracking_preload=initial_error,
                        target_joint_qpos_addresses=c.a_qpos,leg_order=np.asarray(LEGS))
    with np.load(BASE,allow_pickle=False) as z:
        baseline={k:np.array(z[k],copy=True) for k in z.files}
    common=min(len(data['time']),len(baseline['time']))
    assert common>0 and np.allclose(data['time'][:common],baseline['time'][:common],rtol=0,atol=1e-10)
    def stats(d,n):
        e=d['target'][:n]-d['qpos'][:n,c.a_qpos]
        e_change=e-initial_error
        raw=c.actuator.kp[None]*e-c.actuator.kd[None]*d['actual_rate'][:n]
        return dict(samples=n,duration_s=n*.001,
          com_min=float(d['com'][:n,2].min()),com_end=float(d['com'][n-1,2]),
          nonfoot_peak_native=float(d['nonfoot'][:n].max()),
          support_mean_body_weights=float(d['loads'][:n].sum(1).mean()/W),
          support_min_body_weights=float(d['loads'][:n].sum(1).min()/W),
          mean_load_per_leg_body_weights=(d['loads'][:n].mean(0)/W).tolist(),
          max_clearance_per_leg_native=d['padz'][:n].max(0).tolist(),
          clearance_above_015_fraction_per_leg=(d['padz'][:n]>.015).mean(0).tolist(),
          tracking_error_rms_rad=float(np.sqrt(np.mean(e**2))),
          tracking_error_abs_p95_rad=float(np.quantile(np.abs(e),.95)),
          tracking_preload_change_rms_rad=float(np.sqrt(np.mean(e_change**2))),
          tracking_error_rms_per_leg=[float(np.sqrt(np.mean(e[:,ids]**2))) for ids in controller.ids],
          target_drift_norm_rad=float(np.linalg.norm(d['target'][n-1]-initial_target)),
          physical_joint_drift_norm_rad=float(np.linalg.norm(d['qpos'][n-1,c.a_qpos]-initial_qpos[c.a_qpos])),
          actual_joint_velocity_rms_rad_s=float(np.sqrt(np.mean(d['actual_rate'][:n]**2))),
          target_rate_bound_fraction=float(d['bounds_hit'][:n].mean()),
          actual_ctrl_saturation_fraction=float((d['torque_fraction'][:n]>=.999).mean()),
          reconstructed_raw_PD_saturation_fraction=float((np.abs(raw)>=c.actuator.torque_limit[None]).mean()),
          raw_PD_fraction_abs_p95=float(np.quantile(np.abs(raw/c.actuator.torque_limit[None]),.95)),
          torque_timing_note='ctrl fraction is last native substep; reconstructed raw PD uses the 1ms endpoint state')
    report=dict(status='completed',config=settings,actual_ticks=len(data['time']),early_stop=early,
                baseline_actual_ticks=len(baseline['time']),common_prefix_ticks=common,
                baseline_common_prefix=stats(baseline,common),XY_frozen_common_prefix=stats(data,common),
                XY_frozen_full=stats(data,len(data['time'])),
                initial_com=initial_qpos[:3].tolist(),npz_sha256=sha256(path),
                source_identity_sha256=identity['identity_sha256'],wall_s=time.monotonic()-started,
                scope='one reference-only intervention; no root/pose writes after reset, no altered physical forces; 1ms endpoint metrics')
    verify_runtime(identity,Core(adhesion=False));verify(identity)
    with np.load(path,allow_pickle=False) as z:
        assert len(z['time'])==len(data['time'])
        assert np.allclose(np.diff(z['time']),.001,rtol=0,atol=1e-10)
        for key,value in data.items():
            assert np.array_equal(z[key],value),key
            assert np.isfinite(z[key]).all(),key
        assert np.array_equal(z['reference_xy'],np.broadcast_to(rest_xy,z['reference_xy'].shape))
    report['independent_artifact_readback']=True
    atomic_json(OUT/'RESULT.json',report)
    atomic_json(OUT/'IDENTITY_AFTER.json',identity)
    atomic_json(OUT/'OUTPUT_HASHES.json',{p.name:sha256(p) for p in [OUT/'CONFIG.json',OUT/'IDENTITY_BEFORE.json',OUT/'IDENTITY_AFTER.json',path,OUT/'RESULT.json']})
    print(json.dumps(dict(status='completed',actual_ticks=len(data['time']),early_stop=early,
           baseline=report['baseline_common_prefix'],frozen=report['XY_frozen_common_prefix'],
           frozen_full=report['XY_frozen_full'],wall_s=report['wall_s'])),flush=True)

if __name__=='__main__':main()
