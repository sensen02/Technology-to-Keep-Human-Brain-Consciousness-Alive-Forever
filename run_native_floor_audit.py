"""Bounded read-only native positive-control-floor experiment; not collapse attribution."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from engine import canonical_physics as cp, memory_guard as mg
from engine.embodied.closure_bundle import validate_bundle

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs/neural_muscle_closure'
EXPECTED_XML_SHA = '65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6'
ARMS = ('normal', 'spike_output_silence')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dof_groups(model):
    import mujoco
    linear, angular = [], []
    for kind, address in zip(model.jnt_type, model.jnt_dofadr):
        address = int(address)
        if kind == mujoco.mjtJoint.mjJNT_FREE:
            linear.extend(range(address, address+3)); angular.extend(range(address+3, address+6))
        elif kind == mujoco.mjtJoint.mjJNT_SLIDE:
            linear.append(address)
        else:
            angular.extend(range(address, address+(3 if kind == mujoco.mjtJoint.mjJNT_BALL else 1)))
    assert sorted(linear+angular) == list(range(model.nv))
    return dict(translational=linear, angular=angular)

def decompose(model, qpos, qvel, act):
    """Two scratch data objects, one unchanged model, same pose/velocity; never step act0."""
    import mujoco
    values = {}
    for label, activation in (('total', act), ('act0', np.zeros(model.na))):
        d = mujoco.MjData(model)
        d.qpos[:] = qpos; d.qvel[:] = qvel; d.act[:] = activation; d.ctrl[:] = 0
        mujoco.mj_forward(model, d)
        for field in ('actuator_force', 'qfrc_actuator', 'qfrc_passive'):
            values[label+'_'+field] = getattr(d, field).copy()
    for field in ('actuator_force', 'qfrc_actuator', 'qfrc_passive'):
        values['active_delta_'+field] = values['total_'+field]-values['act0_'+field]
    return values

def summarize(arrays, groups):
    f = arrays['active_delta_actuator_force']
    total = arrays['total_actuator_force']
    out = dict(positive_activation_count_per_sample=(arrays['act']>0).sum(axis=1).tolist(),
        activation_min=float(arrays['act'].min()), activation_max=float(arrays['act'].max()),
        max_abs_active_delta_actuator_force_model=float(abs(f).max()),
        max_abs_total_actuator_force_model=float(abs(total).max()),
        max_abs_act0_actuator_force_model=float(abs(arrays['act0_actuator_force']).max()),
        active_to_total_actuator_force_L2_ratio_per_sample=np.divide(np.linalg.norm(f,axis=1),np.linalg.norm(total,axis=1),out=np.zeros(len(f)),where=np.linalg.norm(total,axis=1)>0).tolist(),
        nonzero_active_force_count_per_sample=(f!=0).sum(axis=1).tolist(),
        max_abs_qfrc_passive_pair_difference=float(abs(arrays['active_delta_qfrc_passive']).max()),
        generalized_norms={})
    for group, idx in groups.items():
        out['generalized_norms'][group] = {field:np.linalg.norm(arrays[field][:,idx],axis=1).tolist()
            for field in ('total_qfrc_actuator','act0_qfrc_actuator','active_delta_qfrc_actuator','act0_qfrc_passive')}
    return out

def main(report=None):
    import mujoco
    start = time.monotonic()
    bundle = validate_bundle(OUT)
    source = Path(bundle['path'])
    assert sha(cp.CANONICAL_XML) == cp.CANONICAL_XML_SHA256 == EXPECTED_XML_SHA
    source_hashes = {name:sha(source/name) for name in ('manifest.json','closure_results.json','closure_traces.npz')}
    model = cp.load_model()
    assert model.nu == model.na == 90
    assert np.all(model.actuator_ctrllimited) and np.all(model.actuator_ctrlrange[:,0] == .0001)
    assert not (model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL))
    assert np.all(model.actuator_dyntype == int(mujoco.mjtDyn.mjDYN_MUSCLE))
    assert np.all(model.actuator_gaintype == int(mujoco.mjtGain.mjGAIN_MUSCLE))
    assert not np.any(model.actuator_actearly)
    groups = dof_groups(model)
    frozen_fields = ('actuator_ctrlrange','actuator_dynprm','actuator_gainprm','actuator_biasprm','actuator_ctrllimited')
    frozen = {key:getattr(model,key).copy() for key in frozen_fields}
    arrays, summaries = {}, {}
    with np.load(source/'closure_traces.npz', allow_pickle=False) as saved:
        seated = saved['seated_qpos'].copy()
        for arm in ARMS:
            tr = saved[arm+'_trajectory'].copy()
            assert tr.shape == (10,model.nq+model.nv+model.na)
            local = dict(qpos=tr[:,:model.nq], qvel=tr[:,model.nq:model.nq+model.nv], act=tr[:,-model.na:],
                         time_s=np.arange(1,11)*.005, source_controls=saved[arm+'_controls'].copy())
            pairs = [decompose(model,q,v,a) for q,v,a in zip(local['qpos'],local['qvel'],local['act'])]
            local.update({key:np.asarray([r[key] for r in pairs]) for key in pairs[0]})
            summaries[arm] = summarize(local,groups)
            arrays.update({arm+'_'+key:value for key,value in local.items()})
    d = mujoco.MjData(model)
    d.qpos[:] = seated; d.qvel[:] = 0; d.act[:] = 0; d.ctrl[:] = 0
    mujoco.mj_forward(model,d)
    initial_dot = d.act_dot.copy()
    analytic_dot = np.array([mujoco.mju_muscleDynamics(.0001,0,model.actuator_dynprm[i,:3]) for i in range(model.nu)])
    np.testing.assert_allclose(initial_dot,analytic_dot,rtol=1e-14,atol=0)
    n = round(.005/model.opt.timestep)
    assert n == 50 and model.opt.timestep == .0001
    local = {key:[] for key in ('time_s','qpos','qvel','act','ctrl_readback')}
    pairs = []
    for _ in range(n):
        # No post-step activation reset or model mutation: control remains the initial zero.
        mujoco.mj_step(model,d)
        mujoco.mj_forward(model,d)  # match original closure's per-step observation call
        assert not any(w.number for w in d.warning)
        for key in ('qpos','qvel','act'):
            local[key].append(getattr(d,key).copy())
        local['time_s'].append(d.time); local['ctrl_readback'].append(d.ctrl.copy())
        pairs.append(decompose(model,d.qpos,d.qvel,d.act))
    local = {key:np.asarray(value) for key,value in local.items()}
    local.update({key:np.asarray([r[key] for r in pairs]) for key in pairs[0]})
    assert np.all(local['ctrl_readback']==0) and np.all(local['act']>0)
    np.testing.assert_array_equal(local['qpos'][-1],arrays['spike_output_silence_qpos'][0])
    np.testing.assert_array_equal(local['qvel'][-1],arrays['spike_output_silence_qvel'][0])
    np.testing.assert_array_equal(local['act'][-1],arrays['spike_output_silence_act'][0])
    summaries['zero_ctrl_integration'] = summarize(local,groups)
    arrays.update({'zero_ctrl_integration_'+key:value for key,value in local.items()})
    arrays.update(initial_act_dot=initial_dot,clamped_floor_analytic_act_dot=analytic_dot,
                  actuator_ctrlrange=model.actuator_ctrlrange.copy(),actuator_dynprm=model.actuator_dynprm.copy())
    for key, value in frozen.items(): np.testing.assert_array_equal(getattr(model,key),value)
    assert validate_bundle(OUT) == bundle
    assert all(sha(source/name)==h for name,h in source_hashes.items())
    assert sha(cp.CANONICAL_XML)==EXPECTED_XML_SHA
    assert all(np.isfinite(value).all() for value in arrays.values())
    assert time.monotonic()-start < 60
    dest = OUT/'native_floor_audit.npz'
    np.savez_compressed(dest,**arrays)
    guard = report('completed') if report else None
    if guard and not np.isfinite(guard['previous_soft_limit_mb']): guard['previous_soft_limit_mb']='unlimited'
    result = dict(schema_version=1,status='PASS_NATIVE_FLOOR_OBSERVATION',
        source_bundle=dict(run_id=bundle['run_id'],path=str(source),sha256=source_hashes),
        provenance=dict(code_path=str(Path(__file__).resolve()),code_sha256=sha(__file__),canonical_xml=str(cp.CANONICAL_XML),canonical_xml_sha256=EXPECTED_XML_SHA,mujoco_version=mujoco.__version__),
        protocol=dict(model_edits=False,neural_replay=False,saved_sample_times_s=(np.arange(1,11)*.005).tolist(),
            integration_initialization='source seated_qpos; qvel=act=ctrl=0',steps=n,dt_s=model.opt.timestep,
            ctrl_readback_all_zero=True,all_90_activations_positive_after_every_step=True,
            first_saved_silenced_state_exactly_reproduced=True,initial_act_dot_matches_native_muscleDynamics_at_clamped_floor=True,
            initial_act_dot_min=float(initial_dot.min()),initial_act_dot_max=float(initial_dot.max()),
            integrator_id=int(model.opt.integrator),first_step_activation_min=float(local['act'][0].min()),first_step_activation_max=float(local['act'][0].max()),
            first_step_note='Unmodified native Euler dt=0.0001 s, act_dot=2 /s from muscle activation tau=0.0001*(0.5+1.5*0)=0.00005 s; first act=0.0002 overshoots control floor0.0001. Subsequent native deactivation approaches floor. No manual clamping; no MuJoCo warning.',
            scratch_method='same model qpos qvel; total=saved/integrated act; act0=zero activation mj_forward only; delta=total-act0',
            passive_definition='act0 actuator force is passive muscle bias; qfrc_passive is separate joint/tendon passive force, not contact/gravity; act0 scratch is NOT a dynamically attainable zero-ctrl trajectory',
            force_units='model force; angular generalized components model torque, translational components model force; no combined norm or SI conversion; mass unit unestablished',dof_groups=groups),
        summaries=summaries,
        boundary='Zero decoded control is not exact zero physical activation or exact zero active muscle force under this frozen native positive ctrl floor. This numerical boundary does not establish that the floor explains collapse or meaningful movement.',
        limitations=['Hand-built simulation audit, not a finished product or biological validation.',
            'Same-state subtraction isolates activation-dependent muscle force, not dynamic causal effects of a floor-removal intervention.',
            'No model with modified floor is constructed. No neural parameter, source bundle, or archive is changed.',
            'Historical source initialization has known joint-range violations; faithfully reused, not repaired.'],
        traces=dict(path=str(dest),sha256=sha(dest),keys=sorted(arrays)),memory_guard=guard,wall_s=time.monotonic()-start)
    (OUT/'native_floor_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(wall_s=result['wall_s'],summaries=summaries),indent=2))
    return result

if __name__ == '__main__':
    with mg.guarded_run('native_floor_audit',**mg.BUDGETS['muscle_floor_audit']) as report:
        main(report)
