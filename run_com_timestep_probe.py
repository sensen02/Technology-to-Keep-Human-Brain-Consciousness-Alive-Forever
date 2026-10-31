"""S113/P1/S2-S4: baseline-only native RK4 timestep diagnostic, 3 ms.

Derived from S112 run_com_contact_probe.py (read-only imported capture helpers).
Import is inert: no model loading, neural stepping, or output creation.
Run once into a fresh absolute --output; no deployment or checkpoint replacement.
"""
from pathlib import Path
import argparse
import json
import traceback
import numpy as np
from run_native_rk4_gate_training import (sha, atomic_json, atomic_npz,
    callback_snapshot, require_no_callbacks, raw_observation_record, numeric_model_snapshot)
from run_native_rk4_tick_consumption import MODEL, INITIAL, ENDPOINTS, ROOT
from run_neural_initialized_startup import WEIGHTS
from run_initialized_translation_training import neural_meta, mesh_dependencies
from run_local_pool_response import moment_record, MOMENT_HEADER
from run_com_contact_probe import (com_contact_record, stack_rows,
    diagnostic_report as s112_diagnostic_report, CRITERIA as S112_CRITERIA, GATE_NAMES)

PRE, TICKS = 80, 6
TIMESTEPS = {'dt50us': .00005, 'dt25us': .000025}
SUBSTEPS = {'dt50us': 10, 'dt25us': 20}
EXPECTED_PER_ARM = {
    'dt50us': dict(model_load=1, forward=61, step=60, neural_ticks=86),
    'dt25us': dict(model_load=1, forward=121, step=120, neural_ticks=86)}
EXPECTED = dict(model_load=2, forward=182, step=180, neural_ticks=172)
FROZEN = ROOT/'run_com_contact_probe.py'
REFERENCE = Path('/home/sensen/Desktop/engine修复验收/s112-com-contact-probe/20261024T055500Z/baseline/TRAJECTORY.npz')
OUTPUT_PARENT = Path('/home/sensen/Desktop/engine修复验收/s113-com-timestep/20261024T060500Z')
CRITERIA = dict(S112_CRITERIA, sampling_checks=['native', 'downsample100us'],
    time_scope='impulse 0..1.7ms inclusive; force/velocity FD native indices 1..1.7ms inclusive; instantaneous/Jacobian all samples',
    downsampling='stride 2/4 without filtering, includes 0 and 3ms; recompute gradient edge_order=2 on actual times')


def local_weights(weights):
    weights = np.asarray(weights)
    if weights.shape != (66, 5) or not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 4)):
        raise ValueError('invalid source W; expected finite (66,5) in [0,4]')
    return {label: weights.copy() for label in TIMESTEPS}


def diagnostic_report(arrays):
    """Original scopes/tolerances on actual native and 100us sample times.

    S112's masks and gradients already depend on len(time) and actual time,
    not its original 31-sample schedule. Retain its primary/native FD arrays.
    """
    native = s112_diagnostic_report(arrays)
    dt = float(np.median(np.diff(arrays['time'])))
    stride = int(round(.0001/dt))
    assert stride in (2, 4) and abs(stride*dt-.0001) < 1e-12
    needed = ('time', 'com_momentum_world', 'external_force_world', 'com_position_world',
              'com_mass_times_acceleration_world', 'body_linear_velocity_world',
              'body_jac_velocity_world', 'com_velocity_world', 'body_mass')
    down = {key: arrays[key][::stride].copy() for key in needed}
    assert len(down['time']) == 31
    downsampled = s112_diagnostic_report(down)
    native['criteria'] = downsampled['criteria'] = CRITERIA
    return dict(criteria=CRITERIA, native=native, downsample100us=downsampled,
                downsample_stride=stride, passed=bool(native['passed'] and downsampled['passed']),
                physical_units='NOT_ESTABLISHED', behavior_claim='NONE')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    out = p.parse_args().output
    if not out.is_absolute() or out.exists(): raise ValueError('absolute fresh --output required; no resume')
    import mujoco as mj
    from engine.embodied.adapters import build_neural_tier
    from engine.embodied.command_tier import downstream_union
    from engine.embodied.motor_endpoints import retain_motor_endpoints
    from engine.embodied.sensory_channels import ChannelSensoryEncoder, channel_index_map, LEGS
    from engine.embodied.motor_translation import FiniteMotorTranslator
    from engine.neck_cut_data import build_edges
    from engine.neural_cond import ConductanceNetwork
    from run_neural_muscle_closure import observe_quantities
    if mj.__version__ != '3.9.0': raise ValueError('protocol requires MuJoCo 3.9.0')
    dependencies = mesh_dependencies()
    checkpoint = Path('/home/sensen/Desktop/engine修复验收/s109-initialized-translation-training/20261024T045944Z')
    sources = list(dict.fromkeys([MODEL, INITIAL, WEIGHTS, ENDPOINTS/'MANIFEST.json',
        ENDPOINTS/'FROZEN_WEIGHTS.json', ROOT/'data/flywire/banc_connectome.npz',
        ROOT/'outputs/neural_rebuild/sensory_channel_lookup.json', MOMENT_HEADER,
        Path(mj.__file__).resolve().parent/'include/mujoco/mujoco.h',
        Path(__file__).resolve(), FROZEN, ROOT/'run_pool_synergy_response.py',
        ROOT/'run_local_pool_response.py', ROOT/'run_neural_initialized_startup.py',
        checkpoint/'CHECKPOINT.json', checkpoint/'CHECKPOINT.npz',
        ROOT/'run_initialized_translation_training.py', ROOT/'initialized_training_objective.py',
        ROOT/'run_neural_muscle_closure.py', ROOT/'run_native_rk4_gate_training.py',
        ROOT/'run_native_rk4_tick_consumption.py', ROOT/'run_native_rk4_reproducibility.py',
        REFERENCE, REFERENCE.parent/'RESULT.json', REFERENCE.parent/'MODEL_PROVENANCE.json',
        REFERENCE.parent/'COM_CONTACT_METADATA.json', REFERENCE.parent.parent/'INPUT_HASHES.json',
        *ROOT.glob('engine/**/*.py'), *dependencies]))
    identities = {str(s): sha(s) for s in sources}
    with np.load(WEIGHTS, allow_pickle=False) as z: weights = z['weights'].copy()
    arms = local_weights(weights)
    with np.load(INITIAL, allow_pickle=False) as z:
        q = z['new_qpos'].copy(); sensory_ref = z['sensory_reference_qpos'].copy()
    manifest = json.loads((ENDPOINTS/'MANIFEST.json').read_text())
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'INPUT_HASHES.json', identities)
    atomic_json(out/'MODEL_DEPENDENCIES.json', {str(s): identities[str(s)] for s in dependencies})
    atomic_json(out/'PREREG.json', dict(arms=list(TIMESTEPS), factors={label:1. for label in arms},
        bounds=[0, 4], clipping=False, preserve_zero_mask=True, all_weights_identical=True,
        initialization_ticks=PRE, initialization_ms=40, physical_duration_ms=3,
        physical_steps_per_arm={k:v['step'] for k,v in EXPECTED_PER_ARM.items()},
        physical_samples_per_arm={k:v['forward'] for k,v in EXPECTED_PER_ARM.items()},
        physical_neural_ticks_per_arm=TICKS, timesteps_s=TIMESTEPS, substeps=SUBSTEPS,
        dt_ms={k:v*1000 for k,v in TIMESTEPS.items()}, neural_dt_ms=.5, sensory_hold_ms=5, source_offset_s=.04,
        same_weights=str(WEIGHTS), source_hashes=identities, expected_calls=EXPECTED,
        expected_calls_per_arm=EXPECTED_PER_ARM, closure_criteria=CRITERIA, six_gate_names=GATE_NAMES,
        retained_dt100us_reference=str(REFERENCE), retained_reference_sha256=sha(REFERENCE),
        planned_output_parent=str(OUTPUT_PARENT),
        six_gate_thresholds=dict(height_ratio_min=.85, angle_rad_max=.35, contact_legs_min=3,
             load_ratio_min=.8, nonfoot_depth_max=1e-12, limit_excess_max=1e-6),
        early_window_s=[0, .003], early_load_scale=.8, early_limit_scale=.001,
        early_limit_threshold=1e-6, early_aggregation='trapezoid actual time divided by .003',
        call_counts='explicit only; initial forward plus one after each step; RK4 internals excluded',
        model_changes_vs_xml=['opt.integrator','opt.timestep'], model_changes_between_arms=['opt.timestep'],
        initialization='held actual initial observation; genuine net.step; computational only, no physics',
        physical_initial='same saved qpos; zero qvel/act/ctrl/external loads; no later pose writes',
        clocks='continuous neural/FIR .04s offset; source END before physical START operator split',
        composite=False, selection=False, deployment=False, checkpoint_replacement=False,
        physical_units='NOT_ESTABLISHED', behavior_claim='NONE', automatic_resume=False))
    atomic_npz(out/'DIRECTION.npz', weights=weights, **arms,
               timesteps_s=np.asarray(list(TIMESTEPS.values())), factors=np.ones(2))
    ledger = dict(model_load=0, forward=0, step=0, neural_ticks=0, active=None, completed=[], per_arm={})
    def reserve(**delta):
        for key, value in delta.items():
            ledger[key] += value
            if ledger['active'] is not None: ledger['per_arm'][ledger['active']][key] += value
        atomic_json(out/'CALL_LEDGER.json', ledger)
    reserve(); results = []
    try:
        built = build_neural_tier(); a = built['data']
        tier = retain_motor_endpoints(a, downstream_union(a, built['global_index'], ['720575941433155799','720575941566493282'], hops=2, max_neurons=20000), ['720575941391252886','720575941511883985','720575941497037218','720575941559306355'])
        roots = a['root_ids'][tier['global_index']]; assert len(roots) == 17099
        edges = build_edges(a, tier, built['cut']); groups, _ = channel_index_map(len(roots), roots)
        reference_before = reference_after = None
        for label, arm_weights in arms.items():
            dt, substeps = TIMESTEPS[label], SUBSTEPS[label]
            samples = TICKS*substeps+1
            dest = out/label; dest.mkdir(); ledger['active'] = label
            ledger['per_arm'][label] = {k:0 for k in EXPECTED}
            reserve(model_load=1); m = mj.MjModel.from_xml_path(str(MODEL))
            assert (m.nq, m.nv, m.nu, m.nbody, m.nmesh) == (49,48,90,73,71) and m.opt.timestep == .0001
            before = numeric_model_snapshot(m)
            m.opt.integrator = mj.mjtIntegrator.mjINT_RK4; m.opt.timestep = dt
            after = numeric_model_snapshot(m)
            changed = sorted(key for key in before if before[key] != after[key])
            assert changed == ['opt.integrator','opt.timestep']
            if reference_before is None: reference_before, reference_after = before, after
            assert before == reference_before
            assert {k:v for k,v in after.items() if k != 'opt.timestep'} == {k:v for k,v in reference_after.items() if k != 'opt.timestep'}
            between = sorted(k for k in after if after[k] != reference_after[k])
            assert between == ([] if label == 'dt50us' else ['opt.timestep'])
            atomic_json(dest/'MODEL_PROVENANCE.json', dict(before=before, after=after,
                changed_numeric_fields=changed, changed_vs_first_arm=between,
                numerical_match_except_timestep=True, xml_sha256=sha(MODEL)))
            require_no_callbacks(callback_snapshot(mj))
            d = mj.MjData(m); d.qpos[:] = q  # Sole pose write.
            net = ConductanceNetwork(len(roots), dt_ms=.5)
            net.set_connectivity(edges['We']*.2, edges['Wi']*.2)
            dec = FiniteMotorTranslator(manifest, arm_weights); enc = ChannelSensoryEncoder(groups, len(roots))
            assert not dec.history.any() and net.steps == 0 and net.t_ms == 0
            atomic_npz(dest/'NEURAL_COLD.npz', **{k:v.copy() for k,v in vars(net).items() if isinstance(v,np.ndarray)})
            atomic_json(dest/'NEURAL_COLD_META.json', neural_meta(net))
            connectivity = {}
            for name in ('We', 'Wi'):
                matrix = getattr(net, name).tocsr()
                for field in ('data','indices','indptr'): connectivity[name+'_'+field] = getattr(matrix,field)
                connectivity[name+'_shape'] = np.asarray(matrix.shape)
            atomic_npz(dest/'CONNECTIVITY.npz', **connectivity)
            joints, geom_leg = {}, {}
            for leg in LEGS:
                js = [j for j in range(m.njnt) if (mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j) or '').lower().startswith('joint_'+leg)]
                tib = next(j for j in js if 'tibia' in mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j).lower())
                joints[leg] = dict(q=[int(m.jnt_qposadr[j]) for j in js], tib_q=int(m.jnt_qposadr[tib]), tib_v=int(m.jnt_dofadr[tib]))
            for g in range(m.ngeom):
                gn = (mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g) or '').lower()
                bn = (mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or '').lower()
                for leg in LEGS:
                    if 'tars' in gn+bn and (gn.startswith(leg) or bn.startswith(leg)): geom_leg[g] = leg
            floor, = np.flatnonzero(m.geom_type == mj.mjtGeom.mjGEOM_PLANE)
            floor_ids = {int(floor)}
            named_floor = mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'floor')
            if named_floor >= 0: floor_ids.add(int(named_floor))
            limited = np.flatnonzero(m.jnt_limited); qa = m.jnt_qposadr[limited]
            atomic_json(dest/'COM_CONTACT_METADATA.json', dict(
                body_names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,j) for j in range(m.nbody)],
                geom_names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,j) for j in range(m.ngeom)],
                geom_bodyid=m.geom_bodyid.tolist(), floor_geom_ids=sorted(floor_ids), gravity_world=m.opt.gravity.tolist(),
                body_position='body_xpos is frame origin; body_xipos is individual inertial COM, world',
                velocity_method='mj_objectVelocity(m,d,mjOBJ_BODY,j,result,flg_local=0); world angular then linear; centered at xipos',
                cvel_convention='raw COM-based spatial velocity, subtree-COM-centered; not individual COM linear velocity',
                acceleration_method='primary world kinematic COM: mj_jacBodyCom@qacc + mj_jacDot(xipos,body)@qvel; all Jacobians saved; no gravity convention ambiguity',
                object_acceleration='raw mj_objectAcceleration BODY world rot:lin after mj_rnePostConstraint, secondary only',
                frame='contact frame rows are world basis vectors; world=frame.T@raw',
                contact_sign='raw force/moment on geom2; opposite on geom1',
                external_mask='exactly one side is floor geom or world body0; self contacts excluded',
                torque='signed intrinsic torque + cross(contact_pos-whole_body_COM, signed force)',
                xfrc='world force[0:3], torque[3:6] at xipos; body0 excluded; asserted zero',
                padding='max actual ncon after capture; IDs -1, other padding 0; contact_valid_mask authoritative',
                units='native; physical units NOT_ESTABLISHED'))
            atomic_json(dest/'MOMENT_METADATA.json', dict(mujoco_version=mj.__version__, format='row-addressed sparse CSR',
                raw_shape=[m.nJmom], dense_shape=[m.nu,m.nv], header=str(MOMENT_HEADER), header_sha256=sha(MOMENT_HEADER),
                dense_key='actuator_moment', raw_key='actuator_moment_raw',
                sign='transmission Jacobian d(actuator_length)/d(q DOF)',
                force_identity='qfrc_actuator = actuator_moment.T @ actuator_force',
                equation='Ma = actuator + passive + constraint + applied - bias; xfrc asserted zero',
                actuator_names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)],
                joint_names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,i) for i in range(m.njnt)],
                jnt_range=m.jnt_range.tolist(), jnt_limited=m.jnt_limited.tolist(),
                jnt_qposadr=m.jnt_qposadr.tolist(), jnt_dofadr=m.jnt_dofadr.tolist(), dof_jntid=m.dof_jntid.tolist()))
            reserve(forward=1); mj.mj_forward(m,d)
            initial_observed = observe_quantities(m,d,sensory_ref,joints,geom_leg)
            initial_current, _ = enc.encode(initial_observed)
            atomic_json(dest/'INITIAL_OBSERVATION.json',raw_observation_record(initial_observed)[0])
            warmsp, warmf, warmu = [], [], []
            for j in range(PRE):
                reserve(neural_ticks=1); sp = net.step(current_nA=initial_current).copy()
                u, info = dec.consume(sp[None,:],spike_dt_s=.0005,start_s=j*.0005,end_s=(j+1)*.0005)
                warmsp.append(sp); warmf.append(info['pool_features']); warmu.append(u)
            release_features = np.stack([dec.history[lo:hi].mean(0) for lo,hi in dec.BINS],axis=1)
            pooled = np.clip((release_features*arm_weights).sum(1),0,1)
            release_ctrl = np.zeros(m.nu); mapped = dec.actuator_pool >= 0
            release_ctrl[mapped] = pooled[dec.actuator_pool[mapped]]
            atomic_npz(dest/'INITIALIZATION.npz',spikes=np.asarray(warmsp,dtype=bool),features=np.asarray(warmf),decoded=np.asarray(warmu),
                current=initial_current,history=dec.history,release_features=release_features,release_decoded=release_ctrl,
                weights=arm_weights,body_qpos=d.qpos,body_qvel=d.qvel,body_act=d.act,body_ctrl=d.ctrl,
                **{k:v for k,v in vars(net).items() if isinstance(v,np.ndarray)})
            atomic_json(dest/'NEURAL_RELEASE_META.json',neural_meta(net))
            assert d.time == 0 and np.array_equal(d.qpos,q) and not any(x.any() for x in (d.qvel,d.act,d.ctrl,d.xfrc_applied,d.qfrc_applied))
            def sample():
                contacts = np.zeros(6,dtype=int); force = np.zeros((6,3)); depth = 0.
                for ci,c in enumerate(d.contact):
                    g1,g2 = int(c.geom1),int(c.geom2)
                    if floor not in (g1,g2): continue
                    foot = g2 if g1 == floor else g1
                    if foot in geom_leg:
                        li = LEGS.index(geom_leg[foot]); contacts[li] += int(c.dist < 0)
                        f = np.zeros(6); mj.mj_contactForce(m,d,ci,f)
                        force[li] += (1 if g1 == floor else -1)*c.frame.reshape(3,3).T@f[:3]
                    else: depth = max(depth,-float(c.dist))
                excess = np.maximum(np.maximum(m.jnt_range[limited,0]-d.qpos[qa],d.qpos[qa]-m.jnt_range[limited,1]),0)
                moments = moment_record(m,d)
                assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
                Ma = np.zeros(m.nv); mj.mj_mulM(m,d,Ma,d.qacc)
                rhs = d.qfrc_actuator+d.qfrc_passive+d.qfrc_constraint+d.qfrc_applied-d.qfrc_bias
                return dict(time=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy(),act=d.act.copy(),ctrl=d.ctrl.copy(),
                    contacts=contacts,foot_force=force,nonfoot_depth=depth,limit_excess=float(excess.max(initial=0)),
                    actuator_force=d.actuator_force.copy(),actuator_length=d.actuator_length.copy(),actuator_velocity=d.actuator_velocity.copy(),
                    qacc=d.qacc.copy(),qfrc_actuator=d.qfrc_actuator.copy(),qfrc_passive=d.qfrc_passive.copy(),
                    qfrc_constraint=d.qfrc_constraint.copy(),qfrc_bias=d.qfrc_bias.copy(),qfrc_applied=d.qfrc_applied.copy(),
                    xfrc_applied=d.xfrc_applied.copy(),Ma=Ma,RHS=rhs.copy(),residual=Ma-rhs,
                    actuator_force_residual=moments['actuator_moment'].T@d.actuator_force-d.qfrc_actuator,
                    warnings=np.asarray([w.number for w in d.warning]),**moments,**com_contact_record(mj,m,d,floor_ids))
            rows = [sample()]; currents,spikes,features,commands,schedule = [],[],[],[],[]
            for tick in range(TICKS):
                if tick % 10 == 0:
                    observed = observe_quantities(m,d,sensory_ref,joints,geom_leg)
                    current, _ = enc.encode(observed)
                    atomic_json(dest/f'OBSERVATION_{tick:02d}.json',raw_observation_record(observed)[0])
                currents.append(current.copy()); reserve(neural_ticks=1)
                sp = net.step(current_nA=current).copy()
                start,end = (PRE+tick)*.0005,(PRE+tick+1)*.0005
                u,info = dec.consume(sp[None,:],spike_dt_s=.0005,start_s=start,end_s=end)
                assert abs(net.t_ms/1000-end) < 1e-12 and dec.interval == PRE+tick
                assert abs(d.time-tick*.0005) < 1e-12
                spikes.append(sp); features.append(info['pool_features']); commands.append(u.copy())
                schedule.append(dict(tick=tick,source_start_s=start,source_end_s=end,physical_start_s=tick*.0005,
                    physical_end_s=(tick+1)*.0005,source_offset_s=.04,consume_id=dec.interval))
                d.ctrl[:] = u
                for sub in range(substeps):
                    require_no_callbacks(callback_snapshot(mj))
                    assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
                    reserve(step=1,forward=1); mj.mj_step(m,d); mj.mj_forward(m,d)
                    row = sample(); rows.append(row)
                    assert all(np.isfinite(v).all() for v in row.values()) and not row['warnings'].any()
            arrays = stack_rows(rows)
            arrays.update(spikes=np.asarray(spikes),features=np.asarray(features),applied=np.asarray(commands),
                decoded=np.asarray(commands),current=np.asarray(currents),weights=arm_weights,actuator_pool=dec.actuator_pool.copy())
            assert arrays['qpos'].shape == (samples,49) and arrays['spikes'].shape == (6,17099)
            assert arrays['actuator_moment'].shape == (samples,90,48) and arrays['actuator_moment_raw'].shape == (samples,m.nJmom)
            assert np.allclose(arrays['time'],np.arange(samples)*dt,rtol=0,atol=1e-12)
            assert net.steps == 86 and dec.interval == 85 and abs(dec.elapsed_s-.043) < 1e-12
            assert abs(net.t_ms/1000-.043) < 1e-12 and numeric_model_snapshot(m) == after
            weight_native = float(m.body_mass.sum()*abs(m.opt.gravity[2]))
            load = arrays['foot_force'][:,:,2].sum(1)/weight_native
            quat = arrays['qpos'][:,3:7]; norms = np.linalg.norm(quat,axis=1)
            angle = 2*np.arccos(np.clip(np.abs(quat@quat[0])/(norms*norms[0]),0,1))
            gates = np.column_stack((arrays['qpos'][:,2]/q[2] >= .85,angle <= .35,
                np.count_nonzero(arrays['contacts'] > 0,axis=1) >= 3,load >= .8,
                arrays['nonfoot_depth'] <= 1e-12,arrays['limit_excess'] <= 1e-6))
            arrays.update(six_gates=gates,all_six_gates=gates.all(1),load_ratio=load,root_angle_rad=angle)
            closure = diagnostic_report(arrays)
            atomic_npz(dest/'TRAJECTORY.npz',**arrays); atomic_json(dest/'SCHEDULE.json',schedule)
            atomic_json(dest/'COM_CLOSURE.json',closure)
            atomic_json(dest/'NEURAL_FINAL_META.json',neural_meta(net))
            atomic_npz(dest/'NEURAL_FINAL.npz',**{k:v.copy() for k,v in vars(net).items() if isinstance(v,np.ndarray)},FIR_history=dec.history)
            Jload = float(np.trapezoid((np.maximum(.8-load,0)/.8)**2,x=arrays['time'])/.003)
            Jlimit = float(np.trapezoid((np.maximum(arrays['limit_excess']-1e-6,0)/.001)**2,x=arrays['time'])/.003)
            assert np.isfinite([Jload,Jlimit]).all()
            assert ledger['per_arm'][label] == EXPECTED_PER_ARM[label]
            hashes = {str(f):sha(f) for f in dest.iterdir() if f.is_file()}
            record = dict(label=label,factor=1.,timestep_s=dt,substeps=substeps,samples=samples,
                initialization_ticks=PRE,source_offset_s=.04,weight_native=weight_native,
                Jload=Jload,Jlimit=Jlimit,hashes=hashes,calls=ledger['per_arm'][label].copy(),
                trajectory_sha256=sha(dest/'TRAJECTORY.npz'),initialization_sha256=sha(dest/'INITIALIZATION.npz'),
                direction_sha256=sha(out/'DIRECTION.npz'),neural_final_s=net.t_ms/1000,
                FIR_final_s=dec.elapsed_s,physical_final_s=float(d.time),closure=closure,
                behavior_claim='NONE',selection=False,deployment=False)
            results.append(record); atomic_json(dest/'RESULT.json',record); atomic_json(out/'RESULTS.json',results)
            ledger['completed'].append(label); ledger['active'] = None; reserve()
        assert {k:ledger[k] for k in EXPECTED} == EXPECTED
        assert identities == {str(s):sha(s) for s in sources}
        atomic_json(out/'COMPLETE.json',dict(status='COMPLETE',physical_units='NOT_ESTABLISHED',ledger=ledger,
            closure_passed=all(r['closure']['passed'] for r in results),
            deployment=False,selection=False,checkpoint_replacement=False))
        print(json.dumps(results))
    except BaseException:
        atomic_json(out/'FAILED.json',dict(traceback=traceback.format_exc(),ledger=ledger)); raise


if __name__ == '__main__':
    main()
