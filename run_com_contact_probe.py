"""S112/P1/S2-S4: matched baseline/pool15 COM/contact diagnostic, 3 ms.

Derived from frozen run_pool_synergy_response.py; never modifies its inputs.
Import is inert: no model loading, neural stepping, or output creation.
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

PRE, TICKS, SUBSTEPS = 80, 6, 5
POOL, BIN = 15, 3
FACTORS = {'baseline': 1., 'pool15_up': 1.2}
EXPECTED = dict(model_load=2, forward=62, step=60, neural_ticks=172)
FROZEN = ROOT/'run_pool_synergy_response.py'
GATE_NAMES = ['height', 'angle', 'legs', 'load', 'nonfoot', 'limit']
CRITERIA = dict(window_s=[0., .0017], impulse_absolute_max_native=1e-4,
    impulse_relative_max=.01, impulse_absolute_max_native_requested=1e-4,
    instantaneous_force_absolute_max_native=1e-8, instantaneous_force_relative_max=1e-6,
    relative_denominator_floor_native=1e-12,
    pointwise_force_absolute_max_native=1., pointwise_force_relative_max=.05,
    com_velocity_fd_absolute_max_native=.1, com_velocity_fd_relative_max=.01,
    fd='numpy.gradient edge_order=2; velocity verification excludes endpoints',
    impulse='cumulative trapezoid external force vs COM momentum minus initial; every prefix through 1.7ms',
    relative='impulse: max component residual / max(total_mass*max norm COM velocity over window,1e-12); other: max component residual / max component magnitude of either side floored1e-12',
    diagnostic_failure='Report fail without retry, tolerance relaxation, selection, or suppressing artifacts')


def local_weights(weights):
    weights = np.asarray(weights)
    if weights.shape != (66, 5) or not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 4)):
        raise ValueError('invalid source W; expected finite (66,5) in [0,4]')
    arms = {}
    for label, factor in FACTORS.items():
        w = weights.copy(); w[POOL, BIN] *= factor
        if np.any((w < 0) | (w > 4)):
            raise ValueError('probe outside [0,4]; clipping forbidden: '+label)
        mask = np.ones(w.shape, dtype=bool); mask[POOL, BIN] = False
        assert np.array_equal(w[mask], weights[mask]) and np.array_equal(w == 0, weights == 0)
        arms[label] = w
    return arms


def com_contact_record(mj, m, d, floor_ids):
    """Read live data only. Contact frame rows are world basis vectors.

    mj_objectVelocity(BODY, flg_local=0) is centered at the body's inertial
    frame (xipos), world-oriented, angular then linear. cvel is retained raw:
    it is COM-based spatial velocity in MuJoCo's subtree-COM-centered frame,
    NOT interchangeable with the velocity of each individual body's COM.
    """
    mass = m.body_mass.copy(); total = mass.sum()
    if total <= 0: raise ValueError('nonpositive total mass')
    velocity = np.zeros((m.nbody, 6)); acceleration = np.zeros((m.nbody, 6))
    # Required to populate cacc/cfrc for objectAcceleration; read-only dynamics
    # postprocessing, NOT forward, step, integration, or a control intervention.
    mj.mj_rnePostConstraint(m, d)
    for j in range(m.nbody):
        mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY, j, velocity[j], 0)
        mj.mj_objectAcceleration(m, d, mj.mjtObj.mjOBJ_BODY, j, acceleration[j], 0)
    jacp = np.zeros((m.nbody, 3, m.nv)); jacdot = np.zeros_like(jacp)
    for j in range(1, m.nbody):
        mj.mj_jacBodyCom(m, d, jacp[j], None, j)
        mj.mj_jacDot(m, d, jacdot[j], None, d.xipos[j], j)
    body_jac_velocity = jacp @ d.qvel
    body_acceleration = jacp @ d.qacc + jacdot @ d.qvel
    com = mass @ d.xipos / total
    com_velocity = mass @ velocity[:, 3:] / total
    com_acceleration = mass @ body_acceleration / total
    n = int(d.ncon)
    geom = np.empty((n, 2), dtype=np.int64); body = np.empty((n, 2), dtype=np.int64)
    pos = np.empty((n, 3)); frame = np.empty((n, 3, 3)); dist = np.empty(n)
    raw = np.zeros((n, 6)); world = np.zeros((n, 3)); spin = np.zeros((n, 3))
    torque = np.zeros((n, 3)); external = np.zeros(n, dtype=bool)
    sign = np.zeros(n); dim = np.empty(n, dtype=np.int64); friction = np.empty((n, 5))
    for ci in range(n):
        c = d.contact[ci]; g1, g2 = int(c.geom1), int(c.geom2)
        geom[ci] = [g1, g2]
        if g1 < 0 or g2 < 0: raise ValueError('unsupported non-geom contact')
        body[ci] = m.geom_bodyid[geom[ci]]
        pos[ci] = c.pos; frame[ci] = c.frame.reshape(3, 3); dist[ci] = c.dist
        dim[ci] = c.dim; friction[ci] = c.friction
        mj.mj_contactForce(m, d, ci, raw[ci])
        world[ci] = frame[ci].T @ raw[ci, :3]  # force on geom2
        spin[ci] = frame[ci].T @ raw[ci, 3:]  # intrinsic moment on geom2
        torque[ci] = spin[ci] + np.cross(pos[ci]-com, world[ci])
        fixed1 = g1 in floor_ids or body[ci, 0] == 0
        fixed2 = g2 in floor_ids or body[ci, 1] == 0
        external[ci] = fixed1 != fixed2  # never count self contacts twice
        sign[ci] = (1. if fixed1 else -1.) if external[ci] else 0.
    cf = (sign[:, None]*world).sum(0); ct = (sign[:, None]*torque).sum(0)
    gf = total*m.opt.gravity.copy()
    gt = np.cross(d.xipos-com, mass[:, None]*m.opt.gravity).sum(0)
    # xfrc_applied is world force then torque, at each body's COM. World body
    # is not a dynamic member of the fly; retain its raw row but exclude it.
    xf = d.xfrc_applied[1:, :3].sum(0)
    xt = (d.xfrc_applied[1:, 3:]+np.cross(d.xipos[1:]-com, d.xfrc_applied[1:, :3])).sum(0)
    return dict(body_mass=mass, body_xpos=d.xpos.copy(), body_xipos=d.xipos.copy(),
        body_cvel_raw=d.cvel.copy(), body_spatial_velocity_world=velocity,
        body_linear_velocity_world=velocity[:, 3:].copy(),
        body_object_acceleration_raw6_world=acceleration, body_com_jacp=jacp, body_com_jacdot=jacdot,
        body_jac_velocity_world=body_jac_velocity, body_linear_acceleration_world=body_acceleration,
        com_acceleration_world=com_acceleration, com_mass_times_acceleration_world=total*com_acceleration,
        com_position_world=com,
        com_velocity_world=com_velocity, com_momentum_world=total*com_velocity,
        gravity_force_world=gf, gravity_torque_about_com=gt, contact_force_world=cf,
        contact_torque_about_com=ct, applied_force_world=xf, applied_torque_about_com=xt,
        external_force_world=gf+cf+xf, external_torque_about_com=gt+ct+xt,
        contact_ncon=np.asarray(n), contact_geom_ids=geom, contact_body_ids=body,
        contact_pos_world=pos, contact_frame=frame, contact_dist=dist,
        contact_force_raw6=raw, contact_force_on_geom2_world=world,
        contact_intrinsic_torque_on_geom2_world=spin, contact_torque_on_geom2_about_com=torque,
        contact_external_mask=external, contact_external_sign=sign,
        contact_dim=dim, contact_friction=friction)


def stack_rows(rows):
    """Pad only after live capture; ncon and valid mask distinguish padding."""
    maximum = max(int(r['contact_ncon']) for r in rows)
    arrays = {}
    variable = {k for k in rows[0] if k.startswith('contact_') and k not in
                ('contact_ncon', 'contact_force_world', 'contact_torque_about_com')}
    for key in rows[0]:
        if key not in variable:
            arrays[key] = np.asarray([r[key] for r in rows]); continue
        exemplar = rows[0][key]
        fill = -1 if key in ('contact_geom_ids', 'contact_body_ids') else 0
        a = np.full((len(rows), maximum)+exemplar.shape[1:], fill, dtype=exemplar.dtype)
        for i, row in enumerate(rows): a[i, :len(row[key])] = row[key]
        arrays[key] = a
    arrays['contact_valid_mask'] = np.arange(maximum)[None, :] < arrays['contact_ncon'][:, None]
    return arrays


def diagnostic_report(arrays):
    t = arrays['time']; p = arrays['com_momentum_world']; f = arrays['external_force_world']
    impulse = np.vstack([np.zeros(3), np.cumsum((f[1:]+f[:-1])*.5*np.diff(t)[:, None], axis=0)])
    dp = p-p[0]; residual = dp-impulse
    fdv = np.gradient(arrays['com_position_world'], t, axis=0, edge_order=2)
    fdp = np.gradient(p, t, axis=0, edge_order=2)
    mask = t <= .0017+1e-12; interior = mask.copy(); interior[0] = False
    interior[-1] = False
    def measure(a, b, which, absolute, relative):
        err = float(np.max(np.abs((a-b)[which]), initial=0))
        scale = max(float(np.max(np.abs(a[which]), initial=0)),
                    float(np.max(np.abs(b[which]), initial=0)), 1e-12)
        rel = err/scale
        return dict(max_abs_residual=err, relative_residual=rel, scale=scale,
                    passed=bool(err <= absolute and rel <= relative))
    report = dict(criteria=CRITERIA,
        impulse=measure(dp, impulse, mask, 1e-4, .01),
        instantaneous_force=measure(arrays['com_mass_times_acceleration_world'], f, np.ones(len(t),dtype=bool), 1e-8, 1e-6),
        velocity_jacobian=measure(arrays['body_linear_velocity_world'], arrays['body_jac_velocity_world'], np.ones(len(t),dtype=bool), 1e-8, 1e-6),
        pointwise_force=measure(fdp, f, interior, 1., .05),
        com_velocity_fd=measure(fdv, arrays['com_velocity_world'], interior, .1, .01),
        endpoint_policy='impulse includes t=0; centered finite differences only for pointwise checks',
        physical_units='NOT_ESTABLISHED', behavior_claim='NONE')
    # Preregistered impulse scale is whole-body mass times maximum COM speed,
    # not gravity impulse or generalized root momentum.
    impulse_scale = max(float(arrays['body_mass'][0].sum()*np.linalg.norm(arrays['com_velocity_world'][mask],axis=1).max()), 1e-12)
    report['impulse']['scale'] = impulse_scale
    report['impulse']['relative_residual'] = report['impulse']['max_abs_residual']/impulse_scale
    report['impulse']['passed'] = bool(report['impulse']['max_abs_residual'] <= 1e-4 and report['impulse']['relative_residual'] <= .01)
    report['passed'] = all(report[k]['passed'] for k in ('impulse', 'instantaneous_force', 'velocity_jacobian', 'pointwise_force', 'com_velocity_fd'))
    arrays.update(com_position_fd_velocity_world=fdv, com_momentum_fd_force_world=fdp,
                  external_impulse_world=impulse, com_momentum_change_world=dp,
                  com_impulse_residual_world=residual)
    return report


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
        Path(__file__).resolve(), FROZEN, ROOT/'run_local_pool_response.py',
        ROOT/'run_neural_initialized_startup.py', checkpoint/'CHECKPOINT.json', checkpoint/'CHECKPOINT.npz',
        ROOT/'run_initialized_translation_training.py', ROOT/'initialized_training_objective.py',
        ROOT/'run_neural_muscle_closure.py', ROOT/'run_native_rk4_gate_training.py',
        ROOT/'run_native_rk4_tick_consumption.py', ROOT/'run_native_rk4_reproducibility.py',
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
    atomic_json(out/'PREREG.json', dict(arms=list(FACTORS), pool=POOL, bin=BIN, zero_based=True,
        factors=FACTORS, bounds=[0, 4], clipping=False, preserve_zero_mask=True,
        actual_pool15_weights={label: w[POOL].tolist() for label, w in arms.items()},
        initialization_ticks=PRE, initialization_ms=40, physical_duration_ms=3,
        physical_steps_per_arm=30, physical_samples_per_arm=31, physical_neural_ticks_per_arm=TICKS,
        dt_ms=.1, neural_dt_ms=.5, sensory_hold_ms=5, source_offset_s=.04,
        same_weights=str(WEIGHTS), source_hashes=identities, expected_calls=EXPECTED,
        closure_criteria=CRITERIA, six_gate_names=GATE_NAMES,
        six_gate_thresholds=dict(height_ratio_min=.85, angle_rad_max=.35, contact_legs_min=3,
             load_ratio_min=.8, nonfoot_depth_max=1e-12, limit_excess_max=1e-6),
        early_window_s=[0, .003], early_load_scale=.8, early_limit_scale=.001,
        early_limit_threshold=1e-6, early_aggregation='trapezoid dx=.0001 divided by .003',
        call_counts='explicit only; initial forward plus one after each step; RK4 internals excluded',
        initialization='held actual initial observation; genuine net.step; computational only, no physics',
        physical_initial='same saved qpos; zero qvel/act/ctrl/external loads; no later pose writes',
        clocks='continuous neural/FIR .04s offset; source END before physical START operator split',
        composite=False, selection=False, deployment=False, checkpoint_replacement=False,
        physical_units='NOT_ESTABLISHED', behavior_claim='NONE', automatic_resume=False))
    atomic_npz(out/'DIRECTION.npz', weights=weights, **arms, pool=np.asarray(POOL),
               bin=np.asarray(BIN), factors=np.asarray(list(FACTORS.values())))
    ledger = dict(model_load=0, forward=0, step=0, neural_ticks=0, active=None, completed=[])
    def reserve(**delta):
        for key, value in delta.items(): ledger[key] += value
        atomic_json(out/'CALL_LEDGER.json', ledger)
    reserve(); results = []
    try:
        built = build_neural_tier(); a = built['data']
        tier = retain_motor_endpoints(a, downstream_union(a, built['global_index'], ['720575941433155799','720575941566493282'], hops=2, max_neurons=20000), ['720575941391252886','720575941511883985','720575941497037218','720575941559306355'])
        roots = a['root_ids'][tier['global_index']]; assert len(roots) == 17099
        edges = build_edges(a, tier, built['cut']); groups, _ = channel_index_map(len(roots), roots)
        reference_before = reference_after = None
        for label, arm_weights in arms.items():
            dest = out/label; dest.mkdir(); ledger['active'] = label
            reserve(model_load=1); m = mj.MjModel.from_xml_path(str(MODEL))
            assert (m.nq, m.nv, m.nu, m.nbody, m.nmesh) == (49,48,90,73,71) and m.opt.timestep == .0001
            before = numeric_model_snapshot(m); m.opt.integrator = mj.mjtIntegrator.mjINT_RK4
            after = numeric_model_snapshot(m)
            assert [key for key in before if before[key] != after[key]] == ['opt.integrator']
            if reference_before is None: reference_before, reference_after = before, after
            assert before == reference_before and after == reference_after
            atomic_json(dest/'MODEL_PROVENANCE.json', dict(before=before, after=after,
                changed_numeric_fields=['opt.integrator'], xml_sha256=sha(MODEL)))
            require_no_callbacks(callback_snapshot(mj))
            d = mj.MjData(m); d.qpos[:] = q  # Sole pose write.
            net = ConductanceNetwork(len(roots), dt_ms=.5)
            net.set_connectivity(edges['We']*.2, edges['Wi']*.2)
            dec = FiniteMotorTranslator(manifest, arm_weights); enc = ChannelSensoryEncoder(groups, len(roots))
            assert not dec.history.any() and net.steps == 0 and net.t_ms == 0
            atomic_npz(dest/'NEURAL_COLD.npz', **{k: v.copy() for k, v in vars(net).items() if isinstance(v, np.ndarray)})
            atomic_json(dest/'NEURAL_COLD_META.json', neural_meta(net))
            connectivity = {}
            for name in ('We', 'Wi'):
                matrix = getattr(net, name).tocsr()
                for field in ('data', 'indices', 'indptr'): connectivity[name+'_'+field] = getattr(matrix, field)
                connectivity[name+'_shape'] = np.asarray(matrix.shape)
            atomic_npz(dest/'CONNECTIVITY.npz', **connectivity)
            joints, geom_leg = {}, {}
            for leg in LEGS:
                js = [j for j in range(m.njnt) if (mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or '').lower().startswith('joint_'+leg)]
                tib = next(j for j in js if 'tibia' in mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j).lower())
                joints[leg] = dict(q=[int(m.jnt_qposadr[j]) for j in js], tib_q=int(m.jnt_qposadr[tib]), tib_v=int(m.jnt_dofadr[tib]))
            for g in range(m.ngeom):
                gn = (mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) or '').lower()
                bn = (mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or '').lower()
                for leg in LEGS:
                    if 'tars' in gn+bn and (gn.startswith(leg) or bn.startswith(leg)): geom_leg[g] = leg
            floor, = np.flatnonzero(m.geom_type == mj.mjtGeom.mjGEOM_PLANE)
            floor_ids = {int(floor)}
            named_floor = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, 'floor')
            if named_floor >= 0: floor_ids.add(int(named_floor))
            limited = np.flatnonzero(m.jnt_limited); qa = m.jnt_qposadr[limited]
            atomic_json(dest/'COM_CONTACT_METADATA.json', dict(
                body_names=[mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, j) for j in range(m.nbody)],
                geom_names=[mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, j) for j in range(m.ngeom)],
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
                actuator_names=[mj.mj_id2name(m, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)],
                joint_names=[mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)],
                jnt_range=m.jnt_range.tolist(), jnt_limited=m.jnt_limited.tolist(),
                jnt_qposadr=m.jnt_qposadr.tolist(), jnt_dofadr=m.jnt_dofadr.tolist(), dof_jntid=m.dof_jntid.tolist()))
            reserve(forward=1); mj.mj_forward(m, d)
            initial_observed = observe_quantities(m, d, sensory_ref, joints, geom_leg)
            initial_current, _ = enc.encode(initial_observed)
            atomic_json(dest/'INITIAL_OBSERVATION.json', raw_observation_record(initial_observed)[0])
            warmsp, warmf, warmu = [], [], []
            for j in range(PRE):
                reserve(neural_ticks=1); sp = net.step(current_nA=initial_current).copy()
                u, info = dec.consume(sp[None,:], spike_dt_s=.0005, start_s=j*.0005, end_s=(j+1)*.0005)
                warmsp.append(sp); warmf.append(info['pool_features']); warmu.append(u)
            release_features = np.stack([dec.history[lo:hi].mean(0) for lo, hi in dec.BINS], axis=1)
            pooled = np.clip((release_features*arm_weights).sum(1), 0, 1)
            release_ctrl = np.zeros(m.nu); mapped = dec.actuator_pool >= 0
            release_ctrl[mapped] = pooled[dec.actuator_pool[mapped]]
            atomic_npz(dest/'INITIALIZATION.npz', spikes=np.asarray(warmsp, dtype=bool), features=np.asarray(warmf), decoded=np.asarray(warmu),
                current=initial_current, history=dec.history, release_features=release_features, release_decoded=release_ctrl,
                weights=arm_weights, body_qpos=d.qpos, body_qvel=d.qvel, body_act=d.act, body_ctrl=d.ctrl,
                **{k: v for k, v in vars(net).items() if isinstance(v, np.ndarray)})
            atomic_json(dest/'NEURAL_RELEASE_META.json', neural_meta(net))
            assert d.time == 0 and np.array_equal(d.qpos, q) and not any(x.any() for x in (d.qvel,d.act,d.ctrl,d.xfrc_applied,d.qfrc_applied))
            def sample():
                contacts = np.zeros(6, dtype=int); force = np.zeros((6,3)); depth = 0.
                for ci, c in enumerate(d.contact):
                    g1, g2 = int(c.geom1), int(c.geom2)
                    if floor not in (g1,g2): continue
                    foot = g2 if g1 == floor else g1
                    if foot in geom_leg:
                        li = LEGS.index(geom_leg[foot]); contacts[li] += int(c.dist < 0)
                        f = np.zeros(6); mj.mj_contactForce(m,d,ci,f)
                        force[li] += (1 if g1 == floor else -1)*c.frame.reshape(3,3).T@f[:3]
                    else: depth = max(depth, -float(c.dist))
                excess = np.maximum(np.maximum(m.jnt_range[limited,0]-d.qpos[qa], d.qpos[qa]-m.jnt_range[limited,1]), 0)
                moments = moment_record(m,d)
                assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
                Ma = np.zeros(m.nv); mj.mj_mulM(m,d,Ma,d.qacc)
                rhs = d.qfrc_actuator+d.qfrc_passive+d.qfrc_constraint+d.qfrc_applied-d.qfrc_bias
                return dict(time=float(d.time), qpos=d.qpos.copy(), qvel=d.qvel.copy(), act=d.act.copy(), ctrl=d.ctrl.copy(),
                    contacts=contacts, foot_force=force, nonfoot_depth=depth, limit_excess=float(excess.max(initial=0)),
                    actuator_force=d.actuator_force.copy(), actuator_length=d.actuator_length.copy(), actuator_velocity=d.actuator_velocity.copy(),
                    qacc=d.qacc.copy(), qfrc_actuator=d.qfrc_actuator.copy(), qfrc_passive=d.qfrc_passive.copy(),
                    qfrc_constraint=d.qfrc_constraint.copy(), qfrc_bias=d.qfrc_bias.copy(), qfrc_applied=d.qfrc_applied.copy(),
                    xfrc_applied=d.xfrc_applied.copy(), Ma=Ma, RHS=rhs.copy(), residual=Ma-rhs,
                    actuator_force_residual=moments['actuator_moment'].T@d.actuator_force-d.qfrc_actuator,
                    warnings=np.asarray([w.number for w in d.warning]), **moments,
                    **com_contact_record(mj,m,d,floor_ids))
            rows = [sample()]; currents, spikes, features, commands, schedule = [], [], [], [], []
            for tick in range(TICKS):
                if tick % 10 == 0:
                    observed = observe_quantities(m,d,sensory_ref,joints,geom_leg)
                    current, _ = enc.encode(observed)
                    atomic_json(dest/f'OBSERVATION_{tick:02d}.json', raw_observation_record(observed)[0])
                currents.append(current.copy()); reserve(neural_ticks=1)
                sp = net.step(current_nA=current).copy()
                start, end = (PRE+tick)*.0005, (PRE+tick+1)*.0005
                u, info = dec.consume(sp[None,:], spike_dt_s=.0005, start_s=start, end_s=end)
                assert abs(net.t_ms/1000-end) < 1e-12 and dec.interval == PRE+tick
                assert abs(d.time-tick*.0005) < 1e-12
                spikes.append(sp); features.append(info['pool_features']); commands.append(u.copy())
                schedule.append(dict(tick=tick, source_start_s=start, source_end_s=end, physical_start_s=tick*.0005,
                    physical_end_s=(tick+1)*.0005, source_offset_s=.04, consume_id=dec.interval))
                d.ctrl[:] = u
                for sub in range(SUBSTEPS):
                    require_no_callbacks(callback_snapshot(mj))
                    assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
                    reserve(step=1,forward=1); mj.mj_step(m,d); mj.mj_forward(m,d)
                    row = sample(); rows.append(row)
                    assert all(np.isfinite(v).all() for v in row.values()) and not row['warnings'].any()
            arrays = stack_rows(rows)
            arrays.update(spikes=np.asarray(spikes), features=np.asarray(features), applied=np.asarray(commands),
                decoded=np.asarray(commands), current=np.asarray(currents), weights=arm_weights, actuator_pool=dec.actuator_pool.copy())
            assert arrays['qpos'].shape == (31,49) and arrays['spikes'].shape == (6,17099)
            assert arrays['actuator_moment'].shape == (31,90,48) and arrays['actuator_moment_raw'].shape == (31,m.nJmom)
            assert np.allclose(arrays['time'], np.arange(31)*.0001, rtol=0, atol=1e-12)
            assert net.steps == 86 and dec.interval == 85 and abs(dec.elapsed_s-.043) < 1e-12
            assert abs(net.t_ms/1000-.043) < 1e-12 and numeric_model_snapshot(m) == after
            weight_native = float(m.body_mass.sum()*abs(m.opt.gravity[2]))
            load = arrays['foot_force'][:,:,2].sum(1)/weight_native
            quat = arrays['qpos'][:,3:7]; norms = np.linalg.norm(quat,axis=1)
            angle = 2*np.arccos(np.clip(np.abs(quat@quat[0])/(norms*norms[0]),0,1))
            gates = np.column_stack((arrays['qpos'][:,2]/q[2] >= .85, angle <= .35,
                np.count_nonzero(arrays['contacts'] > 0,axis=1) >= 3, load >= .8,
                arrays['nonfoot_depth'] <= 1e-12, arrays['limit_excess'] <= 1e-6))
            arrays.update(six_gates=gates, all_six_gates=gates.all(1), load_ratio=load, root_angle_rad=angle)
            closure = diagnostic_report(arrays)
            atomic_npz(dest/'TRAJECTORY.npz', **arrays); atomic_json(dest/'SCHEDULE.json', schedule)
            atomic_json(dest/'COM_CLOSURE.json', closure)
            atomic_json(dest/'NEURAL_FINAL_META.json', neural_meta(net))
            atomic_npz(dest/'NEURAL_FINAL.npz', **{k:v.copy() for k,v in vars(net).items() if isinstance(v,np.ndarray)}, FIR_history=dec.history)
            Jload = float(np.trapezoid((np.maximum(.8-load,0)/.8)**2,dx=.0001)/.003)
            Jlimit = float(np.trapezoid((np.maximum(arrays['limit_excess']-1e-6,0)/.001)**2,dx=.0001)/.003)
            assert np.isfinite([Jload,Jlimit]).all()
            hashes = {str(f):sha(f) for f in dest.iterdir() if f.is_file()}
            record = dict(label=label, factor=FACTORS[label], initialization_ticks=PRE, source_offset_s=.04,
                weight_native=weight_native, Jload=Jload, Jlimit=Jlimit, hashes=hashes,
                trajectory_sha256=sha(dest/'TRAJECTORY.npz'), initialization_sha256=sha(dest/'INITIALIZATION.npz'),
                direction_sha256=sha(out/'DIRECTION.npz'), neural_final_s=net.t_ms/1000,
                FIR_final_s=dec.elapsed_s, physical_final_s=float(d.time), closure=closure,
                behavior_claim='NONE', selection=False)
            results.append(record); atomic_json(dest/'RESULT.json',record); atomic_json(out/'RESULTS.json',results)
            ledger['completed'].append(label); ledger['active'] = None; reserve()
        assert {k:ledger[k] for k in EXPECTED} == EXPECTED
        assert identities == {str(s):sha(s) for s in sources}
        atomic_json(out/'COMPLETE.json',dict(status='COMPLETE', physical_units='NOT_ESTABLISHED', ledger=ledger,
            closure_passed=all(r['closure']['passed'] for r in results),
            deployment=False, selection=False, checkpoint_replacement=False))
        print(json.dumps(results))
    except BaseException:
        atomic_json(out/'FAILED.json',dict(traceback=traceback.format_exc(),ledger=ledger)); raise


if __name__ == '__main__':
    main()
