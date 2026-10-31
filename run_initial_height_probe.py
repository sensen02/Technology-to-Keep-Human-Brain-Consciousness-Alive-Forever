"""S114/P1/S2-S4: one initial-height intervention, native RK4 dt25us, 3 ms.

Frozen derivative of run_com_timestep_probe.py; original sources are read-only.
Import is inert. No model loading or output creation until main(). The height
rule uses only saved reference contacts, before any model load. No replay,
pose holding, activation prefill, threshold relaxation, or checkpoint replacement.
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
from run_com_contact_probe import com_contact_record, stack_rows, GATE_NAMES
from run_com_timestep_probe import diagnostic_report, CRITERIA

PRE, TICKS = 80, 6
TIMESTEPS = {'height_up': .000025}
SUBSTEPS = {'height_up': 20}
EXPECTED = dict(model_load=1, forward=121, step=120, neural_ticks=86)
EXPECTED_PER_ARM = {'height_up': EXPECTED.copy()}
FROZEN = ROOT/'run_com_timestep_probe.py'
REFERENCE = Path('/home/sensen/Desktop/engine修复验收/s113-com-timestep/20261024T060500Z/dt25us/TRAJECTORY.npz')
OUTPUT_PARENT = Path('/home/sensen/Desktop/engine修复验收/s114-initial-height/20261024T061500Z')
# Same order used in reference observe_quantities and per-leg force arrays.
FOOT_ORDER = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
NORMAL_ATOL = 1e-12
REFERENCE_ARTIFACTS = ('TRAJECTORY.npz', 'COM_CONTACT_METADATA.json',
    'MOMENT_METADATA.json', 'RESULT.json', 'MODEL_PROVENANCE.json',
    'INITIAL_OBSERVATION.json', 'INITIALIZATION.npz', 'NEURAL_COLD.npz',
    'NEURAL_COLD_META.json', 'NEURAL_RELEASE_META.json', 'CONNECTIVITY.npz',
    'SCHEDULE.json', 'COM_CLOSURE.json', 'NEURAL_FINAL_META.json', 'NEURAL_FINAL.npz')


def local_weights(weights):
    weights = np.asarray(weights)
    if weights.shape != (66, 5) or not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 4)):
        raise ValueError('invalid source W; expected finite (66,5) in [0,4]')
    return {'height_up': weights.copy()}


def initial_height_plan(reference=REFERENCE):
    """Pure saved-array calculation; never calls MuJoCo or a neural network.

    The contact frame normal points geom1 -> geom2. Apply the SAME external
    sign to the raw force and to that normal to obtain ON-fly world values.
    Reconstruct both side classifications rather than trusting padding/labels.
    No eligible contact means dz is undefined, not a zero/default perturbation.
    """
    reference = Path(reference)
    meta = json.loads((reference.parent/'COM_CONTACT_METADATA.json').read_text())
    names, bodies = meta['geom_names'], meta['body_names']
    geom_body = np.asarray(meta['geom_bodyid'], dtype=int)
    floor_ids = set(meta['floor_geom_ids'])
    feet = {}
    for g, name in enumerate(names):
        gn = (name or '').lower(); bn = (bodies[int(geom_body[g])] or '').lower()
        for leg in FOOT_ORDER:
            if 'tars' in gn+bn and (gn.startswith(leg) or bn.startswith(leg)):
                feet[g] = leg
    if not feet or not floor_ids:
        raise ValueError('cannot define dz: missing reference floor/foot metadata')
    with np.load(reference, allow_pickle=False) as z:
        if float(z['time'][0]) != 0:
            raise ValueError('reference initial sample is not time zero')
        old = z['qpos'][0].copy()
        if old.shape != (49,) or not np.isfinite(old).all():
            raise ValueError('invalid reference initial qpos')
        for key in ('qvel', 'act', 'ctrl', 'qfrc_applied', 'xfrc_applied'):
            if z[key][0].any(): raise ValueError('reference release is not zero '+key)
        valid = z['contact_valid_mask'][0].astype(bool)
        if int(valid.sum()) != int(z['contact_ncon'][0]):
            raise ValueError('invalid reference contact padding')
        selected = []
        for ci in np.flatnonzero(valid):
            geom = z['contact_geom_ids'][0, ci].astype(int)
            body = z['contact_body_ids'][0, ci].astype(int)
            if np.any(geom < 0) or np.any(geom >= len(names)) or not np.array_equal(body, geom_body[geom]):
                raise ValueError('invalid reference contact geom/body IDs')
            fixed = [int(g) in floor_ids or int(b) == 0 for g, b in zip(geom, body)]
            external = bool(fixed[0] != fixed[1])
            sign = (1. if fixed[0] else -1.) if external else 0.
            if bool(z['contact_external_mask'][0, ci]) != external or float(z['contact_external_sign'][0, ci]) != sign:
                raise ValueError('reference external mask/sign mismatch')
            if not external: continue
            fly_side = 1 if fixed[0] else 0
            foot = int(geom[fly_side]); ground = int(geom[1-fly_side])
            if foot not in feet or not (ground in floor_ids or int(body[1-fly_side]) == 0): continue
            frame = z['contact_frame'][0, ci].copy()
            raw = z['contact_force_raw6'][0, ci].copy()
            on_fly = sign*(frame.T @ raw[:3])
            if not np.allclose(frame.T @ raw[:3], z['contact_force_on_geom2_world'][0, ci], rtol=0, atol=1e-12):
                raise ValueError('reference raw/world force mismatch')
            dist = float(z['contact_dist'][0, ci])
            normal = sign*frame[0]
            if not np.isfinite(np.r_[raw, frame.ravel(), on_fly, normal, dist]).all():
                raise ValueError('nonfinite reference contact')
            if not (on_fly[2] > 0 and dist < 0): continue
            if not np.allclose(normal, [0., 0., 1.], rtol=0, atol=NORMAL_ATOL):
                raise ValueError('cannot define dz: loadbearing foot-ground normal is not world +z')
            selected.append(dict(contact_id=int(ci), geom_ids=geom.tolist(), body_ids=body.tolist(),
                geom_names=[names[int(g)] for g in geom], body_names=[bodies[int(b)] for b in body],
                fly_side=fly_side, foot_geom_id=foot, ground_geom_id=ground, leg=feet[foot],
                dist=dist, positive_depth=-dist, raw_force6=raw.tolist(), external_sign=sign,
                force_world_on_fly=on_fly.tolist(), raw_Fz_world_on_fly=float(on_fly[2]),
                frame=frame.tolist(), normal_world_on_fly=normal.tolist(),
                contact_valid=True, external=True))
    if not selected:
        raise ValueError('cannot define dz: no valid penetrating loadbearing foot-ground contacts; no arbitrary substitute')
    depth = min(c['positive_depth'] for c in selected)
    dz = .5*depth
    if not np.isfinite(dz) or dz <= 0: raise ValueError('cannot define positive dz')
    new = old.copy(); new[2] += dz
    if not np.array_equal(old[np.arange(49) != 2], new[np.arange(49) != 2]):
        raise AssertionError('non-height pose intervention')
    return dict(reference=str(reference), reference_sha256=sha(reference), reference_sample=0,
        foot_order=list(FOOT_ORDER), foot_geom_leg_map={str(k):v for k,v in feet.items()},
        floor_geom_ids=sorted(floor_ids), normal_world_required=[0., 0., 1.],
        normal_atol=NORMAL_ATOL, normal_rtol=0.,
        selection='valid AND external floor/world-to-tarsus foot AND signed raw world ON-fly Fz>0 AND dist<0; require signed normal +z',
        selected_contacts=selected, selected_contact_ids=[c['contact_id'] for c in selected],
        minimum_positive_depth=depth, rule='dz = 0.5 * minimum positive penetration depth among selected contacts',
        dz=dz, initial_qpos_old=old.tolist(), initial_qpos_new=new.tolist(),
        changed_qpos_indices=[2], no_arbitrary_substitute=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    out = p.parse_args().output
    if not out.is_absolute() or out.exists(): raise ValueError('absolute fresh --output required; no resume')
    # This computation is wholly offline and MUST precede any model loading.
    height_plan = initial_height_plan()
    q = np.asarray(height_plan['initial_qpos_new'], dtype=float)
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
    assert tuple(LEGS) == FOOT_ORDER
    dependencies = mesh_dependencies()
    checkpoint = Path('/home/sensen/Desktop/engine修复验收/s109-initialized-translation-training/20261024T045944Z')
    reference_inputs = json.loads((REFERENCE.parent.parent/'INPUT_HASHES.json').read_text())
    sources = list(dict.fromkeys([MODEL, INITIAL, WEIGHTS, ENDPOINTS/'MANIFEST.json',
        ENDPOINTS/'FROZEN_WEIGHTS.json', ROOT/'data/flywire/banc_connectome.npz',
        ROOT/'outputs/neural_rebuild/sensory_channel_lookup.json', MOMENT_HEADER,
        Path(mj.__file__).resolve().parent/'include/mujoco/mujoco.h',
        Path(__file__).resolve(), FROZEN, ROOT/'run_com_contact_probe.py', ROOT/'run_pool_synergy_response.py',
        ROOT/'run_local_pool_response.py', ROOT/'run_neural_initialized_startup.py',
        checkpoint/'CHECKPOINT.json', checkpoint/'CHECKPOINT.npz',
        ROOT/'run_initialized_translation_training.py', ROOT/'initialized_training_objective.py',
        ROOT/'run_neural_muscle_closure.py', ROOT/'run_native_rk4_gate_training.py',
        ROOT/'run_native_rk4_tick_consumption.py', ROOT/'run_native_rk4_reproducibility.py',
        REFERENCE.parent.parent/'INPUT_HASHES.json', REFERENCE.parent.parent/'PREREG.json',
        REFERENCE.parent.parent/'COMPLETE.json',
        *[REFERENCE.parent/name for name in REFERENCE_ARTIFACTS],
        *[Path(name) for name in reference_inputs], *ROOT.glob('engine/**/*.py'), *dependencies]))
    identities = {str(s): sha(s) for s in sources}
    # Frozen reference source chain must still match, not silently re-anchor.
    if any(identities[name] != digest for name, digest in reference_inputs.items()):
        raise ValueError('reference input/source identity changed')
    if height_plan['reference_sha256'] != identities[str(REFERENCE)]: raise ValueError('reference changed during offline plan')
    reference_result = json.loads((REFERENCE.parent/'RESULT.json').read_text())
    if reference_result['trajectory_sha256'] != identities[str(REFERENCE)]: raise ValueError('reference trajectory hash mismatch')
    for name, digest in reference_result['hashes'].items():
        if sha(Path(name)) != digest: raise ValueError('reference artifact changed: '+name)
    with np.load(WEIGHTS, allow_pickle=False) as z: weights = z['weights'].copy()
    arms = local_weights(weights)
    with np.load(INITIAL, allow_pickle=False) as z:
        sensory_ref = z['sensory_reference_qpos'].copy()
        assert np.array_equal(z['new_qpos'], height_plan['initial_qpos_old'])
    manifest = json.loads((ENDPOINTS/'MANIFEST.json').read_text())
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'INPUT_HASHES.json', identities)
    atomic_json(out/'MODEL_DEPENDENCIES.json', {str(s): identities[str(s)] for s in dependencies})
    atomic_json(out/'REFERENCE_ARTIFACTS.json', dict(reference_root=str(REFERENCE.parent),
        artifacts={str(REFERENCE.parent/name):identities[str(REFERENCE.parent/name)] for name in REFERENCE_ARTIFACTS},
        purpose='retained read-only dt25us baseline release, raw observations, cold/live neural/FIR states, connectivity, trajectory and model for comparison verifier'))
    atomic_json(out/'PREREG.json', dict(arms=list(TIMESTEPS), factors={'height_up':1.},
        height_plan=height_plan, bounds=[0,4], clipping=False, preserve_zero_mask=True,
        all_weights_identical=True, initialization_ticks=PRE, initialization_ms=40, physical_duration_ms=3,
        physical_steps_per_arm={'height_up':120}, physical_samples_per_arm={'height_up':121},
        physical_neural_ticks_per_arm=TICKS, timesteps_s=TIMESTEPS, substeps=SUBSTEPS,
        dt_ms={'height_up':.025}, neural_dt_ms=.5, sensory_hold_ms=5, source_offset_s=.04,
        same_weights=str(WEIGHTS), source_hashes=identities, expected_calls=EXPECTED,
        expected_calls_per_arm=EXPECTED_PER_ARM, closure_criteria=CRITERIA, six_gate_names=GATE_NAMES,
        retained_dt25us_reference=str(REFERENCE), retained_reference_sha256=sha(REFERENCE),
        planned_output_parent=str(OUTPUT_PARENT),
        six_gate_thresholds=dict(height_ratio_min=.85, angle_rad_max=.35, contact_legs_min=3,
            load_ratio_min=.8, nonfoot_depth_max=1e-12, limit_excess_max=1e-6),
        early_window_s=[0,.003], early_load_scale=.8, early_limit_scale=.001,
        early_limit_threshold=1e-6, early_aggregation='trapezoid actual time divided by .003',
        call_counts='explicit only; initial forward plus one after each step; RK4 internals excluded',
        model_changes_vs_xml=['opt.integrator','opt.timestep'], model_changes_vs_reference=[],
        initial_condition_change='reference initial qpos copy; qpos[2] += dz only; all joint angles unchanged',
        initialization='fresh cold net/FIR; held ACTUAL NEW initial observation encoded anew; genuine net.step; computational only, no physics',
        physical_initial='new saved qpos; zero qvel/act/ctrl/external loads; sole initial pose write, no later pose writes',
        clocks='continuous neural/FIR .04s offset; source END before physical START operator split',
        capture_completion='COMPLETE even if scientific diagnostic/gates fail; no retries or threshold relaxation',
        composite=False, selection=False, deployment=False, checkpoint_replacement=False,
        physical_units='NOT_ESTABLISHED', behavior_claim='NONE', automatic_resume=False))
    atomic_npz(out/'DIRECTION.npz', weights=weights, **arms, timesteps_s=np.asarray([.000025]),
        factors=np.ones(1), dz=np.asarray(height_plan['dz']),
        initial_qpos_old=np.asarray(height_plan['initial_qpos_old']), initial_qpos_new=q)
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
        reference_provenance = json.loads((REFERENCE.parent/'MODEL_PROVENANCE.json').read_text())
        for label, arm_weights in arms.items():
            dt, substeps = TIMESTEPS[label], SUBSTEPS[label]; samples = TICKS*substeps+1
            dest = out/label; dest.mkdir(); ledger['active'] = label
            ledger['per_arm'][label] = {k:0 for k in EXPECTED}
            reserve(model_load=1); m = mj.MjModel.from_xml_path(str(MODEL))
            assert (m.nq,m.nv,m.nu,m.nbody,m.nmesh) == (49,48,90,73,71) and m.opt.timestep == .0001
            before = numeric_model_snapshot(m)
            m.opt.integrator = mj.mjtIntegrator.mjINT_RK4; m.opt.timestep = dt
            after = numeric_model_snapshot(m)
            changed = sorted(key for key in before if before[key] != after[key])
            assert changed == ['opt.integrator','opt.timestep']
            assert before == reference_provenance['before'] and after == reference_provenance['after']
            atomic_json(dest/'MODEL_PROVENANCE.json', dict(before=before, after=after,
                changed_numeric_fields=changed, changed_vs_reference=[], numerical_match_to_reference=True,
                xml_sha256=sha(MODEL)))
            require_no_callbacks(callback_snapshot(mj))
            d = mj.MjData(m); d.qpos[:] = q  # Sole pose write.
            net = ConductanceNetwork(len(roots), dt_ms=.5)
            net.set_connectivity(edges['We']*.2, edges['Wi']*.2)
            dec = FiniteMotorTranslator(manifest, arm_weights); enc = ChannelSensoryEncoder(groups, len(roots))
            assert not dec.history.any() and net.steps == 0 and net.t_ms == 0
            atomic_npz(dest/'NEURAL_COLD.npz', **{k:v.copy() for k,v in vars(net).items() if isinstance(v,np.ndarray)})
            atomic_json(dest/'NEURAL_COLD_META.json', neural_meta(net))
            connectivity = {}
            for name in ('We','Wi'):
                matrix = getattr(net,name).tocsr()
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
            assert {str(k):v for k,v in geom_leg.items()} == height_plan['foot_geom_leg_map']
            floor, = np.flatnonzero(m.geom_type == mj.mjtGeom.mjGEOM_PLANE)
            floor_ids = {int(floor)}
            named_floor = mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'floor')
            if named_floor >= 0: floor_ids.add(int(named_floor))
            assert sorted(floor_ids) == height_plan['floor_geom_ids']
            limited = np.flatnonzero(m.jnt_limited); qa = m.jnt_qposadr[limited]
            atomic_json(dest/'COM_CONTACT_METADATA.json', dict(
                body_names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,j) for j in range(m.nbody)],
                geom_names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,j) for j in range(m.ngeom)],
                geom_bodyid=m.geom_bodyid.tolist(), floor_geom_ids=sorted(floor_ids), gravity_world=m.opt.gravity.tolist(),
                foot_order=list(LEGS), foot_geom_leg_map={str(k):v for k,v in geom_leg.items()},
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
            record = dict(label=label,factor=1.,dz=height_plan['dz'],timestep_s=dt,substeps=substeps,samples=samples,
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
