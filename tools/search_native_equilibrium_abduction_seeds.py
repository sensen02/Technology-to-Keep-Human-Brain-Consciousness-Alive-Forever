"""Preregistered four-seed abducted-footprint static equilibrium test.

Each of the four SEED_TABLE.json seeds gets exactly one identical least_squares run over
109 variables (42 native hinge angles + root z + 66 shared muscle pools), with the round
6/7 objective, bounds and hard gates unchanged. No plant edit, no mj_step, no rollout,
no neural call, no render.
"""
from pathlib import Path
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import sys, json, hashlib, time, traceback
import numpy as np
import mujoco
from scipy.optimize import least_squares
from flygym.compose.fly.musculoskeletal import _load_mjcf

S = Path('/home/sensen/Desktop/engine修复验收/s5-detailed-handover-measurement/20261028T121000Z')
O = Path('/home/sensen/Desktop/engine修复验收/s5-equilibrium-abduction-seeds/20261101T201309Z')
R6 = Path('/home/sensen/Desktop/engine修复验收/s5-native-equilibrium-search/20261101T193630Z')
EXPECTED = '3470c72f7e68f181ec1a80a5dc5fbff182b750091e7a1ddc7ccfbfb590b80cb6'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
FACTORS = (0.0, 0.25, 0.5, 0.75)
REF_H_OVER_V = 0.19921933195031086
SCIPY_KEYS = ('method', 'max_nfev', 'jac', 'diff_step', 'ftol', 'xtol', 'gtol', 'x_scale', 'loss')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False,
                            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()) + '\n')


def parameter_digest(m):
    rows = {}
    for prefix, obj in [('model', m), ('option', m.opt)]:
        for name in dir(obj):
            if name.startswith('_'):
                continue
            value = getattr(obj, name)
            if isinstance(value, np.ndarray):
                a = np.ascontiguousarray(value)
                rows[prefix + '.' + name] = dict(shape=list(a.shape), dtype=str(a.dtype),
                                                 sha256=hashlib.sha256(a.tobytes()).hexdigest())
            elif isinstance(value, (int, float, bool, str)):
                rows[prefix + '.' + name] = value
    encoded = json.dumps(rows, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest(), rows


def safe_div(num, den):
    if float(den) == 0.0:
        return None
    return float(num) / float(den)


def joint_limit_table(q42, names, lower, upper):
    d_lo = q42 - lower
    d_hi = upper - q42
    sides = np.where(d_lo < d_hi, 'lower', 'upper')
    near = np.minimum(d_lo, d_hi)
    rows = [dict(index=int(i), joint=str(names[i]), q=float(q42[i]), lower=float(lower[i]),
                 upper=float(upper[i]), distance_to_lower=float(d_lo[i]), distance_to_upper=float(d_hi[i]),
                 nearest_side=str(sides[i]), nearest_distance=float(near[i]),
                 is_coxa_roll_yaw=bool('Coxa' in str(names[i]) and str(names[i]).split('_')[-1] in ('roll', 'yaw')))
            for i in range(len(q42))]
    order = np.argsort(near, kind='stable')
    coxa = [r_ for r_ in rows if r_['is_coxa_roll_yaw']]
    return dict(all42=rows, nearest8=[rows[int(i)] for i in order[:8]],
                min_distance=float(near.min()), argmin_joint=str(names[int(np.argmin(near))]),
                argmin_side=str(sides[int(np.argmin(near))]),
                joints_within_0p01=int(np.sum(near <= .01)), joints_within_0p05=int(np.sum(near <= .05)),
                coxa_roll_yaw_min=float(min(r_['nearest_distance'] for r_ in coxa)),
                coxa_roll_yaw_within_0p01=int(sum(r_['nearest_distance'] <= .01 for r_ in coxa)))


def wrench_report(foot_force, contact_pos, contact_force, weight, com, root_pos):
    net = foot_force.sum(0)
    net_rows = contact_force.sum(0) if len(contact_force) else np.zeros(3)
    about_com = np.cross(contact_pos - com, contact_force).sum(0) if len(contact_force) else np.zeros(3)
    about_root = np.cross(contact_pos - root_pos, contact_force).sum(0) if len(contact_force) else np.zeros(3)
    expected = np.array([0., 0., weight])
    return dict(
        net_foot_force=net, net_contact_force_from_rows=net_rows,
        rows_minus_legs_abs=float(np.max(np.abs(net - net_rows))) if len(contact_force) else None,
        per_leg_vertical_force=foot_force[:, 2],
        zero_or_negative_leg_count=int(np.sum(foot_force[:, 2] <= 0)),
        vertical_over_weight=safe_div(net[2], weight),
        horizontal_norm=float(np.linalg.norm(net[:2])),
        horizontal_over_vertical=safe_div(np.linalg.norm(net[:2]), net[2]),
        horizontal_components=net[:2].tolist(),
        moment_about_candidate_com=about_com, moment_about_root=about_root,
        moment_norm_over_weight_com=safe_div(np.linalg.norm(about_com), weight),
        moment_norm_over_weight_root=safe_div(np.linalg.norm(about_root), weight),
        equilibrium_expectation='static equilibrium requires net contact force = (0,0,+W) and zero net moment '
                                'about the body centre of mass',
        force_error_vector=(net - expected).tolist(),
        force_error_norm_over_weight=float(np.linalg.norm(net - expected) / weight))


def main():
    assert (O / 'PREREGISTRATION.json').is_file(), 'Preregistration must exist before execution'
    assert (O / 'SEED_TABLE.json').is_file(), 'Seed table must exist before execution'
    assert not (O / 'EXECUTION.json').exists() and not (O / 'EVALUATIONS.jsonl').exists(), 'Never restart this attempt'
    prereg = json.loads((O / 'PREREGISTRATION.json').read_text())
    table = json.loads((O / 'SEED_TABLE.json').read_text())
    assert sha(O / 'SEED_TABLE.json') == prereg['seed_table_sha256'], 'Seed table drifted from the preregistered hash'
    assert [r['retained_fraction_k'] for r in table['seeds']] == list(FACTORS)
    assert all(k in prereg['solver'] for k in SCIPY_KEYS)
    solver_kwargs = {k: prereg['solver'][k] for k in SCIPY_KEYS}
    dump(O / 'SOLVER_KWARGS.json', solver_kwargs)
    source = Path(__file__).resolve()
    (O / 'EXECUTED_search_native_equilibrium_abduction_seeds.py').write_bytes(source.read_bytes())
    inputs = {str(p): sha(p) for p in [S / 'MODEL.xml', S / 'BODY.npz', S / 'MAPPING.json',
                                       O / 'PREREGISTRATION.json', O / 'SEED_TABLE.json', source]}
    dump(O / 'INPUT_IDENTITIES.json', inputs)
    counts = dict(model_load_attempt=0, model_load=0, mj_resetData=0, forward=0, optimizer_callbacks=0,
                  final_validation=0, geom_distance=0, contact_force=0, least_squares=0,
                  body_step=0, neural_step=0, render=0)
    started = time.time()
    error = None
    per_seed = {}
    classification = None
    any_pass = False
    model_before = model_after = None
    model_unchanged = False
    m = None
    try:
        assert inputs[str(S / 'MODEL.xml')] == EXPECTED
        counts['model_load_attempt'] += 1
        m = _load_mjcf(str(S / 'MODEL.xml')).compile()
        counts['model_load'] += 1
        assert (m.nq, m.nv, m.nu, m.na) == (49, 48, 90, 90)
        model_before, model_rows_before = parameter_digest(m)
        dump(O / 'MODEL_PARAMETERS_BEFORE.json', dict(sha256=model_before, fields=model_rows_before))
        d = mujoco.MjData(m)
        with np.load(S / 'BODY.npz') as z:
            seed_index = int(np.argmin(abs(z['time'] - .15)))
            seed_time = float(z['time'][seed_index])
            qseed = z['qpos'][seed_index].copy()
        assert abs(seed_time - .15) < 1e-10
        hinges = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
        assert len(hinges) == 42 and np.all(m.jnt_limited[hinges])
        qa = m.jnt_qposadr[hinges].astype(int)
        va = m.jnt_dofadr[hinges].astype(int)
        assert np.array_equal(np.sort(qa), np.arange(7, 49))
        assert np.array_equal(m.actuator_actadr, np.arange(90))
        hinge_names = [m.joint(int(j)).name for j in hinges]
        hlo = m.jnt_range[hinges, 0].copy()
        hhi = m.jnt_range[hinges, 1].copy()
        groups = {}
        mapping = json.loads((S / 'MAPPING.json').read_text())['rows']
        for row in mapping:
            assert m.actuator(int(row['actuator_index'])).name == row['actuator']
            if row['local_indices']:
                groups.setdefault(tuple(row['local_indices']), []).append(int(row['actuator_index']))
        assert len(groups) == 66
        G = np.zeros((90, 66))
        for j, ids in enumerate(groups.values()):
            G[ids, j] = 1
        mapped = G.sum(1) > 0
        assert mapped.sum() == 74 and np.all(G.sum(1) <= 1) and int((~mapped).sum()) == 16
        floor_id = m.geom('floor').id
        assert m.geom_type[floor_id] == mujoco.mjtGeom.mjGEOM_PLANE
        foot = {}
        for g in range(m.ngeom):
            name = m.geom(g).name
            bn = m.body(int(m.geom_bodyid[g])).name
            legs = [i for i, L in enumerate(LEGS) if (name.startswith(L) or bn.startswith(L)) and 'Tarsus' in name + bn]
            if len(legs) == 1:
                foot[g] = legs[0]
        assert len(foot) == 30
        nonfoot = np.array([g for g in range(m.ngeom)
                            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and g not in foot], int)
        assert len(nonfoot) == 41
        side_bodies, side_sign = [], []
        for b in range(m.nbody):
            name = m.body(b).name
            if any(name.startswith(L) for L in LEGS) and ('Tibia' in name or 'Tarsus' in name):
                side_bodies.append(b)
                side_sign.append(1. if name.startswith(('LF', 'LM', 'LH')) else -1.)
        side_bodies = np.array(side_bodies, int)
        side_sign = np.array(side_sign)
        assert len(side_bodies) == 36
        thorax = m.body('Thorax').id
        weight = float(m.body_mass.sum() * abs(m.opt.gravity[2]))
        seed_pose = np.r_[qseed[qa], qseed[2]]
        zlo, zhi = qseed[2] - .25, qseed[2] + .25
        plo, phi = .0001, 1.
        lower = np.r_[hlo, zlo, np.full(66, plo)]
        upper = np.r_[hhi, zhi, np.full(66, phi)]
        dump(O / 'VARIABLES.json', dict(
            seed_index=seed_index, seed_time=seed_time, hinge_names=hinge_names,
            hinge_qpos_addresses=qa, hinge_dof_addresses=va, joint_lower=hlo, joint_upper=hhi,
            root_z_range=[zlo, zhi], pool_range=[plo, phi], pool_initial=.1, lower=lower, upper=upper,
            seed_pose_43=seed_pose, group_matrix=G, mapped_actuator_mask=mapped,
            foot_geom_to_leg={m.geom(g).name: LEGS[l] for g, l in foot.items()},
            nonfoot_geoms=[m.geom(int(g)).name for g in nonfoot],
            side_bodies=[m.body(int(b)).name for b in side_bodies],
            side_frame='Thorax local y axis, LF/LM/LH positive, RF/RM/RH negative',
            weight_native=weight, objective=prereg['objective'], solver=prereg['solver'],
            variable_layout='x[0:42]=42 hinge angles in hinge_names order; x[42]=root z; x[43:109]=66 pool activations'))
        eval_log = O / 'EVALUATIONS.jsonl'

        def evaluate_factory(k):
            state = dict(best=None, best_feasible=None, evals=0)

            def evaluate(x, role='optimizer', retain=False):
                if role == 'optimizer':
                    counts['optimizer_callbacks'] += 1
                else:
                    counts['final_validation'] += 1
                state['evals'] += 1
                mujoco.mj_resetData(m, d)
                counts['mj_resetData'] += 1
                d.qpos[:] = qseed
                d.qpos[qa] = x[:42]
                d.qpos[2] = x[42]
                d.qvel[:] = 0
                activation = np.full(90, plo)
                activation[mapped] = (G @ x[43:])[mapped]
                d.act[:] = activation
                d.ctrl[:] = activation
                d.qfrc_applied[:] = 0
                d.xfrc_applied[:] = 0
                mujoco.mj_forward(m, d)
                counts['forward'] += 1
                distances = np.array([mujoco.mj_geomDistance(m, d, int(g), floor_id, 1000., None) for g in nonfoot])
                counts['geom_distance'] += len(nonfoot)
                yaxis = d.xmat[thorax].reshape(3, 3)[:, 1]
                side = ((d.xpos[side_bodies] - d.xpos[thorax]) @ yaxis) * side_sign
                ff = np.zeros((6, 3))
                nf = np.zeros(3)
                nonfoot_max = 0.
                contact_rows, contact_frames, contact_local, contact_all, pairs = [], [], [], [], []
                for ci in range(d.ncon):
                    c = d.contact[ci]
                    g1, g2 = int(c.geom1), int(c.geom2)
                    pairs.append('%d-%d' % (g1, g2))
                    f = np.zeros(6)
                    mujoco.mj_contactForce(m, d, ci, f)
                    counts['contact_force'] += 1
                    force = c.frame.reshape(3, 3).T @ f[:3] * (1 if g1 == floor_id else -1)
                    is_floor = floor_id in (g1, g2)
                    contact_all.append(np.r_[g1, g2, c.dist, c.pos, force, float(is_floor)])
                    if not is_floor:
                        continue
                    contact_frames.append(np.asarray(c.frame).copy())
                    contact_local.append(f.copy())
                    other = g2 if g1 == floor_id else g1
                    if other in foot:
                        ff[foot[other]] += force
                    else:
                        nf += force
                        nonfoot_max = max(nonfoot_max, float(np.max(abs(force))))
                    contact_rows.append(np.r_[g1, g2, c.dist, c.pos, force])
                topology = 'ncon=%d|%s' % (d.ncon, ';'.join(sorted(pairs)))
                load = float(ff[:, 2].sum() / weight)
                r_qacc = d.qacc / 9810.
                r_dist = 100 * np.maximum(0., .02 - distances)
                r_side = 100 * np.maximum(0., .02 - side)
                r_load = np.array([10 * (load - 1)])
                r_reg = 1e-6 * (x[:43] - seed_pose)
                r = np.r_[r_qacc, r_dist, r_side, r_load, r_reg]
                if not np.isfinite(r).all() or not np.isfinite(d.actuator_force).all():
                    np.savez_compressed(O / ('NONFINITE_FAILURE_k%s.npz' % k), x=x, qpos=d.qpos, qacc=d.qacc, residual=r)
                    raise FloatingPointError('Nonfinite static evaluation')
                cost = float(r @ r)
                blocks = dict(
                    qacc48=dict(n=48, sse=float(r_qacc @ r_qacc), n_nonzero=int(np.count_nonzero(r_qacc))),
                    nonfoot41=dict(n=41, sse=float(r_dist @ r_dist), n_nonzero=int(np.count_nonzero(r_dist))),
                    side36=dict(n=36, sse=float(r_side @ r_side), n_nonzero=int(np.count_nonzero(r_side))),
                    load1=dict(n=1, sse=float(r_load @ r_load), n_nonzero=int(np.count_nonzero(r_load))),
                    regularizer43=dict(n=43, sse=float(r_reg @ r_reg), n_nonzero=int(np.count_nonzero(r_reg))))
                row = dict(seed_k=float(k), evaluation=counts['forward'], seed_evaluation=state['evals'], role=role,
                           cost=cost, blocks=blocks, block_sse_sum=float(sum(b['sse'] for b in blocks.values())),
                           max_abs_qacc=float(abs(d.qacc).max()), min_nonfoot_distance=float(distances.min()),
                           min_side=float(side.min()), foot_load_ratio=load, nonfoot_force_max_abs=nonfoot_max,
                           act_dot_max_abs=float(abs(d.act_dot).max()), warnings=int(d.warning.number.sum()),
                           geom_distance_cutoff_hits=int(np.sum(distances >= 1000.)), ncon=int(d.ncon),
                           contact_topology=topology, solver_niter=int(d.solver_niter[0]),
                           x=[float(v) for v in x])
                observed_feasible = (row['max_abs_qacc'] <= .1 and np.all(distances >= .02) and np.all(side >= .02)
                                     and nonfoot_max == 0 and .99 <= load <= 1.01
                                     and row['act_dot_max_abs'] <= 1e-10 and row['warnings'] == 0
                                     and np.all(x >= lower) and np.all(x <= upper)
                                     and row['geom_distance_cutoff_hits'] == 0)
                if role == 'optimizer' and observed_feasible and (state['best_feasible'] is None
                                                                or cost < state['best_feasible']['cost']):
                    state['best_feasible'] = dict(cost=cost, evaluation=counts['forward'],
                                                  scope='observed physics gates only; not selected instead of the '
                                                        'minimum objective; independent validation still required')
                row['observed_physics_feasible'] = bool(observed_feasible)
                with eval_log.open('a') as f:
                    f.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
                if role == 'optimizer' and (state['best'] is None or cost < state['best']['cost']):
                    state['best'] = dict(cost=cost, x=x.copy(), evaluation=counts['forward'])
                    np.savez_compressed(O / ('BEST_INCREMENTAL_k%s.npz' % k), x=x, residual=r, qpos=d.qpos,
                                        qvel=d.qvel, activation=d.act, ctrl=d.ctrl, qacc=d.qacc, foot_force=ff,
                                        nonfoot_distances=distances, body_side=side, evaluation=counts['forward'],
                                        cost=cost)
                if retain:
                    moment = np.zeros((m.nu, m.nv))
                    assert len(d.moment_rownnz) == m.nu
                    for j in range(m.nu):
                        s0 = int(d.moment_rowadr[j])
                        n = int(d.moment_rownnz[j])
                        moment[j, d.moment_colind[s0:s0 + n]] = d.actuator_moment[s0:s0 + n]
                    com = (m.body_mass[:, None] * d.xipos).sum(0) / m.body_mass.sum()
                    return r, row, dict(
                        x=x.copy(), qpos=d.qpos.copy(), qvel=d.qvel.copy(), activation=d.act.copy(),
                        ctrl=d.ctrl.copy(), act_dot=d.act_dot.copy(), qacc=d.qacc.copy(),
                        qfrc_actuator=d.qfrc_actuator.copy(), qfrc_passive=d.qfrc_passive.copy(),
                        qfrc_bias=d.qfrc_bias.copy(), qfrc_constraint=d.qfrc_constraint.copy(),
                        qfrc_applied=d.qfrc_applied.copy(), xfrc_applied=d.xfrc_applied.copy(),
                        actuator_force=d.actuator_force.copy(), foot_force=ff, nonfoot_force=nf,
                        nonfoot_distances=distances, body_side=side,
                        contact_rows=np.array(contact_rows, float).reshape(-1, 9),
                        contact_frames=np.array(contact_frames, float).reshape(-1, 9),
                        contact_local_force=np.array(contact_local, float).reshape(-1, 6),
                        contact_all=np.array(contact_all, float).reshape(-1, 10),
                        residual=r.copy(), moment=moment, com_world=com, root_pos=d.qpos[:3].copy(),
                        thorax_pos=d.xpos[thorax].copy(), thorax_yaxis=yaxis.copy(),
                        side_body_pos=d.xpos[side_bodies].copy(), side_sign=side_sign.copy(),
                        topology=topology, solver_niter=int(d.solver_niter[0]), ncon=int(d.ncon))
                return r
            evaluate.state = state
            return evaluate

        for factor, srow in zip(FACTORS, table['seeds']):
            k = float(factor)
            x0 = np.asarray(srow['x0_109'], float)
            assert np.all(x0 >= lower) and np.all(x0 <= upper), 'preregistered seed outside declared bounds'
            evaluate = evaluate_factory(k)
            record = dict(retained_fraction_k=k, n_seed_projections=int(srow['n_projections']),
                          seed_projections=srow['projections'], x0=x0, solver=None, final=None,
                          gate=None, gate_passed=False, solver_error=None, evaluations=0)
            try:
                counts['least_squares'] += 1
                sol = least_squares(evaluate, x0, bounds=(lower, upper), **solver_kwargs)
                record['solver'] = dict(success=bool(sol.success), status=int(sol.status), message=str(sol.message),
                                        nfev=int(sol.nfev), njev=int(sol.njev), optimality=float(sol.optimality),
                                        cost_half_sse=float(sol.cost), solver_x=[float(v) for v in sol.x],
                                        evaluations_for_this_seed=int(evaluate.state['evals']),
                                        selected_evaluation=evaluate.state['best']['evaluation'],
                                        selected_sse=evaluate.state['best']['cost'],
                                        best_feasible_observed=evaluate.state['best_feasible'])
            except Exception:
                record['solver_error'] = traceback.format_exc()
            record['evaluations'] = int(evaluate.state['evals'])
            if evaluate.state['best'] is None:
                per_seed[k] = record
                dump(O / ('SEED_%s_SUMMARY.json' % k), record)
                continue
            d = mujoco.MjData(m)  # independent fresh data, same single compiled model
            r, row, arrays = evaluate(evaluate.state['best']['x'], role='final_validation', retain=True)
            reconstruction = arrays['moment'].T @ arrays['actuator_force']
            reconstruction_error = float(abs(reconstruction - arrays['qfrc_actuator']).max())
            vertex_distances = []
            plane_normal = d.geom_xmat[floor_id].reshape(3, 3)[:, 2]
            plane_point = d.geom_xpos[floor_id]
            for g in nonfoot:
                mesh = int(m.geom_dataid[g])
                s0 = int(m.mesh_vertadr[mesh])
                n = int(m.mesh_vertnum[mesh])
                verts = m.mesh_vert[s0:s0 + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                vertex_distances.append(float(np.min((verts - plane_point) @ plane_normal)))
            vertex_distances = np.asarray(vertex_distances)
            crosscheck = float(np.max(abs(vertex_distances - arrays['nonfoot_distances'])))
            q42 = arrays['qpos'][qa]
            limits = joint_limit_table(q42, hinge_names, hlo, hhi)
            wr = wrench_report(arrays['foot_force'], arrays['contact_rows'][:, 3:6],
                               arrays['contact_rows'][:, 6:9], weight, arrays['com_world'], arrays['root_pos'])
            x = arrays['x']
            # Only the 74 mapped actuator rows carry a pool value; the 16 unmapped rows are
            # structurally zero and are checked separately by unmapped_floor_16.
            pool_values = (G @ x[43:])[mapped]
            limit_rows = {r_['joint']: r_ for r_ in limits['all42']}
            gates = dict(
                all48_acceleration=bool(np.all(abs(arrays['qacc']) <= .1)),
                nonfoot_clearance=bool(np.all(arrays['nonfoot_distances'] >= .02)),
                nonfoot_force_zero=bool(row['nonfoot_force_max_abs'] == 0.),
                foot_load=bool(.99 <= row['foot_load_ratio'] <= 1.01),
                joint_bounds=bool(np.all(q42 >= hlo) and np.all(q42 <= hhi)
                                  and np.array_equal(arrays['qpos'][qa], x[:42])),
                root_z_bounds=bool(zlo <= arrays['qpos'][2] <= zhi and arrays['qpos'][2] == x[42]),
                body_sides=bool(np.all(arrays['body_side'] >= .02)),
                activation_bounds=bool(np.all(arrays['activation'] >= plo) and np.all(arrays['activation'] <= phi)),
                pool_binding_66=bool(np.all(pool_values >= plo) and np.all(pool_values <= phi)
                                     and np.all(G.sum(1) <= 1)
                                     and np.array_equal(arrays['activation'][mapped], pool_values)),
                unmapped_floor_16=bool(np.all(arrays['activation'][~mapped] == plo) and int((~mapped).sum()) == 16),
                act_ctrl_equal=bool(np.array_equal(arrays['activation'], arrays['ctrl'])),
                activation_steady=bool(np.max(abs(arrays['act_dot'])) <= 1e-10),
                external_forces_zero=bool(not np.any(arrays['qfrc_applied']) and not np.any(arrays['xfrc_applied'])),
                moment_reconstruction=bool(reconstruction_error <= 1e-9),
                finite=bool(all(np.isfinite(a).all() for a in [arrays['x'], arrays['qpos'], arrays['qacc'],
                                                              arrays['activation'], arrays['residual'],
                                                              arrays['actuator_force'], arrays['foot_force'],
                                                              arrays['nonfoot_distances'], arrays['body_side']])),
                no_warnings=bool(not np.any(d.warning.number)),
                model_parameters_unchanged=bool(parameter_digest(m)[0] == model_before),
                xml_unchanged=bool(sha(S / 'MODEL.xml') == EXPECTED),
                geom_distance_cutoff_zero=bool(row['geom_distance_cutoff_hits'] == 0),
                mesh_plane_crosscheck=bool(crosscheck <= 1e-8),
                all109_within_declared_bounds=bool(np.all(x >= lower) and np.all(x <= upper)))
            saved = dict(arrays)
            saved.update(moment=arrays['moment'], reconstructed_qfrc_actuator=reconstruction, group_matrix=G,
                         hinge_qpos_addresses=qa, compiled_vertex_distances=vertex_distances,
                         seed_pose_43=seed_pose, weight_native=weight, mapped_actuator_mask=mapped)
            np.savez_compressed(O / ('SEED_%s_FINAL_STATE.npz' % k), **saved)
            dump(O / ('SEED_%s_MESH_DISTANCE_CROSSCHECK.json' % k),
                 dict(max_abs_error=crosscheck, cutoff_hits=int(np.sum(arrays['nonfoot_distances'] >= 1000.)),
                      compiled_vertex_distance=vertex_distances, geom_distance=arrays['nonfoot_distances']))
            row_summary = {kk: vv for kk, vv in row.items() if kk != 'x'}
            record.update(
                final=dict(row=row_summary, blocks=row['blocks'], limit_table=limits, wrench=wr,
                           pool_min_margin=float(min(np.min(pool_values - plo), np.min(phi - pool_values))),
                           root_height_margin=float(min(arrays['qpos'][2] - zlo, zhi - arrays['qpos'][2])),
                           reconstruction_error=reconstruction_error, vertex_crosscheck=crosscheck,
                           coxa_values={n: float(limit_rows[n]['q']) for n in table['contract_order']},
                           abduction=dict(
                               seed_sum_abs_coxa_roll_yaw=float(srow['sum_abs_abduction_roll_yaw_x0']),
                               candidate_sum_abs_coxa_roll_yaw=float(sum(abs(limit_rows[n]['q'])
                                                                         for n in table['contract_order'])),
                               candidate_max_abs_coxa_roll=float(max(abs(limit_rows[n]['q'])
                                                                     for n in table['contract_order']
                                                                     if n.endswith('roll'))),
                               candidate_max_abs_coxa_yaw=float(max(abs(limit_rows[n]['q'])
                                                                    for n in table['contract_order']
                                                                    if n.endswith('yaw'))),
                               coxa_roll_yaw_min_limit_distance=limits['coxa_roll_yaw_min']),
                           lm_coxa_roll=limit_rows['joint_LMCoxa_roll'], rh_coxa_roll=limit_rows['joint_RHCoxa_roll'],
                           lm_coxa_yaw=limit_rows['joint_LMCoxa_yaw']),
                gate=dict(passed=all(gates.values()), gates=gates), gate_passed=bool(all(gates.values())),
                x=x, final_qacc_root=[float(v) for v in arrays['qacc'][:6]])
            per_seed[k] = record
            dump(O / ('SEED_%s_SUMMARY.json' % k), record)
            print('SEED', k, 'sse', row['cost'], 'maxqacc', row['max_abs_qacc'], 'load', row['foot_load_ratio'],
                  'hv', wr['horizontal_over_vertical'], 'gate', record['gate_passed'], 'nearest',
                  limits['argmin_joint'], limits['argmin_side'], round(limits['min_distance'], 6), flush=True)

        summary_rows = []
        for k in FACTORS:
            rec = per_seed.get(float(k))
            if rec is None or rec.get('final') is None:
                summary_rows.append(dict(retained_fraction_k=float(k), available=False,
                                         solver_error=(rec or {}).get('solver_error', 'missing')))
                continue
            f = rec['final']
            summary_rows.append(dict(
                retained_fraction_k=float(k), available=True, seed_projections=rec['n_seed_projections'],
                seed_sum_abs_coxa_roll_yaw=f['abduction']['seed_sum_abs_coxa_roll_yaw'],
                candidate_sum_abs_coxa_roll_yaw=f['abduction']['candidate_sum_abs_coxa_roll_yaw'],
                coxa_values=f['coxa_values'],
                sse=f['row']['cost'], max_abs_qacc=f['row']['max_abs_qacc'],
                vertical_over_weight=f['wrench']['vertical_over_weight'],
                horizontal_over_vertical=f['wrench']['horizontal_over_vertical'],
                moment_norm_over_weight_com=f['wrench']['moment_norm_over_weight_com'],
                moment_norm_over_weight_root=f['wrench']['moment_norm_over_weight_root'],
                min_nonfoot_distance=f['row']['min_nonfoot_distance'], min_side=f['row']['min_side'],
                nonfoot_force_max_abs=f['row']['nonfoot_force_max_abs'], ncon=f['row']['ncon'],
                contact_topology=f['row']['contact_topology'],
                min_joint_limit_distance=f['limit_table']['min_distance'],
                min_joint_limit_joint=f['limit_table']['argmin_joint'],
                min_joint_limit_side=f['limit_table']['argmin_side'],
                coxa_roll_yaw_min_limit_distance=f['abduction']['coxa_roll_yaw_min_limit_distance'],
                joints_within_0p01=f['limit_table']['joints_within_0p01'],
                joints_within_0p05=f['limit_table']['joints_within_0p05'],
                lm_coxa_roll_distance=f['lm_coxa_roll']['nearest_distance'],
                lm_coxa_roll_side=f['lm_coxa_roll']['nearest_side'],
                rh_coxa_roll_distance=f['rh_coxa_roll']['nearest_distance'],
                rh_coxa_roll_side=f['rh_coxa_roll']['nearest_side'],
                lm_coxa_yaw_distance=f['lm_coxa_yaw']['nearest_distance'],
                root_height_margin=f['root_height_margin'], pool_min_margin=f['pool_min_margin'],
                blocks={b: f['blocks'][b]['sse'] for b in f['blocks']},
                gate_passed=rec['gate_passed'],
                failed_gates=[g for g, v in rec['gate']['gates'].items() if not v],
                solver=dict(success=rec['solver']['success'], status=rec['solver']['status'],
                            message=rec['solver']['message'], nfev=rec['solver']['nfev'], njev=rec['solver']['njev'],
                            optimality=rec['solver']['optimality'], evaluations=rec['solver']['evaluations_for_this_seed'])))
        avail = [r for r in summary_rows if r['available']]
        ref_row = None
        if (R6 / 'FINAL_STATE.npz').is_file():
            with np.load(R6 / 'FINAL_STATE.npz') as z6:
                q6 = z6['qpos']
                lim6 = joint_limit_table(q6[qa], hinge_names, hlo, hhi)
                rows6 = z6['contact_rows']
                net6 = rows6[:, 6:9].sum(0)
                m6 = np.cross(rows6[:, 3:6] - q6[:3], rows6[:, 6:9]).sum(0)
                l6 = {r_['joint']: r_ for r_ in lim6['all42']}
                ref_row = dict(label='round6_search60_recomputed_here',
                               sse=float(z6['residual'] @ z6['residual']),
                               max_abs_qacc=float(abs(z6['qacc']).max()),
                               vertical_over_weight=float(net6[2] / weight),
                               horizontal_over_vertical=float(np.linalg.norm(net6[:2]) / net6[2]),
                               moment_norm_over_weight_root=float(np.linalg.norm(m6) / weight),
                               moment_about_com='round6 FINAL_STATE.npz stores no xipos/COM, so the COM moment is '
                                                'not recomputed and is not fabricated',
                               min_joint_limit_distance=lim6['min_distance'], min_joint_limit_joint=lim6['argmin_joint'],
                               min_joint_limit_side=lim6['argmin_side'],
                               lm_coxa_roll_distance=l6['joint_LMCoxa_roll']['nearest_distance'],
                               lm_coxa_roll_side=l6['joint_LMCoxa_roll']['nearest_side'],
                               rh_coxa_roll_distance=l6['joint_RHCoxa_roll']['nearest_distance'],
                               rh_coxa_roll_side=l6['joint_RHCoxa_roll']['nearest_side'],
                               joints_within_0p01=lim6['joints_within_0p01'],
                               joints_within_0p05=lim6['joints_within_0p05'], ncon=int(len(rows6)),
                               reference_sealed_values=prereg['evidence_inputs_read_only']
                               ['reference_metrics_for_comparison_only'])
        if avail:
            any_pass = any(r['gate_passed'] for r in avail)
            h_min = min(r['horizontal_over_vertical'] for r in avail)
            worst = min(avail, key=lambda r: r['horizontal_over_vertical'])
            if any_pass:
                label = 'A_supported_with_candidate'
            elif h_min <= .5 * REF_H_OVER_V:
                label = 'B_partial_support'
            elif all(r['horizontal_over_vertical'] >= .9 * REF_H_OVER_V for r in avail):
                label = 'C_rejected'
            else:
                label = 'D_weak_or_inconclusive'
            recurring = [r['retained_fraction_k'] for r in avail
                         if (r['lm_coxa_roll_side'] == 'upper' and r['lm_coxa_roll_distance'] <= .01)
                         or (r['rh_coxa_roll_side'] == 'lower' and r['rh_coxa_roll_distance'] <= .01)]
            classification = dict(
                label=label, min_horizontal_over_vertical=float(h_min),
                seed_with_min_horizontal=float(worst['retained_fraction_k']),
                reference_horizontal_over_vertical=REF_H_OVER_V,
                ratio_of_min_to_reference=float(h_min / REF_H_OVER_V),
                thresholds=dict(partial_support_at_or_below=.5 * REF_H_OVER_V,
                                rejected_at_or_above=.9 * REF_H_OVER_V),
                any_seed_passed_all_gates=bool(any_pass),
                seeds_with_recurring_same_two_coxa_roll_binding=recurring,
                recurring_binding_test_triggered=bool(len(recurring) >= 3),
                recurring_binding_sentence=(
                    'The same two coxa_roll limits (joint_LMCoxa_roll upper bound, joint_RHCoxa_roll lower bound) are '
                    're-bound within 0.01 rad in %d of %d seeds that started from contracted abduction, which points '
                    'to a joint-range / foot-position mismatch rather than to missing activation.'
                    % (len(recurring), len(avail))) if len(recurring) >= 3 else
                    ('Only %d of %d contracted seeds re-bind those two coxa_roll limits within 0.01 rad.'
                     % (len(recurring), len(avail))),
                rule='preregistered before execution; not adjusted after the numbers were known')
        dump(O / 'PER_SEED_COMPARISON.json', dict(
            scope='One run per seed, identical objective/bounds/solver; per-seed selected candidate = strict minimum '
                  'complete SSE over that seed\'s evaluations including finite-difference probes',
            rows=summary_rows, round6_reference=ref_row, classification=classification))
        dump(O / 'FINAL_GATE.json', dict(
            any_seed_passed_all_gates=bool(any_pass),
            per_seed={str(r['retained_fraction_k']): dict(passed=r['gate_passed'], failed=r['failed_gates'])
                      for r in avail},
            classification=classification,
            note='A static engineering screen only; no candidate is an autonomous behaviour claim.'))
        model_after, rows_after = parameter_digest(m)
        model_unchanged = model_before == model_after
        dump(O / 'MODEL_PARAMETERS_AFTER.json', dict(sha256=model_after, fields=rows_after))
    except Exception:
        error = traceback.format_exc()
        dump(O / 'FAILURE.json', dict(error=error, counts=counts))
    finally:
        if m is not None and model_before is not None and model_after is None:
            model_after, rows_after = parameter_digest(m)
            model_unchanged = model_before == model_after
            dump(O / 'MODEL_PARAMETERS_AFTER.json', dict(sha256=model_after, fields=rows_after))
        unchanged = all(Path(p).is_file() and sha(p) == h for p, h in inputs.items())
        execution = dict(counts=counts, wall_seconds=time.time() - started, source_hashes_unchanged=unchanged,
                         model_parameters_before=model_before, model_parameters_after=model_after,
                         model_parameters_unchanged=model_unchanged,
                         per_seed=({str(k): dict(evaluations=per_seed[k].get('evaluations'),
                                                 nfev=(per_seed[k].get('solver') or {}).get('nfev'),
                                                 njev=(per_seed[k].get('solver') or {}).get('njev'),
                                                 selected_sse=(per_seed[k].get('solver') or {}).get('selected_sse'),
                                                 gate_passed=per_seed[k].get('gate_passed'),
                                                 solver_error=bool(per_seed[k].get('solver_error')))
                                     for k in per_seed}),
                         body_steps=counts['body_step'], neural_steps=counts['neural_step'],
                         classification=(classification or {}).get('label'),
                         exit_code=0 if error is None and unchanged else 1, error=error)
        dump(O / 'EXECUTION.json', execution)
        print(json.dumps(execution, ensure_ascii=False, indent=2), flush=True)
    return 0 if error is None else 1


if __name__ == '__main__':
    sys.exit(main())
