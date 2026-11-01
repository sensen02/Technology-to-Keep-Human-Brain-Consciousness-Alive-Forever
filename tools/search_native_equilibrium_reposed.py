"""S2/S5 round 8: native3470 reposed (steady-prewarm seed) joint + root-attitude + shared-pool
activation static soft-contact equilibrium search. One preregistered audit, then at most two
bounded least-squares segments. No physical integration, no neural steps, no model edits,
no rollout, frozen inputs read-only.

Modes (all must run with /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B):
  (no flag)      main sequence: invariance re-check -> baseline Jacobian audit -> segment A -> B
  --invariance   step-1 trivial-direction (horizontal translation) check only
  --verify       separate-process, arrays-only re-check plus REPORT/MANIFEST; no new optimization
"""
from pathlib import Path
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import sys, json, time, traceback, hashlib
import numpy as np
import mujoco
from scipy.optimize import least_squares
from flygym.compose.fly.musculoskeletal import _load_mjcf
from search_native_muscle_equilibrium import S, EXPECTED, LEGS, sha, dump, parameter_digest

O = Path('/home/sensen/Desktop/engine修复验收/s5-native-equilibrium-reposed/20261101T201309Z')
R6 = Path('/home/sensen/Desktop/engine修复验收/s5-native-equilibrium-search/20261101T193630Z')
R7 = Path('/home/sensen/Desktop/engine修复验收/s5-equilibrium-derivative-continuation/20261101T200110Z')
R1 = Path('/home/sensen/Desktop/engine修复验收/s5-equilibrium-wrench-diagnosis/20261101T200110Z')

STEADY = (0.25, 1.30)
TARGET_TIME = 1.2
NX = 112
BLOCKS = {'qacc': (0, 48), 'clearance': (48, 89), 'side': (89, 125), 'load': (125, 126),
          'regularizer': (126, 172)}
NROW = BLOCKS['regularizer'][1]
EPS_AUDIT = 1e-7
ACC_GATE = 0.1

INVARIANCE_TOLERANCE = dict(qacc=1e-9, foot_force=1e-8, nonfoot_distance=1e-9, nonfoot_force=1e-8,
                            contact_position=1e-12)
SHIFTS = ((0.05, 0.05), (-0.05, 0.0), (0.0, 0.05), (0.05, 0.0), (0.0, -0.05))
CONTACT_ROW_SCHEMA = ['geom1', 'geom2', 'c.dist', 'c.pos[0..2]', 'world_force[0..2]', 'local_force[0..2]']


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def quat_from_rotvec(v):
    v = np.asarray(v, float)
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.r_[np.cos(theta/2), v * (np.sin(theta/2)/theta)]


class Experiment:
    """Single compiled model, single MjData for optimization; fresh MjData only for final checks."""

    def __init__(self, mode):
        self.mode = mode
        self.counts = dict(model_load_attempt=1, model_load=0, mj_resetData=0, forward=0,
                           geom_distance=0, contact_force=0, least_squares=0, jacobian_calls=0,
                           body_step=0, neural_step=0)
        self.roles = {}
        self.best = None
        self.best_acc = None
        self.feasible = []
        self.segment_best = {}
        self.segment_arrays = {}
        self.topology_events = []
        self.last_signature = None
        self.last_pairs = None
        assert sha(S/'MODEL.xml') == EXPECTED, 'MODEL.xml sha mismatch'
        self.m = _load_mjcf(str(S/'MODEL.xml')).compile()
        self.counts['model_load'] = 1
        m = self.m
        assert (m.nq, m.nv, m.nu, m.na) == (49, 48, 90, 90)
        self.model_before, self.before_rows = parameter_digest(m)
        self.d = mujoco.MjData(m)
        hinges = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
        assert len(hinges) == 42 and np.all(m.jnt_limited[hinges])
        self.hinges = hinges
        self.qa = m.jnt_qposadr[hinges].astype(int)
        self.va = m.jnt_dofadr[hinges].astype(int)
        assert np.array_equal(np.sort(self.qa), np.arange(7, 49))
        assert np.array_equal(m.actuator_actadr, np.arange(90))
        free = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        assert len(free) == 1
        self.root = int(m.jnt_bodyid[free[0]])
        groups, self.pool_names = {}, []
        for row in json.loads((S/'MAPPING.json').read_text())['rows']:
            assert m.actuator(int(row['actuator_index'])).name == row['actuator']
            if row['local_indices']:
                groups.setdefault(tuple(row['local_indices']), []).append(int(row['actuator_index']))
        assert len(groups) == 66
        G = np.zeros((90, 66))
        for j, ids in enumerate(groups.values()):
            G[ids, j] = 1
            self.pool_names.append([m.actuator(int(i)).name for i in ids])
        self.G = G
        self.mapped = G.sum(1) > 0
        assert int(self.mapped.sum()) == 74 and np.all(G.sum(1) <= 1)
        self.floor = m.geom('floor').id
        assert m.geom_type[self.floor] == mujoco.mjtGeom.mjGEOM_PLANE
        foot = {}
        for g in range(m.ngeom):
            name, bn = m.geom(g).name, m.body(int(m.geom_bodyid[g])).name
            legs = [i for i, L in enumerate(LEGS)
                    if (name.startswith(L) or bn.startswith(L)) and 'Tarsus' in name + bn]
            if len(legs) == 1:
                foot[g] = legs[0]
        assert len(foot) == 30
        self.foot = foot
        self.nonfoot = np.array([g for g in range(m.ngeom)
                                 if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and g not in foot], int)
        assert len(self.nonfoot) == 41
        side_bodies, side_sign = [], []
        for b in range(m.nbody):
            name = m.body(b).name
            if any(name.startswith(L) for L in LEGS) and ('Tibia' in name or 'Tarsus' in name):
                side_bodies.append(b)
                side_sign.append(1.0 if name.startswith(('LF', 'LM', 'LH')) else -1.0)
        self.sideb = np.array(side_bodies, int)
        self.sign = np.array(side_sign)
        assert len(self.sideb) == 36
        self.thorax = m.body('Thorax').id
        self.weight = float(m.body_mass.sum()*abs(m.opt.gravity[2]))
        v6 = json.loads((R6/'VARIABLES.json').read_text())
        assert np.array_equal(self.G, np.array(v6['group_matrix']))
        assert {m.geom(g).name: LEGS[l] for g, l in self.foot.items()} == v6['foot_geom_to_leg']
        assert [m.geom(int(g)).name for g in self.nonfoot] == v6['nonfoot_geoms']
        assert [m.body(int(b)).name for b in self.sideb] == v6['side_bodies']
        assert self.weight == v6['weight_native']
        assert np.array_equal(self.qa, np.array(v6['hinge_qpos_addresses'], int))
        with np.load(S/'BODY.npz') as z:
            self.body_time = z['time'].copy()
            self.body_qpos = z['qpos'].copy()
            self.body_act = z['activation'].copy()
            self.body_footforce = z['foot_force'].copy()
            self.body_footcontact = z['foot_contact'].copy()
            self.body_nonfootcount = z['nonfoot_count'].copy()
            self.body_warning = z['warning'].copy()
        self.seed = self.select_seeds()
        self.build_variables()
        self.log = None
        self.write_states = (mode == 'run')

    # ---------------------------------------------------------------- seeds
    def select_seeds(self):
        t = self.body_time
        fz = self.body_footforce[:, :, 2]
        nlegs = (fz > 0).sum(1)
        steady = (t >= STEADY[0]) & (t < STEADY[1])
        eligible = np.flatnonzero(steady & (nlegs == 6))
        literal = int(np.argmin(abs(t - TARGET_TIME)))
        info = dict(requested_time=TARGET_TIME, steady_window=list(STEADY),
                    steady_frames=int(steady.sum()),
                    steady_frame_leg_count_histogram={str(k): int(np.sum(steady & (nlegs == k)))
                                                      for k in range(7)},
                    eligible_six_leg_frames=int(eligible.size),
                    literal=dict(index=literal, time=float(t[literal]), leg_count=int(nlegs[literal]),
                                 per_leg_vertical_force=fz[literal].tolist(),
                                 foot_contact=self.body_footcontact[literal].tolist(),
                                 nonfoot_count=int(self.body_nonfootcount[literal]),
                                 activation_min=float(self.body_act[literal].min()),
                                 activation_max=float(self.body_act[literal].max()),
                                 qpos=self.body_qpos[literal].tolist(),
                                 used_as_main_initial_point=False,
                                 note='literal t=1.2 frame of the unmodified recording; %d legs have '
                                      'strictly positive vertical ground force, so it does not satisfy the '
                                      'task premise and is not the main initial point' % nlegs[literal]),
                    rule='nearest frame to t=1.2 inside the steady prewarm window with all six legs '
                         'carrying strictly positive vertical ground force')
        assert eligible.size > 0, 'no six-leg positive-vertical-force frame inside the steady window'
        chosen = int(eligible[np.argmin(abs(t[eligible] - TARGET_TIME))])
        info.update(chosen_index=chosen, chosen_time=float(t[chosen]),
                    chosen_time_delta=float(t[chosen] - TARGET_TIME),
                    chosen_leg_count=int(nlegs[chosen]),
                    chosen_per_leg_vertical_force=fz[chosen].tolist(),
                    chosen_foot_force=self.body_footforce[chosen].tolist(),
                    chosen_foot_contact=self.body_footcontact[chosen].tolist(),
                    chosen_nonfoot_count=int(self.body_nonfootcount[chosen]),
                    chosen_warning=int(self.body_warning[chosen].sum()),
                    chosen_activation=self.body_act[chosen].tolist(),
                    chosen_qpos=self.body_qpos[chosen].tolist())
        if chosen != literal:
            info['deviation'] = ('literal t=1.2 has %d loaded legs (task premise requires 6); main initial '
                                 'pose uses the nearest qualifying steady frame t=%.4f instead. The literal '
                                 'frame values remain recorded under literal.') % (nlegs[literal], t[chosen])
        self.seed_qpos = self.body_qpos[chosen].copy()
        self.seed_quat = self.seed_qpos[3:7].copy()
        return info

    def build_variables(self):
        m, q = self.m, self.seed_qpos
        self.lo = np.r_[m.jnt_range[self.hinges, 0], q[2] - .25, np.full(3, -.25), np.full(66, .0001)]
        self.hi = np.r_[m.jnt_range[self.hinges, 1], q[2] + .25, np.full(3, .25), np.ones(66)]
        raw = np.r_[q[self.qa], q[2], np.zeros(3), np.full(66, .1)]
        x0 = raw.copy()
        x0[:42] = np.clip(x0[:42], self.lo[:42], self.hi[:42])
        assert np.all(x0 >= self.lo) and np.all(x0 <= self.hi)
        self.raw_initial = raw
        self.x0 = x0
        self.pose_target = np.r_[q[self.qa], q[2], np.zeros(3)]
        names = ([m.joint(int(j)).name for j in self.hinges] + ['root_z']
                 + ['root_rotvec_x', 'root_rotvec_y', 'root_rotvec_z']
                 + ['pool_%02d' % i for i in range(66)])
        assert len(names) == NX
        self.names = names

    def seed_artifacts(self):
        q, m = self.seed_qpos, self.m
        changed = np.flatnonzero(self.x0 != self.raw_initial)
        projection = dict(
            raw_seed_qpos=q, seed_time=self.seed['chosen_time'], seed_index=self.seed['chosen_index'],
            raw_initial=self.raw_initial, solver_initial=self.x0,
            changes=[dict(variable=int(i), name=self.names[i],
                          joint=m.joint(int(self.hinges[i])).name if i < 42 else None,
                          old=float(self.raw_initial[i]), new=float(self.x0[i]),
                          delta=float(self.x0[i] - self.raw_initial[i])) for i in changed],
            native_joint_range=dict(lower=m.jnt_range[self.hinges, 0].tolist(),
                                    upper=m.jnt_range[self.hinges, 1].tolist()),
            reason='preregistered minimal projection to declared native joint bounds applied to the initial '
                   'point only; the model, the raw seed pose and the regularizer reference are unchanged')
        variables = dict(
            seed_time=self.seed['chosen_time'], seed_index=self.seed['chosen_index'],
            variable_names=self.names, variable_count=NX,
            hinge_qpos_addresses=self.qa, hinge_dof_addresses=self.va,
            order=['joint angle x 42', 'root z', 'root attitude rotvec (body frame, relative to seed quat) x 3',
                   'shared pool activation x 66'],
            root_xy_fixed=[float(q[0]), float(q[1])],
            root_xy_justification='horizontal translation invariance measured in INVARIANCE.json at '
                                  'preregistered float tolerance; root x/y therefore not search variables',
            seed_quat=self.seed_quat, pose_target=self.pose_target,
            lower=self.lo, upper=self.hi, initial=self.x0, raw_initial=self.raw_initial,
            group_matrix=self.G, pool_actuator_names=self.pool_names, unmapped_count=int(16),
            foot_geom_to_leg={m.geom(g).name: LEGS[l] for g, l in self.foot.items()},
            nonfoot_geoms=[m.geom(int(g)).name for g in self.nonfoot],
            side_bodies=[m.body(int(b)).name for b in self.sideb], side_frame='Thorax local y axis',
            weight_native=self.weight, residual_blocks={k: list(v) for k, v in BLOCKS.items()},
            residual_rows=NROW, contact_row_schema=CONTACT_ROW_SCHEMA)
        old = dict()
        i = int(np.argmin(abs(self.body_time - .15)))
        qo = self.body_qpos[i]
        old_frozen = json.loads((R6/'SEED_BOUND_PROJECTION.json').read_text())
        old_raw = np.r_[qo[self.qa], qo[2], np.full(66, .1)]
        old_x0 = old_raw.copy()
        old_x0[:42] = np.clip(old_x0[:42], self.lo[:42], self.hi[:42])
        frozen_x0 = np.array(old_frozen['solver_initial'])
        old = dict(
            time=float(self.body_time[i]), index=i, qpos=qo, raw_initial=old_raw, solver_initial=old_x0,
            frozen_round6_solver_initial=frozen_x0,
            reproduction_max_abs_error=float(np.max(abs(old_x0 - frozen_x0))),
            projection=[dict(variable=int(j), name=self.names[j], old=float(old_raw[j]), new=float(old_x0[j]))
                        for j in np.flatnonzero(old_x0 != old_raw)],
            status='optional comparison only; the main sequence uses the steady prewarm seed',
            note='the round-6 initial point had 109 variables; the first 43 entries and the 66 activations '
                 'must match bitwise, the three new attitude variables are zero at this seed')
        return projection, variables, old

    # ------------------------------------------------------------ evaluation
    def evaluate(self, x, role='optimizer', retain=False, probe=None, compact=None):
        m, d = self.m, self.d
        x = np.asarray(x, float)
        self.roles[role] = self.roles.get(role, 0) + 1
        mujoco.mj_resetData(m, d)
        self.counts['mj_resetData'] += 1
        qpos = self.seed_qpos.copy()
        qpos[self.qa] = x[:42]
        qpos[2] = x[42]
        qpos[3:7] = quat_mul(self.seed_quat, quat_from_rotvec(x[43:46]))
        d.qpos[:] = qpos
        d.qvel[:] = 0
        activation = np.full(90, .0001)
        activation[self.mapped] = (self.G @ x[46:])[self.mapped]
        d.act[:] = activation
        d.ctrl[:] = activation
        d.qfrc_applied[:] = 0
        d.xfrc_applied[:] = 0
        mujoco.mj_forward(m, d)
        self.counts['forward'] += 1
        distances = np.array([mujoco.mj_geomDistance(m, d, int(g), self.floor, 1000., None)
                              for g in self.nonfoot])
        self.counts['geom_distance'] += len(self.nonfoot)
        side = ((d.xpos[self.sideb] - d.xpos[self.thorax]) @ d.xmat[self.thorax].reshape(3, 3)[:, 1]) * self.sign
        ff = np.zeros((6, 3))
        nf = np.zeros(3)
        nfmax = 0.0
        topology, floor_rows = [], []
        for ci in range(d.ncon):
            c = d.contact[ci]
            a, b = int(c.geom1), int(c.geom2)
            topology.append([a, b, int(c.dim), int(c.efc_address)])
            if self.floor not in (a, b):
                continue
            f = np.zeros(6)
            mujoco.mj_contactForce(m, d, ci, f)
            self.counts['contact_force'] += 1
            world = c.frame.reshape(3, 3).T @ f[:3] * (1 if a == self.floor else -1)
            other = b if a == self.floor else a
            if other in self.foot:
                ff[self.foot[other]] += world
            else:
                nf += world
                nfmax = max(nfmax, float(abs(world).max()))
            floor_rows.append(np.r_[a, b, c.dist, c.pos, world, f[:3]])
        load = float(ff[:, 2].sum()/self.weight)
        r = np.r_[d.qacc/9810., 100*np.maximum(0., .02 - distances), 100*np.maximum(0., .02 - side),
                  10*(load - 1), 1e-6*(x[:46] - self.pose_target)]
        normal = d.geom_xmat[self.floor].reshape(3, 3)[:, 2]
        origin = d.geom_xpos[self.floor]
        vd = []
        for g in self.nonfoot:
            mesh = int(m.geom_dataid[g])
            a = int(m.mesh_vertadr[mesh])
            n = int(m.mesh_vertnum[mesh])
            verts = m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            vd.append(float(np.min((verts - origin) @ normal)))
        vd = np.array(vd)
        cross = float(abs(vd - distances).max())
        topology = sorted(topology)
        signature = hashlib.sha256(json.dumps(topology).encode()).hexdigest()
        finite = bool(all(np.isfinite(a).all() for a in
                          (d.qpos, d.qvel, d.qacc, d.act, d.act_dot, d.actuator_force, distances, side, r)))
        gates = dict(
            all48_acceleration=bool(np.all(abs(d.qacc) <= ACC_GATE)),
            foot_load=bool(.99 <= load <= 1.01),
            nonfoot_clearance=bool(np.all(distances >= .02)),
            nonfoot_force_zero=bool(nfmax == 0.),
            body_sides=bool(np.all(side >= .02)),
            all112_bounds=bool(np.all(x >= self.lo) and np.all(x <= self.hi)),
            joint_bounds=bool(np.all(x[:42] >= m.jnt_range[self.hinges, 0])
                              and np.all(x[:42] <= m.jnt_range[self.hinges, 1])),
            root_z_bounds=bool(self.lo[42] <= x[42] <= self.hi[42]),
            root_attitude_bounds=bool(np.all(x[43:46] >= self.lo[43:46]) and np.all(x[43:46] <= self.hi[43:46])),
            pool_bounds=bool(np.all(x[46:] >= .0001) and np.all(x[46:] <= 1.)),
            activation_bounds=bool(np.all(d.act >= .0001) and np.all(d.act <= 1.)),
            pool_binding=bool(np.array_equal(d.act[self.mapped], (self.G @ x[46:])[self.mapped])),
            unmapped_floor=bool(np.all(d.act[~self.mapped] == .0001)),
            act_ctrl_equal=bool(np.array_equal(d.act, d.ctrl)),
            activation_steady=bool(float(np.max(abs(d.act_dot))) <= 1e-10),
            external_forces_zero=bool(not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)),
            finite=finite,
            no_warnings=bool(not np.any(d.warning.number)),
            geom_distance_cutoff_zero=bool(np.all(distances < 1000.)),
            mesh_plane_crosscheck=bool(cross <= 1e-8),
        )
        cost = float(r @ r)
        block_sse = {name: float(r[a:b] @ r[a:b]) for name, (a, b) in BLOCKS.items()}
        feasible = bool(all(gates.values()))
        row = dict(evaluation=self.counts['forward'], role=role, cost=cost, block_sse=block_sse,
                   max_abs_qacc=float(abs(d.qacc).max()), foot_load_ratio=load,
                   min_nonfoot_distance=float(distances.min()), min_side=float(side.min()),
                   nonfoot_force_max_abs=nfmax, act_dot_max_abs=float(np.max(abs(d.act_dot))),
                   warnings=int(d.warning.number.sum()),
                   geom_distance_cutoff_hits=int(np.sum(distances >= 1000.)), crosscheck_error=cross,
                   solver_niter=d.solver_niter.tolist(), ncon=int(d.ncon),
                   floor_contact_count=len(floor_rows), contact_topology=topology,
                   topology_signature=signature, gates=gates,
                   observed_physics_feasible=feasible)
        if probe is not None:
            row['probe'] = probe
        if compact is None:
            compact = role.endswith('jac_probe')
        if not compact:
            row.update(x=x.tolist(), residual=r.tolist(), qacc=d.qacc.tolist(),
                       floor_contacts=[[float(v) for v in fr] for fr in floor_rows])
        if signature != self.last_signature:
            prev = set(map(tuple, self.last_pairs)) if self.last_pairs else set()
            cur = set(map(tuple, topology))
            self.topology_events.append(dict(
                evaluation=row['evaluation'], role=role, signature=signature, previous_signature=self.last_signature,
                ncon=int(d.ncon), floor_contact_count=len(floor_rows),
                appeared=sorted([list(p) for p in cur - prev]), disappeared=sorted([list(p) for p in prev - cur]),
                note='contact topology changes here; the objective is not differentiable at this evaluation'))
            self.last_signature = signature
            self.last_pairs = topology
        if self.log is not None:
            self.log.write(json.dumps(row, allow_nan=False) + '\n')
            self.log.flush()
        if not finite and self.write_states:
            np.savez_compressed(O/'NONFINITE_FAILURE.npz', x=x, qpos=d.qpos, qacc=d.qacc, residual=r)
            raise FloatingPointError('nonfinite evaluation at forward=%d role=%s' % (row['evaluation'], role))
        need = (retain or feasible or (self.best is None or cost < self.best['cost'])
                or (self.best_acc is None or row['max_abs_qacc'] < self.best_acc['max_abs_qacc']))
        arrays = None
        if need:
            arrays = dict(x=x.copy(), residual=r.copy(), qpos=d.qpos.copy(), qvel=d.qvel.copy(),
                          activation=d.act.copy(), ctrl=d.ctrl.copy(), act_dot=d.act_dot.copy(),
                          qacc=d.qacc.copy(), qfrc_applied=d.qfrc_applied.copy(),
                          xfrc_applied=d.xfrc_applied.copy(), actuator_force=d.actuator_force.copy(),
                          qfrc_actuator=d.qfrc_actuator.copy(), qfrc_constraint=d.qfrc_constraint.copy(),
                          qfrc_passive=d.qfrc_passive.copy(), qfrc_bias=d.qfrc_bias.copy(),
                          foot_force=ff, nonfoot_force=nf, nonfoot_distances=distances,
                          compiled_vertex_distances=vd, body_side=side, solver_niter=d.solver_niter.copy(),
                          ncon=np.array([d.ncon]), root_xpos=d.xpos[self.root].copy(),
                          subtree_com=d.subtree_com.copy(), group_matrix=self.G,
                          hinge_qpos_addresses=self.qa, seed_qpos=self.seed_qpos,
                          contact_rows=(np.array(floor_rows, float).reshape(-1, 12) if floor_rows
                                        else np.zeros((0, 12))),
                          contact_topology=(np.array(topology, int).reshape(-1, 4) if topology
                                            else np.zeros((0, 4), int)),
                          evaluation=np.array([row['evaluation']]), cost=np.array([cost]))
        if self.best is None or cost < self.best['cost']:
            self.best = dict(cost=cost, evaluation=row['evaluation'], role=role, x=x.copy())
            if self.write_states:
                np.savez_compressed(O/'BEST_INCREMENTAL.npz', **arrays)
        if self.best_acc is None or row['max_abs_qacc'] < self.best_acc['max_abs_qacc']:
            self.best_acc = dict(max_abs_qacc=row['max_abs_qacc'], evaluation=row['evaluation'],
                                 role=role, cost=cost)
            if self.write_states:
                np.savez_compressed(O/'BEST_MAXQACC_STATE.npz', **arrays)
        if feasible:
            self.feasible.append(dict(evaluation=row['evaluation'], role=role, cost=cost,
                                      max_abs_qacc=row['max_abs_qacc'], foot_load_ratio=load))
            if self.write_states:
                np.savez_compressed(O/('FEASIBLE_%03d.npz' % len(self.feasible)), **arrays)
            print('FEASIBLE_CANDIDATE', json.dumps(self.feasible[-1]), flush=True)
        for tag in ('SEG_A', 'SEG_B'):
            if role.startswith(tag):
                cur = self.segment_best.get(tag)
                if cur is None or cost < cur['cost']:
                    self.segment_best[tag] = dict(cost=cost, evaluation=row['evaluation'], role=role)
                    self.segment_arrays[tag] = arrays
        return (r, row, arrays) if retain else r

    # ------------------------------------------------------------- jacobian
    def jacobian_explicit(self, at, tag, eps, steps_store):
        m = NX
        base = self.evaluate(at, tag + '_f')
        J = np.empty((NROW, m))
        plus = np.empty((NROW, m))
        minus = np.empty((NROW, m))
        signatures, niters = [], []
        for j in range(m):
            if at[j] - eps >= self.lo[j] and at[j] + eps <= self.hi[j]:
                p = at.copy(); p[j] += eps
                q = at.copy(); q[j] -= eps
                fp = self.evaluate(p, tag + '_jac_probe',
                                   probe=dict(column=j, name=self.names[j], step=eps, mode='central', direction='+'))
                fm = self.evaluate(q, tag + '_jac_probe',
                                   probe=dict(column=j, name=self.names[j], step=eps, mode='central', direction='-'))
                plus[:, j], minus[:, j] = fp, fm
                J[:, j] = (fp - fm)/(2*eps)
                steps_store[j] = dict(mode='central', step=eps)
            else:
                sign = 1 if self.hi[j] - at[j] >= at[j] - self.lo[j] else -1
                room = (self.hi[j] - at[j]) if sign == 1 else (at[j] - self.lo[j])
                h = min(eps, room/2)
                assert h > 0
                p = at.copy(); p[j] += sign*h
                q = at.copy(); q[j] += 2*sign*h
                f1 = self.evaluate(p, tag + '_jac_probe',
                                   probe=dict(column=j, name=self.names[j], step=float(h), mode='one_sided_inward',
                                              direction='inward1'))
                f2 = self.evaluate(q, tag + '_jac_probe',
                                   probe=dict(column=j, name=self.names[j], step=float(2*h), mode='one_sided_inward',
                                              direction='inward2'))
                J[:, j] = sign*(-3*base + 4*f1 - f2)/(2*h)
                plus[:, j] = J[:, j]
                minus[:, j] = J[:, j]
                steps_store[j] = dict(mode='one_sided_inward', step=float(h))
            signatures.append(self.last_signature)
            niters.append(int(self.d.solver_niter[0]))
        norms = np.linalg.norm(J, axis=0)
        mismatch = np.linalg.norm(plus - minus, axis=0)/np.maximum(
            np.maximum(np.linalg.norm(plus, axis=0), np.linalg.norm(minus, axis=0)), 1e-8)
        info = dict(jacobian=J, plus=plus, minus=minus, base=base, column_norms=norms,
                    one_sided_relative_difference=mismatch, steps=steps_store,
                    column_signatures=signatures, column_solver_niter=niters)
        return info

    # ------------------------------------------------------------ invariance
    def snapshot(self, qpos, activation):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        self.counts['mj_resetData'] += 1
        d.qpos[:] = qpos
        d.qvel[:] = 0
        d.act[:] = activation
        d.ctrl[:] = activation
        d.qfrc_applied[:] = 0
        d.xfrc_applied[:] = 0
        mujoco.mj_forward(m, d)
        self.counts['forward'] += 1
        distances = np.array([mujoco.mj_geomDistance(m, d, int(g), self.floor, 1000., None)
                              for g in self.nonfoot])
        self.counts['geom_distance'] += len(self.nonfoot)
        ff = np.zeros((6, 3))
        nf = np.zeros(3)
        positions = []
        for ci in range(d.ncon):
            c = d.contact[ci]
            a, b = int(c.geom1), int(c.geom2)
            if self.floor not in (a, b):
                continue
            f = np.zeros(6)
            mujoco.mj_contactForce(m, d, ci, f)
            self.counts['contact_force'] += 1
            world = c.frame.reshape(3, 3).T @ f[:3] * (1 if a == self.floor else -1)
            other = b if a == self.floor else a
            if other in self.foot:
                ff[self.foot[other]] += world
            else:
                nf += world
            positions.append(np.r_[c.pos, world])
        return dict(qacc=d.qacc.copy(), foot_force=ff, nonfoot_distances=distances, nonfoot_force=nf,
                    contacts=np.array(positions, float).reshape(-1, 6),
                    solver_niter=d.solver_niter.copy(), ncon=int(d.ncon), qpos=d.qpos.copy())

    def invariance(self):
        cases, ok = [], True
        poses = {}
        i_old = int(np.argmin(abs(self.body_time - .15)))
        poses['original_settle_t0.15'] = (i_old, self.body_qpos[i_old].copy())
        poses['steady_seed'] = (self.seed['chosen_index'], self.seed_qpos.copy())
        activations = dict(saved_recording_activation=None, search_initial_activation=np.full(90, .1))
        for pose_label, (idx, qpos) in poses.items():
            activations['saved_recording_activation'] = self.body_act[idx].copy()
            for act_label, activation in activations.items():
                base = self.snapshot(qpos, activation)
                for dx, dy in SHIFTS:
                    shifted_qpos = qpos.copy()
                    shifted_qpos[0] += dx
                    shifted_qpos[1] += dy
                    shifted = self.snapshot(shifted_qpos, activation)
                    same_shape = shifted['contacts'].shape == base['contacts'].shape
                    dq = float(np.max(abs(shifted['qacc'] - base['qacc'])))
                    df = float(np.max(abs(shifted['foot_force'] - base['foot_force'])))
                    dd = float(np.max(abs(shifted['nonfoot_distances'] - base['nonfoot_distances'])))
                    dn = float(np.max(abs(shifted['nonfoot_force'] - base['nonfoot_force'])))
                    dc = (float(np.max(abs((shifted['contacts'][:, :3] - base['contacts'][:, :3])
                                           - np.array([dx, dy, 0.])))) if same_shape else None)
                    niter_same = bool(np.array_equal(shifted['solver_niter'], base['solver_niter']))
                    passed = (dq <= INVARIANCE_TOLERANCE['qacc']
                              and df <= INVARIANCE_TOLERANCE['foot_force']
                              and dd <= INVARIANCE_TOLERANCE['nonfoot_distance']
                              and dn <= INVARIANCE_TOLERANCE['nonfoot_force']
                              and same_shape and niter_same and shifted['ncon'] == base['ncon']
                              and dc is not None and dc <= INVARIANCE_TOLERANCE['contact_position'])
                    ok = ok and passed
                    cases.append(dict(pose=pose_label, pose_index=idx, activation=act_label,
                                      shift=[dx, dy], max_abs_qacc_diff=dq, max_abs_foot_force_diff=df,
                                      max_abs_nonfoot_distance_diff=dd, max_abs_nonfoot_force_diff=dn,
                                      max_abs_contact_position_shift_error=dc,
                                      solver_niter_identical=niter_same, ncon_base=base['ncon'],
                                      ncon_shifted=shifted['ncon'], ncon_match=shifted['ncon'] == base['ncon'],
                                      contact_row_count=len(base['contacts']), passed=bool(passed),
                                      base_max_abs_qacc=float(np.max(abs(base['qacc']))),
                                      base_foot_load_ratio=float(base['foot_force'][:, 2].sum()/self.weight)))
        result = dict(scope='two-forward horizontal-translation invariance on the unmodified 3470 model; '
                            'each case is one baseline mj_forward and one shifted mj_forward with identical '
                            'remaining qpos, qvel=0, act=ctrl and zeroed applied forces',
                      tolerance=INVARIANCE_TOLERANCE, shifts=[list(s) for s in SHIFTS],
                      real_forward_count=self.counts['forward'],
                      pose_sources=dict(original_settle='BODY.npz frame nearest t=0.15',
                                        steady_seed='BODY.npz frame selected by SEED_SELECTION rule'),
                      all_passed=bool(ok), cases=cases,
                      decision=('root x/y are a verified trivial (float-tolerance invariant) direction: fix '
                                'them at the steady-seed values and continue' if ok else
                                'INVARIANCE FAILED: stop, do not fix root x/y, report to parent'))
        return result

    # ---------------------------------------------------------------- audit
    def audit(self, eps=EPS_AUDIT):
        x0 = self.x0
        started = self.counts['forward'] + 1
        base_r, base_row, base_arrays = self.evaluate(x0, 'AUDIT_base', retain=True)
        np.savez_compressed(O/'AUDIT_BASE_STATE.npz', **base_arrays)
        steps = {}
        info = self.jacobian_explicit(x0, 'AUDIT', eps, steps)
        np.savez_compressed(O/'AUDIT_JAC.npz', jacobian=info['jacobian'], base_residual=base_r,
                            column_norms=info['column_norms'],
                            one_sided_relative_difference=info['one_sided_relative_difference'],
                            steps=np.array([steps[j]['step'] for j in range(NX)]))
        norms, mismatch, J = info['column_norms'], info['one_sided_relative_difference'], info['jacobian']
        base_sig = base_row['topology_signature']
        switched = [j for j in range(NX) if info['column_signatures'][j] != base_sig]
        key = []
        for j in [42, 43, 44, 45] + list(np.argsort(-norms)[:6]) + list(np.argsort(-mismatch)[:3]):
            j = int(j)
            if j not in key:
                key.append(j)
        key = key[:12]
        key_rows = []
        for j in key:
            rel, one6 = None, None
            if x0[j] - 1e-6 >= self.lo[j] and x0[j] + 1e-6 <= self.hi[j]:
                p = x0.copy(); p[j] += 1e-6
                q = x0.copy(); q[j] -= 1e-6
                fp = self.evaluate(p, 'AUDIT_key_probe', probe=dict(column=j, name=self.names[j], step=1e-6,
                                                                   mode='central', direction='+'))
                fm = self.evaluate(q, 'AUDIT_key_probe', probe=dict(column=j, name=self.names[j], step=1e-6,
                                                                   mode='central', direction='-'))
                one6 = (fp - fm)/(2*1e-6)
                rel = float(np.linalg.norm(one6 - J[:, j])/max(np.linalg.norm(J[:, j]), np.linalg.norm(one6), 1e-8))
            key_rows.append(dict(column=j, name=self.names[j], value=float(x0[j]),
                                 lower=float(self.lo[j]), upper=float(self.hi[j]),
                                 column_norm=float(norms[j]), one_sided_relative_difference=float(mismatch[j]),
                                 central_eps1e7_sse=float(J[:, j] @ J[:, j]),
                                 central_eps1e6_sse=(float(one6 @ one6) if one6 is not None else None),
                                 relative_difference_1e6_vs_1e7=rel,
                                 topology_changed=bool(info['column_signatures'][j] != base_sig),
                                 solver_niter=int(info['column_solver_niter'][j])))
        fraction = len(switched)/NX
        aborted = bool(fraction > .5)
        result = dict(
            baseline=dict(evaluation=base_row['evaluation'], cost=base_row['cost'],
                          block_sse=base_row['block_sse'], max_abs_qacc=base_row['max_abs_qacc'],
                          foot_load_ratio=base_row['foot_load_ratio'],
                          min_nonfoot_distance=base_row['min_nonfoot_distance'],
                          min_side=base_row['min_side'], gates=base_row['gates'],
                          topology_signature=base_sig, ncon=base_row['ncon'],
                          solver_niter=base_row['solver_niter']),
            absolute_eps=eps, columns=NX,
            audit_forward_span=[started, self.counts['forward']],
            column_norms=norms.tolist(), one_sided_relative_difference=mismatch.tolist(),
            max_column_norm=float(norms.max()), median_column_norm=float(np.median(norms)),
            key_columns=key_rows,
            topology=dict(switched_column_count=len(switched), switched_columns=switched,
                          fraction=float(fraction), abort_threshold=.5, aborted=aborted,
                          note='each entry is the contact topology signature of that column probe; a '
                               'switched column means the finite difference there spans two contact '
                               'topologies and is not a derivative of a smooth map'),
            steps_modes={m: int(sum(1 for v in steps.values() if v['mode'] == m))
                         for m in set(v['mode'] for v in steps.values())},
            x_scale_rule='floor=max(1e-8, max_column_norm*1e-8); x_scale=clip(1/max(norm,floor),1e-8,1e8)')
        floor = max(1e-8, float(norms.max())*1e-8)
        result['x_scale'] = np.clip(1/np.maximum(norms, floor), 1e-8, 1e8).tolist()
        result['column_norm_floor'] = floor
        dump(O/'AUDIT_JAC_STABILITY.json', result)
        return result, info, base_row

    # -------------------------------------------------------------- segments
    def run_segment(self, tag, start, x_scale, start_cost=None, eps=EPS_AUDIT):
        steps = {}
        jacs = []
        self.counts['least_squares'] += 1

        def jac(at):
            self.counts['jacobian_calls'] += 1
            info = self.jacobian_explicit(at, tag, eps, steps)
            jacs.append(info['jacobian'])
            np.savez_compressed(O/(tag + '_JAC_%02d.npz' % len(jacs)), jacobian=info['jacobian'],
                                base_residual=info['base'], column_norms=info['column_norms'],
                                one_sided_relative_difference=info['one_sided_relative_difference'],
                                plus=info['plus'], minus=info['minus'])
            return info['jacobian']

        before = time.time()
        sol = least_squares(lambda at: self.evaluate(at, tag + '_f'), start, jac=jac,
                            bounds=(self.lo, self.hi), x_scale=x_scale, method='trf', loss='linear',
                            max_nfev=120, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        _, solver_row, solver_arrays = self.evaluate(sol.x, tag + '_solver_x_validation', retain=True)
        np.savez_compressed(O/(tag + '_SOLVER_X_STATE.npz'), **solver_arrays)
        best = self.segment_best.get(tag)
        if self.segment_arrays.get(tag) is not None:
            np.savez_compressed(O/(tag + '_BEST_STATE.npz'), **self.segment_arrays[tag])
        result = dict(tag=tag, success=bool(sol.success), status=int(sol.status), message=str(sol.message),
                      nfev=int(sol.nfev), njev=int(sol.njev), optimality=float(sol.optimality),
                      cost_half_sse=float(sol.cost), wall_seconds=time.time() - before,
                      jacobian_calls=len(jacs), max_nfev=120, absolute_eps=eps,
                      jacobian_steps_modes={m: int(sum(1 for v in steps.values() if v['mode'] == m))
                                            for m in set(v['mode'] for v in steps.values())},
                      start=start, start_sse=start_cost, solver_x=sol.x,
                      solver_x_metrics=dict(cost=solver_row['cost'], max_abs_qacc=solver_row['max_abs_qacc'],
                                            foot_load_ratio=solver_row['foot_load_ratio'],
                                            min_nonfoot_distance=solver_row['min_nonfoot_distance'],
                                            min_side=solver_row['min_side'], gates=solver_row['gates'],
                                            evaluation=solver_row['evaluation'],
                                            topology_signature=solver_row['topology_signature']),
                      segment_best=best,
                      note='solver.x is the optimizer endpoint, not the reported selection; selection is the '
                           'strict minimum SSE over every evaluation')
        dump(O/(tag + '_RESULT.json'), result)
        return result, sol, solver_row

    # ---------------------------------------------------------------- gates
    def final_gate(self, x, prefix='FINAL'):
        self.d = mujoco.MjData(self.m)
        r, row, arrays = self.evaluate(x, 'final_validation', retain=True)
        m, d = self.m, self.d
        moment = np.zeros((m.nu, m.nv))
        for j in range(m.nu):
            a = int(d.moment_rowadr[j])
            n = int(d.moment_rownnz[j])
            moment[j, d.moment_colind[a:a + n]] = d.actuator_moment[a:a + n]
        reconstruction = moment.T @ d.actuator_force
        err = float(abs(reconstruction - d.qfrc_actuator).max())
        after, rows = parameter_digest(m)
        gates = dict(row['gates'])
        gates.update(moment_reconstruction=bool(err <= 1e-9), model_parameters_unchanged=bool(after == self.model_before),
                     xml_unchanged=bool(sha(S/'MODEL.xml') == EXPECTED))
        arrays.update(moment=moment, reconstructed_qfrc_actuator=reconstruction,
                      group_matrix=self.G, seed_qpos=self.seed_qpos, hinge_qpos_addresses=self.qa)
        np.savez_compressed(O/(prefix + '_STATE.npz'), **arrays)
        result = dict(passed=bool(all(gates.values())), gates=gates, metrics=row,
                      moment_reconstruction_error=err, model_parameters_before=self.model_before,
                      model_parameters_after=after,
                      fresh_evaluation='new independent MjData, mj_resetData, full state reconstruction from x, '
                                       'single mj_forward; same single compiled model')
        dump(O/(prefix + '_GATE.json'), result)
        dump(O/(prefix + '_MODEL_PARAMETERS.json'), dict(before=self.model_before, after=after, fields=rows))
        dump(O/'MESH_DISTANCE_CROSSCHECK.json',
             dict(max_abs_error=float(np.max(abs(arrays['compiled_vertex_distances'] - arrays['nonfoot_distances']))),
                  cutoff_hits=int(np.sum(arrays['nonfoot_distances'] >= 1000.)),
                  geom_distance=arrays['nonfoot_distances'], compiled_vertex_distance=arrays['compiled_vertex_distances'],
                  method='compiled mesh vertices transformed by geom pose vs mj_geomDistance; zero extra calls'))
        dump(O/'WRENCH_DECOMPOSITION.json', self.wrench(arrays))
        return result, arrays, row

    def wrench(self, arrays):
        rows = arrays['contact_rows']
        pos, force = rows[:, 3:6], rows[:, 6:9]
        net = force.sum(0)
        root = arrays['root_xpos']
        com = arrays['subtree_com'][self.root]
        m_root = np.cross(pos - root, force).sum(0)
        m_com = np.cross(pos - com, force).sum(0)
        fz = arrays['foot_force'][:, 2]
        return dict(
            scope='exact arithmetic on the final-state arrays of this run; contact points are the real '
                  'soft-contact positions saved by the same forward pass (no approximation, no extra forward)',
            weight_native=self.weight, contact_row_schema=CONTACT_ROW_SCHEMA,
            n_contact_rows=int(len(rows)), contact_distances=rows[:, 2].tolist(),
            contact_positions=pos.tolist(), contact_world_forces=force.tolist(),
            per_contact_vertical_force=force[:, 2].tolist(),
            net_contact_force=net.tolist(),
            net_contact_force_from_rows_check=float(abs(net - arrays['foot_force'].sum(0)).max()),
            vertical_over_weight=float(net[2]/self.weight),
            horizontal_over_vertical=float(np.linalg.norm(net[:2])/net[2]),
            moment_about_root=m_root.tolist(), moment_about_root_over_weight=float(np.linalg.norm(m_root)/self.weight),
            moment_about_root_subtree_com=m_com.tolist(),
            moment_about_root_subtree_com_over_weight=float(np.linalg.norm(m_com)/self.weight),
            root_xpos=root.tolist(), root_subtree_com=com.tolist(),
            per_leg_vertical_force=fz.tolist(), zero_or_negative_leg_count=int(np.sum(fz <= 0)),
            root_qacc=arrays['qacc'][:6].tolist(), max_abs_qacc=float(np.max(abs(arrays['qacc']))),
            solver_niter=arrays['solver_niter'].tolist(),
            note='moment columns are exact for this candidate pose (subtree COM taken from the same forward); '
                 'unlike the round-7 wrench diagnosis these are not approximations about a seed COM')

    def compare_previous(self, final_row):
        out = {'round8_current': dict(max_abs_qacc=final_row['max_abs_qacc'],
                                      foot_load_ratio=final_row['foot_load_ratio'],
                                      min_nonfoot_distance=final_row['min_nonfoot_distance'],
                                      min_side=final_row['min_side'], cost=final_row['cost'])}
        for tag, directory in (('round6_search60', R6), ('round7_continuation150', R7)):
            item = {}
            with np.load(directory/'FINAL_STATE.npz') as z:
                item['max_abs_qacc'] = float(np.max(abs(z['qacc'])))
                item['foot_load_ratio'] = float(z['foot_force'][:, 2].sum()/self.weight)
                item['min_nonfoot_distance'] = float(np.min(z['nonfoot_distances']))
                item['min_side'] = float(np.min(z['body_side']))
            gate = json.loads((directory/'FINAL_GATE.json').read_text())
            item['gate_passed'] = bool(gate['passed'])
            item['failed_gates'] = [k for k, v in gate['gates'].items() if not v]
            out[tag] = item
        out['scope'] = 'read-only comparison of saved FINAL_STATE/FINAL_GATE of earlier rounds; no re-run'
        dump(O/'SEGMENT_COMPARISON.json', out)
        return out

    def dump_topology(self):
        dump(O/'TOPOLOGY_TRANSITIONS.json',
             dict(scope='reconstructed from the evaluation sequence of this run (including every finite-difference '
                        'probe); each entry marks a contact topology signature change, i.e. an evaluation where the '
                        'objective is not differentiable',
                  transitions=self.topology_events,
                  count=len(self.topology_events),
                  note='the reported points after such a change must not be read as an interior optimum of a '
                       'smooth map'))


def load(experiment_mode):
    e = Experiment(experiment_mode)
    projection, variables, old = e.seed_artifacts()
    return e, projection, variables, old


def inputs_identity(extra=()):
    files = [Path(__file__).resolve(), S/'MODEL.xml', S/'BODY.npz', S/'MAPPING.json',
             O/'PREREGISTRATION.json', R6/'VARIABLES.json', R6/'SEED_BOUND_PROJECTION.json',
             R6/'FINAL_STATE.npz', R7/'FINAL_STATE.npz']
    files.extend(Path(p) for p in extra)
    return {str(p): sha(p) for p in files}


def invariance_mode():
    assert (O/'PREREGISTRATION.json').is_file(), 'preregistration must exist before execution'
    assert not (O/'INVARIANCE.json').exists() and not (O/'EXECUTION.json').exists()
    e = None
    error = None
    started = time.time()
    try:
        e, projection, variables, old = load('invariance')
        dump(O/'MODEL_PARAMETERS_BEFORE.json', dict(sha256=e.model_before, fields=e.before_rows))
        dump(O/'SEED_SELECTION.json', e.seed)
        result = e.invariance()
        dump(O/'INVARIANCE.json', result)
        execution = dict(stage='step1_invariance', error=None, counts=e.counts, wall_seconds=time.time() - started,
                         all_passed=result['all_passed'], exit_code=0 if result['all_passed'] else 1)
    except Exception:
        error = traceback.format_exc()
        dump(O/'INVARIANCE_FAILURE.json', dict(error=error, counts=e.counts if e else None))
        execution = dict(stage='step1_invariance', error=error, counts=e.counts if e else None,
                         wall_seconds=time.time() - started, all_passed=False, exit_code=1)
    dump(O/'EXECUTION_INVARIANCE.json', execution)
    print(json.dumps(execution, ensure_ascii=False, indent=2), flush=True)
    return execution['exit_code']


def run():
    assert (O/'PREREGISTRATION.json').is_file(), 'preregistration must exist before execution'
    assert (O/'INVARIANCE.json').is_file(), 'step-1 invariance record must exist before the main sequence'
    assert not (O/'EXECUTION.json').exists() and not (O/'EVALUATIONS.jsonl').exists(), 'never restart'
    prereg = json.loads((O/'PREREGISTRATION.json').read_text())
    import shutil
    shutil.copyfile(Path(__file__).resolve(), O/'search_native_equilibrium_reposed.py')
    inputs = inputs_identity()
    dump(O/'INPUT_IDENTITIES.json', inputs)
    e = None
    error = None
    gate = None
    started = time.time()
    results = {}
    pre_log = {}
    try:
        e, projection, variables, old = load('run')
        dump(O/'MODEL_PARAMETERS_BEFORE.json', dict(sha256=e.model_before, fields=e.before_rows))
        dump(O/'SEED_SELECTION.json', e.seed)
        dump(O/'SEED_BOUND_PROJECTION.json', projection)
        dump(O/'VARIABLES.json', variables)
        dump(O/'SEED_COMPARISON.json', old)
        assert old['reproduction_max_abs_error'] <= 1e-12, 'round-6 initial point reproduction failed'
        inv = e.invariance()
        dump(O/'INVARIANCE_MAIN.json', inv)
        assert inv['all_passed'], 'invariance check failed inside the main sequence; abort'
        pre_log = dict(forward=int(e.counts['forward']), mj_resetData=int(e.counts['mj_resetData']),
                       geom_distance=int(e.counts['geom_distance']),
                       contact_force=int(e.counts['contact_force']),
                       note='counts consumed by the step-1 invariance re-check; those forwards are recorded in '
                            'INVARIANCE_MAIN.json, not in EVALUATIONS.jsonl')
        e.log = (O/'EVALUATIONS.jsonl').open('x')
        audit, audit_info, base_row = e.audit()
        results['audit'] = dict(aborted=audit['topology']['aborted'], baseline_metrics=audit['baseline'])
        if audit['topology']['aborted']:
            dump(O/'SEGMENT_SKIPPED.json', dict(reason='audit probe contact topology switched for more than the '
                                                      'preregistered 50 percent of columns; baseline is not '
                                                      'differentiable, no optimizer segment launched',
                                               fraction=audit['topology']['fraction']))
        else:
            x_scale = np.array(audit['x_scale'])
            A, solA, rowA = e.run_segment('SEG_A', e.x0.copy(), x_scale, start_cost=base_row['cost'])
            results['segment_A'] = {k: v for k, v in A.items() if k not in ('solver_x', 'start')}
            shares = {}
            for name, (a, b) in BLOCKS.items():
                shares[name] = dict(start=float(base_row['block_sse'][name]/base_row['cost']),
                                    end=float(rowA['block_sse'][name]/rowA['cost']))
            change = max(abs(v['end'] - v['start']) for v in shares.values())
            dominant_start = max(base_row['block_sse'], key=base_row['block_sse'].get)
            dominant_end = max(rowA['block_sse'], key=rowA['block_sse'].get)
            ratio = (rowA['max_abs_qacc']/base_row['max_abs_qacc']) if base_row['max_abs_qacc'] else float('inf')
            triggers = dict(no_feasible_candidate=len(e.feasible) == 0,
                            block_share_shift_ge_0_05=bool(change >= .05),
                            dominant_block_changed=bool(dominant_start != dominant_end),
                            qacc_ratio_ge_10_or_le_0_1=bool(ratio >= 10 or ratio <= .1))
            launch = bool(triggers['no_feasible_candidate'] and
                          (triggers['block_share_shift_ge_0_05'] or triggers['dominant_block_changed']
                           or triggers['qacc_ratio_ge_10_or_le_0_1']))
            decision = dict(block_shares=shares, max_abs_share_change=float(change),
                            dominant_block_start=dominant_start, dominant_block_end=dominant_end,
                            max_abs_qacc_start=base_row['max_abs_qacc'],
                            max_abs_qacc_seg_A_end=rowA['max_abs_qacc'], qacc_ratio=float(ratio),
                            triggers=triggers, launch_segment_B=launch,
                            start_point_of_B=(e.segment_best.get('SEG_A') or {}).get('evaluation'),
                            reason='preregistered criterion: residual block composition change between the start '
                                   'and the end of segment A while no observed-feasible candidate exists')
            dump(O/'SEGMENT_B_DECISION.json', decision)
            results['segment_B_decision'] = {k: v for k, v in decision.items() if k != 'block_shares'}
            if launch:
                startB = e.segment_arrays['SEG_A']['x'].copy()
                B, solB, rowB = e.run_segment('SEG_B', startB, x_scale, start_cost=e.segment_best['SEG_A']['cost'])
                results['segment_B'] = {k: v for k, v in B.items() if k not in ('solver_x', 'start')}
            else:
                dump(O/'SEGMENT_B_SKIPPED.json', dict(reason='preregistered trigger not satisfied',
                                                      triggers=triggers, decision=decision))
        gate, final_arrays, final_row = e.final_gate(e.best['x'])
        results['final'] = dict(passed=gate['passed'], failed=[k for k, v in gate['gates'].items() if not v],
                                metrics=dict(cost=final_row['cost'], max_abs_qacc=final_row['max_abs_qacc'],
                                             foot_load_ratio=final_row['foot_load_ratio'],
                                             min_nonfoot_distance=final_row['min_nonfoot_distance'],
                                             min_side=final_row['min_side'],
                                             block_sse=final_row['block_sse'],
                                             solver_niter=final_row['solver_niter'],
                                             topology_signature=final_row['topology_signature']))
        results['comparison'] = e.compare_previous(final_row)
        e.dump_topology()
        dump(O/'FEASIBLE_LIST.json', dict(count=len(e.feasible), candidates=e.feasible,
                                          note='every evaluation satisfying all hard gates, in evaluation order; '
                                               'saved as FEASIBLE_*.npz with contact points and forces'))
        dump(O/'BEST_MAXQACC.json', dict(best_max_abs_qacc=e.best_acc))
        dump(O/'SELECTION.json', dict(global_min_sse=e.best, best_max_abs_qacc=e.best_acc,
                                      per_segment_best=e.segment_best,
                                      rule='strict minimum of the complete objective SSE over every real '
                                           'evaluation including finite-difference probes; no gate-driven '
                                           'reselection'))
    except Exception:
        error = traceback.format_exc()
        dump(O/'FAILURE.json', dict(error=error, counts=e.counts if e else None,
                                    roles=e.roles if e else None))
    finally:
        if e is not None and e.log is not None:
            e.log.close()
        if e is not None:
            after, rows = parameter_digest(e.m)
            dump(O/'MODEL_PARAMETERS_AFTER.json', dict(sha256=after, fields=rows))
            unchanged_model = bool(after == e.model_before)
            if e.topology_events:
                e.dump_topology()
        else:
            unchanged_model = False
        unchanged = all(sha(Path(p)) == h for p, h in inputs.items())
        execution = dict(error=error, counts=e.counts if e else None, roles=e.roles if e else None,
                         pre_log_counts=pre_log, wall_seconds=time.time() - started, results=results,
                         input_hashes_unchanged=unchanged, model_parameters_unchanged=unchanged_model,
                         observed_feasible=e.feasible if e else None,
                         engineering_static_gate_passed=bool(gate and gate['passed'] and unchanged
                                                             and unchanged_model and error is None),
                         exit_code=0 if error is None and unchanged else 1)
        dump(O/'EXECUTION.json', execution)
        print(json.dumps({k: v for k, v in execution.items() if k != 'observed_feasible'},
                         ensure_ascii=False, indent=2, default=lambda a: a.tolist()), flush=True)
    return execution['exit_code']


def verify():
    assert (O/'EXECUTION.json').is_file(), 'main sequence must have run'
    assert not (O/'INDEPENDENT_VERIFICATION.json').exists(), 'never restart verification'
    e = None
    error = None
    checks = {}
    errors = {}
    try:
        e, projection, variables, old = load('verify')
        with np.load(O/'FINAL_STATE.npz') as z:
            saved = {k: z[k].copy() for k in z.files}
        e.log = (O/'VERIFY_EVALUATIONS.jsonl').open('x')
        r, row, arrays = e.evaluate(saved['x'], 'independent_validation', retain=True)
        np.savez_compressed(O/'VERIFY_STATE.npz', **arrays)
        common = [k for k in saved if k in arrays and getattr(saved[k], 'size', 0) and
                  np.shape(saved[k]) == np.shape(arrays[k])]
        errors = {k: float(np.max(abs(np.asarray(saved[k], float) - np.asarray(arrays[k], float))))
                  for k in common if np.asarray(saved[k]).dtype.kind in 'fiu' and k != 'evaluation'}
        e.log.close()
        rows = [json.loads(line) for line in (O/'EVALUATIONS.jsonl').open()]
        execution = json.loads((O/'EXECUTION.json').read_text())
        gate_saved = json.loads((O/'FINAL_GATE.json').read_text())
        wrench_saved = json.loads((O/'WRENCH_DECOMPOSITION.json').read_text())
        topology_saved = json.loads((O/'TOPOLOGY_TRANSITIONS.json').read_text())
        empty = np.zeros((0, 12))
        prev_sig, recomputed, pairs_at = None, [], None
        for r_ in rows:
            if r_['topology_signature'] != prev_sig:
                recomputed.append(dict(evaluation=r_['evaluation'], role=r_['role'],
                                       signature=r_['topology_signature']))
                prev_sig = r_['topology_signature']
        saved_seq = [(t_['evaluation'], t_['signature']) for t_ in topology_saved['transitions']]
        recomputed_seq = [(t_['evaluation'], t_['signature']) for t_ in recomputed]
        wrench_rebuilt = e.wrench(saved)
        full_rows = [r_ for r_ in rows if 'residual' in r_]
        best_row = min(rows, key=lambda r_: r_['cost'])
        checks = dict(
            fresh_arrays_match=bool(errors and max(errors.values()) <= 1e-8),
            minimum_objective_selected=bool(abs(best_row['cost'] - float(saved['residual'] @ saved['residual'])) <= 1e-12),
            all_counts_match=bool(len(rows) == execution['counts']['forward'] - execution['pre_log_counts']['forward']
                                  == execution['counts']['mj_resetData'] - execution['pre_log_counts']['mj_resetData']),
            no_physical_or_neural_steps=bool(execution['counts']['body_step'] == execution['counts']['neural_step'] == 0),
            block_sse_adds_to_cost=bool(all(abs(r_['cost'] - sum(r_['block_sse'].values())) <= 1e-9*max(1., r_['cost'])
                                             for r_ in rows)),
            residual_cost_consistent=bool(all(abs(float(np.dot(r_['residual'], r_['residual'])) - r_['cost']) <= 1e-12
                                              for r_ in full_rows)),
            all_rows_in_bounds=bool(all(np.all(np.array(r_['x']) >= e.lo) and np.all(np.array(r_['x']) <= e.hi)
                                        for r_ in full_rows)),
            probe_rows_recorded=bool(all('block_sse' in r_ and 'gates' in r_ and 'topology_signature' in r_
                                         and 'solver_niter' in r_ for r_ in rows)),
            topology_transitions_match=bool(saved_seq == recomputed_seq),
            wrench_matches_saved=bool(np.allclose(np.array(wrench_saved['net_contact_force']),
                                                  np.array(wrench_rebuilt['net_contact_force']), atol=1e-12, rtol=0)
                                      and abs(wrench_saved['vertical_over_weight'] - wrench_rebuilt['vertical_over_weight']) <= 1e-12
                                      and np.allclose(np.array(wrench_saved['moment_about_root']),
                                                      np.array(wrench_rebuilt['moment_about_root']), atol=1e-12, rtol=0)),
            wrench_row_consistency=bool(wrench_saved['net_contact_force_from_rows_check'] <= 1e-12),
            final_gates_recomputed=bool(all(gate_saved['gates'][k] == row['gates'][k]
                                            for k in row['gates'] if k in gate_saved['gates'])),
            gate_booleans_match_arrays=bool(all(gate_saved['gates'][k] ==
                                                bool(row['gates'][k]) for k in gate_saved['gates'] if k in row['gates'])),
            model_parameters_unchanged=bool(gate_saved['gates']['model_parameters_unchanged']
                                            and json.loads((O/'EXECUTION.json').read_text())['model_parameters_unchanged']),
            input_hashes_unchanged=bool(all(sha(Path(p)) == h for p, h in
                                            json.loads((O/'INPUT_IDENTITIES.json').read_text()).items())),
            no_rollout_files=bool(not any(p.name.upper().startswith(('TRAJ', 'ROLLOUT', 'VIDEO'))
                                          for p in O.iterdir())),
        )
        passed = all(checks.values())
        result = dict(passed=passed, checks=checks, array_repeat_errors=errors,
                      counts=e.counts, row_count=len(rows), full_row_count=len(full_rows),
                      probe_row_count=len(rows) - len(full_rows),
                      selection_evaluation=best_row['evaluation'], selection_role=best_row['role'],
                      selection_cost=best_row['cost'],
                      observed_feasible_count=sum(1 for r_ in rows if r_['observed_physics_feasible']),
                      static_gate_passed=gate_saved['passed'],
                      failed_gates=[k for k, v in gate_saved['gates'].items() if not v],
                      metrics=gate_saved['metrics'],
                      wrench=dict(vertical_over_weight=wrench_saved['vertical_over_weight'],
                                  horizontal_over_vertical=wrench_saved['horizontal_over_vertical'],
                                  moment_about_root_over_weight=wrench_saved['moment_about_root_over_weight'],
                                  moment_about_root_subtree_com_over_weight=wrench_saved['moment_about_root_subtree_com_over_weight'],
                                  per_leg_vertical_force=wrench_saved['per_leg_vertical_force']),
                      scope='separate process, new model load, no new optimization; this re-checks saved arrays and '
                            'records only')
        dump(O/'INDEPENDENT_VERIFICATION.json', result)
        e.log.close() if not e.log.closed else None
    except Exception:
        error = traceback.format_exc()
        dump(O/'VERIFICATION_FAILURE.json', dict(error=error, checks=checks, errors=errors))
    if e is not None and e.log is not None and not e.log.closed:
        e.log.close()
    manifest_targets = [p for p in sorted(O.rglob('*')) if p.is_file()
                        and p.name not in ('MANIFEST_SHA256.json', 'REPORT.zh-CN.md')]
    report = build_report(checks, errors, error)
    (O/'REPORT.zh-CN.md').write_text(report)
    manifest = {str(p): dict(sha256=sha(p), importance=importance(p.name)) for p in manifest_targets}
    manifest[str(O/'REPORT.zh-CN.md')] = dict(sha256=sha(O/'REPORT.zh-CN.md'), importance='重要辅助')
    dump(O/'MANIFEST_SHA256.json', manifest)
    print(json.dumps(dict(passed=all(checks.values()) if checks else False, error=error,
                          checks=checks), ensure_ascii=False, indent=2), flush=True)
    return 0 if (checks and all(checks.values()) and error is None) else 1


CORE = ('PREREGISTRATION.json', 'search_native_equilibrium_reposed.py', 'INVARIANCE.json',
        'INVARIANCE_MAIN.json', 'SEED_SELECTION.json', 'SEED_BOUND_PROJECTION.json', 'SEED_COMPARISON.json',
        'VARIABLES.json', 'EVALUATIONS.jsonl', 'AUDIT_JAC_STABILITY.json', 'AUDIT_JAC.npz',
        'AUDIT_BASE_STATE.npz', 'TOPOLOGY_TRANSITIONS.json', 'SEGMENT_A_RESULT.json', 'SEGMENT_A_BEST_STATE.npz',
        'SEGMENT_A_SOLVER_X_STATE.npz', 'SEGMENT_B_DECISION.json', 'FINAL_STATE.npz', 'FINAL_GATE.json',
        'WRENCH_DECOMPOSITION.json', 'FEASIBLE_LIST.json', 'SELECTION.json', 'INDEPENDENT_VERIFICATION.json')


def importance(name):
    if name in CORE or name.startswith(('SEGMENT_B_', 'SEG_B_', 'SEG_A_JAC_', 'FEASIBLE_', 'BEST_', 'VERIFY_')):
        return '核心必需'
    if name.endswith('.npz') or name.endswith('.jsonl'):
        return '核心必需'
    return '重要辅助'


def build_report(checks, errors, error):
    def read(name):
        p = O/name
        return json.loads(p.read_text()) if p.is_file() else {}
    execution = read('EXECUTION.json')
    gate = read('FINAL_GATE.json')
    inv = read('INVARIANCE_MAIN.json')
    selection = read('SELECTION.json')
    feasible = read('FEASIBLE_LIST.json')
    decision = read('SEGMENT_B_DECISION.json')
    audit = read('AUDIT_JAC_STABILITY.json')
    seed = read('SEED_SELECTION.json')
    proj = read('SEED_BOUND_PROJECTION.json')
    old = read('SEED_COMPARISON.json')
    wrench = read('WRENCH_DECOMPOSITION.json')
    transitions = read('TOPOLOGY_TRANSITIONS.json')
    comparison = read('SEGMENT_COMPARISON.json')
    verification = read('INDEPENDENT_VERIFICATION.json')
    t = []
    t.append('# S2/S5 第8轮：稳态预热初姿态 ＋ 根姿态自由度 的原生静态软接触平衡搜索\n\n')
    t.append('工程静态硬门：**%s**；独立纯数组复核：**%s**。零身体步、零神经步、无 rollout、无视频。'
             '未通过全部硬门即未完成「静态平衡候选」，局部失败不证明全局无解，绝对力值生物标定仍 NOT_ESTABLISHED。\n\n'
             % ('通过' if gate.get('passed') else '未通过',
                '通过' if (checks and all(checks.values()) and not error) else '未通过'))
    t.append('## 1 平凡方向核验（先做，已记录）\n\n')
    t.append('在原始 t=0.15 沉降姿态与稳态预热 seed 姿态上，各用「基准 mj_forward ＋ 只平移根 x/y 的一次 mj_forward」对，'
             '比较全 48 维 qacc、6 足接触力、41 个非足 mesh 到 floor 的距离、非足合力、ncon 与 solver_niter。\n\n')
    t.append('- 结论：**%s**。' % ('水平平移是浮点容差内的平凡不变方向' if inv.get('all_passed') else '核验失败'))
    if inv.get('cases'):
        dq = max(c['max_abs_qacc_diff'] for c in inv['cases'])
        df = max(c['max_abs_foot_force_diff'] for c in inv['cases'])
        dd = max(c['max_abs_nonfoot_distance_diff'] for c in inv['cases'])
        dc = max(c['max_abs_contact_position_shift_error'] for c in inv['cases'])
        t.append('实测最大偏差：Δqacc %.3e、Δ足力 %.3e、Δ非足净空 %.3e、接触点平移误差 %.3e（容差 %.1e/%.1e/%.1e/%.1e）；'
                 '全部 %d 组 ncon 与 solver_niter 相同。\n' % (dq, df, dd, dc,
                 inv['tolerance']['qacc'], inv['tolerance']['foot_force'], inv['tolerance']['nonfoot_distance'],
                 inv['tolerance']['contact_position'], len(inv['cases'])))
        t.append('- 因此**根 x/y 固定为 seed 值**（%.6f, %.6f），不进变量、不进正则；这不是锁根模型，自由根模型照旧，'
                 '全 48 维 qacc 仍逐项照检。\n\n' % tuple(json.loads((O/'VARIABLES.json').read_text())['root_xy_fixed']))
    t.append('## 2 变量与初值定义\n\n')
    v = json.loads((O/'VARIABLES.json').read_text())
    t.append('- 112 变量：x[0:42] 42 个受限 hinge 关节角（qpos 7..48）；x[42] 根 z；x[43:46] 相对 seed 四元数的 body-frame '
             '旋转矢量（±0.25 rad，初值恰为 0 即现姿态）；x[46:112] 66 个共享池激活（[1e-4,1]）。16 个未映射肌肉保持 1e-4 floor。'
             '根 x/y 固定。plant/geometry/lengthrange/gain/bias/dyn/接触参数一律未动。\n\n')
    t.append('- 初值帧：请求 t=1.2，实际取稳态预热窗 [%.2f,%.2f) 内**六腿竖直力全为正**且最接近 1.2 的帧 t=%.4f（index %d），'
             '逐腿 Fz=%s。%s\n' % (seed['steady_window'][0], seed['steady_window'][1], seed['chosen_time'],
                                   seed['chosen_index'],
                                   [round(x, 4) for x in seed['chosen_per_leg_vertical_force']],
                                   ('**偏差记录**：' + seed['deviation']) if 'deviation' in seed else ''))
    if seed.get('literal'):
        lt = seed['literal']
        t.append('- 字面 t=1.2 帧（未被用作主初值）：逐腿 Fz=%s，只有 %d 条腿受力；该帧数值已完整保存在 SEED_SELECTION.json。\n'
                 % ([round(x, 4) for x in lt['per_leg_vertical_force']], lt['leg_count']))
    t.append('- 关节 bounds 投影（未悄悄改初值）：%d 处，%s。\n' % (len(proj.get('changes', [])),
             '；'.join('%s %.6f→%.6f' % (c['name'], c['old'], c['new']) for c in proj.get('changes', [])) or '无'))
    t.append('- t=0.15 老初值仅作对照：按同一代码重建与第6轮 solver_initial 的最大差 %.3e（要求 ≤1e-12）。\n\n'
             % old.get('reproduction_max_abs_error', float('nan')))
    t.append('## 3 目标、门与求解结构\n\n')
    t.append('- 目标与第6/7轮完全一致：48 维 qacc/9810、41 个非足 mesh 真实 floor 距离 ≥.02 惩罚×100、36 点 Thorax 坐标系分侧 ≥.02 '
             '惩罚×100、总足 Fz/W−1 ×10、姿态正则 1e-6（现覆盖 46 个姿态变量）。共 %d 行。\n' % NROW)
    t.append('- 每次评估 resetData、qvel=0、ctrl=act、外力清零，真实软接触决定力；每次评估记录真实 forward 计数、接触拓扑签名、'
             '五块 SSE、solver_niter 与 20 项硬门布尔。\n\n')
    if audit:
        b = audit['baseline']
        t.append('## 4 基线评估与 112 列绝对 1e-7 雅可比稳定性\n\n')
        t.append('- 基线（x0）：SSE %.6f，max|qacc| %.4f，足 Fz/W %.6f，最小非足净空 %.5f，最小分侧 %.5f，ncon %d，'
                 'solver_niter %s。\n' % (b['cost'], b['max_abs_qacc'], b['foot_load_ratio'],
                                          b['min_nonfoot_distance'], b['min_side'], b['ncon'], b['solver_niter']))
        t.append('- 列范数最大 %.4g、中位 %.4g；单边相对差最大 %.4g（=1 表示该列左右斜率反向，典型是跨限位/跨接触拓扑）。\n'
                 % (audit['max_column_norm'], audit['median_column_norm'],
                    max(audit['one_sided_relative_difference'])))
        t.append('- 探针级接触拓扑：%d/%d 列发生拓扑切换（比例 %.4f，预登记中止阈值 0.5 → %s）。\n'
                 % (audit['topology']['switched_column_count'], NX, audit['topology']['fraction'],
                    '中止优化' if audit['topology']['aborted'] else '继续'))
        if audit['topology']['switched_columns']:
            t.append('  切换列（不可微处）：%s。\n' % audit['topology']['switched_columns'])
        t.append('- 关键列 1e-6 与 1e-7 中心差分对比：\n\n```json\n%s\n```\n\n'
                 % json.dumps([{k: r[k] for k in ('column', 'name', 'column_norm', 'one_sided_relative_difference',
                                                  'relative_difference_1e6_vs_1e7', 'topology_changed',
                                                  'solver_niter')} for r in audit['key_columns']],
                              ensure_ascii=False, indent=1))
    t.append('## 5 两段有界最小二乘\n\n')
    for tag in ('SEG_A_RESULT.json', 'SEG_B_RESULT.json'):
        r = read(tag)
        if r:
            sb = r.get('segment_best') or {}
            t.append('- %s：max_nfev %d，实际 nfev %d / njev %d，status %s（%s），optimality %.4g，'
                     'solver.x SSE %.6f（max|qacc| %.4f，足 Fz/W %.6f）；段内严格最小 SSE %.6f（evaluation %s）。'
                     'solver.x 只是求解器终点，不当作最优性证书。\n'
                     % (r['tag'], r['max_nfev'], r['nfev'], r['njev'], r['status'], r['message'],
                        r['optimality'], r['cost_half_sse']*2,
                        r['solver_x_metrics']['max_abs_qacc'], r['solver_x_metrics']['foot_load_ratio'],
                        sb.get('cost', float('nan')), sb.get('evaluation')))
    if decision:
        t.append('- 段间依据（残差块构成）：起点占比 %s；A 段末占比 %s；最大占比变化 %.4f；主导块 %s→%s；'
                 'max|qacc| %.4f→%.4f（倍数 %.3g）；触发 B 段：**%s**。\n\n'
                 % ({k: round(x['start'], 4) for k, x in decision['block_shares'].items()},
                    {k: round(x['end'], 4) for k, x in decision['block_shares'].items()},
                    decision['max_abs_share_change'], decision['dominant_block_start'],
                    decision['dominant_block_end'], decision['max_abs_qacc_start'],
                    decision['max_abs_qacc_seg_A_end'], decision['qacc_ratio'],
                    decision['launch_segment_B']))
    t.append('## 6 最终候选与硬门\n\n')
    t.append('选择规则：全部评估（含差分探针）中**完整目标 SSE 严格最小**者，不按门重选。最终点 evaluation %s（role %s），SSE %.6f。\n\n'
             % (json.dumps(selection.get('global_min_sse', {}).get('evaluation')),
                selection.get('global_min_sse', {}).get('role'),
                selection.get('global_min_sse', {}).get('cost', float('nan'))))
    if gate:
        m = gate['metrics']
        t.append('| 指标 | 数值 | 门 |\n|---|---|---|\n')
        t.append('| max\\|qacc(48)\\| | %.6f | ≤ 0.1 |\n' % m['max_abs_qacc'])
        t.append('| 足 Fz/W | %.6f | [0.99,1.01] |\n' % m['foot_load_ratio'])
        t.append('| 最小非足净空 | %.6f | ≥ 0.02 |\n' % m['min_nonfoot_distance'])
        t.append('| 最小分侧 | %.6f | ≥ 0.02 |\n' % m['min_side'])
        t.append('| 非足力最大绝对值 | %.6g | == 0 |\n' % m['nonfoot_force_max_abs'])
        t.append('| max\\|act_dot\\| | %.3e | ≤ 1e-10 |\n' % m['act_dot_max_abs'])
        t.append('\n未通过项：%s。\n\n' % (', '.join([k for k, v in gate['gates'].items() if not v]) or '无'))
        t.append('逐项门：\n\n```json\n%s\n```\n\n' % json.dumps(gate['gates'], ensure_ascii=False, indent=1))
    t.append('- observed_feasible（全部硬门同时成立）候选数：**%d**。%s\n\n'
             % (feasible.get('count', 0),
                '候选已保存且**只报告父，未自行积分**。' if feasible.get('count', 0) else
                '没有任何候选通过全部硬门；因此本轮不交付静态平衡候选，也不进入交接。'))
    if wrench:
        t.append('## 7 缺口写成功率/力矩（含真实接触点）\n\n')
        t.append('- 接触点已保存（%d 行，schema %s），合力校验与足力一致残差 %.2e。\n'
                 % (wrench['n_contact_rows'], wrench['contact_row_schema'], wrench['net_contact_force_from_rows_check']))
        t.append('- 竖直合力/体重 = **%.6f**；水平合力/竖直合力 = **%.6f**；对根位置力矩/体重 = **%.6f**；'
                 '对根子树质心力矩/体重 = **%.6f**。\n'
                 % (wrench['vertical_over_weight'], wrench['horizontal_over_vertical'],
                    wrench['moment_about_root_over_weight'], wrench['moment_about_root_subtree_com_over_weight']))
        t.append('- 逐腿竖直力 %s；零/负竖直力腿数 %d。本轮力矩用同一前向给出的 d.subtree_com，是**真实值**，'
                 '不是第7轮用 seed COM 的近似。\n\n'
                 % ([round(x, 4) for x in wrench['per_leg_vertical_force']], wrench['zero_or_negative_leg_count']))
    if transitions.get('count'):
        t.append('## 8 接触拓扑切换位置（不可微点）\n\n')
        t.append('共 %d 处。前若干处：\n\n' % transitions['count'])
        for ev in transitions['transitions'][:12]:
            t.append('- evaluation %s（role %s）：ncon %d，floor 接触 %d，新增 %s，消失 %s。\n'
                     % (ev['evaluation'], ev['role'], ev['ncon'], ev['floor_contact_count'],
                        ev['appeared'][:4], ev['disappeared'][:4]))
        t.append('\n在这些位置目标函数**不可微**，无论是最小 SSE 点还是 solver 终点都不能读成光滑映射的内点最优。\n\n')
    if comparison:
        t.append('## 9 与第6/7轮对比（只读）\n\n```json\n%s\n```\n\n' % json.dumps(comparison, ensure_ascii=False, indent=1))
    if checks:
        t.append('## 10 独立纯数组复核（新进程、不新优化）\n\n```json\n%s\n```\n\n'
                 % json.dumps(checks, ensure_ascii=False, indent=1))
        t.append('数组重建最大差：%s。\n\n' % json.dumps({k: round(v, 9) for k, v in (errors or {}).items()}, ensure_ascii=False))
    if error:
        t.append('## 复核异常\n\n```\n%s\n```\n\n' % error)
    t.append('## 11 重要程度与清理\n\n')
    t.append('**核心必需**：PREREGISTRATION.json、本工具源码（执行版副本 search_native_equilibrium_reposed.py）、'
             'INVARIANCE.json / INVARIANCE_MAIN.json、SEED_SELECTION.json、SEED_BOUND_PROJECTION.json、VARIABLES.json、'
             'EVALUATIONS.jsonl、AUDIT_JAC_STABILITY.json 与 AUDIT_JAC.npz、TOPOLOGY_TRANSITIONS.json、'
             'SEGMENT_A/B 的 RESULT/决策/雅可比/best/solver.x 状态、FINAL_STATE.npz、FINAL_GATE.json、'
             'WRENCH_DECOMPOSITION.json、FEASIBLE_LIST.json 与 FEASIBLE_*.npz、SELECTION.json、BEST_MAXQACC_STATE.npz、'
             'INDEPENDENT_VERIFICATION.json。\n\n')
    t.append('**重要辅助**：MODEL_PARAMETERS_BEFORE/AFTER、INPUT_IDENTITIES、EXECUTION、SEGMENT_COMPARISON、SEED_COMPARISON、'
             'MESH_DISTANCE_CROSSCHECK、BEST_INCREMENTAL、VERIFY_STATE、VERIFY_EVALUATIONS、REPORT、MANIFEST_SHA256。\n\n')
    t.append('**清理**：本轮只在新建目录内写文件；未改任何冻结目录、未改原工具、未改 GOAL_TRACKER/ROADMAP、未 git 提交。'
             '写预登记前的一次性 scouting 前向对只用于确认水平平移方向，不保留文件、不参与结论。'
             '本轮没有产生确认可弃的中间文件，因此没有删除操作（如实说明，不宣称未执行的清理）。\n')
    return ''.join(t)


if __name__ == '__main__':
    if '--invariance' in sys.argv:
        sys.exit(invariance_mode())
    if '--verify' in sys.argv:
        sys.exit(verify())
    sys.exit(run())
