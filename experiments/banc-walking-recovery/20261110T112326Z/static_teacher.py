"""Quasistatic inverse-statics/position-IK algorithm teacher for S2/S5.

Only private FK data are written. Contact lambdas are optimization variables,
never applied forces. Outputs go through the unchanged finite-PD rate interface.
Native units; absolute SI NOT_ESTABLISHED. This is not walking evidence.
"""
from __future__ import annotations

import numpy as np
import mujoco as mj
from scipy.optimize import lsq_linear
from scipy.spatial import ConvexHull, QhullError

from physics import LEG_ORDER, W, MAX_RATE
from wave_reference import WaveReference


DEFAULT_CONFIG = dict(mode='measuredwave', gravity_ff=True,
                      stance_force_pref='uniform', step_s=.8, cycles=1,
                      lift=.12, stride=.36, target_tau_s=.05,
                      ik_ridge=.001, ik_iterations=5, command_dt_s=.005,
                      contact_transition_height=.03)


class HoldReference:
    def __init__(self, rest):
        self.rest = np.array(rest, copy=True)
        self.metadata = dict(label='hold actual settled feet; no motion or preload')

    def sample(self, t):
        if not np.isfinite(t):
            raise ValueError('time must be finite')
        return dict(target_xy=self.rest[:, :2].copy(), target_z=np.zeros(6),
                    velocity_xy=np.zeros((6, 2)), velocity_z=np.zeros(6),
                    stance=np.ones(6, dtype=bool), source_time_s=None, phase='hold')


def motor_target(q_des, qdot_des, tau_ff, kp, kd, bounds):
    """PD inverse, followed by native target bounds; no hidden physical writes."""
    unbounded = q_des + (tau_ff + kd * qdot_des) / kp
    return np.clip(unbounded, bounds[:, 0], bounds[:, 1]), unbounded


class StaticTeacher:
    """action(c, elapsed_s) -> (normalized 42-rate, JSON-ready diagnostics).

    Construct after settling. ``reference`` is replaceable with any object whose
    sample(t) follows WaveReference's contract. q_des is a full private qpos copy;
    its free root stays at the initial pose. The caller owns command timing.
    """
    def __init__(self, c, config=None):
        supplied = {} if config is None else dict(config)
        unknown = set(supplied) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError('Unknown static teacher options: '+str(sorted(unknown)))
        self.config = {**DEFAULT_CONFIG, **supplied}
        cfg = self.config
        if cfg['mode'] not in ('hold', 'measuredwave'):
            raise ValueError('mode must be hold or measuredwave')
        if cfg['stance_force_pref'] not in ('uniform', 'current'):
            raise ValueError('stance_force_pref must be uniform or current')
        for k in ('target_tau_s', 'ik_ridge', 'command_dt_s', 'step_s', 'stride', 'contact_transition_height'):
            if not np.isfinite(cfg[k]) or cfg[k] <= 0:
                raise ValueError(k+' must be finite and positive')
        if not np.isfinite(cfg['lift']) or cfg['lift'] < 0:
            raise ValueError('lift must be finite and nonnegative')
        for k in ('ik_iterations', 'cycles'):
            if isinstance(cfg[k], (bool, np.bool_)) or not isinstance(cfg[k], (int, np.integer)) or cfg[k] < 1:
                raise ValueError(k+' must be a positive integer')
        if not isinstance(cfg['gravity_ff'], (bool, np.bool_)):
            raise ValueError('gravity_ff must be boolean')
        self._validate_physics(c)
        self.model = c.plant
        self.shadow = mj.MjData(self.model)
        self.q_des = c.d.qpos.copy()  # actual settled angles, NOT actuator target/q0
        self.initial_qpos = self.q_des.copy()
        self.bounds = self.model.jnt_range[c.actuator.joint_ids].copy()
        self.target_bounds = c.actuator.target_bounds_rad.copy()
        self.qa, self.va = c.a_qpos.copy(), c.a_dof.copy()
        self.initial_rot = c.d.xmat[c.th].reshape(3, 3).copy()
        self.initial_origin = c.d.xpos[c.th].copy()
        measure = c.measure()
        self.rest = measure['foot'].copy()
        self.rest_pad_bottom = measure['padz'].copy()
        self.leg_act = []
        for leg in LEG_ORDER:
            indices = np.array([i for i, j in enumerate(c.actuator.joint_ids)
                                if (self.model.joint(int(j)).name or '').startswith('joint_'+leg)], dtype=int)
            if len(indices) != 7 or not c.pads[leg]:
                raise ValueError('Expected seven actuated hinges and pad spheres for '+leg)
            self.leg_act.append(indices)
        if cfg['mode'] == 'hold':
            self.reference = HoldReference(self.rest)
        else:
            self.reference = WaveReference(self.rest, stride_native=cfg['stride'],
                                           lift_native=cfg['lift'], step_s=cfg['step_s'],
                                           cycles=cfg['cycles'])
        self.metadata = dict(label='inverse-statics plus position IK algorithm teacher',
            config=dict(cfg), leg_order=list(LEG_ORDER),
            desired_pose_initialization='actual settled qpos, not persistent actuator target',
            shadow='private MjData FK only, free root fixed at initial actual pose',
            reference=self.reference.metadata,
            root_balance='1000*(A*fz/W - (bias-passive)[:6]/W); preferred fraction weight 0.1; bounds [0,2]',
            force_allocation='teacher-only planned load: all-six baseline fit, then caps baseline_fraction*activation for partial feet, 2 for full stance; activation=1-smoothstep(clip(reference_height/contact_transition_height,0,1)); zero caps omitted; no forces applied',
            torque='tau_ff=(bias-passive-sum(J_z.T*fz))[actuated_dofs]; gravity_ff false sets feedforward to zero',
            target='clip(q_des+(tau_ff+kd*qdot_des)/kp, native target bounds)',
            rate='qdot_des+(desired_motor_target-current_target)/target_tau_s; clip +/-2 and target bounds over command_dt_s',
            units='native; absolute SI NOT_ESTABLISHED',
            limitations=['Vertical quasistatic support ignores horizontal friction and angular contact wrench',
                         'Actual velocity bias is retained; inertia is omitted from predicted statics',
                         'Unbacked planned supports are hypothetical, explicitly flagged',
                         'COM within support hull and small root residual are necessary, not sufficient',
                         'Finite motor torque/rate and target bounds can prevent tracking',
                         'No gait success or neural autonomy follows from these mathematical checks'])
        self._contact_activation = np.ones(6)
        self.last_diagnostics = None

    @staticmethod
    def _validate_physics(c):
        if c.adhesion_on or np.any(c.d.xfrc_applied != 0) or np.any(c.d.qfrc_applied != 0):
            raise ValueError('StaticTeacher requires no adhesion and zero applied external forces')
        if np.any(c.actuator.kp <= 0):
            raise ValueError('Positive finite PD stiffness required')

    def _fk(self):
        self.shadow.qpos[:] = self.q_des
        # FK only: no dynamics/constraints/step even on the private data.
        mj.mj_kinematics(self.model, self.shadow)
        mj.mj_comPos(self.model, self.shadow)

    def _leg_kinematics(self, c, li, jacobian=True):
        geoms = c.pads[LEG_ORDER[li]]
        d, m = self.shadow, self.model
        centers = d.geom_xpos[geoms]
        lowest = int(np.argmin(centers[:, 2] - m.geom_size[geoms, 0]))
        xy = (self.initial_rot.T @ (centers.mean(axis=0)-self.initial_origin))[:2]
        bottom = centers[lowest, 2] - m.geom_size[geoms[lowest], 0]
        position = np.r_[xy, bottom]
        if not jacobian:
            return position
        jacs = []
        for g in geoms:
            jac = np.zeros((3, m.nv))
            mj.mj_jac(m, d, jac, None, d.geom_xpos[g], int(m.geom_bodyid[g]))
            jacs.append(jac)
        jac = np.vstack([(self.initial_rot.T @ np.mean(jacs, axis=0))[:2], jacs[lowest][2]])
        return position, jac[:, self.va[self.leg_act[li]]]

    def _ik(self, c, desired):
        previous = self.q_des[self.qa].copy()
        self._fk()
        for _ in range(self.config['ik_iterations']):
            for li, indices in enumerate(self.leg_act):
                position, jac = self._leg_kinematics(c, li)
                error = desired[li] - position
                delta = jac.T @ np.linalg.solve(jac @ jac.T + self.config['ik_ridge'] * np.eye(3), error)
                self.q_des[self.qa[indices]] = np.clip(
                    self.q_des[self.qa[indices]] + delta,
                    self.bounds[indices, 0], self.bounds[indices, 1])
                self._fk()
        achieved = np.array([self._leg_kinematics(c, li, False) for li in range(6)])
        velocity_raw = (self.q_des[self.qa] - previous) / self.config['command_dt_s']
        return achieved, np.clip(velocity_raw, -MAX_RATE, MAX_RATE), velocity_raw

    def _allocation(self, c, stance):
        m, d = self.model, c.d
        points_by_leg = [[] for _ in range(6)]
        for ci in range(d.ncon):
            ct = d.contact[ci]
            ga, gb = int(ct.geom1), int(ct.geom2)
            if not (c.planes[ga] or c.planes[gb]):
                continue
            geom = gb if c.planes[ga] else ga
            li = int(c.geom_leg[geom])
            if li >= 0:
                points_by_leg[li].append(ct.pos.copy())
        activation = np.asarray(self._contact_activation, float)
        if activation.shape != (6,) or not np.isfinite(activation).all() or np.any((activation < 0.) | (activation > 1.)):
            raise ValueError('Contact activation must be finite (6,) within [0,1]')
        columns, points, bodies, backed = [], [], [], []
        for li in range(6):
            geoms = c.pads[LEG_ORDER[li]]
            low = min(geoms, key=lambda g: d.geom_xpos[g, 2]-m.geom_size[g, 0])
            present = bool(points_by_leg[li])
            point = np.mean(points_by_leg[li], axis=0) if present else d.geom_xpos[low].copy()-np.array([0., 0., m.geom_size[low, 0]])
            body = int(m.geom_bodyid[low])
            if any(int(m.geom_bodyid[g]) != body for g in geoms):
                raise ValueError('Equivalent contact point requires pads on same rigid foot')
            jac = np.zeros((3, m.nv))
            mj.mj_jac(m, d, jac, None, point, body)
            columns.append(jac[2].copy())
            points.append(point)
            bodies.append(body)
            backed.append(present)
        b = (d.qfrc_bias-d.qfrc_passive)[:6] / W
        all_jz = np.array(columns).reshape((-1, m.nv)).T
        A = all_jz[:6]
        pref = np.full(6, 1./6.)
        if self.config['stance_force_pref'] == 'current':
            pref = np.clip(c.measure()['loads'] / W, 0., 2.)
        baseline = lsq_linear(np.vstack([1000.*A, .1*np.eye(6)]),
                              np.r_[1000.*b, .1*pref], bounds=(0., 2.), method='bvls')
        if not baseline.success:
            raise RuntimeError('Baseline vertical force allocation failed: '+baseline.message)
        caps = np.where(activation == 1., 2., baseline.x*activation)
        # BVLS requires strictly positive bound width. Tiny cap omission has a
        # bounded <=1e-14 fraction effect and retains the zero-force endpoint.
        support = np.flatnonzero(caps > 1e-14)
        fractions = np.zeros(6)
        if len(support):
            fit = lsq_linear(np.vstack([1000.*A[:, support], .1*np.eye(len(support))]),
                             np.r_[1000.*b, .1*pref[support]],
                             bounds=(np.zeros(len(support)), caps[support]), method='bvls')
            if not fit.success:
                raise RuntimeError('Vertical force allocation failed: '+fit.message)
            fractions[support] = fit.x
        residual = A @ fractions-b
        fz = fractions * W
        points = [points[li] for li in support]
        bodies = [bodies[li] for li in support]
        backed = [backed[li] for li in support]
        inside = False
        if len(points) >= 3:
            try:
                hull = ConvexHull(np.asarray(points)[:, :2])
                inside = bool(np.all(hull.equations[:, :2] @ d.subtree_com[c.th, :2] + hull.equations[:, 2] <= 1e-10))
            except QhullError:
                pass
        residual_norm = float(np.linalg.norm(residual))
        unbacked = [int(li) for li, present in zip(support, backed) if not present]
        diag = dict(support_indices=support.tolist(), planned_stance=stance.tolist(),
                     contact_activation=activation.tolist(),
                     baseline_per_leg_fz_over_W=baseline.x.tolist(),
                     force_fraction_upper_bounds=caps.tolist(),
                     planned_load_is_teacher_only=True,
                     reference_air_support_indices=np.flatnonzero((activation > 0.) & (activation < 1.) & (fractions > 0.)).tolist(),
                     force_solver_is_physics_validation=False,
                    actual_contact_counts=[len(p) for p in points_by_leg],
                    actual_contact_indices=[i for i, p in enumerate(points_by_leg) if p],
                    unbacked_support_indices=unbacked,
                    support_contact_points_world=np.asarray(points).reshape((-1, 3)).tolist(),
                    support_contact_body_ids=bodies, per_leg_fz_native=fz.tolist(),
                    per_leg_fz_over_W=(fz/W).tolist(), root_balance_residual_over_W=residual.tolist(),
                    root_balance_residual_norm_over_W=residual_norm,
                    root_balance_residual_max_over_W=float(np.max(np.abs(residual))),
                    com_xy_inside_support_hull=inside,
                    unsupported=bool(residual_norm > .02 or not inside or unbacked),
                    support_is_hypothetical=bool(unbacked),
                    root_balance_row_units='root native force/moment divided by W; no length rescaling')
        return all_jz @ (fractions*W), diag

    def action(self, c, t):
        if c.plant is not self.model:
            raise ValueError('Teacher is bound to its original Core/model')
        self._validate_physics(c)
        sample = self.reference.sample(t)
        refxy, refz = np.asarray(sample['target_xy'], float), np.asarray(sample['target_z'], float)
        stance = np.asarray(sample['stance'], bool).copy()
        if refxy.shape != (6, 2) or refz.shape != (6,) or stance.shape != (6,) or not np.isfinite(np.r_[refxy.ravel(), refz]).all():
            raise ValueError('Invalid reference target shapes/values')
        stance &= refz <= 1e-12
        height_fraction = np.clip(refz/self.config['contact_transition_height'], 0., 1.)
        self._contact_activation = 1.-height_fraction**2*(3.-2.*height_fraction)
        desired = np.column_stack([self.rest[:, :2] + refxy-self.rest[:, :2], self.rest_pad_bottom + refz])
        achieved, qdot_des, velocity_raw = self._ik(c, desired)
        q_contact, support_diag = self._allocation(c, stance)
        tau_ff_full = c.d.qfrc_bias-c.d.qfrc_passive-q_contact
        tau_raw = tau_ff_full[self.va] if self.config['gravity_ff'] else np.zeros(42)
        limit = c.actuator.torque_limit
        tau_ff = np.clip(tau_raw, -limit, limit)
        target, target_unbounded = motor_target(self.q_des[self.qa], qdot_des, tau_ff,
                                                c.actuator.kp, c.actuator.kd, self.target_bounds)
        raw_rate = qdot_des + (target-c.actuator.target_rad)/self.config['target_tau_s']
        cap = np.minimum(MAX_RATE, c.actuator.max_rate_rad_s)
        rate = np.clip(raw_rate, -cap, cap)
        dt = self.config['command_dt_s']
        rate = np.clip(rate, (self.target_bounds[:, 0]-c.actuator.target_rad)/dt,
                       (self.target_bounds[:, 1]-c.actuator.target_rad)/dt)
        raw_torque = c.actuator.kp*(target-c.d.qpos[self.qa])-c.actuator.kd*c.d.qvel[self.va]
        next_target = np.clip(c.actuator.target_rad+dt*rate, self.target_bounds[:, 0], self.target_bounds[:, 1])
        next_raw_torque = c.actuator.kp*(next_target-c.d.qpos[self.qa])-c.actuator.kd*c.d.qvel[self.va]
        diag = dict(time_s=float(t), phase=sample.get('phase'), source_time_s=sample.get('source_time_s'),
                    desired_foot_xy_bottom_z=desired.tolist(), achieved_foot_xy_bottom_z=achieved.tolist(),
                    ik_error=(desired-achieved).tolist(), ik_error_norm_per_leg=np.linalg.norm(desired-achieved, axis=1).tolist(),
                    ik_joint_bound_indices=np.flatnonzero(np.any(np.isclose(self.q_des[self.qa, None], self.bounds, atol=1e-10, rtol=0), axis=1)).tolist(),
                    q_des_act=self.q_des[self.qa].tolist(), qdot_des=qdot_des.tolist(),
                    qdot_des_unclipped=velocity_raw.tolist(),
                    qdot_des_clip_count=int(np.sum(np.abs(velocity_raw) > MAX_RATE)),
                    tau_ff_unclipped=tau_raw.tolist(), tau_ff=tau_ff.tolist(),
                    tau_ff_native_clip_indices=np.flatnonzero(tau_ff != tau_raw).tolist(),
                    desired_motor_target=target.tolist(), motor_target_unbounded=target_unbounded.tolist(),
                    motor_target_bound_indices=np.flatnonzero(target != target_unbounded).tolist(),
                    actual_state_raw_torque_at_desired_target=raw_torque.tolist(),
                    actual_state_raw_torque_native_exceed_indices=np.flatnonzero(np.abs(raw_torque) > limit).tolist(),
                    actual_state_raw_torque_at_next_target=next_raw_torque.tolist(),
                    next_target_torque_native_exceed_indices=np.flatnonzero(np.abs(next_raw_torque) > limit).tolist(),
                    rate_rad_s=rate.tolist(), rate_clip_count=int(np.sum(rate != raw_rate)),
                    tau_ff_root_unactuated=tau_ff_full[:6].tolist(),
                    **support_diag)
        self.last_diagnostics = diag
        return rate / MAX_RATE, diag
