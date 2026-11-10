"""S5 full-body incremental target actuator (engineering interface, not a brain).

42-vectors are in model actuator order, NOT sorted joint or qpos order.
Call update(rate, command_dt_s) exactly once per command tick; call torque(data)
at EVERY physics substep and assign its result to data.ctrl. Neither method
writes physical state, external forces, or data.ctrl. reset(data) captures the
current joint pose, without resetting physics; use it only at declared boundaries.

Zero rate holds the last target using local finite PD: this is actuator memory
and mechanical feedback, not evidence of continuing neural activity. A motor
cut instead sets data.ctrl[:] = 0 and bypasses torque(). Compare zero-rate hold,
motor cut, neural/sensory cut and matched replay to identify neural contribution.
state_dict reports command effects but makes no neural provenance claim. Angles
are radians, time seconds; torque/gain units remain native / SI NOT_ESTABLISHED.

This module does not train, run trajectories, or modify the frozen source XML.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco as mj
import numpy as np

DEFAULT_MODEL_PATH = Path(
    '/home/sensen/Desktop/engine修复验收/s5-joint-torque-plant/'
    '20261105T060730Z/MODEL_JOINT_TORQUE.xml'
)


def validate_full_model(model):
    """Reject fixtures/reduced plants; do not silently repair model physics."""
    if (model.nq, model.nv, model.nu, model.nmesh, model.neq) != (49, 48, 42, 71, 0):
        raise ValueError('Expected original 49/48/42/71 full plant, no equalities')
    roots = np.flatnonzero(model.jnt_type == mj.mjtJoint.mjJNT_FREE)
    if len(roots) != 1 or model.jnt_qposadr[roots[0]] != 0:
        raise ValueError('Expected one free root at qpos address zero')
    if not np.array_equal(model.opt.gravity, [0., 0., -9801.]):
        raise ValueError('Expected full original gravity [0, 0, -9801]')
    flags = int(model.opt.disableflags)
    forbidden = int(mj.mjtDisableBit.mjDSBL_CONTACT) | int(mj.mjtDisableBit.mjDSBL_GRAVITY)
    if flags & forbidden:
        raise ValueError('Contact and gravity must remain enabled')
    if not np.any(model.geom_contype) or not np.any(model.geom_conaffinity):
        raise ValueError('Collision masks must remain enabled')
    if not np.any(model.geom_type == mj.mjtGeom.mjGEOM_PLANE):
        raise ValueError('Original ground plane is required')


def load_full_model(xml_path=DEFAULT_MODEL_PATH, *, visible_meshes=True):
    """Compile with FlyGym asset resolution; return MjModel (no physics steps).

    All masses/inertias, contact/friction, joints, passive forces, timestep and
    motor parameters are preserved. visible_meshes only makes mesh geom alpha
    opaque in memory; no XML is exported or overwritten. Material alpha is also
    made opaque for materials referenced by mesh geoms. Default gravity is the
    source's -9801, NOT SI -9.81. No weld or auxiliary root control is added.
    """
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    path = Path(xml_path).expanduser().resolve(strict=True)
    before = hashlib.sha256(path.read_bytes()).digest()
    model = _load_mjcf(path).compile()
    validate_full_model(model)
    if visible_meshes:
        mask = model.geom_type == mj.mjtGeom.mjGEOM_MESH
        model.geom_rgba[mask, 3] = 1.
        materials = np.unique(model.geom_matid[mask])
        model.mat_rgba[materials[materials >= 0], 3] = 1.
    if hashlib.sha256(path.read_bytes()).digest() != before:
        raise RuntimeError('Source XML changed during compilation')
    return model


def _vector(value, name, *, nonnegative=False):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full(42, float(arr))
    if arr.shape != (42,) or not np.isfinite(arr).all():
        raise ValueError(f'{name} must be a finite scalar or shape (42,)')
    if nonnegative and np.any(arr < 0):
        raise ValueError(f'{name} must be nonnegative')
    return arr.copy()


class IncrementalActuator:
    """Bounded integrator plus tau=clip(kp*(target-q)-kd*qvel, +/-limit).

    kp/kd/torque_limit/max_rate_rad_s accept a scalar or 42-vector. Defaults
    reproduce the local joint_load.py scaling: kp=limit/0.2, kd=limit*0.005,
    speed limit 2 rad/s. These are starting engineering gains, not a validated
    whole-body standing controller. Limits cannot exceed native motor limits.
    Target bounds default to native joint ranges; custom (42,2) bounds must
    be a subset. No velocity feedforward, gravity compensation, or root force.
    """

    def __init__(self, model, data, *, command_dt_s=0.001, kp=None, kd=None,
                 torque_limit=None, max_rate_rad_s=2., target_bounds_rad=None):
        validate_full_model(model)
        self.model = model
        self.command_dt_s = float(command_dt_s)
        if not np.isfinite(self.command_dt_s) or self.command_dt_s <= 0:
            raise ValueError('command_dt_s must be positive and finite')
        ratio = self.command_dt_s / model.opt.timestep
        if ratio < 1 or not np.isclose(ratio, round(ratio), rtol=0, atol=1e-9):
            raise ValueError('Command period must be an integer number of physics steps')
        self.joint_ids = model.actuator_trnid[:, 0].copy()
        if (not np.all(model.actuator_trntype == mj.mjtTrn.mjTRN_JOINT)
                or len(set(self.joint_ids.tolist())) != 42
                or np.any(model.jnt_type[self.joint_ids] != mj.mjtJoint.mjJNT_HINGE)):
            raise ValueError('Expected 42 distinct hinge joint transmissions')
        self.qpos_addresses = model.jnt_qposadr[self.joint_ids].copy()
        self.dof_addresses = model.jnt_dofadr[self.joint_ids].copy()
        if np.any(self.dof_addresses < 6):
            raise ValueError('Actuation of the free root is forbidden')
        gear = np.zeros((42, 6)); gear[:, 0] = 1.
        if (not np.array_equal(model.actuator_gear, gear)
                or np.any(model.actuator_dyntype != mj.mjtDyn.mjDYN_NONE)
                or np.any(model.actuator_gaintype != mj.mjtGain.mjGAIN_FIXED)
                or np.any(model.actuator_gainprm[:, 0] != 1.)
                or np.any(model.actuator_biastype != mj.mjtBias.mjBIAS_NONE)):
            raise ValueError('torque() requires stateless unit-gain, unit-gear motors')
        if not np.all(model.actuator_ctrllimited):
            raise ValueError('All motors must have native control limits')
        native_limit = np.minimum(-model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        force_limited = model.actuator_forcelimited.astype(bool)
        native_limit[force_limited] = np.minimum(native_limit[force_limited], np.minimum(
            -model.actuator_forcerange[force_limited, 0], model.actuator_forcerange[force_limited, 1]))
        self.torque_limit = _vector(native_limit if torque_limit is None else torque_limit,
                                    'torque_limit', nonnegative=True)
        if np.any(native_limit <= 0) or np.any(self.torque_limit > native_limit):
            raise ValueError('Torque limits must not exceed native symmetric motor limits')
        self.kp = _vector(self.torque_limit / .2 if kp is None else kp, 'kp', nonnegative=True)
        self.kd = _vector(self.torque_limit * .005 if kd is None else kd, 'kd', nonnegative=True)
        self.max_rate_rad_s = _vector(max_rate_rad_s, 'max_rate_rad_s', nonnegative=True)
        native_bounds = model.jnt_range[self.joint_ids].copy()
        self.target_bounds_rad = np.asarray(native_bounds if target_bounds_rad is None
                                            else target_bounds_rad, dtype=float).copy()
        b = self.target_bounds_rad
        if (not np.all(model.jnt_limited[self.joint_ids]) or b.shape != (42, 2)
                or not np.isfinite(b).all() or np.any(b[:, 0] >= b[:, 1])
                or np.any(b[:, 0] < native_bounds[:, 0]) or np.any(b[:, 1] > native_bounds[:, 1])):
            raise ValueError('Target bounds must lie within native finite hinge ranges')
        self.reset(data)

    def _pose(self, data):
        if data.qpos.shape != (self.model.nq,) or data.qvel.shape != (self.model.nv,):
            raise ValueError('Data dimensions do not match model')
        q = data.qpos[self.qpos_addresses].copy()
        v = data.qvel[self.dof_addresses].copy()
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            raise ValueError('Nonfinite joint state')
        return q, v

    def reset(self, data):
        """Capture current angles, clipped to bounds; never write data state."""
        q, _ = self._pose(data)
        self.target_rad = np.clip(q, self.target_bounds_rad[:, 0], self.target_bounds_rad[:, 1])
        self.initial_target_rad = self.target_rad.copy()
        self.reset_projection_rad = self.target_rad - q
        self.command_ticks = 0
        self.last_input_rate_rad_s = np.zeros(42)
        self.last_limited_rate_rad_s = np.zeros(42)
        self.last_effective_rate_rad_s = np.zeros(42)
        self.rate_clip_mask = np.zeros(42, dtype=bool)
        self.angle_clip_mask = np.zeros(42, dtype=bool)
        return self.target_rad.copy()

    def update(self, delta_rate_rad_s, dt_s):
        """Integrate ONE fixed command tick, rate-limit then clamp (no windup).

        dt_s is checked, not silently replaced: callers must not update once
        per neural or physics substep. Zero input preserves the accumulated
        target exactly. Returns a copy; malformed/nonfinite inputs fail atomically.
        """
        dt = float(dt_s)
        if not np.isfinite(dt) or not np.isclose(dt, self.command_dt_s, rtol=0, atol=1e-12):
            raise ValueError('dt_s must equal the configured fixed command_dt_s')
        rate = np.asarray(delta_rate_rad_s, dtype=float)
        if rate.shape != (42,) or not np.isfinite(rate).all():
            raise ValueError('delta_rate_rad_s must be a finite shape (42,) vector')
        limited = np.clip(rate, -self.max_rate_rad_s, self.max_rate_rad_s)
        proposed = self.target_rad + limited * self.command_dt_s
        target = np.clip(proposed, self.target_bounds_rad[:, 0], self.target_bounds_rad[:, 1])
        self.last_input_rate_rad_s = rate.copy()
        self.last_limited_rate_rad_s = limited
        self.last_effective_rate_rad_s = (target - self.target_rad) / self.command_dt_s
        self.rate_clip_mask = rate != limited
        self.angle_clip_mask = proposed != target
        self.target_rad = target
        self.command_ticks += 1
        return target.copy()

    def torque(self, data):
        """Return finite, bounded native joint torques, suitable for data.ctrl.

        Pure evaluation: does not advance target/time or write any physical
        field. Recompute at each physics step so damping uses current velocity.
        """
        q, v = self._pose(data)
        with np.errstate(over='raise', invalid='raise'):
            raw = self.kp * (self.target_rad - q) - self.kd * v
        if not np.isfinite(raw).all():
            raise ValueError('PD torque is nonfinite')
        return np.clip(raw, -self.torque_limit, self.torque_limit)

    def state_dict(self):
        """JSON-ready snapshot; returned lists cannot mutate actuator state."""
        result = dict(schema_version=1, command_dt_s=self.command_dt_s,
                      command_ticks=self.command_ticks,
                      command_time_s=self.command_ticks * self.command_dt_s,
                      joint_names=[self.model.joint(int(j)).name for j in self.joint_ids],
                      command_source='caller supplied; neural provenance not established',
                      holding_source='local finite PD plus persistent target memory',
                      torque_units='native; absolute SI NOT_ESTABLISHED')
        for name in ('joint_ids', 'qpos_addresses', 'dof_addresses', 'kp', 'kd',
                     'torque_limit', 'max_rate_rad_s', 'target_bounds_rad', 'target_rad',
                     'initial_target_rad', 'reset_projection_rad', 'last_input_rate_rad_s',
                     'last_limited_rate_rad_s', 'last_effective_rate_rad_s',
                     'rate_clip_mask', 'angle_clip_mask'):
            result[name] = getattr(self, name).tolist()
        result['command_target_shift_rad'] = (self.target_rad - self.initial_target_rad).tolist()
        return result
