"""ONLYCODE training teacher candidate. No heavy imports or execution at import.
Parent reviews this API before a separate finite physical diagnostic runner.
World anchors / desired yaw are engineering teacher targets, never root writes.
"""
from pathlib import Path
import math
import sys

BASE = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
CONDITIONS = {'stop': 0., 'left_pivot': .15, 'right_pivot': -.15}
BODY_SEGMENTS = ('Head', 'Thorax', 'A1A2', 'A3', 'A4', 'A5', 'A6')


def blend(u):
    u = max(0., min(1., float(u)))
    return u*u*u*(10.+u*(-15.+6.*u))


def ramp_integral(t, ramp_s=1.):
    """Integral of quintic ramp, continuous from zero; units seconds."""
    if not math.isfinite(t) or t < 0 or ramp_s <= 0:
        raise ValueError('Finite nonnegative time and positive ramp')
    u = min(t/ramp_s, 1.)
    return ramp_s*(2.5*u**4-3.*u**5+u**6) + max(0., t-ramp_s)


def schedule(t, order, step_s=.8):
    if not math.isfinite(t) or t < 0 or step_s <= 0 or sorted(order) != list(range(6)):
        raise ValueError('Finite teacher time, positive step and six-leg order')
    slot = int(math.floor((t+1e-12)/step_s))
    return slot, int(order[slot % 6]), max(0., min(1., (t-slot*step_s)/step_s))


def pin_baseline():
    # Reject collisions; old snapshot remains unmodified.
    for name in ('physics', 'static_teacher', 'wave_reference'):
        if name in sys.modules and Path(sys.modules[name].__file__).resolve() != BASE/(name+'.py'):
            raise ValueError('Baseline import collision: ' + name)
    if str(BASE) not in sys.path:
        sys.path.insert(0, str(BASE))


def body_length_from_compiled_mesh(c):
    """Native Head/Thorax/abdomen mesh extent on INITIAL actual Thorax x axis.
    Call once in the common initial settled pose; save returned membership/extent.
    No leg, wing, eye, antenna or global model.extent enters this definition.
    """
    import numpy as np
    axis = c.d.xmat[c.th].reshape(3, 3)[:, 0].copy()
    origin = c.d.xpos[c.th].copy()
    values = []
    records = []
    for segment in BODY_SEGMENTS:
        body_id = c.plant.body(segment).id
        candidates = [i for i in range(c.plant.ngeom) if int(c.plant.geom_bodyid[i]) == body_id and int(c.plant.geom_dataid[i]) >= 0 and (c.plant.geom(i).name or '') == segment+'_geom']
        if len(candidates) != 1:
            raise ValueError('Exactly one named body mesh required: ' + segment)
        gid = candidates[0]
        mid = int(c.plant.geom_dataid[gid])
        if not 0 <= mid < c.plant.nmesh:
            raise ValueError('Compiled mesh ID')
        start, count = int(c.plant.mesh_vertadr[mid]), int(c.plant.mesh_vertnum[mid])
        local = np.asarray(c.plant.mesh_vert[start:start+count], dtype=float)
        rotation = c.d.geom_xmat[gid].reshape(3, 3)
        world = local @ rotation.T + c.d.geom_xpos[gid]
        projected = (world-origin) @ axis
        if not len(projected) or not np.isfinite(projected).all():
            raise ValueError('Finite compiled body vertices required')
        values.extend(projected.tolist())
        records.append(dict(segment=segment, geom_id=gid, mesh_id=mid, vertices=count, projected_min=float(projected.min()), projected_max=float(projected.max())))
    length = max(values)-min(values)
    if not math.isfinite(length) or length <= 0:
        raise ValueError('Invalid body length')
    return dict(length_native=length, initial_origin=origin.tolist(), initial_axis=axis.tolist(), projected_min=min(values), projected_max=max(values), segments=records, absolute_SI='NOT_ESTABLISHED')


class PivotReference:
    """Six sequential swings; five planned stances, or six fixed stances for stop.
    Targets are body coordinates of WORLD anchors in the DESIRED teacher frame.
    The controller's private FK is separately expressed at actual body pose.
    This mismatch creates physical contact torque through finite joint servos;
    the desired frame never updates the actual free root.
    """
    def __init__(self, c, teacher, condition, order):
        import numpy as np
        if condition not in CONDITIONS:
            raise ValueError('Only three preregistered conditions')
        self.np = np
        self.omega = CONDITIONS[condition]
        self.condition = condition
        self.step_s, self.lift = .8, .12
        self.order = tuple(int(i) for i in order)
        self.origin = c.d.xpos[c.th].copy()
        self.rot0 = c.d.xmat[c.th].reshape(3, 3).copy()
        self.rest = teacher.rest.copy()
        self.bottom0 = teacher.rest_pad_bottom.copy()
        self.anchors = self.rest @ self.rot0.T + self.origin
        self.slot = -1
        self.last_t = None
        self.swing_start = self.swing_end = None
        self.last_snapshot = None
        self.metadata = dict(kind='TRAINING_ONLY_WORLD_ANCHOR_PIVOT_REFERENCE_V1', condition=condition, requested_yaw_rate=self.omega, step_s=self.step_s, lift_native=self.lift, order_indices=list(self.order), ramp_s=1., desired_translation='fixed handover actual Thorax origin', actual_root_writes=False, external_forces=False, measured_animal_pivot=False)

    def rotation(self, t):
        np = self.np
        theta = self.omega*ramp_integral(t)
        cs, sn = math.cos(theta), math.sin(theta)
        return np.array([[cs, -sn, 0.], [sn, cs, 0.], [0., 0., 1.]]) @ self.rot0

    def sample(self, t):
        np = self.np
        if self.last_t is not None and (t <= self.last_t or abs(t-self.last_t-.005) > 1e-8):
            raise ValueError('Contiguous parent 5ms teacher decisions required')
        self.last_t = float(t)
        target_world = self.anchors.copy()
        stance = np.ones(6, dtype=bool)
        heights = np.zeros(6)
        slot, leg, u = schedule(t, self.order, self.step_s)
        if self.omega != 0.:
            if slot != self.slot:
                if slot != self.slot+1:
                    raise ValueError('No skipped swing slots')
                if self.slot >= 0:
                    old_leg = self.order[self.slot % 6]
                    self.anchors[old_leg] = self.swing_end.copy()
                self.slot = slot
                self.swing_start = self.anchors[leg].copy()
                # Foot placement ahead by half the next stance duration (4s).
                landing_t = (slot+1)*self.step_s
                lead_t = landing_t + 2.
                self.swing_end = self.origin + self.rotation(lead_t) @ self.rest[leg]
            target_world = self.anchors.copy()
            target_world[leg] = (1.-blend(u))*self.swing_start + blend(u)*self.swing_end
            heights[leg] = self.lift*16.*u*u*(1.-u)*(1.-u)
            stance[leg] = False
        desired_rot = self.rotation(t)
        target_body = (target_world-self.origin) @ desired_rot
        # StaticTeacher consumes separate horizontal body target and WORLD bottom z.
        # Keep the flat-ground bottom baseline invariant; add only swing clearance.
        self.last_snapshot = dict(time_s=float(t), slot=slot, swing_leg=None if self.omega == 0. else leg, phase_u=u, world_anchors=self.anchors.tolist(), desired_foot_world=target_world.tolist(), desired_yaw_relative=self.omega*ramp_integral(t), desired_origin=self.origin.tolist(), desired_rot=desired_rot.tolist(), support_planned=stance.tolist(), planned_clearance=heights.tolist())
        return dict(target_xy=target_body[:, :2], target_z=heights, stance=stance, source_time_s=None, phase='stop_hold' if self.omega == 0. else 'pivot_sequential_swing', derivative_fields='not_provided_StaticTeacher_uses_actual_q_des_finite_difference')


def make_teacher(c, condition, *, order_indices):
    """Parent-only API, call ONCE at physical 5.8s after unchanged old warmup.
    Returns an algorithm teacher, never an ANN. Does not integrate or reset Core.
    """
    pin_baseline()
    import numpy as np
    from static_teacher import StaticTeacher

    class ActualPoseStaticTeacher(StaticTeacher):
        def action(self, c, t):
            # Only private FK target gets actual nonactuated pose; actual data untouched.
            previous_joints = self.q_des[self.qa].copy()
            self.q_des[:] = c.d.qpos
            self.q_des[self.qa] = previous_joints
            self.initial_rot = c.d.xmat[c.th].reshape(3, 3).copy()
            self.initial_origin = c.d.xpos[c.th].copy()
            qpos_before = c.d.qpos.copy()
            qvel_before = c.d.qvel.copy()
            xforce_before = c.d.xfrc_applied.copy()
            qforce_before = c.d.qfrc_applied.copy()
            action, diag = super().action(c, t)
            if not (np.array_equal(qpos_before, c.d.qpos) and np.array_equal(qvel_before, c.d.qvel) and np.array_equal(xforce_before, c.d.xfrc_applied) and np.array_equal(qforce_before, c.d.qfrc_applied)):
                raise RuntimeError('Teacher must not write actual generalized state or extra forces')
            diag.update(teacher_reference=self.reference.last_snapshot, actual_pose_shadow=True, desired_root_reference_only=True, no_actual_root_or_external_force_write=True)
            return action, diag

    teacher = ActualPoseStaticTeacher(c, dict(mode='hold', gravity_ff=True, stance_force_pref='uniform', step_s=.8, cycles=2, stride=.36, lift=.12, contact_transition_height=.01))
    teacher.reference = PivotReference(c, teacher, condition, order_indices)
    teacher.metadata.update(candidate=teacher.reference.metadata, shadow='PRIVATE FK actual nonactuated pose each decision; continuous desired actuated joints', scope='Training teacher physical feasibility only; vertical statics omits horizontal friction wrench; no neural success')
    return teacher
