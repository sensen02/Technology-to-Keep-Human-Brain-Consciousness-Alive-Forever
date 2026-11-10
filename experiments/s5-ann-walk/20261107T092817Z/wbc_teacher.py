"""Training-only full-body dynamics QP. No physics state writes or neural claim.

Original finite-PD motors implement the requested torque through their bounded
incremental target. QP.npz records optimal, reachable and actually applied torque;
solving a QP alone is not evidence of physical walking or actuator feasibility.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation
import mujoco as mj

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'deps'))
import osqp
SOURCE = HERE.parent/'20261107T091243Z/anchor_teacher.py'
spec = importlib.util.spec_from_file_location('wbc_original_runtime', SOURCE)
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
n = b.n
OUT = Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T092817Z')
b.OUT = OUT
W = n.dsm.WEIGHT
P = dict(policy_dt=.005, blend_seconds=.1, contact_threshold_W=.005,
         body_velocity_gain=5., height_kp=100., height_kd=10., orientation_kp=30., orientation_kd=5.,
         stance_kp=100., stance_kd=15., swing_kp=100., swing_kd=10., posture_kp=20., posture_kd=3.,
         body_weight=20., angular_weight=5., contact_weight=50., swing_weight=2., posture_weight=.02,
         acceleration_regularizer=1e-5, torque_regularizer=1e-4, force_regularizer=1e-4,
         task_acceleration_scale=100., angular_acceleration_scale=100., max_normal_force_W=2.,
         solver_eps=1e-6, solver_max_iter=4000, dynamics_tolerance_W=1e-3, constraint_tolerance=1e-4)
INSTANCES, ACTUATORS = [], []
OriginalActuator = b.IncrementalActuator


class ObservedActuator(OriginalActuator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ACTUATORS.append(self)


class WholeBodyTeacher(b.AnchorTeacher):
    def __init__(self, p, ref):
        super().__init__(p, ref)
        assert hasattr(mj, 'mj_jacDot'), 'Requires native attached-material-point Jacobian derivative'
        self.nx = 108; self.nc = 120  # dynamics48, torque42, normal6, pyramid24
        self.S = np.zeros((48, 42)); self.S[p.va, np.arange(42)] = 1.
        self.scale = np.r_[np.full(3, 9801.), np.full(45, 100.), p.limits, np.full(18, W)]
        self.contact_anchor = np.zeros((6, 3)); self.have_anchor = np.zeros(6, bool)
        self.previous_stance = np.zeros(6, bool)
        self.rows = []; self.solver = None
        rr, cc = np.triu_indices(self.nx); order = np.lexsort((rr, cc))
        self.pr, self.pc = rr[order], cc[order]
        self.ar = np.tile(np.arange(self.nc), self.nx)
        self.ac = np.repeat(np.arange(self.nx), self.nc)
        INSTANCES.append(self)

    def jacobian(self, d, point, bid):
        jp = np.zeros((3, 48)); jr = np.zeros_like(jp)
        dp = np.zeros_like(jp); dr = np.zeros_like(jp)
        mj.mj_jac(self.m, d, jp, jr, point, int(bid))
        mj.mj_jacDot(self.m, d, dp, dr, point, int(bid))
        return jp, jr, dp@d.qvel, dr@d.qvel

    def contacts(self, d, stance):
        J = np.zeros((6, 3, 48)); jd = np.zeros((6, 3)); points = np.zeros((6, 3))
        load = np.zeros(6); mu = np.full(6, np.inf); force = np.zeros(6)
        for i in range(d.ncon):
            co = d.contact[i]; ga, gb = int(co.geom1), int(co.geom2)
            if not (self.p.planes[ga] or self.p.planes[gb]):
                continue
            geom = gb if self.p.planes[ga] else ga
            li = int(self.p.feet[geom])
            if li < 0:
                continue
            mj.mj_contactForce(self.m, d, i, force)
            normal = float((co.frame.reshape(3, 3).T@force[:3])[2])*(1 if self.p.planes[ga] else -1)
            if normal <= 0:
                continue
            jp, _, jdotv, _ = self.jacobian(d, co.pos, self.m.geom_bodyid[geom])
            J[li] += normal*jp; jd[li] += normal*jdotv
            points[li] += normal*co.pos; load[li] += normal
            mu[li] = min(mu[li], float(co.friction[0]), float(co.friction[1]))
        for li in range(6):
            if load[li] > 0:
                J[li] /= load[li]; jd[li] /= load[li]; points[li] /= load[li]
        active = stance & (load > P['contact_threshold_W']*W)
        mu[~np.isfinite(mu)] = 0.
        return J, jd, points, load, mu, active

    def solve(self, H, g, A, lo, hi):
        values = dict(Px=H[self.pr, self.pc], Ax=A.T.ravel(), q=g, l=lo, u=hi)
        if self.solver is None:
            self.solver = osqp.OSQP()
            self.solver.setup(P=sparse.csc_matrix((values['Px'], (self.pr, self.pc)), shape=H.shape),
                q=g, A=sparse.csc_matrix((values['Ax'], (self.ar, self.ac)), shape=A.shape),
                l=lo, u=hi, verbose=False, eps_abs=P['solver_eps'], eps_rel=P['solver_eps'],
                max_iter=P['solver_max_iter'], polishing=True, warm_starting=True)
        else:
            self.solver.update(**values)
        return self.solver.solve(raise_error=False)

    def desired(self, d, t):
        assert self.ready and ACTUATORS
        before = self.p.snapshot(d)
        actuator = ACTUATORS[-1]; start = time.monotonic()
        stance = (self.phase(t)+self.offset) % 1 < 5/6
        entry = stance & ~self.previous_stance; self.previous_stance = stance.copy()
        self.anchor_pos[entry] = d.xpos[self.p.bids][entry]
        self.anchor_rot[entry] = d.xmat[self.p.bids].reshape(6, 3, 3)[entry]
        Jc, jdc, points, load, mu, active = self.contacts(d, stance)
        self.have_anchor[~stance] = False
        capture = active & (~self.have_anchor | entry)
        self.contact_anchor[capture] = points[capture]; self.have_anchor[capture] = True
        self.reference_position += max(0., t-self.last_t)*self.velocity; self.last_t = t
        nominal = self.ref.warm(t)
        virtual = d.qpos[:7].copy(); virtual[:3] += b.PDT*self.velocity
        # Only scratch FK uses a moving virtual body and source swing geometry.
        scratch = self.scratch; scratch.qpos[:] = d.qpos; scratch.qvel[:] = 0.
        scratch.qpos[:7] = virtual; scratch.qpos[self.p.qa] = nominal
        mj.mj_forward(self.m, scratch)
        goals = scratch.xpos[self.p.bids].copy()
        goalrot = scratch.xmat[self.p.bids].reshape(6, 3, 3).copy()
        M = np.zeros((48, 48)); mj.mj_fullM(self.m, M, d.qM)
        bias = d.qfrc_bias-d.qfrc_passive
        D = np.zeros((48, self.nx)); D[:, :48] = M; D[:, 48:90] = -self.S
        for li in range(6):
            if active[li]:
                D[:, 90+3*li:93+3*li] = -Jc[li].T
        A = np.zeros((self.nc, self.nx)); lo = np.full(self.nc, -np.inf); hi = np.full(self.nc, np.inf)
        scaled = D*self.scale[None, :]
        normalization = np.maximum(np.linalg.norm(scaled, axis=1), 1e-8)
        A[:48] = scaled/normalization[:, None]; lo[:48] = hi[:48] = -bias/normalization
        A[48:90, 48:90] = np.diag(self.scale[48:90])
        lo[48:90] = -actuator.torque_limit; hi[48:90] = actuator.torque_limit
        for li in range(6):
            c = 90+3*li; row = 90+5*li
            A[row, c+2] = W; lo[row] = 0.; hi[row] = P['max_normal_force_W']*W if active[li] else 0.
            for j, (axis, sign) in enumerate(((0, 1), (0, -1), (1, 1), (1, -1))):
                A[row+1+j, c+axis] = sign*W; A[row+1+j, c+2] = -mu[li]/np.sqrt(2.)*W
                hi[row+1+j] = 0.
        H = np.diag(np.r_[np.full(48, P['acceleration_regularizer']),
                              np.full(42, P['torque_regularizer']), np.full(18, P['force_regularizer'])])
        g = np.zeros(self.nx)
        def cost(J, target, weight):
            L = np.zeros((len(target), self.nx)); L[:, :48] = J*self.scale[:48]
            L /= P['task_acceleration_scale']; target = np.asarray(target)/P['task_acceleration_scale']
            H[:] += weight*(L.T@L); g[:] -= weight*(L.T@target)
        jp, jr, jdp, jdr = self.jacobian(d, d.xpos[self.p.thorax], self.p.thorax)
        linear = jp@d.qvel; angular = jr@d.qvel
        ades = P['body_velocity_gain']*(self.velocity-linear)
        ades[2] = P['height_kp']*(self.height-d.qpos[2])-P['height_kd']*linear[2]
        orient = (self.upright*b.root_rotation(d.qpos).inv()).as_rotvec()
        rdes = P['orientation_kp']*orient-P['orientation_kd']*angular
        cost(jp, np.clip(ades, -100., 100.)-jdp, P['body_weight'])
        cost(jr, np.clip(rdes, -100., 100.)-jdr, P['angular_weight'])
        for li in range(6):
            if active[li]:
                acc = -P['stance_kp']*(points[li]-self.contact_anchor[li])-P['stance_kd']*(Jc[li]@d.qvel)
                cost(Jc[li], acc-jdc[li], P['contact_weight'])
            elif not stance[li]:
                jf, _, jdf, _ = self.jacobian(d, d.xpos[self.p.bids[li]], self.p.bids[li])
                acc = P['swing_kp']*(goals[li]-d.xpos[self.p.bids[li]])-P['swing_kd']*(jf@d.qvel)
                cost(jf, acc-jdf, P['swing_weight'])
        posture = np.eye(48)[self.p.va]
        cost(posture, P['posture_kp']*(nominal-d.qpos[self.p.qa])-P['posture_kd']*d.qvel[self.p.va], P['posture_weight'])
        status = 'EXCEPTION'; iterations = 0; fallback = True
        solution = np.zeros(self.nx); dynamics = bias.copy(); violation = 0.
        try:
            result = self.solve(H, g, A, lo, hi)
            status = result.info.status; iterations = result.info.iter
            if result.x is not None and np.isfinite(result.x).all():
                solution = result.x*self.scale
                dynamics = D@solution+bias
                bounded = A@result.x
                violation = float(max(np.max(lo-bounded), np.max(bounded-hi), 0.))
                fallback = not (result.info.status_val in (1, 2)
                    and np.max(abs(dynamics))/W <= P['dynamics_tolerance_W']
                    and violation <= P['constraint_tolerance']
                    and np.max(abs(solution[48:90])-actuator.torque_limit) <= 1e-4)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            status = type(exc).__name__+': '+str(exc)
        tau = solution[48:90]
        torque_target = d.qpos[self.p.qa]+(tau+actuator.kd*d.qvel[self.p.va])/actuator.kp
        alpha = float(np.clip((t-b.HANDOVER)/P['blend_seconds'], 0., 1.))
        desired = nominal if fallback else (1-alpha)*nominal+alpha*torque_target
        if not np.isfinite(desired).all():
            desired = nominal; fallback = True; status += ':INVALID_TARGET'
        desired = np.clip(desired, self.ref.lo, self.ref.hi)
        rate = np.clip((desired-actuator.target_rad)/b.PDT, -b.CAP, b.CAP)
        reachable = np.clip(actuator.target_rad+b.PDT*rate, self.ref.lo, self.ref.hi)
        predicted = np.clip(actuator.kp*(reachable-d.qpos[self.p.qa])-actuator.kd*d.qvel[self.p.va],
                            -actuator.torque_limit, actuator.torque_limit)
        assert np.array_equal(before, self.p.snapshot(d)), 'QP changed actual state'
        self.rows.append(dict(time=t, status=status, iterations=iterations, fallback=fallback, blend=alpha,
            qacc=solution[:48].copy(), tau_optimal=tau.copy(), contact_force=solution[90:].reshape(6, 3).copy(),
            dynamics_residual=dynamics.copy(), dynamics_residual_W=float(np.max(abs(dynamics))/W),
            constraint_violation=violation, actual_contact_load=load, friction_mu=mu,
            friction_pyramid_mu=mu/np.sqrt(2.), activelegs=active, active_contact=active,
            averaged_contact_J=Jc*active[:, None, None], averaged_Jdotv=jdc*active[:, None],
            dynamics_rhs=-bias.copy(),
            contact_material_velocity=np.einsum('lij,j->li', Jc, d.qvel), contact_point=points,
            contact_anchor=self.contact_anchor.copy(), contact_anchor_capture=capture,
            unmodeled_swing_reaction=load*(~stance), tau_before_command=actuator.torque(d),
            tau_predicted_reachable=predicted, tau_reachable_minus_optimal=predicted-tau,
            target_rate=rate, torque_target=torque_target.copy(), commanded_target=desired.copy(),
            qacc_body_task=ades.copy(), angular_acceleration_task=rdes.copy(),
            actual_root=d.qpos[:7].copy(), actual_qvel=d.qvel.copy(), solve_wall_s=time.monotonic()-start))
        residual = np.c_[np.linalg.norm(goals-d.xpos[self.p.bids], axis=1), np.linalg.norm(Jc@d.qvel, axis=1)]
        return desired, dict(stance=stance, stance_entry=entry, virtual_root=virtual,
            reference_body_position=self.reference_position.copy(), anchor_position=self.anchor_pos.copy(),
            anchor_rotation=self.anchor_rot.copy(), foot_goal_position=goals, foot_goal_rotation=goalrot,
            ik_residual=residual, body_feedback_velocity=np.zeros(2))


b.AnchorTeacher = WholeBodyTeacher
b.IncrementalActuator = ObservedActuator
original_prereg = b.prereg


def prereg(folder, args):
    original_prereg(folder, args)
    path = folder/'PREREGISTRATION.json'; protocol = json.loads(path.read_text())
    protocol.update(stage='S5 training-only whole-body dynamics QP feasibility',
        algorithm='Hard M*qacc+bias-passive=S*tau+Jc.T*f, unactuated root; soft body, contact acceleration, swing and nominal-posture costs',
        parameters=P, solver=dict(name='osqp', version=osqp.__version__, fixed_variables=108,
            variables='qacc48,tau42,worldcontactforce3x6; inactivelegforces0',
            scaling='qacc linear9801/angular100; tau/nativecaps; force/W; dynamicsrows normalized by scaled row norm',
            jacobian_derivative='native mj_jacDot actual attached material points; no finite difference'),
        contacts='Actual positive ground-contact normal-force weighted Jacobian/Jdotv/point per leg; represents fixed proportional force distribution across real tarsal contacts. Nominal swing or load<=.005W forces constrained0; unmodeled swing reaction logged.',
        friction='Original minimum of two actual contact sliding coefficients; conservative inscribed square |fx|,|fy|<=mu/sqrt2*fz,0<=fz<=2W; no model changes',
        actuation='q_tau=q+(tau+kd*qvel)/kp; .1s nominal-to-QP blend then nativejointclip; original30rad/s integrated target and zero-DtargetPD32; optimized torque may be unreachable',
        fallback='Current original nominal desired if solve/status/finite/dynamics/nativebounds fail. Every policy solve and fallback retained in QP.npz; fallback prevents pure-QP success claim.',
        diagnostics='QP.npz optimal, reachable-at-current-state and actual BODY endpoint torque, dynamics residual/status/contactforces/activecontacts/rootstate. TEACHER.ik_residual=[footgoalpositionerror,weightedmaterialspeed], NOT IK errors.',
        previous_anchor_algorithm='not used: no pose IK. Bodyreference only soft acceleration objective, never actualstate writes.',
        additional_sources={str(q):n.sha(q) for q in [Path(__file__), SOURCE,
            Path(osqp.__file__), HERE/'DEPENDENCIES.json', HERE/'deps/osqp/interface.py',
            Path(osqp.ext_builtin.__file__)]})
    n.dump(path, protocol)


b.prereg = prereg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds', type=float, default=4.)
    ap.add_argument('--tag', default='wbc_screen')
    ap.add_argument('--feedback-off', action='store_true')
    args = ap.parse_args(); result = None
    try:
        result = b.simulate(args)
    finally:
        rows = [r for teacher in INSTANCES for r in teacher.rows]
        if rows:
            folder = OUT/args.tag
            fields = {key:np.asarray([r[key] for r in rows]) for key in rows[0]}
            fields['policy_times'] = fields['time'].copy()
            bodypath = folder/'BODY.npz'
            if bodypath.exists():
                with np.load(bodypath) as body:
                    indices = np.rint(fields['time']/b.PDT).astype(int)
                    fields['tau_actual_after_policy'] = body['ctrl'][indices]
                    fields['tau_actual_minus_optimal'] = fields['tau_actual_after_policy']-fields['tau_optimal']
            np.savez_compressed(folder/'QP.npz', **fields)
            if result is not None:
                result.update(qp_fallback_count=int(fields['fallback'].sum()), qp_ticks=len(rows),
                    qp_solved_without_fallback=bool(not fields['fallback'].any()),
                    qp_max_dynamics_residual_W=float(fields['dynamics_residual_W'].max()),
                    qp_trace_sha256=n.sha(folder/'QP.npz'))
                n.dump(folder/'RESULT.json', result)


if __name__ == '__main__':
    main()
