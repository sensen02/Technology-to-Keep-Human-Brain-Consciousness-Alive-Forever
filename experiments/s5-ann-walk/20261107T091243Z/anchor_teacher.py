"""S5 training-only world-anchor teacher; no neural claim or physical state writes.

Only fresh() initializes the physical state. All FK/IK and virtual-root writes
below use a separate MjData. Execute physics only after independent parent review.
"""
import argparse
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = ROOT/'experiments/s5-ann-incremental/20261107T010000Z'
SOURCE = ROOT/'experiments/s5-ann-walk/20261107T083316Z/run_aligned.py'
OUT = Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T091243Z')
WAVE = OUT.parent/'20261107T081009Z/METADATA.json'
INERT = OUT.parent/'20261107T073258Z/MODEL.npz'
spec = importlib.util.spec_from_file_location('anchor_aligned_reference', SOURCE)
aligned = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aligned)
n = aligned.n  # Readable periodic two-neighbor PCHIP Reference, never n.simulate.
from incremental_actuator import IncrementalActuator, DEFAULT_MODEL_PATH
import mujoco as mj
sys.path.insert(0, str(Path(__file__).resolve().parent))
from proprioception import features, contract

DT, CDT, PDT, CAP, HANDOVER = .0000125, .001, .005, 30., 1.
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
KNOBS = dict(frequency_hz=1., desired_speed=.25, position_gain=.5,
             position_feedback_speed_cap=.1, height_gain=1., height_speed_cap=.05,
             tilt_tau=.5, tilt_speed_cap=.2, ik_steps=3, ik_joint_step=.08,
             ik_damping=.002, orientation_length=.15, kp_scale=32., rate_cap=CAP)


def root_rotation(q):
    return Rotation.from_quat(np.roll(q[3:7], -1))


def clipnorm(v, cap):
    return v * min(1., cap/max(float(np.linalg.norm(v)), 1e-15))


class AnchorTeacher:
    """Virtual-root bounded IK; stance anchors change only at measured entry."""
    def __init__(self, p, ref):
        self.p, self.ref, self.m = p, ref, p.m
        self.scratch = mj.MjData(p.m)
        self.wave = json.loads(WAVE.read_text())
        assert tuple(self.wave['legs']) == LEGS
        assert self.wave['parameters']['duty'] == 5/6
        self.offset = np.asarray(self.wave['phase_offsets'])
        self.indices = []
        actual = [p.m.joint(int(j)).name for j in p.jids]
        for leg in LEGS:
            ix = [i for i, name in enumerate(actual) if name.startswith('joint_'+leg)]
            assert len(ix) == 7, (leg, [actual[i] for i in ix])
            assert all(p.m.jnt_type[p.jids[i]] == mj.mjtJoint.mjJNT_HINGE for i in ix)
            self.indices.append(np.asarray(ix))
        assert len(set(np.concatenate(self.indices))) == 42
        self.ready = False
        self.jp, self.jr = np.zeros((3, p.m.nv)), np.zeros((3, p.m.nv))

    def phase(self, t):
        # Exactly Reference.warm's post-ramp phase; no phase reset at handover.
        return (t-n.RAMP/2)*KNOBS['frequency_hz']

    def capture(self, d, t):
        self.height = float(d.qpos[2])
        rot = root_rotation(d.qpos)
        self.yaw = float(rot.as_euler('xyz')[2])
        self.upright = Rotation.from_euler('z', self.yaw)
        self.velocity = KNOBS['desired_speed']*np.array([np.cos(self.yaw), np.sin(self.yaw), 0.])
        self.origin = d.qpos[:3].copy()
        self.reference_position = self.origin.copy()
        self.anchor_pos = d.xpos[self.p.bids].copy()
        self.anchor_rot = d.xmat[self.p.bids].reshape(6, 3, 3).copy()
        self.previous_stance = (self.phase(t)+self.offset) % 1 < 5/6
        self.last_t = t
        self.ready = True

    def desired(self, d, t):
        assert self.ready and self.scratch is not d
        stance = (self.phase(t)+self.offset) % 1 < 5/6
        entry = stance & ~self.previous_stance
        self.anchor_pos[entry] = d.xpos[self.p.bids][entry]
        self.anchor_rot[entry] = d.xmat[self.p.bids].reshape(6, 3, 3)[entry]
        self.previous_stance = stance.copy()
        self.reference_position += max(0., t-self.last_t)*self.velocity
        self.last_t = t
        feedback = clipnorm(KNOBS['position_gain']*(self.reference_position[:2]-d.qpos[:2]),
                            KNOBS['position_feedback_speed_cap'])
        virtual = d.qpos[:7].copy()
        virtual[:2] += PDT*(self.velocity[:2]+feedback)
        virtual[2] += PDT*np.clip(KNOBS['height_gain']*(self.height-d.qpos[2]),
                                   -KNOBS['height_speed_cap'], KNOBS['height_speed_cap'])
        actual_rot = root_rotation(d.qpos)
        correction = clipnorm((self.upright*actual_rot.inv()).as_rotvec()/KNOBS['tilt_tau'],
                              KNOBS['tilt_speed_cap'])
        virtual[3:7] = np.roll((Rotation.from_rotvec(PDT*correction)*actual_rot).as_quat(), 1)
        s = self.scratch
        s.qpos[:] = d.qpos
        s.qpos[:7] = virtual
        s.qpos[self.p.qa] = self.ref.warm(t)
        mj.mj_forward(self.m, s)
        # Original aligned source geometry (clearance .25), at actual-root-based
        # virtual pose, supplies both swing position AND consistent orientation.
        goals = s.xpos[self.p.bids].copy()
        rotations = s.xmat[self.p.bids].reshape(6, 3, 3).copy()
        goals[stance], rotations[stance] = self.anchor_pos[stance], self.anchor_rot[stance]
        s.qpos[self.p.qa] = np.clip(d.qpos[self.p.qa], self.ref.lo, self.ref.hi)
        residual = np.zeros((6, 2))
        for li, ix in enumerate(self.indices):
            qa, va = self.p.qa[ix], self.p.va[ix]
            bid = self.p.bids[li]
            for _ in range(KNOBS['ik_steps']):
                mj.mj_forward(self.m, s)
                dp = goals[li]-s.xpos[bid]
                dr = Rotation.from_matrix(rotations[li]@s.xmat[bid].reshape(3, 3).T).as_rotvec()
                mj.mj_jac(self.m, s, self.jp, self.jr, s.xpos[bid], bid)
                J = np.vstack((self.jp[:, va], KNOBS['orientation_length']*self.jr[:, va]))
                error = np.r_[dp, KNOBS['orientation_length']*dr]
                step = J.T@np.linalg.solve(J@J.T+KNOBS['ik_damping']**2*np.eye(6), error)
                s.qpos[qa] = np.clip(s.qpos[qa]+np.clip(step, -KNOBS['ik_joint_step'], KNOBS['ik_joint_step']),
                                     self.ref.lo[ix], self.ref.hi[ix])
            mj.mj_forward(self.m, s)
            residual[li] = [np.linalg.norm(goals[li]-s.xpos[bid]), np.linalg.norm(
                Rotation.from_matrix(rotations[li]@s.xmat[bid].reshape(3, 3).T).as_rotvec())]
        desired = s.qpos[self.p.qa].copy()
        assert np.isfinite(desired).all() and not s.warning.number.any()
        return desired, dict(stance=stance, stance_entry=entry, virtual_root=virtual,
            reference_body_position=self.reference_position.copy(), anchor_position=self.anchor_pos.copy(),
            anchor_rotation=self.anchor_rot.copy(), foot_goal_position=goals,
            foot_goal_rotation=rotations, ik_residual=residual, body_feedback_velocity=feedback)


def source_hashes():
    """Hash all imported project sources, including unregistered wrapper modules."""
    paths = {Path(__file__), SOURCE, WAVE, n.GAIT, INERT, DEFAULT_MODEL_PATH}
    seen = set()
    def visit(module):
        if id(module) in seen:
            return
        seen.add(id(module))
        path = getattr(module, '__file__', None)
        if path and str(path).startswith(str(ROOT)) and '/site-packages/' not in str(path):
            paths.add(Path(path))
            for value in vars(module).values():
                if isinstance(value, types.ModuleType):
                    visit(value)
    visit(aligned)
    for module in list(sys.modules.values()):
        if module is not None:
            visit(module)
    return {str(p.resolve()): n.sha(p) for p in sorted(paths)}


def prereg(folder, args):
    protocol = dict(stage='S5 physical teacher feasibility ONLY', control='teacher', neural_success=False,
        seconds=args.seconds, assist_window=[.8, 1.], release_window=[1., args.seconds],
        feedback_off=args.feedback_off, knobs=KNOBS, wave_metadata=json.loads(WAVE.read_text()),
        physics=dict(nq=49, nv=48, nu=42, nmesh=71, neq=0, gravity=[0, 0, -9801],
                     dt=DT, command_dt=CDT, policy_dt=PDT, torque_caps='original unchanged',
                     kp='limit/.2*32', kd='limit*.005*sqrt32', derivative_target=0,
                     root_writes='scratch MjData only; never actual after fresh initialization',
                     forces='no qfrc_applied/xfrc_applied; no gravity/friction/mass changes'),
        reference='aligned periodic two-neighbor PCHIP, 1Hz, ramp .5s; first 1s original teacher',
        anchors='actual Tarsus5 world position/rotation at1s, then measured once per nominal stance entry',
        swing='unaltered aligned-source FK at virtual root; source .25 clearance, no added lift',
        ik='3 damped Newton steps per leg;7 actuator-order hinges resolved by name; native bounds',
        body_reference='internally integrated .25native/s along captured yaw; optimization target ONLY',
        matched_control='--feedback-off uses original reference throughout, no body or anchor correction',
        assessment='full [1,end] native D1-D8/D4b unchanged; independent replay/steps/heading/material-slip required; support alone never walking',
        slip=dict(max_material_slip_to_body_speed=.5, evaluation_stride_native=40),
        heading=dict(mean_yaw_deg_s=2., range_rad=.5, straightness=.9, lateral_to_forward=.2),
        failure='no forward progress is failure; no scripted displacement or neural claim',
        inert_model='old MODEL used only by Policy.observe logger; no action/predict/weight evaluation',
        sensory_contract=contract(),
        source_hashes=source_hashes(), versions=dict(numpy=np.__version__, mujoco=mj.__version__),
        units='native; absolute SI force NOT_ESTABLISHED')
    n.dump(folder/'PREREGISTRATION.json', protocol)  # MUST precede fresh physics integration.


def simulate(args):
    if not np.isfinite(args.seconds) or args.seconds < 4 or abs(args.seconds/CDT-round(args.seconds/CDT)) > 1e-8:
        raise ValueError('Need at least4s total/full3s release, integer1ms duration')
    if not args.tag or Path(args.tag).name != args.tag or args.tag in ('.', '..'):
        raise ValueError('tag must be a single directory name')
    folder = OUT/args.tag
    folder.mkdir(parents=True, exist_ok=False)
    p = n.Plant(); m = p.m
    assert (m.nq, m.nv, m.nu, m.nmesh, m.neq) == (49, 48, 42, 71, 0)
    assert m.opt.timestep == DT and np.array_equal(m.opt.gravity, [0., 0., -9801.])
    ref = n.Reference(p)
    teacher = AnchorTeacher(p, ref)
    prereg(folder, args)
    d = p.fresh()
    a = IncrementalActuator(m, d, kp=p.limits/.2*KNOBS['kp_scale'],
                            kd=p.limits*.005*np.sqrt(KNOBS['kp_scale']), max_rate_rad_s=CAP)
    history = n.Policy(INERT)  # observe() only; the old network never produces control.
    nt = int(round(args.seconds/CDT))
    native, commands = np.empty((nt*80, 20)), np.empty((nt, 127))
    body, targets, actions, sensory, teachers = [], [], [], [], []
    rate = np.zeros(42); start = time.monotonic()
    for k in range(nt):
        t = float(d.time)
        if k % 5 == 0:
            history.observe(a.target_rad)
            before = p.snapshot(d)
            if k == 1000:
                teacher.capture(d, t)
                np.savez_compressed(folder/'HANDOVER.npz', history=history.buf, target=a.target_rad,
                                    physics_integration=before)
            desired = ref.warm(t)
            meta = dict(stance=(teacher.phase(t)+teacher.offset) % 1 < 5/6,
                        virtual_root=d.qpos[:7].copy(), reference_body_position=d.qpos[:3].copy(),
                        anchor_position=d.xpos[p.bids].copy(), anchor_rotation=d.xmat[p.bids].reshape(6, 3, 3).copy(),
                        foot_goal_position=d.xpos[p.bids].copy(), foot_goal_rotation=d.xmat[p.bids].reshape(6, 3, 3).copy(),
                        ik_residual=np.zeros((6, 2)), stance_entry=np.zeros(6, bool), body_feedback_velocity=np.zeros(2))
            if k >= 1000 and not args.feedback_off:
                desired, meta = teacher.desired(d, t)
            rate = np.clip((desired-a.target_rad)/PDT, -CAP, CAP)
            assert np.array_equal(before, p.snapshot(d)), 'controller changed actual physical state'
            if k == 1000:
                np.savez_compressed(folder/'HANDOVER_AFTER.npz', physics_integration=p.snapshot(d))
            velocity = np.zeros(6)
            mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY, p.thorax, velocity, 0)
            R = d.xmat[p.thorax].reshape(3, 3)
            loads = p.contact(d)[1:7]
            sensory.append(dict(time=t, features=features(p, d, a.target_rad, loads),
                qpos=d.qpos.copy(), qvel=d.qvel.copy(), q=d.qpos[p.qa].copy(),
                joint_velocity=d.qvel[p.va].copy(), loads=loads/n.dsm.WEIGHT,
                root=d.qpos[:7].copy(), gravity_body=R.T@np.array([0., 0., -1.]),
                body_velocity_angular_linear=np.r_[R.T@velocity[:3], R.T@velocity[3:]],
                world_velocity_angular_linear=velocity.copy()))
            teachers.append(dict(time=t, phase=teacher.phase(t), active=k >= 1000 and not args.feedback_off,
                                 joint_desired=desired.copy(), target=a.target_rad.copy(), rate=rate.copy(), **meta))
            targets.append(a.target_rad.copy()); actions.append(rate.copy())
        commands[k] = np.r_[t, a.target_rad, rate, np.zeros(42)]
        a.update(rate, CDT)
        for sub in range(80):
            d.ctrl[:] = a.torque(d)
            assert not d.qfrc_applied.any() and not d.xfrc_applied.any(), 'auxiliary force forbidden'
            mj.mj_step(m, d); mj.mj_forward(m, d)
            assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and not d.warning.number.any()
            native[k*80+sub] = p.contact(d)
        if k % 5 == 4:
            body.append(p.bodyrow(d))
        if (k+1) % 1000 == 0:
            print(json.dumps(dict(tag=args.tag, sim_s=(k+1)/1000, wall_s=time.monotonic()-start)), flush=True)
    for name, rows in [('BODY', body), ('SENSORY', sensory), ('TEACHER', teachers)]:
        np.savez_compressed(folder/(name+'.npz'), **{key:np.asarray([r[key] for r in rows]) for key in rows[0]})
    np.savez_compressed(folder/'NATIVE.npz', rows=native)
    np.savez_compressed(folder/'COMMANDS.npz', rows=commands, kp=a.kp, kd=a.kd, limits=a.torque_limit, q0=p.q0[p.qa])
    np.savez_compressed(folder/'POLICY.npz', targets=targets, actions=actions)
    assist, release = native[(native[:, 0] >= .8) & (native[:, 0] < 1)], native[native[:, 0] >= 1]
    gate = n.dsm.evaluate_native(dict(foot_support_mean=float(assist[:, 1:7].sum(1).mean()/n.dsm.WEIGHT),
                                    com_z_mean=float(assist[:, 10].mean())), release)
    gate['checks']['D1_no_nonfoot_ground_force'] = bool(release[:, 7].max() == 0)
    gate['passed'] = all(gate['checks'].values())
    progress = float((d.qpos[:3]-teacher.origin)@(teacher.velocity/KNOBS['desired_speed']))
    result = dict(tag=args.tag, controller='teacher', seconds=args.seconds, release_seconds=args.seconds-HANDOVER,
        validation=gate, finite=bool(np.isfinite(native).all()), warnings=int(d.warning.number.sum()),
        model=str(INERT), model_hash=n.sha(INERT), body_hash=n.sha(folder/'BODY.npz'),
        warm_history_preserved=True, physics_handover_bitwise=True, original_brain='not run; algorithm training teacher',
        disturb=False, feedback_off=args.feedback_off, neural_success=False, release_forward_progress=progress,
        forward_progress_positive=progress > 0, physical_walking_verified=False,
        status='PENDING_INDEPENDENT_SLIP_STEPS_HEADING_REPLAY' if progress > 0 else 'FAILED_NO_FORWARD_PROGRESS',
        wall_s=time.monotonic()-start, folder=str(folder))
    n.dump(folder/'RESULT.json', result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds', type=float, default=4.)
    ap.add_argument('--tag', default='anchor_screen')
    ap.add_argument('--feedback-off', action='store_true')
    simulate(ap.parse_args())
