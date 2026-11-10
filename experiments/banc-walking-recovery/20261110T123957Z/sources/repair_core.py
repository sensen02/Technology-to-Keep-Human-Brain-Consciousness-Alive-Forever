"""Single-source control core for the repaired pipeline (declared derived world).

Why this file exists: the audit found that training, evaluation and the renderer each
implemented their own schedule for the frozen base network, the history buffer, the
observation order and the action clipping. This module is the only control path now;
every entry (probe, trainer, evaluator, renderer) calls Core.tick.

Declared changes versus the audited pipeline:
1. one schedule: the frozen base-network target is no longer part of the action path at
   all. The policy owns the full actuator authority (+/-2 rad/s, the actuator's own
   limit), so a policy output can never be masked by a saturated tracking term.
2. the 294 history features are the last 7 commanded joint targets (lags over a K=40
   ring buffer of own commands), not the base network's predictions. Same width.
3. the six foot loads are handed to the sensory encoder in the order that encoder
   declares (LF LM LH RF RM RH); the previous code handed them in actuator order.
4. per-leg pad geometry is looked up by prefix over all three triad spheres.
5. forward motion is reported as signed thorax-frame x velocity, not a displacement
   magnitude.

Nothing here is a biological claim. Body: derived triad pad body (new hash, ledger in
experiments/banc-triad-pad/20261110T010000Z/TRIAD_LEDGER.json). Actuator limit is the
model's own ctrlrange; absolute SI is NOT_ESTABLISHED.
"""
from pathlib import Path
import importlib.util
import sys

import numpy as np
import mujoco as mj

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
ACT = ROOT / 'experiments/s5-ann-incremental/20261107T010000Z/incremental_actuator.py'
SENSORY = ROOT / 'experiments/s5-ann-walk/20261107T091243Z/proprioception.py'
BASE = ROOT / 'experiments/banc-stand-retarget/20261108T190000Z/MODEL_MEASURED_STAND.npz'
PAD_XML = ROOT / 'experiments/banc-triad-pad/20261110T010000Z/MODEL_TRIAD_x200.xml'
POSE_FILE = ROOT / 'experiments/banc-triad-stand-pose/20261110T030000Z/STAND_POSE_TRIAD_1.45.json'

WEIGHT = 24.446354755878005
PDT = .001          # policy tick
BASE_EVERY = 5      # 200 Hz base target, kept only for warm-start data generation
MAX_RATE = 2.0      # the actuator's own default rate limit (incremental_actuator.py)
MU = 1.5
ADH_FRAC = .25      # declared adhesion amplitude per leg, fraction of body weight
STANCE_LOAD = .1 * WEIGHT

LEG_ORDER = ['LF', 'LH', 'LM', 'RF', 'RH', 'RM']        # actuator / contact array order
SENSORY_ORDER = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']    # order declared by the encoder
SENS_IDX = [LEG_ORDER.index(L) for L in SENSORY_ORDER]
NL = 6
NJ = 42
PROMPT = NL
# Declared engineering encoding (PREREG_OBS_V2.json): the replacement ANN observation gains
# per-leg pad bottom height and pad vertical velocity, because the 183-feature sensory
# vector has no channel for the foot's own clearance. This is not a connectome cell and
# does not touch encoder_v2 or its 556-cell port contract.
PAD_H_DIV, PAD_V_DIV = 0.05, 10.0
PAD_EXTRA = 2 * NL
OBS = 183 + PAD_EXTRA + 294 + PROMPT


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


class Shim:
    """Adapter that proprioception.features expects."""

    def __init__(self, m):
        self.m = m
        self.jids = m.actuator_trnid[:, 0]
        self.qa = m.jnt_qposadr[self.jids]
        self.va = m.jnt_dofadr[self.jids]
        self.q0 = m.key_qpos[0]
        self.thorax = m.body('Thorax').id


class Core:
    """One plant, one schedule. Call tick(action) once per 1 ms policy tick."""

    def __init__(self, *, act_mod=None, sensory=None, mu=MU, adhesion=True):
        self.act_mod = act_mod or load_module('rc_act', ACT)
        self.sensory = sensory or load_module('rc_sens', SENSORY)
        self.plant = self.act_mod.load_full_model(PAD_XML)
        self.plant.geom_friction[:, 0] = mu
        self.d = mj.MjData(self.plant)
        self.mu = float(mu)
        self.adhesion_on = bool(adhesion)
        self.actuator = self.act_mod.IncrementalActuator(self.plant, self.d)
        self.th = self.plant.body('Thorax').id
        pose = __import__('json').loads(POSE_FILE.read_text())
        self.pose_q = np.asarray(pose['q_rad'], float)
        self.pose_z = float(pose['root_z'])
        self.a_qpos = self.plant.jnt_qposadr[self.actuator.joint_ids]
        self.a_dof = self.plant.jnt_dofadr[self.actuator.joint_ids]
        self.q0 = self.plant.key_qpos[0][self.a_qpos].copy()
        self.planes = self.plant.geom_type == mj.mjtGeom.mjGEOM_PLANE
        names = [self.plant.geom(i).name or '' for i in range(self.plant.ngeom)]
        self.pads = {L: [i for i, nm in enumerate(names) if nm.startswith(L + 'Tarsus5_pad')] for L in LEG_ORDER}
        geom_leg = np.full(self.plant.ngeom, -1)
        for li, L in enumerate(LEG_ORDER):
            for g in self.pads[L]:
                geom_leg[g] = li
        self.geom_leg = geom_leg
        self.shim = Shim(self.plant)
        self.K = 40
        self.lags = np.array([0, 8, 24, 32, 36, 38, 39])
        self._jac = np.zeros((3, self.plant.nv))
        self.t = 0
        self.reset()

    # ---- state -----------------------------------------------------------------
    def reset(self):
        mj.mj_resetDataKeyframe(self.plant, self.d, 0)
        self.d.qpos[self.a_qpos] = self.pose_q
        self.d.qpos[2] = self.pose_z
        mj.mj_forward(self.plant, self.d)
        self.actuator.reset(self.d)
        self.d.xfrc_applied[:] = 0.
        self.buf = np.repeat(self.pose_q[None], self.K, axis=0)
        self._padz_prev = None
        self.t = 0

    def reset_from(self, qpos, qvel):
        mj.mj_resetDataKeyframe(self.plant, self.d, 0)
        self.d.qpos[:] = qpos
        self.d.qvel[:] = qvel
        mj.mj_forward(self.plant, self.d)
        self.actuator.reset(self.d)
        self.d.xfrc_applied[:] = 0.
        self.buf = np.repeat(self.d.qpos[self.a_qpos][None], self.K, axis=0)
        self._padz_prev = None
        self.t = 0

    # ---- measurement -----------------------------------------------------------
    def measure(self):
        loads = np.zeros(NL)
        nonfoot = 0.
        wt = np.zeros(NL)
        den = np.zeros(NL)
        vsum = np.zeros(NL)
        ncon = np.zeros(NL, int)
        for ci in range(self.d.ncon):
            c = self.d.contact[ci]
            ga, gb = int(c.geom1), int(c.geom2)
            if not (self.planes[ga] or self.planes[gb]):
                continue
            geom = gb if self.planes[ga] else ga
            f = np.zeros(6)
            mj.mj_contactForce(self.plant, self.d, ci, f)
            normal = max(0., float((c.frame.reshape(3, 3).T @ f[:3])[2]) * (1 if self.planes[ga] else -1))
            leg = int(self.geom_leg[geom])
            if leg < 0:
                nonfoot += normal
                continue
            loads[leg] += normal
            mj.mj_jac(self.plant, self.d, self._jac, None, np.asarray(c.pos), int(self.plant.geom_bodyid[geom]))
            v = self._jac @ self.d.qvel
            sp = float(np.hypot(v[0], v[1]))
            wt[leg] += normal * sp
            vsum[leg] += sp
            den[leg] += normal
            ncon[leg] += 1
        speed = np.where(den > 0, wt / np.maximum(den, 1e-12), 0.)
        raw_speed = np.where(ncon > 0, vsum / np.maximum(ncon, 1), 0.)
        padz = np.zeros(NL)
        foot = np.zeros((NL, 3))
        rot = self.d.xmat[self.th].reshape(3, 3)
        for li, L in enumerate(LEG_ORDER):
            zs, ps = [], []
            for g in self.pads[L]:
                zs.append(float(self.d.geom_xpos[g][2]) - float(self.plant.geom_size[g][0]))
                ps.append(np.asarray(self.d.geom_xpos[g]))
            padz[li] = min(zs) if zs else 0.
            p = np.mean(ps, axis=0) if ps else np.zeros(3)
            foot[li] = rot.T @ (p - self.d.xpos[self.th])
        return dict(loads=loads, nonfoot=nonfoot, speed=speed, raw_speed=raw_speed,
                    padz=padz, foot=foot, com=float(self.d.subtree_com[self.th][2]),
                    contact=ncon)

    def forward_body_velocity(self):
        rot = self.d.xmat[self.th].reshape(3, 3)
        return float((rot.T @ self.d.qvel[:3])[0])

    # ---- action path -----------------------------------------------------------
    def obs(self, prompt=None):
        m = self.measure()
        proprio = self.sensory.features(self.shim, self.d, self.actuator.target_rad,
                                        m['loads'][SENS_IDX])
        padz = np.asarray(m['padz'], float)
        if self._padz_prev is None:
            pad_rate = np.zeros(NL)
        else:
            pad_rate = (padz - self._padz_prev) / PDT
        self._padz_prev = padz.copy()
        clearance = np.r_[np.clip(padz / PAD_H_DIV, -10., 10.),
                          np.clip(pad_rate / PAD_V_DIV, -10., 10.)]
        x = np.concatenate([proprio, clearance, self.buf[self.lags].reshape(-1),
                            np.zeros(PROMPT) if prompt is None else np.asarray(prompt, float)])
        assert x.shape == (OBS,)
        return x

    def apply_adhesion(self, loads):
        self.d.xfrc_applied[:] = 0.
        if not self.adhesion_on:
            return
        cnt = np.maximum(self.contact_count(), 1)
        rising = np.zeros(NL, bool)
        for li, L in enumerate(LEG_ORDER):
            for g in self.pads[L]:
                mj.mj_jac(self.plant, self.d, self._jac, None, np.asarray(self.d.geom_xpos[g]),
                          int(self.plant.geom_bodyid[g]))
                if float((self._jac @ self.d.qvel)[2]) > 0.:
                    rising[li] = True
                    break
        amp = ADH_FRAC * WEIGHT
        for li in range(NL):
            if loads[li] <= STANCE_LOAD or rising[li]:
                continue
            share = amp / cnt[li]
            for ci in range(self.d.ncon):
                c = self.d.contact[ci]
                ga, gb = int(c.geom1), int(c.geom2)
                if not (self.planes[ga] or self.planes[gb]):
                    continue
                geom = gb if self.planes[ga] else ga
                if self.geom_leg[geom] != li:
                    continue
                body = int(self.plant.geom_bodyid[geom])
                f = np.array([0., 0., -share])
                r = np.asarray(c.pos) - self.d.xpos[body]
                self.d.xfrc_applied[body, 0:3] += f
                self.d.xfrc_applied[body, 3:6] += np.cross(r, f)

    def contact_count(self):
        n = np.zeros(NL, int)
        for ci in range(self.d.ncon):
            c = self.d.contact[ci]
            ga, gb = int(c.geom1), int(c.geom2)
            if not (self.planes[ga] or self.planes[gb]):
                continue
            geom = gb if self.planes[ga] else ga
            if 0 <= geom < self.geom_leg.size and self.geom_leg[geom] >= 0:
                n[self.geom_leg[geom]] += 1
        return n

    def tick(self, action):
        """One 1 ms policy tick. action in [-1,1]^42 -> rate = action*MAX_RATE."""
        return self.tick_rate(np.clip(np.asarray(action, float) * MAX_RATE, -MAX_RATE, MAX_RATE))

    def tick_rate(self, rate):
        """One 1 ms tick from an explicit joint-rate command (already in rad/s)."""
        rate = np.clip(np.asarray(rate, float), -MAX_RATE, MAX_RATE)
        self.requested_rate = rate.copy()
        self.actuator.update(rate, PDT)
        for k in range(80):
            if k % 40 == 0:
                loads = self.measure()['loads']
                self.apply_adhesion(loads)
            self.d.ctrl[:] = self.actuator.torque(self.d)
            mj.mj_step(self.plant, self.d)
            mj.mj_forward(self.plant, self.d)
        if self.d.warning.number.any():
            raise RuntimeError('MuJoCo warning')
        self.buf = np.vstack([self.buf[1:], self.actuator.target_rad[None]])
        self.t += 1
        return self.measure()


# ---- frozen base network: warm-start data source only --------------------------
class BaseNet:
    """Our own frozen measured-motion MLP. Used only to imitate from; never evaluated."""

    def __init__(self, path=BASE):
        z = np.load(path)
        self.z = z
        self.K = int(z['K'])
        self.lags = np.asarray(z['lags'])
        self.q0 = None
        self.buf = None

    def reset(self, q0):
        self.q0 = np.asarray(q0, float).copy()
        self.buf = np.repeat(self.q0[None], self.K, axis=0)

    def target(self):
        z = self.z
        x = self.buf[self.lags].reshape(-1)
        h0 = np.tanh(x @ z['w0'] + z['b0'])
        h1 = np.tanh(h0 @ z['w1'] + z['b1'])
        t = z['center'] + z['scale'] * (h1 @ z['w2'] + z['b2'])
        self.buf = np.vstack([self.buf[1:], t[None]])
        return t
