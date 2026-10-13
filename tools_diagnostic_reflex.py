"""DIAGNOSTIC ONLY: can this BODY walk at all, when a designer-written feedback law drives it?

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This file is a CAPABILITY TEST of the body, skeleton and muscle set.  It writes its own
state-feedback law (a reflex) and optimises its weights against a locomotion score.  That makes it
exactly the thing the project forbids inside the autonomous loop:

  * the commands are NOT the fly's nervous system output - they are a hand-written controller;
  * the score contains an external forward objective - the animal has no such objective here.

So this tool is kept OUT of the autonomous loop on purpose:
  * it never writes into any autonomous artefact and never touches the canonical model;
  * its outputs are labelled DIAGNOSTIC_CONTROL_NOT_AUTONOMOUS everywhere, and its numbers must be
    read as "what a designer-written feedback law can get out of this body", never as "what the fly
    can do";
  * the autonomous-loop result is quoted next to it for contrast, and the two must never be mixed.

WHY IT IS STILL WORTH RUNNING
-----------------------------
It answers one question that decides where the next effort goes: is the WIRING DESIGN the thing that
blocks walking, or can nothing walk on this body?  If an optimised reflex cannot walk either, no
amount of annotation work will produce walking, and the effort should move elsewhere.  If it can,
then a locomotion solution exists inside this body, and the connectome wiring hypothesis is what
needs changing.

THE FEEDBACK LAW (deliberately small, mirror-symmetric)
-------------------------------------------------------
Per leg, the controller observes only body-available signals, taken from the same class-level
proxies the autonomous loop already uses:
    load      = tarsus contact force / body weight      (campaniform class)
    touch     = binary tarsus contact                   (tactile class)
    ft        = femur-tibia angle                       (chordotonal class position)
plus the body's own pose error and velocity in x and y.
Activation for a muscle of leg L is
    a = clip(bias + k_load*load + k_touch*touch + k_ft*ft + k_h*(h - h_ref)
             + k_vx*vx + k_vy*vy + k_gx*gx + k_gy*gy, 0, 1)
with the SAME weights for all six legs (mirror symmetry), the reference pose taken from the supplied
balance stance, and no time schedule, no phase variable and no target trajectory.  Every one of these
signals is a body state; nothing is a stored plan.

Diagnostic only.  No model file is written, no runtime authorisation is granted.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'diagnostics'))

import diag_static_support_feasibility as r15  # noqa: E402
import diag_posture_support_search as r16      # noqa: E402

STANCE = HERE / 'outputs/design_runtime_round36/stance_for_runner.json'
MARGIN = HERE / 'outputs/stance_margin_round37/stance_margin.json'
OUT = HERE / 'outputs/diagnostic_reflex_round37'
XML = r15.XML
PARAM_J = ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll',
           'Trochanter_yaw', 'Trochanter_pitch', 'Trochanter_roll', 'Tibia_pitch')


class Body:
    """The fly body plus the observation and command channels the reflex uses."""

    def __init__(self, stance_path: Path, penetration_mm: float = -0.050, xml: Path = None):
        import mujoco as mj
        from flygym.compose.fly.musculoskeletal import _load_mjcf
        self.mj = mj
        self.xml = Path(xml).resolve() if xml is not None else XML
        self.is_variant = xml is not None
        self.model = _load_mjcf(str(self.xml)).compile()
        self.data = mj.MjData(self.model)
        self.model.actuator_ctrlrange[:, 0] = 0.0
        names = [self.model.actuator(i).name or f'act{i}' for i in range(self.model.nu)]
        self.names = names
        self.nu = self.model.nu
        self.floor = r16.floor_geom_id(self.model, mj)
        self.tarsus_geoms = [g for g in range(self.model.ngeom)
                             if 'Tarsus' in (mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, g) or '')]
        self.tarsus_names = {mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, g)
                             for g in self.tarsus_geoms}
        self.thorax = self.model.body('Thorax').id
        self.body_weight = float(mj.mj_getTotalmass(self.model)
                                 * abs(float(self.model.opt.gravity[2])))
        stance = json.loads(Path(stance_path).read_text())
        mj.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qvel[:] = 0
        mj.mj_forward(self.model, self.data)
        key = self.data.qpos.copy()
        qpos = key.copy()
        offs = stance.get('joint_offsets')
        if offs:
            for jn, v in offs.items():
                jid = self.model.joint(jn).id
                lo, hi = float(self.model.jnt_range[jid][0]), float(self.model.jnt_range[jid][1])
                qpos[self.model.jnt_qposadr[jid]] = float(np.clip(
                    key[self.model.jnt_qposadr[jid]] + float(v), lo, hi))
        for jn, v in stance.get('joints_rad', {}).items():
            qpos[self.model.jnt_qposadr[self.model.joint(jn).id]] = float(v)
        seated, self.root_z, _ = r16.seat_root(self.model, self.data, mj, qpos,
                                               self.tarsus_geoms, self.floor, penetration_mm)
        self.seated = np.asarray(seated, float)

        # actuator index structure: leg actuators are the muscle channels that belong to a leg
        self.leg_act = {leg: [] for leg in r16.LEGS}
        self.joint_act = {leg: {j: [] for j in PARAM_J} for leg in r16.LEGS}
        # Which parameterised joint does each actuator act on?  The tendon's attachment sites sit on
        # the bodies a muscle spans, which does NOT single out one joint (86 of 90 actuators touch
        # more than one coxa joint body).  The generalised force DOES: assign the actuator to the
        # joint on which its moment arm is largest at the balance pose.  Because the reflex is linear
        # around that pose, grouping by the dominant joint is what "the gain for that joint" means.
        R0, _ = r15.actuator_moment_matrix(self.model, self.data, mj)
        jcol = {}
        for leg in r16.LEGS:
            for j in PARAM_J:
                jcol[(leg, j)] = int(self.model.jnt_dofadr[self.model.joint(f'joint_{leg}{j}').id])
        for a in range(self.nu):
            nm = names[a]
            if nm[:2] not in self.leg_act:
                continue
            leg = nm[:2]
            self.leg_act[leg].append(a)
            best = max(PARAM_J, key=lambda j: abs(float(R0[jcol[(leg, j)], a])))
            self.joint_act[leg][best].append(a)
        self.leg_act = {k: np.asarray(v, int) for k, v in self.leg_act.items()}
        self.joint_act = {leg: {j: np.asarray(v, int) for j, v in d.items()}
                          for leg, d in self.joint_act.items()}
        self.joint_coverage = sum(len(v) for d in self.joint_act.values() for v in d.values())
        if self.joint_coverage != len(names):
            raise ValueError(f'muscle->joint grouping covers {self.joint_coverage} of {len(names)} '
                             'actuators; every muscle channel must map to exactly one joint')
        self.tibia_joint_qadr = {leg: int(self.model.jnt_qposadr[
            self.model.joint(f'joint_{leg}Tibia_pitch').id]) for leg in r16.LEGS}
        self.tibia_ref = {leg: float(self.seated[self.tibia_joint_qadr[leg]]) for leg in r16.LEGS}

    def reset(self):
        mj = self.mj
        mj.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.time = 0.0
        self.data.qpos[:] = self.seated
        self.data.qvel[:] = 0
        mj.mj_forward(self.model, self.data)

    def observe(self):
        """Per-leg class-level proxies plus the body's own pose error and velocity."""
        mj, model, data = self.mj, self.model, self.data
        foot_force = {leg: 0.0 for leg in r16.LEGS}
        for k in range(data.ncon):
            con = data.contact[k]
            g2 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, int(con.geom2)) or ''
            f = np.zeros(6)
            mj.mj_contactForce(model, data, k, f)
            leg = g2[:2] if g2[:2] in r16.LEGS else None
            if leg:
                foot_force[leg] += float(abs(f[0]))
        q = np.asarray(data.qpos[3:7], float)
        up_z = float(1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2))
        obs = {
            'height': float(data.xpos[self.thorax][2]),
            'up_z': up_z,
            'vx': float(data.qvel[0]), 'vy': float(data.qvel[1]), 'vz': float(data.qvel[2]),
            'gx': float(data.qpos[0]), 'gy': float(data.qpos[1]),
            'gpitch': float(1.0 - up_z),
            'foot_force_norm': {leg: min(foot_force[leg] / max(self.body_weight, 1e-12), 1.0)
                                for leg in r16.LEGS},
            'touch': {leg: 1.0 if foot_force[leg] > 1e-9 else 0.0 for leg in r16.LEGS},
        }
        return obs

    def apply(self, ctrl):
        self.data.ctrl[:] = ctrl


N_JOINT = 7            # the parameterised leg joints, in PARAM_J order
N_FEEDBACK = 9         # load, touch, ft, h, vz, vx, vy, gx, gy (the last five may be per side)


def theta_layout():
    """Named layout, so the parameter vector stays auditable.

    per-leg bias         : 6 legs x 7 joints = 42   (joint-specific resting activation)
    per-side feedback    : 2 sides x 9 signals = 18 (left and right may differ; fore/mid/hind share)
    global up_z gain     : 1
    total                : 61
    """
    return {'bias': (0, 42), 'feedback': (42, 42 + 18), 'upz': 42 + 18}


def reflex_ctrl(body: Body, theta: np.ndarray, obs: dict, h_ref: float) -> np.ndarray:
    """a = clip(bias[leg,joint] + feedback[side] . signals + k_upz*(1-up_z), 0, 1).

    Every term is a body-available, class-level signal: leg load (campaniform class), tarsus touch
    (tactile class), femur-tibia angle (chordotonal class), plus the body's own pose error and
    velocity.  There is no phase variable, no schedule, no target trajectory and no stored plan.
    """
    lay = theta_layout()
    b0, b1 = lay['bias']; f0, f1 = lay['feedback']; u0 = lay['upz']
    ctrl = np.zeros(body.nu)
    upz_gain = float(theta[u0])
    common = np.array([obs['height'] - h_ref, obs['vz'], obs['vx'], obs['vy'],
                       obs['gx'], obs['gy']], float)
    for leg in r16.LEGS:
        side = 0 if leg[0] == 'L' else 1
        w = np.asarray(theta[f0 + side * N_FEEDBACK: f0 + (side + 1) * N_FEEDBACK], float)
        load = obs['foot_force_norm'][leg]
        touch = obs['touch'][leg]
        ft = float(body.data.qpos[body.tibia_joint_qadr[leg]]) - body.tibia_ref[leg]
        signals = np.concatenate(([load, touch, ft], common))
        li = r16.LEGS.index(leg)
        for j, jn in enumerate(PARAM_J):
            acts = body.joint_act[leg][jn]
            if acts.size == 0:
                continue
            val = (theta[b0 + li * N_JOINT + j]
                   + float(w @ signals) + upz_gain * (1.0 - obs['up_z']))
            ctrl[acts] = np.clip(val, 0.0, 1.0)
    return ctrl


def rollout(body: Body, theta: np.ndarray, seconds: float, h_ref: float,
            record_every: int = 20, perturb_at_s: float = None, perturb_vy: float = 0.0):
    """Integrate the body under the reflex.  Returns the locomotion metrics and a trace.

    `perturb_at_s`/`perturb_vy` add ONE pure lateral velocity impulse to the root at a fixed time.
    That is a physical displacement, not a command: it is how the file asks whether the reflex
    responds to a disturbance at all, or merely runs its own course while the body tips."""
    body.reset()
    mj, model, data = body.mj, body.model, body.data
    n_step = int(round(seconds / model.opt.timestep))
    every = max(1, n_step // 400)
    trace, up_frac_n, up_n, tarsus_only_n = [], 0, 0, 0
    heights, xs, ys = [], [], []
    perturbed = False
    for i in range(n_step):
        if (perturb_at_s is not None and not perturbed and data.time >= perturb_at_s):
            data.qvel[1] += float(perturb_vy)
            perturbed = True
        obs = body.observe()
        body.apply(reflex_ctrl(body, theta, obs, h_ref))
        mj.mj_step(model, data)
        if i % every == 0 or i == n_step - 1:
            o = body.observe()
            facts = r16.pose_facts(model, data, mj, body.tarsus_names, body.thorax)
            heights.append(o['height']); xs.append(o['gx']); ys.append(o['gy'])
            up_n += 1
            if o['up_z'] >= 0.9:
                up_frac_n += 1
            if facts['contacts_are_feet_only_tarsus']:
                tarsus_only_n += 1
            trace.append({'t': float(data.time), 'height_mm': o['height'],
                          'up_z': o['up_z'], 'x_mm': o['gx'], 'y_mm': o['gy'],
                          'vy': o['vy'],
                          'contacts': facts['contact_count'],
                          'tarsus_only': bool(facts['contacts_are_feet_only_tarsus'])})
    x0, y0 = xs[0], ys[0]
    dist = float(np.hypot(xs[-1] - x0, ys[-1] - y0))
    finite = [(x, y) for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)]
    if len(finite) > 1:
        dist = max(dist, float(np.hypot(finite[-1][0] - finite[0][0],
                                        finite[-1][1] - finite[0][1])))
    return {
        'distance_mm': dist,
        'upright_fraction': up_frac_n / max(up_n, 1),
        'tarsus_only_fraction': tarsus_only_n / max(up_n, 1),
        'height_final_mm': float(heights[-1]), 'height_min_mm': float(min(heights)),
        'height_max_mm': float(max(heights)),
        'trace': trace,
    }


def score(metrics: dict, upright_min: float = 0.8, tarsus_min: float = 0.9) -> float:
    """Distance, GATED by the project's contract but with a smooth interior, so the optimiser has a
    gradient before it clears the gate:

        score = distance * upright_fraction * tarsus_only_fraction        (smooth part)
        reported 'contract_pass' is the hard gate: upright >= 0.8 and tarsus_only >= 0.9

    The gate is what the project's frozen contract asks for; the smooth part only shapes the search.
    A run that never clears the gate has NOT walked, whatever the smooth number says.
    """
    return float(metrics['distance_mm']) * float(metrics['upright_fraction']) \
        * float(metrics['tarsus_only_fraction'])


def contract_pass(metrics: dict, upright_min: float = 0.8, tarsus_min: float = 0.9) -> bool:
    return (metrics['upright_fraction'] >= upright_min
            and metrics['tarsus_only_fraction'] >= tarsus_min)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--xml', type=Path, default=None,
                    help='body asset to use (default: the canonical standing asset). Any other '
                         'value is a VARIANT and is recorded as such.')
    ap.add_argument('--stance-file', type=Path, default=STANCE,
                    help='starting stance JSON; `joint_offsets` are relative to the XML keyframe and '
                         '`joints_rad` are absolute')
    ap.add_argument('--margin-design', type=Path, default=MARGIN,
                    help='design JSON whose best pose is used as the starting stance (its `x` is a '
                         'set of joint OFFSETS from the XML keyframe)')
    ap.add_argument('--out-dir', type=Path, default=OUT)
    ap.add_argument('--seconds', type=float, default=0.5)
    ap.add_argument('--evals', type=int, default=600)
    ap.add_argument('--wall-budget-s', type=float, default=1800.0)
    ap.add_argument('--seed-json', type=Path, default=None)
    ap.add_argument('--validate-seconds', type=float, default=1.0,
                    help='re-run the best reflex over a longer window')
    ap.add_argument('--perturb-at-s', type=float, default=0.2)
    ap.add_argument('--perturb-vy', type=float, default=3.0,
                    help='lateral velocity impulse on the root (mm/s) used in validation')
    args = ap.parse_args()
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    body = Body(args.stance_file, xml=args.xml)
    h_ref = float(body.seated[2])
    xml_before = r15.digest(XML)

    # ---------------------------------------------------------------------------------
    # A trivial baseline first: hold the balance-pose activation constant.
    # ---------------------------------------------------------------------------------
    validation: dict = {}
    zero = np.zeros(61)
    base_m = rollout(body, zero, args.seconds, h_ref)
    base_m['contract_pass'] = contract_pass(base_m)
    print(json.dumps({'baseline_zero_reflex': {k: v for k, v in base_m.items() if k != 'trace'},
                      'contract_pass': contract_pass(base_m)}, indent=1), flush=True)

    # ---------------------------------------------------------------------------------
    # Seed the reflex from the static balance solution: the mean activation of the actuators
    # belonging to each joint becomes that joint's resting bias.  This gives the optimiser a
    # starting point that at least supports the body, instead of learning support from scratch.
    # ---------------------------------------------------------------------------------
    def seeded_theta():
        th = np.zeros(61)
        src = None
        for path, get in ((MARGIN, lambda d: d.get('best')),):
            try:
                d = json.loads(path.read_text())
                if d.get('best', {}).get('feasible'):
                    src = d['best']
            except Exception:
                pass
        if src and src.get('activation'):
            act = src['activation']
            for leg in r16.LEGS:
                li = r16.LEGS.index(leg)
                for j, jn in enumerate(PARAM_J):
                    acts = body.joint_act[leg][jn]
                    if acts.size:
                        th[li * 7 + j] = float(np.mean([act[body.names[a]] for a in acts]))
        return th, src

    seed_th, seed_src = seeded_theta()
    starts = [zero, seed_th]

    rng = np.random.default_rng(7)
    best = {'score': score(base_m), 'theta': zero.tolist(), 'metrics': base_m,
            'contract_pass': contract_pass(base_m)}
    hist = []
    evals = 0

    def try_theta(th):
        nonlocal evals
        m = rollout(body, np.asarray(th, float), args.seconds, h_ref)
        evals += 1
        s = score(m)
        m['contract_pass'] = contract_pass(m)
        hist.append({'eval': evals, 'score': s, 'distance_mm': m['distance_mm'],
                     'upright_fraction': m['upright_fraction'],
                     'tarsus_only_fraction': m['tarsus_only_fraction'],
                     'contract_pass': m['contract_pass']})
        return s, m

    # a few hand-set plausible reflexes on top of the seed
    for k_load in (-0.5, 0.5):
        for k_vx in (-0.05, 0.05):
            th = seed_th.copy(); th[42] = k_load; th[42 + 4] = k_vx
            starts.append(th)
    for th in starts:
        s, m = try_theta(th)
        if s > best['score']:
            best = {'score': s, 'theta': th.tolist(), 'metrics': m,
                    'contract_pass': m['contract_pass']}

    step0 = 0.10
    cur = np.asarray(best['theta'], float)
    cur_s = best['score']
    n_pass = 0
    while evals < args.evals and time.time() - t0 < args.wall_budget_s and n_pass < 6:
        n_pass += 1
        improved = False
        for k in range(cur.size):
            for d in (+step0, -step0):
                if evals >= args.evals or time.time() - t0 > args.wall_budget_s:
                    break
                cand = cur.copy(); cand[k] += d
                sc, m = try_theta(cand)
                if sc > best['score']:
                    best = {'score': sc, 'theta': cand.tolist(), 'metrics': m,
                            'contract_pass': m['contract_pass']}
                if sc > cur_s + 1e-9:
                    cur, cur_s, improved = cand, sc, True
        if not improved:
            step0 *= 0.5
            if step0 < 0.005:
                break
        print(json.dumps({'pass': n_pass, 'evals': evals, 'step': step0,
                          'best_score': round(best['score'], 4),
                          'best_distance_mm': round(best['metrics']['distance_mm'], 3),
                          'best_upright': round(best['metrics']['upright_fraction'], 3),
                          'best_tarsus_only': round(best['metrics']['tarsus_only_fraction'], 3),
                          'contract_pass_any': any(h.get('contract_pass') for h in hist),
                          'wall_s': round(time.time() - t0, 1)}), flush=True)
    if validation.get('best'):
        v = validation.get('best', {})
        print(json.dumps({'validation_seconds': validation['seconds'],
                          'validation_distance_mm': v.get('distance_mm'),
                          'validation_upright': v.get('upright_fraction'),
                          'validation_tarsus_only': v.get('tarsus_only_fraction'),
                          'validation_contract_pass': v.get('contract_pass'),
                          'lateral_impulse': {k: vv for k, vv in
                                              validation.get('lateral_impulse', {}).items()
                                              if k not in ('trace_with', 'trace_without')}},
                         indent=1), flush=True)
    print(json.dumps({'evaluations': evals, 'wall_s': round(time.time() - t0, 1),
                      'best_score': best['score'],
                      'contract_pass_any': any(h.get('contract_pass') for h in hist),
                      'best_distance_mm': best['metrics']['distance_mm'],
                      'best_upright_fraction': best['metrics']['upright_fraction'],
                      'best_tarsus_only_fraction': best['metrics']['tarsus_only_fraction']},
                     indent=1), flush=True)

    # ---------------------------------------------------------------------------------
    # VALIDATION of the best reflex found: a longer window, and one lateral velocity impulse.
    # This is where "it moved 2.5 mm" gets separated from "it controlled its attitude".
    # ---------------------------------------------------------------------------------
    best_th = np.asarray(best['theta'], float)
    if args.validate_seconds > 0:
        long_run = rollout(body, best_th, args.validate_seconds, h_ref)
        long_run['contract_pass'] = contract_pass(long_run)
        zero_long = rollout(body, zero, args.validate_seconds, h_ref)
        zero_long['contract_pass'] = contract_pass(zero_long)
        val = {'seconds': args.validate_seconds,
               'best': {k: v for k, v in long_run.items() if k != 'trace'},
               'zero_reflex': {k: v for k, v in zero_long.items() if k != 'trace'},
               'best_trace': long_run['trace']}
        if args.perturb_vy != 0.0:
            p1 = rollout(body, best_th, args.validate_seconds, h_ref,
                         perturb_at_s=args.perturb_at_s, perturb_vy=args.perturb_vy)
            p0 = rollout(body, best_th, args.validate_seconds, h_ref,
                         perturb_at_s=args.perturb_at_s, perturb_vy=0.0)
            dev = [abs(a['y_mm'] - b['y_mm']) for a, b in zip(p1['trace'], p0['trace'])]
            half = len(dev) // 2
            val['lateral_impulse'] = {
                'at_s': args.perturb_at_s, 'vy_mm_s': args.perturb_vy,
                'y_deviation_first_half_mm': float(max(dev[1:half] or [0.0])),
                'y_deviation_second_half_mm': float(max(dev[half:] or [0.0])),
                'y_deviation_final_mm': float(dev[-1]),
                'up_z_end_with_impulse': p1['trace'][-1]['up_z'],
                'up_z_end_without_impulse': p0['trace'][-1]['up_z'],
                'response': ('the deviation shrinks: the reflex reacts to the disturbance'
                             if max(dev[half:] or [0.0]) < max(dev[1:half] or [0.0])
                             else 'the deviation does not shrink: no attitude response visible'),
                'trace_with': p1['trace'], 'trace_without': p0['trace']}
        validation = val

    report = {
        'status': 'DIAGNOSTIC_CONTROL_NOT_AUTONOMOUS',
        'validation_of_best': validation,
        'warning': 'This is a DESIGNER-WRITTEN feedback law optimised against a locomotion score. '
                   'It is NOT the fly\'s nervous system output and the score is an EXTERNAL forward '
                   'objective the animal does not have here. Its numbers measure the BODY\'s ceiling, '
                   'never the fly\'s ability, and must not be reported as an autonomous result.',
        'question': 'can this body, skeleton and muscle set walk at all, when a designer-written '
                    'reflex drives it?',
        'body_model': str(body.xml), 'is_variant_model': bool(body.is_variant),
        'canonical_body_model': str(XML),
        'stance': {'source': str(args.stance_file.resolve()),
                   'margin_design': str(args.margin_design.resolve()),
                   'root_z_mm': h_ref,
                   'adhesion': 'NONE: the canonical body has no adhesion actuator and none is used'},
        'reflex': {
            'law': 'a = clip(bias[leg,joint] + w[side] . [load, touch, ft-ft_ref, h-h_ref, vz, vx, '
                   'vy, gx, gy] + k_upz*(1-up_z), 0, 1)',
            'n_parameters': 61,
            'layout': {'bias': '6 legs x 7 joints', 'feedback': '2 sides x 9 signals',
                       'upz': 1},
            'seeded_from': ('the zero-adhesion muscle-only balance solution '
                            '(outputs/stance_margin_round37)' if seed_src else 'no seed available'),
            'mirror_symmetric_across_legs': False,
            'signals_are_body_available_class_level_proxies': True,
            'no_target_trajectory': True, 'no_phase_or_schedule_variable': True},
        'score': 'distance * upright_fraction * tarsus_only_fraction (smooth search objective); '
                 'the project\'s hard contract is reported separately as contract_pass '
                 '(upright >= 0.8 and tarsus_only >= 0.9)',
        'seconds_per_rollout_s': args.seconds,
        'baseline_zero_reflex': {k: v for k, v in base_m.items() if k != 'trace'},
        'contract_pass_any_evaluation': any(h.get('contract_pass') for h in hist),
        'best': {'score': best['score'], 'contract_pass': best.get('contract_pass'),
                 'theta': best['theta'],
                 'metrics': {k: v for k, v in best['metrics'].items() if k != 'trace'},
                 'trace': best['metrics']['trace']},
        'history': hist,
        'evaluations': evals, 'wall_seconds': float(time.time() - t0),
        'source_sha256': {str(XML): r15.digest(XML), str(body.xml): r15.digest(body.xml),
                          str(args.stance_file): r15.digest(args.stance_file)},
        'limitations': [
            'An optimised reflex is a controller and an external forward objective; it is forbidden '
            'inside the autonomous loop and is used here only as a body-capability measurement.',
            'Coordinate search over 11 parameters is a weak optimiser; a negative result bounds '
            'nothing about stronger optimisers or richer reflex families.',
            'No model file is modified and no runtime authorisation is granted.'],
    }
    assert r15.digest(XML) == xml_before
    (out / 'reflex.json').write_text(json.dumps(report, indent=2))
    # re-run the best parameters once more so the saved trace is definitely the reported one
    print(out)


if __name__ == '__main__':
    main()
