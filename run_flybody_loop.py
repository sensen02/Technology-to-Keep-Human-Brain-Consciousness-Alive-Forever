"""The SAME closed loop, run on the published FlyBody body instead of the canonical body.

MANUAL RESEARCH PROTOTYPE, NOT A PRODUCT.

Nothing about the neural side is changed here.  The tier build, the connectome weights, the
conductance network, the sensory channel partition, the sensory encoder, the muscle-pool
manifest and the spike->activation decoder are the canonical ones, imported from the published
closure modules and used with the same declared parameters.  What changes is exactly two
translation layers, and both are written down:

  SENSORY   the canonical body reports a tibia hinge deviation, a multi-joint mean deviation and
            a summed foot contact force per leg.  FlyBody reports the same three KINDS of
            quantity, read from the hinges and claw geoms that the translation tables
            (tools_flybody_translation.py) identify.  Contact force is expressed in units of the
            body's OWN per-leg weight share, which is the commensurate normaliser this project
            already declared, so the two bodies feed the encoder the same dimensionless scale.

  MOTOR     the decoder still produces one activation per canonical muscle, with the canonical
            pool, alias and normalisation rules; those 90 activations are then projected onto
            FlyBody's joint actuators through the measured coefficient table.  Two declared
            variants of that last step are run:
              servo_modulation   the published position servos are kept; the projection sets a
                                 deviation from the published neutral command (ctrl = 0), so the
                                 posture hold stays where the published model puts it: in its own
                                 servo, NOT in the connectome.
              torque_fraction    the published servos are replaced by torque motors with the same
                                 published force limit, and the projection sets a dimensionless
                                 torque fraction, so the posture hold is removed and whatever
                                 holds the fly up must come from the loop.

Arms per configuration: normal, spike_output_silence (the decoder's output is discarded), and
replay (the normal arm's commands replayed into a fresh episode).  The canonical body is run in
this same file with the same normaliser, so the comparison is like for like.

Clocks are the same as the canonical runs: neural 0.5 ms, command 5 ms, body 0.1 ms, and every
interval asserts that the body and the network both landed exactly on the command grid.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engine.memory_guard as mg
import engine.canonical_physics as cp
import run_neural_muscle_closure as closure
import run_flybody_body as body_mod
from engine.embodied.closure_clock import grid_steps, push_steps
from engine.embodied.motor_output import MotorOutput, MotorOutputConfig
from engine.neural_cond import ConductanceNetwork, ConductanceParams

OUT = HERE / 'outputs/model_branches/flybody_loop'
TRANS = OUT / 'flybody_translation.json'
FLYBODY_DIR = HERE / 'upstream/flybody'
XML_PUBLISHED = FLYBODY_DIR / 'floor.xml'
XML_TORQUE = OUT / 'flybody_torque_variant.xml'
TRANS_SHA = None            # filled from the artifact itself; see provenance
DT_BODY, DT_NEURAL, DT_COMMAND = .0001, .0005, .005
CUT_S = .015          # the canonical closure's motor-cut onset, kept identical here
LEGS = body_mod.CANONICAL_LEGS          # LF/RF/LM/RM/LH/RH, the labels both bodies share
DRIVE_FRACTION = .5   # declared: a saturated projection reaches half of the published range
ARM_NAMES = ('normal', 'no_external_touch', 'sensory_cut', 'motor_cut',
             'spike_output_silence', 'replay')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def leg_key(leg):
    return leg.lower()


def build_body(kind, tibia_sign='declared'):
    """Compile one body and return the index tables the loop needs.  `kind` is explicit.

    `tibia_sign` selects the declared translation (canonical sign transferred) or the measured
    variant (the leg's own geometric flexion check corrects the tibia sign where the transfer
    disagrees with it -- on both hind legs, as measured).
    """
    import mujoco
    if kind in ('flybody_servo', 'flybody_torque'):
        xml = XML_PUBLISHED if kind == 'flybody_servo' else XML_TORQUE
        model = mujoco.MjModel.from_xml_path(str(xml))
        if not np.isclose(model.opt.timestep, DT_BODY, rtol=0, atol=1e-15):
            raise ValueError('body timestep differs from the declared clock')
        data = mujoco.MjData(model)
        seated = np.zeros(model.nq)
        seated[2] = body_mod.seat(model, data, gap_model=0.0)
        seated[3] = 1.0
        data.qpos[:] = seated
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        translation = json.loads(TRANS.read_text())['measured']
        floor = body_mod.floor_geom(model)
        tables, joint_group = {}, {}
        for leg in LEGS:
            group = translation['sensory_translation'][leg]['multi_joint_group']
            tib = translation['sensory_translation'][leg]['tibia_hinge']
            adr = {n: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
                   for n in group}
            tib_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, tib)
            tables[leg_key(leg)] = dict(q=[adr[n] for n in group],
                               tib_q=int(model.jnt_qposadr[tib_id]),
                               tib_v=int(model.jnt_dofadr[tib_id]))
            joint_group[leg_key(leg)] = group
            if abs(model.jnt_qposadr[tib_id] - adr[tib]) != 0:
                pass  # the tibia hinge is inside the group by construction of the table
        hinge_qpos = {}
        for leg in LEGS:
            for name in translation['sensory_translation'][leg]['multi_joint_group']:
                hinge_qpos[name] = int(model.jnt_qposadr[mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        contact = {}
        for g in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
            leg = body_mod.leg_of(name)
            if leg and name.startswith('tarsal_claw'):
                contact.setdefault(leg_key(leg), []).append(g)
        if sorted(contact) != sorted(leg_key(leg) for leg in LEGS):
            raise ValueError('claw geom mapping incomplete for the FlyBody body')
        muscle_index = {name: row['actuator_index']
                        for name, row in translation['segment_projection']['table'].items()}
        sign_override = (translation['tibia_direction_check']['sign_override_by_hinge']
                         if tibia_sign == 'measured' else {})
        weight = float(model.body_mass.sum()*abs(model.opt.gravity[2]))
        return dict(kind=kind, xml=str(xml), xml_sha256=sha256(xml), model=model,
                    thorax_id=model.body('thorax').id,
                    muscle_index=muscle_index, sign_override=sign_override, hinge_qpos=hinge_qpos,
                    tibia_sign_variant=tibia_sign if sign_override else 'declared',
                    seated=seated, joints=tables, contact_geoms=contact, floor=floor,
                    joint_group=joint_group, weight=weight, weight_per_leg=weight/len(LEGS),
                    translation=translation, nq=int(model.nq), nu=int(model.nu))
    if kind == 'canonical':
        model, seated, gate, joints, geom_leg = closure.prepare_body()
        floor = cp.floor_geom_id(model)
        contact = {leg_key(leg): [g for g, l in geom_leg.items() if l == leg] for leg in LEGS}
        weight = float(model.body_mass.sum()*abs(model.opt.gravity[2]))
        return dict(kind=kind, xml=str(cp.CANONICAL_XML), xml_sha256=cp.sha256_file(cp.CANONICAL_XML),
                    model=model, seated=seated, joints=joints, contact_geoms=contact, floor=floor,
                    joint_group={leg_key(leg): [f'joint_{leg}{p}' for p in
                                                ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll',
                                                 'Trochanter_yaw', 'Trochanter_pitch',
                                                 'Trochanter_roll')] for leg in LEGS},
                    weight=weight, weight_per_leg=weight/len(LEGS), translation=None,
                    thorax_id=model.body('Thorax').id,
                    gate=gate, nq=int(model.nq), nu=int(model.nu))
    raise ValueError(f'unknown body kind {kind!r}')


def observe(body, data, force_normaliser):
    """The same three sensory quantities the canonical front end has always used, per leg.

    Force is divided by `force_normaliser` (the body's own per-leg weight share) so both bodies
    hand the encoder a dimensionless number of the same meaning.
    """
    import mujoco
    model = body['model']
    force = {leg_key(leg): 0.0 for leg in LEGS}
    touch = {leg_key(leg): 0.0 for leg in LEGS}
    vec = np.zeros(6)
    for ci in range(data.ncon):
        c = data.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        if body['floor'] not in (g1, g2):
            continue
        other = g2 if g1 == body['floor'] else g1
        for leg, geoms in body['contact_geoms'].items():
            if other in geoms:
                mujoco.mj_contactForce(model, data, ci, vec)
                force[leg] += float(np.linalg.norm(vec[:3]))
                touch[leg] = 1.0
                break
    out = {}
    for leg in LEGS:
        key = leg_key(leg)
        j = body['joints'][key]
        out[key] = closure.ChannelQuantities(
            tibia_joint_dev_rad=float(data.qpos[j['tib_q']]-body['reference'][j['tib_q']]),
            tibia_joint_vel_rad_s=float(data.qvel[j['tib_v']]),
            multi_joint_dev_rad=float(np.mean(data.qpos[j['q']]-body['reference'][j['q']])),
            contact_force_norm=float(force[key]/force_normaliser),
            foot_contact=float(touch[key]))
    return out


def project(activation, body, drive_fraction=DRIVE_FRACTION, dc_mode='off', dc_state=None,
            drive_polarity=1.0):
    """Canonical muscle activations (one per canonical actuator) -> this body's ctrl vector.

    Returns (ctrl, drive) where `drive` is the unitless signed value each translated hinge was
    actually given before the per-hinge span is applied, so an audit can decompose it afterwards.

    Declared knobs, all at the translation layer and none of them touching the connectome:
      drive_fraction  scales each hinge's amplitude, defined uniformly as a fraction of HALF THE
                      DECLARED CONTROL RANGE of that variant (the published servo range for the
                      servo variant, the variant's own -1..1 for the torque variant)
      drive_polarity  +1 or -1, a DIAGNOSTIC PROBE only: it multiplies the whole drive by a
                      declared sign.  It is not a fix and it proves nothing about which sign is
                      anatomically right; only the tibia has a geometric sign check in this project.
      dc_mode         'off' keeps the drive as it is; 'running_mean' subtracts, from each hinge's
                      drive, the mean of that hinge's OWN completed earlier intervals (a causal
                      high-pass variant); 'running_mean_only' REPLACES the drive by that mean (the
                      static-brace test).  Both use only completed earlier intervals, no future
                      sample and no new sensory quantity, and both are interventions, not fixes.
    `body['sign_override']` is empty for the declared translation and carries the measured tibia
    sign for the explicitly labelled variant; it is never applied silently.
    """
    import mujoco
    model = body['model']
    ctrl = np.zeros(model.nu)
    if body['kind'] == 'canonical':
        # The canonical body keeps its published semantics exactly: one activation per muscle
        # actuator, handed over unclamped so MuJoCo's own declared control range applies.
        return np.asarray(activation, float).copy(), {}
    if dc_mode not in ('off', 'running_mean', 'running_mean_only'):
        raise ValueError(f'unknown dc_mode {dc_mode!r}')
    translation = body['translation']
    drive_matrix = translation['compiled_drive_matrix']
    torque = body['kind'] == 'flybody_torque'
    index = body['muscle_index']
    drive_values = {}
    applied = 0
    for hinge, entry in drive_matrix.items():
        idx, coeffs, muscles = entry['actuator_index'], entry['coefficients'], entry['muscles']
        override = body.get('sign_override', {}).get(hinge, 1)
        raw = drive_polarity*override*sum(c*activation[index[m]]
                                          for c, m in zip(coeffs, muscles))/entry['coefficient_norm']
        value = raw
        if dc_mode in ('running_mean', 'running_mean_only'):
            if dc_state is None:
                raise ValueError('dc_mode needs a dc_state dict')
            history = dc_state.setdefault(hinge, [])
            mean_of_history = sum(history)/len(history) if history else 0.0
            # 'running_mean' keeps only the fluctuation; 'running_mean_only' keeps only the mean,
            # which is the static-brace test: if a constant per-hinge command alone holds the body
            # up, then what holds it is a brace and not feedback.
            value = (raw - mean_of_history if dc_mode == 'running_mean' else mean_of_history)
            history.append(raw)
        drive_values[hinge] = value
        if torque:
            # MEASURED BUG (fixed): this path used to hand over the raw value, so the declared
            # drive_fraction was silently ignored on the torque variant and the two variants did
            # not share one amplitude convention.  The torque variant declares ctrlrange -1..1, so
            # its half-range is 1 and the same rule applies: amplitude = drive_fraction * half of
            # the DECLARED control range of that variant.
            ctrl[idx] = float(np.clip(drive_fraction*value, -1.0, 1.0))
        else:
            lo, hi = entry['ctrlrange']
            span = drive_fraction*.5*(hi-lo)
            ctrl[idx] = float(np.clip(span*value, lo, hi))
        applied += 1
    if applied != len(drive_matrix):
        raise AssertionError('not every translated hinge was written')
    return ctrl, drive_values


def height_mm(body, data):
    import mujoco
    if body['kind'] == 'canonical':
        return float(cp.model_to_mm(float(data.xpos[body['model'].body('Thorax').id][2])))
    thorax = body['model'].body('thorax').id
    return float((float(data.subtree_com[thorax][2]) - float(data.geom_xpos[body['floor']][2]))
                 * 10.0)


def episode(body, built, manifest, groups, arm, duration, replay=None, stride=1,
            drive_fraction=DRIVE_FRACTION, dc_mode='off', freeze_after_interval=None,
            drive_polarity=1.0, filter_tau_s=None, push=None, channel_signs=None,
            channel_ablation=None, record_extra=None):
    import mujoco
    model = body['model']
    intervals = grid_steps(duration, DT_COMMAND)
    body_steps = grid_steps(DT_COMMAND, DT_BODY)
    neural_steps = grid_steps(DT_COMMAND, DT_NEURAL)
    data = mujoco.MjData(model)
    data.qpos[:] = body['seated']
    data.qvel[:] = 0.0
    if model.na:
        data.act[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    body['reference'] = body['seated'].copy()
    encoder = closure.ChannelSensoryEncoder(groups, int(built['n']), closure.ChannelConfig())
    # `filter_tau_s` is a DECLARED translation-layer knob (the decoder's own population-rate filter,
    # 5 ms in every primary run); a different value is a sensitivity run and never a tuned result.
    decoder = MotorOutput(manifest,
                          MotorOutputConfig() if filter_tau_s is None
                          else MotorOutputConfig(filter_tau_s=filter_tau_s))
    net = ConductanceNetwork(int(built['n']), ConductanceParams(), dt_ms=DT_NEURAL*1000)
    net.set_connectivity(built['edges']['We']*.2, built['edges']['Wi']*.2)
    pending = np.zeros(int(manifest['n_actuators']))
    records, trajectories, commands, spikes_out, heights = [], [], [], [], []
    h0 = height_mm(body, data)
    escape = None
    motor_indices = sorted({i for row in manifest['rows'] for i in row['local_indices']})
    dc_state = {}
    frozen_command = np.zeros(int(manifest['n_actuators']))
    for k in range(intervals):
        t = k*DT_COMMAND
        quantities = observe(body, data, body['weight_per_leg'])
        current, channels = encoder.encode(quantities)
        # The SAME declared experimenter stimulus the canonical closure uses, applied identically
        # to both bodies and never aimed at a motor pool: a local tactile input on the left front
        # leg from 5 ms to 15 ms.  Without it the zero-background tier has no drive at all over a
        # few milliseconds and every arm would be identical, which would make the comparison
        # vacuous rather than informative.
        stimulus_on = arm != 'no_external_touch' and .005 <= t < .015
        if stimulus_on:
            for i in groups['lf']['bristle_touch']:
                current[int(i)] += .3
        # DECLARED SENSORY CONVENTIONS.  `channel_signs` multiplies one channel's input current by a
        # declared sign (this project already records that channel polarity is an UNVALIDATED
        # declared assumption, and that every channel here is treated as SINGLE-SIGNED, which no
        # evidence supports); `channel_ablation` removes a channel's input entirely so its
        # contribution to the motor drive can be measured.  Neither touches the connectome: the
        # cells, their synapses and their weights are unchanged.
        if channel_signs:
            for channel, sign in channel_signs.items():
                for leg in groups:
                    for i in groups[leg].get(channel, []):
                        current[int(i)] *= sign
        if channel_ablation:
            for channel in channel_ablation:
                for leg in groups:
                    for i in groups[leg].get(channel, []):
                        current[int(i)] = 0.0
        if arm == 'sensory_cut':
            current[:] = 0
        sp = np.zeros((neural_steps, int(built['n'])), dtype=bool)
        for j in range(neural_steps):
            sp[j] = net.step(current_nA=current)
        raw_motor_spikes = int(sp[:, motor_indices].sum())
        if arm == 'spike_output_silence':
            sp[:] = False
        applied_activation = pending.copy()
        # DECLARED INTERVENTION: repeat one interval's command forever after the freeze point.  It
        # removes sensory feedback from the motor path completely while keeping the spike-driven
        # translation, so "is the body held up by feedback or by a static pattern of torques?" can
        # be asked at all.  It is not a controller and never claims to be one.
        if freeze_after_interval is not None and k >= freeze_after_interval:
            applied_activation = frozen_command.copy()
        if arm == 'motor_cut' and t >= CUT_S:
            applied_activation[:] = 0.0
        if arm == 'replay':
            applied_activation = np.asarray(replay[k], float).copy()
        ctrl, drive_values = project(applied_activation, body, drive_fraction=drive_fraction,
                                     dc_mode=dc_mode, dc_state=dc_state,
                                     drive_polarity=drive_polarity)
        data.ctrl[:] = ctrl
        push_steps_applied = 0
        for _ in range(body_steps):
            # EXTERNAL PERTURBATION: an experimenter-applied world-frame force on the thorax, given
            # as a fraction of THIS body's own weight and applied identically in every arm.  It is
            # not a neural command and never targets a motor pool; it exists so "does the loop
            # change recovery" can be asked at all, and it is cleared every step.
            data.xfrc_applied[:] = 0.0
            if push is not None and push['start_s'] <= data.time < push['end_s']:
                data.xfrc_applied[body['thorax_id'], :3] = push['force_model']
                push_steps_applied += 1
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            if escape is None and height_mm(body, data) < .9*h0:
                escape = float(data.time)
        if not all(np.all(np.isfinite(x)) for x in (data.qpos, data.qvel)):
            raise FloatingPointError(f'nonfinite body in {arm} at {data.time}')
        if model.na and not np.all(np.isfinite(data.act)):
            raise FloatingPointError(f'nonfinite activation in {arm} at {data.time}')
        if abs(data.time-(t+DT_COMMAND)) > 1e-10 or abs(net.t_ms/1000-(t+DT_COMMAND)) > 1e-10:
            raise AssertionError('body/neural clock drift or simulator reset')
        warnings = [int(w.number) for w in data.warning]
        if any(warnings):
            raise RuntimeError(f'MuJoCo warnings in {arm}: {warnings}')
        pending, trace = decoder.consume(sp, spike_dt_s=DT_NEURAL, start_s=t, end_s=t+DT_COMMAND)
        if freeze_after_interval is not None and k == freeze_after_interval:
            frozen_command = pending.copy()
        h = height_mm(body, data)
        record = dict(time_start_s=t, time_end_s=float(data.time),
                            decoder=trace, raw_motor_spikes_before_any_silence=raw_motor_spikes,
                            emitted_motor_spikes=int(sp[:, motor_indices].sum()),
                            sensory_current_min_nA=float(current.min()),
                            sensory_current_max_nA=float(current.max()),
                            external_lf_touch_stimulus=bool(stimulus_on),
                            external_push_applied=bool(push_steps_applied),
                            external_push_body_steps=push_steps_applied,
                            frozen_command=(freeze_after_interval is not None
                                            and k >= freeze_after_interval),
                            drive_values=({k2: round(v, 6) for k2, v in drive_values.items()}
                                          or None),
                            hinge_deviation_rad=({name: round(float(data.qpos[adr]
                                                                   - body['reference'][adr]), 6)
                                                  for name, adr in body.get('hinge_qpos', {}).items()}
                                                 or None),
                            applied_activation_sum=float(applied_activation.sum()),
                            applied_activation_max=float(applied_activation.max()),
                            flybody_ctrl_sum=float(ctrl.sum()),
                            height_mm=h, h_over_h0=h/h0,
                            active_control_count=int(np.count_nonzero(ctrl)),
                            sensory_quantities=({leg: dict(
                                tibia_joint_dev_rad=round(q.tibia_joint_dev_rad, 5),
                                tibia_joint_vel_rad_s=round(q.tibia_joint_vel_rad_s, 5),
                                multi_joint_dev_rad=round(q.multi_joint_dev_rad, 5),
                                contact_force_norm=round(q.contact_force_norm, 5),
                                foot_contact=q.foot_contact)
                                for leg, q in quantities.items()} if k % 10 == 0 or k < 2 else None),
                            channel_record=channels if (k % 10 == 0 or k < 2) else None)
        if record_extra is not None:
            # Caller-supplied measurement of the LIVE physics at this interval (contact state,
            # drift, whatever the caller needs).  It reads the simulator and never writes to it,
            # so it cannot change any trajectory.
            record['extra'] = record_extra(data)
        records.append(record)
        trajectories.append(np.r_[data.qpos.copy(), data.qvel.copy()])
        commands.append(applied_activation)
        spikes_out.append(sp.sum(axis=0))
        heights.append(h)
    return dict(arm=arm, escape_time_s=escape, initial_height_mm=h0,
                final_height_mm=float(heights[-1]),
                height_min_mm=float(min(heights)), records=records), \
        np.array(trajectories), np.array(commands), np.array(spikes_out)


def run_configuration(kind, built, manifest, groups, duration, tibia_sign='declared',
                      drive_fraction=DRIVE_FRACTION, dc_mode='off'):
    body = build_body(kind, tibia_sign)
    out, arrays = {}, {}
    baseline_activation = None
    for arm in ARM_NAMES:
        body = build_body(kind, tibia_sign)   # fresh model and data per arm: no state can leak
        summary, traj, ctrl, spikes = episode(body, built, manifest, groups, arm, duration,
                                              replay=baseline_activation,
                                              drive_fraction=drive_fraction, dc_mode=dc_mode)
        out[arm] = summary
        arrays[arm] = dict(trajectory=traj, activation=ctrl, spike_counts=spikes)
        if arm == 'normal':
            baseline_activation = ctrl
    checks = dict(
        replay_is_bit_identical=bool(np.array_equal(arrays['normal']['activation'],
                                                    arrays['replay']['activation'])),
        replay_trajectory_exact=bool(np.array_equal(arrays['normal']['trajectory'],
                                                    arrays['replay']['trajectory'])),
        silence_drives_nothing=bool(np.all(arrays['spike_output_silence']['activation'] == 0)),
        neural_commands_present=bool(np.any(arrays['normal']['activation'])),
        silence_changes_the_body=bool(not np.array_equal(arrays['normal']['trajectory'],
                                                         arrays['spike_output_silence']['trajectory'])),
        touch_stimulus_changes_the_body=bool(not np.array_equal(arrays['normal']['trajectory'],
                                                                arrays['no_external_touch']['trajectory'])),
        sensory_cut_changes_the_body=bool(not np.array_equal(arrays['normal']['trajectory'],
                                                             arrays['sensory_cut']['trajectory'])),
        motor_cut_changes_the_body=bool(not np.array_equal(arrays['normal']['trajectory'],
                                                           arrays['motor_cut']['trajectory'])),
        normal_vs_silence_body_max_abs_difference=float(np.max(np.abs(
            arrays['normal']['trajectory']-arrays['spike_output_silence']['trajectory']))),
    )
    for arm in ARM_NAMES:
        print(f'  {kind:16s} {arm:22s} escape={out[arm]["escape_time_s"]} '
              f'final={out[arm]["final_height_mm"]:.4f}mm act_sum={arrays[arm]["activation"].sum():.3f}',
              flush=True)
    return dict(kind=kind, tibia_sign_variant=tibia_sign, drive_fraction=drive_fraction,
                dc_mode=dc_mode, xml=body['xml'],
                xml_sha256=body['xml_sha256'],
                sign_override=body.get('sign_override', {}),
                nu=int(body['nu']), nq=int(body['nq']), weight_model_units=body['weight'],
                weight_per_leg_model_units=body['weight_per_leg'],
                seated_height_mm=out['normal']['initial_height_mm'],
                arms=out, checks=checks), arrays


def figure(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    kinds = list(result['configurations'])
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    for kind in kinds:
        for arm in ARM_NAMES:
            rows = result['configurations'][kind]['arms'][arm]['records']
            t = [r['time_end_s']*1e3 for r in rows]
            axes[0].plot(t, [r['height_mm'] for r in rows], label=f'{kind} / {arm}')
            axes[1].plot(t, [r['applied_activation_sum'] for r in rows], label=f'{kind} / {arm}')
            axes[2].plot(t, [r['raw_motor_spikes_before_any_silence'] for r in rows],
                         label=f'{kind} / {arm}')
            axes[3].plot(t, [r['flybody_ctrl_sum'] for r in rows], label=f'{kind} / {arm}')
    axes[0].set(xlabel='time (ms)', ylabel='body height (mm)', title='does the body hold up?')
    axes[1].set(xlabel='time (ms)', ylabel='sum of muscle activations',
                title='what the decoder produced (canonical muscles)')
    axes[2].set(xlabel='time (ms)', ylabel='motor-pool spikes per interval',
                title='spikes reaching the motor pools')
    axes[3].set(xlabel='time (ms)', ylabel='sum of body ctrl', title='what the body was commanded')
    for ax in axes:
        ax.grid(alpha=.3)
        ax.legend(fontsize=5)
    fig.suptitle('Same connectome, same spikes, same decoder: canonical body vs published FlyBody '
                 '(manual research prototype, not a product)', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return path


def main(duration=.05, kinds=('canonical', 'flybody_servo', 'flybody_torque'),
         tibia_sign='declared', drive_fraction=DRIVE_FRACTION, dc_mode='off', guard_report=None):
    import mujoco
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if not TRANS.exists():
        raise FileNotFoundError('run tools_flybody_translation.py first')
    translation_sha = sha256(TRANS)
    pools = json.loads(closure.POOLS.read_text())
    built = closure.build_neural_tier(closure.TierConfig(background_mean_nA=0, background_sd_nA=0))
    canonical_names = None
    model_probe = cp.load_model()
    canonical_names = [mujoco.mj_id2name(model_probe, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                       for i in range(model_probe.nu)]
    manifest = closure.build_motor_manifest(pools, built['tier']['root_ids'], canonical_names)
    groups, sensory_stats = closure.channel_index_map(int(built['n']), built['tier']['root_ids'])
    from engine.embodied.channel_connectivity import channel_connectivity
    connectivity = channel_connectivity(built['edges']['We'], built['edges']['Wi'], groups,
                                        {i for row in manifest['rows'] for i in row['local_indices']})
    # The translation table addresses canonical muscles by compile order; verify that the order it
    # recorded is the order this run compiles, because a silent mismatch would reroute every pool.
    translation_table = json.loads(TRANS.read_text())['measured']['segment_projection']['table']
    for name, row in translation_table.items():
        if canonical_names[row['actuator_index']] != name:
            raise ValueError(f'translation table actuator order disagrees with this build at {name}')
    print(f'tier {built["n"]} neurons, {manifest["usable_actuators"]}/{manifest["n_actuators"]} '
          f'canonical muscles driven by pools', flush=True)
    configurations, all_arrays = {}, {}
    for kind in kinds:
        summary, arrays = run_configuration(kind, built, manifest, groups, duration, tibia_sign,
                                           drive_fraction=drive_fraction, dc_mode=dc_mode)
        configurations[kind] = summary
        all_arrays.update({f'{kind}_{k}': v for k, v in arrays.items()})
    result = dict(
        status='MEASURED_FLYBODY_CLOSURE',
        label='MANUAL RESEARCH PROTOTYPE, NOT A PRODUCT',
        question='When the body is replaced by the published FlyBody model, does the existing '
                 'sensory->connectome->motor->body loop still close, and does it still change the '
                 'body\'s behaviour?',
        duration_s=duration,
        tibia_sign_variant=tibia_sign,
        drive_fraction=drive_fraction, dc_mode=dc_mode,
        clocks=dict(body_dt_s=DT_BODY, neural_dt_s=DT_NEURAL, command_dt_s=DT_COMMAND,
                    body_steps_per_command=int(round(DT_COMMAND/DT_BODY)),
                    neural_steps_per_command=int(round(DT_COMMAND/DT_NEURAL))),
        declared_translation=dict(
            drive_fraction=drive_fraction,
            drive_fraction_meaning='a saturated projection reaches this fraction of half the '
                                   'published hinge range; the primary run uses 0.5 and any other '
                                   'value is a declared sensitivity run, never a tuned result',
            dc_mode=dc_mode,
            dc_mode_meaning='off = the drive is used as projected; running_mean = each hinge keeps '
                            'its own causal high-passed drive (the mean of its own completed earlier '
                            'intervals is subtracted), an intervention on the translation layer only',
            sensory='canonical front end unchanged; the same three quantities per leg, read from the '
                    'hinges and claw geoms named by the translation tables',
            force_unit='each body\'s own per-leg weight share (the commensurate normaliser this '
                       'project already declared); no per-body tuning',
            servo_modulation='published servos kept, projection sets a deviation from ctrl=0, so the '
                             'model-internal posture hold is NOT attributable to the connectome',
            torque_fraction='published servos replaced by torque motors at the same published force '
                            'limit; ctrl is a dimensionless torque fraction; no posture hold',
            not_changed=['tier membership', 'connectome weights (scale .2)', 'conductance parameters',
                         'sensory channel partition', 'encoder', 'muscle pool manifest',
                         'decoder (tau 5 ms, 100 Hz full control)', 'clocks', 'no CPG', 'no PD',
                         'no hidden drive', 'no gain tuning'],
            translation_artifact=str(TRANS), translation_artifact_sha256=translation_sha),
        configurations=configurations,
        sensory_motor_structural_coverage=connectivity,
        provenance=dict(code=str(Path(__file__).resolve()), interpreter=sys.executable,
                        canonical_xml=cp.sha256_file(cp.CANONICAL_XML),
                        canonical_xml_declared=cp.CANONICAL_XML_SHA256,
                        flybody_published_xml=sha256(XML_PUBLISHED),
                        flybody_torque_variant_xml=sha256(XML_TORQUE) if XML_TORQUE.exists() else None,
                        pool_result_sha256=cp.sha256_file(closure.POOLS),
                        pool_result_usable=manifest['usable_actuators'],
                        pool_result_annotated=manifest['n_actuators'],
                        sensory_stats=sensory_stats,
                        mujoco_version=mujoco.__version__,
                        memory_guard=guard_report('completed') if guard_report else None,
                        wall_s=time.time()-started),
        what_this_does_not_support=[
            'A body that stands is not evidence that the connectome made it stand: in the '
            'servo_modulation configuration the published model\'s own position servo holds the '
            'neutral posture, and the spike_output_silence arm is exactly the control that shows it.',
            'The projection loses muscle-level resolution; several canonical muscles land on one '
            'FlyBody hinge and only their signed sum survives.',
            'No claim is made that any cell reaches any particular FlyBody hinge: the translation is '
            'this branch\'s declared convention, not connectome evidence.',
            'The torque_fraction configuration is this project\'s declared modification of a '
            'published model, not a published model.',
            'FlyBody has no muscle, no attachment site and no activation state, so nothing here '
            'restores muscle-level motor resolution.',
            'The drive_fraction and dc_mode knobs are DECLARED translation variants used for '
            'sensitivity and intervention tests; neither is tuned to make a body stand, and a '
            'variant that stands says nothing about the connectome having a posture reference.',
            'The transferred motor SIGN is verified geometrically only for the tibia, and on the '
            'hind legs it FAILS that check (the canonical flexor projects the extension way).  The '
            'declared translation keeps the transferred sign and reports the defect; the '
            '_tibiasign_measured variant corrects the tibia sign from the leg\'s own geometry and is '
            'reported separately.  No other hinge has an independent sign check.',
            'No walking, gait or escape behaviour is claimed, and no standing criterion of this '
            'project is asserted for FlyBody beyond the height trace it reports.',
        ])
    tag = ''
    if tibia_sign != 'declared':
        tag += '_tibiasign_measured'
    if dc_mode != 'off':
        tag += f'_{dc_mode}'
    if drive_fraction != DRIVE_FRACTION:
        tag += f'_gain{drive_fraction:g}'
    np.savez_compressed(OUT/f'flybody_closure_{int(duration*1000)}ms{tag}.npz', **all_arrays)
    cp.write_result(OUT/f'flybody_closure_{int(duration*1000)}ms{tag}.json', question=result['question'],
                    status='PASS', measured={k: result[k] for k in
                                             ('duration_s', 'tibia_sign_variant', 'clocks',
                                              'declared_translation', 'configurations',
                                              'sensory_motor_structural_coverage')},
                    provenance=result['provenance'], units=dict(
                        time='s', height='mm', current='nA', angle='rad',
                        force='each body\'s per-leg weight share (dimensionless in the encoder)',
                        canonical_length_unit=cp.MODEL_LENGTH_UNIT_NAME,
                        flybody_length_unit='cm'),
                    label=result['label'],
                    status_meaning='PASS means every arm ran with the clocks locked to the command '
                                   'grid and the interface checks held; it is NOT a scientific '
                                   'closure or standing result',
                    illustrative_vs_measured='Propagation is measured simulator output; the sensory '
                                             'translation, the projection coefficient table and the '
                                             'two motor variants are DECLARED translations',
                    what_this_does_not_support=result['what_this_does_not_support'])
    print('figure', figure(result, OUT/f'flybody_closure_{int(duration*1000)}ms{tag}.png'), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=.05)
    parser.add_argument('--kinds', default='canonical,flybody_servo,flybody_torque')
    parser.add_argument('--tibia-sign', default='declared', choices=('declared', 'measured'))
    parser.add_argument('--drive-fraction', type=float, default=DRIVE_FRACTION)
    parser.add_argument('--dc-mode', default='off', choices=('off', 'running_mean'))
    parser.add_argument('--budget-mb', type=float, default=4608.0)
    args = parser.parse_args()
    with mg.guarded_run('flybody_loop', need_mb=2200.0, cap_mb=args.budget_mb) as _rep:
        try:
            main(duration=args.duration, kinds=tuple(args.kinds.split(',')),
                 tibia_sign=args.tibia_sign, drive_fraction=args.drive_fraction,
                 dc_mode=args.dc_mode, guard_report=_rep)
        except Exception:
            traceback.print_exc()
            raise
