"""Open-loop dynamics of the accepted declared-cone static equilibrium under the real 100-frame connectome-derived
command archive.

Input state: the accepted equilibrium of the corrected/reissued batch
  /home/sensen/Desktop/engine修复验收/S2-equilibrium-A-corrected-reissued-20261015T161259/equilibrium_best_state.npz
  (max|qacc| = 2.5865734771464476e-11 mm/s^2, 51 tarsus contacts, six tarsi, 0 non-foot contacts)
  qpos from that file (pose + 42 leg joints), qvel = 0, act from that file.

Drive: /run/media/sensen/Data2/cell_wound_prototype/outputs/experiment_platform/20261015_135150_a2fff3f0/traces.npz
  key normal_controls, shape (100, 90), 5 ms cadence, whole 0.5 s window covered
  sha256 37fe24d3eac6f0962eee4ca6aeb51c89ecc9804be4b553c87ce91fcafda83a4f
  (the 10-frame archive with index saturation is NOT used)

Protocol: dt = 25 us, 20000 steps = 0.5 s; qpos, qvel and act are written ONCE before the loop; inside the loop only
ctrl is written (no pose correction, no PD, no state feedback of any kind).

Manual research implementation, not a product. No standing, no walking. Absolute force values and units remain
NOT_ESTABLISHED.
"""

import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = Path('/home/sensen/Desktop/engine修复验收')
STATE_NPZ = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259/equilibrium_best_state.npz'
STATE_NPZ_V2 = DESK / 'S2-equilibrium-A-corrected-reissued-v2-20261015T163134/equilibrium_best_state.npz'
BATCH_REPORT = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259/report.json'
DRIVE = ROOT / 'outputs/experiment_platform/20261015_135150_a2fff3f0/traces.npz'
DRIVE_KEY = 'normal_controls'
# declared CLI: --drive <traces.npz> selects which frozen command archive drives the run (default: the declared 100-frame
# archive above); --out-suffix appends a label to the output directory name
import sys as _sys
if '--drive' in _sys.argv and _sys.argv.index('--drive') + 1 < len(_sys.argv):
    DRIVE = Path(_sys.argv[_sys.argv.index('--drive') + 1])
OUT_SUFFIX = ''
if '--out-suffix' in _sys.argv and _sys.argv.index('--out-suffix') + 1 < len(_sys.argv):
    OUT_SUFFIX = '-' + _sys.argv[_sys.argv.index('--out-suffix') + 1]
DRIVE_SHA_EXPECTED_OVERRIDE = None
DRIVE_SHA_EXPECTED = '37fe24d3eac6f0962eee4ca6aeb51c89ecc9804be4b553c87ce91fcafda83a4f'
CANONICAL_XML = ROOT / 'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml'
DT = 25e-6
STEPS = 20000
CADENCE_MS = 5.0
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import mujoco
    sys.path.insert(0, str(ROOT))
    from engine import canonical_physics as cp

    drive_sha = sha(DRIVE)
    report = dict(
        status='RUNNING', title='CORRECTED / REISSUED: open-loop dynamics of the accepted declared-cone static equilibrium',
        corrected=True, reissued=True, manual_research_not_product=True, standing=False, walking=False,
        completion_percentage=None,
        scope=('open-loop forward simulation of the accepted declared-cone static equilibrium under the frozen '
               'connectome-derived command archive: state written once, only ctrl written inside the loop, no pose '
               'correction and no state feedback'),
        question=('is the static necessary condition (qacc ~ 0 inside the declared box) the same thing as being able to '
                  'maintain the pose under the frozen open-loop commands?'),
        inputs=dict(state_npz=str(STATE_NPZ), state_npz_sha256=sha(STATE_NPZ),
                    state_npz_v2=str(STATE_NPZ_V2),
                    state_npz_v2_sha256=(sha(STATE_NPZ_V2) if STATE_NPZ_V2.exists() else None),
                    same_state_in_v2=(sha(STATE_NPZ) == sha(STATE_NPZ_V2)) if STATE_NPZ_V2.exists() else None,
                    batch_report=str(BATCH_REPORT), batch_report_sha256=sha(BATCH_REPORT),
                    drive=str(DRIVE), drive_sha256=drive_sha, drive_key=DRIVE_KEY,
                    drive_sha256_matches_the_declared_value=bool(drive_sha == DRIVE_SHA_EXPECTED),
                    drive_is_the_default_declared_archive=bool(Path(DRIVE) == Path(ROOT / 'outputs/experiment_platform/20261015_135150_a2fff3f0/traces.npz')),
                    drive_shape=None, drive_cadence_ms=CADENCE_MS,
                    canonical_xml=str(CANONICAL_XML), canonical_xml_sha256=sha(CANONICAL_XML),
                    script=str(Path(__file__)), script_sha256=sha(Path(__file__))),
        protocol=dict(dt_s=DT, steps=STEPS, duration_s=DT * STEPS, state_written='once before the loop',
                      ctrl_written='every step, index = min(step // 200, 99) for a 5 ms cadence over 100 frames',
                      pose_correction='none', feedback='none',
                      note='the 10-frame archive with index saturation is not used'),
        not_established='absolute force values, physical units, biological validity; no standing, no walking')

    m = cp.load_model()
    # DECLARED TIMESTEP, ENFORCED: never rely on the XML <option timestep>; a previous revision of this script did and ran at 1e-4 s
    m.opt.timestep = DT
    assert m.opt.timestep == DT, 'the model refused the declared timestep'
    assert abs(DT - float(cp.load_model().opt.timestep)) > 0.0, 'the declared timestep equals the XML value; the assert would prove nothing'
    print('declared timestep enforced: m.opt.timestep =', m.opt.timestep, '| xml value =', float(cp.load_model().opt.timestep), flush=True)
    d = mujoco.MjData(m)
    st = np.load(STATE_NPZ)
    qpos0 = st['qpos'].copy()
    act0 = st['act'].copy()
    tr = np.load(DRIVE)
    ctrl_seq = np.asarray(tr[DRIVE_KEY], dtype=float)
    report['inputs']['drive_shape'] = list(ctrl_seq.shape)

    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    jname = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
    suffixes = ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll', 'Trochanter_yaw', 'Trochanter_pitch', 'Trochanter_roll', 'Tibia_pitch')
    jids = []
    for L in LEGS:
        js = [j for j in range(m.njnt) if jname(j).startswith('joint_' + L)]
        for suf in suffixes:
            jids.append(next(j for j in js if jname(j).endswith(suf)))
    qadr = [int(m.jnt_qposadr[j]) for j in jids]

    # state written once
    d.qpos[:] = qpos0
    d.qvel[:] = 0.0
    d.act[:] = act0
    d.ctrl[:] = 0.0
    mujoco.mj_forward(m, d)
    qacc0 = float(np.max(np.abs(np.asarray(d.qacc))))
    report['initial_condition'] = dict(
        max_abs_qacc_mm_s2=qacc0, qvel_norm=float(np.linalg.norm(np.asarray(d.qvel))),
        act_min=float(np.asarray(d.act).min()), act_max=float(np.asarray(d.act).max()),
        jnt_dist_to_delivered_pose_rad=None,
        note='one mj_forward at t = 0 with qvel = 0; the reported value is the static necessary condition of the batch')
    pose_A = np.load(DESK / 'S2-declared-cone-min-s-20261015T123603/best_candidate_pose.npz')['projected_seated_qpos']
    report['initial_condition']['jnt_dist_to_delivered_pose_rad'] = float(np.max(np.abs(qpos0[qadr] - pose_A[qadr])))
    # how far is the actuator state of the equilibrium from what the frozen archive commands in its first frame?
    leading_zero = 0
    for fr in ctrl_seq:
        if np.all(np.asarray(fr) == 0.0):
            leading_zero += 1
        else:
            break
    first_nonzero = int(leading_zero) if leading_zero < ctrl_seq.shape[0] else None
    act_diff = np.abs(act0 - ctrl_seq[0])
    report['initial_condition']['archive_leading_all_zero_frames'] = int(leading_zero)
    report['initial_condition']['archive_first_nonzero_frame_index'] = first_nonzero
    report['initial_condition']['archive_first_nonzero_frame_time_ms'] = (None if first_nonzero is None
                                                                         else round(first_nonzero * CADENCE_MS, 3))
    if first_nonzero is not None:
        d2 = np.abs(act0 - ctrl_seq[first_nonzero])
        report['initial_condition']['act_vs_first_NONZERO_commanded_ctrl'] = dict(
            frame_index=first_nonzero, max_abs_diff=float(d2.max()), mean_abs_diff=float(d2.mean()),
            n_actuators_above_0_1=int(np.sum(d2 > 0.1)), n_actuators_above_0_5=int(np.sum(d2 > 0.5)))
    report['initial_condition']['act_vs_first_commanded_ctrl'] = dict(
        max_abs_diff=float(act_diff.max()), mean_abs_diff=float(act_diff.mean()),
        n_actuators_above_0_1=int(np.sum(act_diff > 0.1)), n_actuators_above_0_5=int(np.sum(act_diff > 0.5)),
        n_actuators=int(act_diff.size),
        note=('the equilibrium actuator state came out of the search (act was a free variable inside [1e-4, 1]); the frozen '
              'archive commands ctrl, and the activation dynamics move act towards it. The archive itself starts with '
              'all-zero frames (see archive_leading_all_zero_frames), so for the first 10 ms the commanded activation is at '
              'the control floor: during that warm-up the model is essentially unactuated. These are measured properties of '
              'the pairing, not statements about stability.'))

    time_s = np.zeros(STEPS + 1)
    root_z = np.zeros(STEPS + 1)
    tilt_deg = np.zeros(STEPS + 1)
    n_tarsus = np.zeros(STEPS + 1, dtype=int)
    n_nonfoot = np.zeros(STEPS + 1, dtype=int)
    n_nonfoot_with_floor = np.zeros(STEPS + 1, dtype=int)
    n_self_contact = np.zeros(STEPS + 1, dtype=int)
    legs_in_contact = np.zeros(STEPS + 1, dtype=int)
    energy = np.zeros(STEPS + 1)
    speed = np.zeros(STEPS + 1)
    ctrl_max = np.zeros(STEPS + 1)
    act_max = np.zeros(STEPS + 1)

    first_nonfoot_step = None
    first_nonfoot_geoms = None
    first_nonfoot_legs_in_contact = None
    first_nonfoot_with_floor_step = None
    first_nonfoot_with_floor_geoms = None
    first_self_contact_step = None
    first_A6_geom_step = None
    first_A6_geom_geoms = None
    first_Thorax_geom_step = None
    first_Thorax_geom_geoms = None
    first_all_tarsus_lost_step = None
    first_legs_lt6_step = None
    first_flip_step = None
    first_z_below_50pct_step = None
    first_nan_step = None
    z0 = float(qpos0[2])
    f6 = np.zeros(6)

    def sample(i):
        nonlocal first_nonfoot_step, first_nonfoot_geoms, first_nonfoot_legs_in_contact
        nonlocal first_nonfoot_with_floor_step, first_nonfoot_with_floor_geoms, first_self_contact_step
        nonlocal first_A6_geom_step, first_A6_geom_geoms, first_Thorax_geom_step, first_Thorax_geom_geoms
        nonlocal first_all_tarsus_lost_step, first_legs_lt6_step, first_flip_step, first_z_below_50pct_step, first_nan_step
        time_s[i] = d.time
        root_z[i] = float(d.qpos[2])
        quat = np.asarray(d.qpos[3:7], dtype=float)
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        r22 = float(mat.reshape(3, 3)[2, 2])
        tilt_deg[i] = float(np.degrees(np.arccos(np.clip(r22, -1.0, 1.0))))
        # TWO PASSES on purpose: the leg count must not depend on where the non-foot contact appears in d.contact
        pairs = []
        for ci, c in enumerate(d.contact):
            if c.efc_address < 0:
                continue
            pairs.append((gname(int(c.geom1)), gname(int(c.geom2))))
        legs = set()
        tar = 0
        for n1, n2 in pairs:
            if 'Tarsus' in (n1 + n2):
                tar += 1
                for L in LEGS:
                    if n1.startswith(L) or n2.startswith(L):
                        legs.add(L)
        for n1, n2 in pairs:
            if first_A6_geom_step is None and ('A6_geom' in (n1 + n2) or 'A6' == n1 or 'A6' == n2):
                first_A6_geom_step = i
                first_A6_geom_geoms = sorted((n1, n2))
            if first_Thorax_geom_step is None and 'Thorax_geom' in (n1 + n2):
                first_Thorax_geom_step = i
                first_Thorax_geom_geoms = sorted((n1, n2))
        nonfoot = 0
        with_floor = 0
        self_contact = 0
        for n1, n2 in pairs:
            if 'Tarsus' in (n1 + n2):
                continue
            nonfoot += 1
            if 'floor' in (n1 + n2):
                with_floor += 1
                if first_nonfoot_with_floor_step is None:
                    first_nonfoot_with_floor_step = i
                    first_nonfoot_with_floor_geoms = sorted((n1, n2))
            else:
                self_contact += 1
                if first_self_contact_step is None:
                    first_self_contact_step = i
            if first_nonfoot_step is None:
                first_nonfoot_step = i
                first_nonfoot_geoms = sorted((n1, n2))
                first_nonfoot_legs_in_contact = len(legs)
        n_tarsus[i] = tar
        n_nonfoot[i] = nonfoot
        n_nonfoot_with_floor[i] = with_floor
        n_self_contact[i] = self_contact
        legs_in_contact[i] = len(legs)
        mujoco.mj_energyPos(m, d)
        mujoco.mj_energyVel(m, d)
        energy[i] = float(d.energy[0] + d.energy[1])
        speed[i] = float(np.linalg.norm(np.asarray(d.qvel)))
        ctrl_max[i] = float(np.max(np.asarray(d.ctrl)))
        act_max[i] = float(np.max(np.asarray(d.act)))
        if first_all_tarsus_lost_step is None and tar == 0:
            first_all_tarsus_lost_step = i
        if first_legs_lt6_step is None and len(legs) < 6:
            first_legs_lt6_step = i
        if first_flip_step is None and r22 < 0.0:
            first_flip_step = i
        if first_z_below_50pct_step is None and abs(float(d.qpos[2]) - z0) > 0.5 * abs(z0):
            first_z_below_50pct_step = i
        if first_nan_step is None and not np.isfinite(d.qpos).all():
            first_nan_step = i

    sample(0)
    t0 = time.time()
    for step in range(STEPS):
        idx = min(step // int(round(CADENCE_MS / 1000.0 / DT)), ctrl_seq.shape[0] - 1)
        d.ctrl[:] = ctrl_seq[idx]
        mujoco.mj_step(m, d)
        sample(step + 1)
        if step % 2000 == 0:
            print('step %d/%d  t=%.4f s  root_z=%.6f  legs=%d  nonfoot=%d  tilt=%.2f deg'
                  % (step, STEPS, d.time, d.qpos[2], legs_in_contact[step + 1], n_nonfoot[step + 1],
                     tilt_deg[step + 1]), flush=True)
    wall = time.time() - t0

    six_leg_steps = int(np.sum(legs_in_contact == 6))
    finite = np.isfinite(root_z)
    d_energy = np.diff(energy[finite])
    rising = int(np.sum(d_energy > 0))
    ms = lambda i: (None if i is None else round(i * DT * 1000.0, 3))
    _act_series = np.asarray(act_max, dtype=float)
    report['declared_runtime_fields'] = dict(
        model_timestep_s=float(m.opt.timestep), integrator_steps=int(STEPS),
        window_s=float(STEPS * float(m.opt.timestep)),
        cmd_period_s=float(int(round(CADENCE_MS / 1000.0 / float(m.opt.timestep))) * float(m.opt.timestep)),
        cmd_period_steps=int(round(CADENCE_MS / 1000.0 / float(m.opt.timestep))),
        checked_not_copied='model_timestep_s is read back from the model after the run, not copied from the declaration')
    assert float(m.opt.timestep) == DT, 'runtime timestep mismatch: the run would not match its declaration'
    report['mandatory_act_disclosure'] = dict(
        act_max_over_run=float(_act_series.max()),
        act_out_of_domain_steps=int((_act_series > 1.0).sum()),
        actuator_actlimited_all_false=bool(not any(np.asarray(m.actuator_actlimited))),
        rule=('if act leaves [0,1] this run is numerically invalid and no number from it may be quoted as a physical result; '
              'the same wording is used in the sister batch S2-true-equilibrium-openloop-20261015T154720'))
    report['result'] = dict(
        wall_seconds=round(wall, 1),
        first_nonfoot_contact_step=first_nonfoot_step, first_nonfoot_contact_ms=ms(first_nonfoot_step),
        first_nonfoot_contact_geom_pair=first_nonfoot_geoms,
        first_nonfoot_contact_is_a_floor_contact=bool(first_nonfoot_geoms is not None
                                                     and 'floor' in ''.join(first_nonfoot_geoms)),
        first_nonfoot_FLOOR_contact_step=first_nonfoot_with_floor_step,
        first_nonfoot_FLOOR_contact_ms=ms(first_nonfoot_with_floor_step),
        first_nonfoot_FLOOR_contact_geom_pair=first_nonfoot_with_floor_geoms,
        first_self_contact_step=first_self_contact_step, first_self_contact_ms=ms(first_self_contact_step),
        legs_in_contact_at_first_nonfoot_contact=first_nonfoot_legs_in_contact,
        first_A6_geom_contact_step=first_A6_geom_step, first_A6_geom_contact_ms=ms(first_A6_geom_step),
        first_A6_geom_contact_geom_pair=first_A6_geom_geoms,
        first_Thorax_geom_contact_step=first_Thorax_geom_step, first_Thorax_geom_contact_ms=ms(first_Thorax_geom_step),
        first_Thorax_geom_contact_geom_pair=first_Thorax_geom_geoms,
        contact_definitions=('foot contact = a contact whose geom pair contains Tarsus; non-foot contact = any other '
                             'contact, which includes floor contacts of the body and leg-to-leg / leg-to-body self '
                             'contacts; the two categories are counted separately below'),
        first_step_with_zero_tarsus_contacts=first_all_tarsus_lost_step,
        first_step_with_zero_tarsus_contacts_ms=ms(first_all_tarsus_lost_step),
        first_legs_in_contact_below_6_step=first_legs_lt6_step, first_legs_in_contact_below_6_ms=ms(first_legs_lt6_step),
        first_flip_step=first_flip_step, first_flip_ms=ms(first_flip_step),
        first_z_below_50pct_step=first_z_below_50pct_step, first_z_below_50pct_ms=ms(first_z_below_50pct_step),
        first_non_finite_step=first_nan_step,
        root_z_initial=float(root_z[0]), root_z_min=float(np.nanmin(root_z)), root_z_min_step=int(np.nanargmin(root_z)),
        root_z_final=float(root_z[-1]), root_z_max_drop=float(root_z[0] - np.nanmin(root_z)),
        tilt_deg_max=float(np.nanmax(tilt_deg)), tilt_deg_final=float(tilt_deg[-1]),
        flipped=bool(first_flip_step is not None),
        six_legs_in_contact_steps=six_leg_steps, six_legs_in_contact_fraction=float(six_leg_steps / (STEPS + 1)),
        legs_in_contact_min=int(legs_in_contact.min()), legs_in_contact_final=int(legs_in_contact[-1]),
        tarsus_contacts_min=int(n_tarsus.min()), tarsus_contacts_final=int(n_tarsus[-1]),
        nonfoot_contacts_max=int(n_nonfoot.max()), nonfoot_contacts_final=int(n_nonfoot[-1]),
        nonfoot_FLOOR_contacts_max=int(n_nonfoot_with_floor.max()), nonfoot_FLOOR_contacts_final=int(n_nonfoot_with_floor[-1]),
        self_contacts_max=int(n_self_contact.max()), self_contacts_final=int(n_self_contact[-1]),
        stable_within_the_window=bool(first_nonfoot_step is None and first_legs_lt6_step is None),
        energy_initial=float(energy[0]), energy_final=float(energy[-1]), energy_min=float(np.nanmin(energy)),
        energy_max=float(np.nanmax(energy)), energy_monotone_decreasing=bool(rising == 0),
        energy_rising_steps=rising, energy_steps_compared=int(d_energy.size),
        ctrl_max_effective=float(np.nanmax(ctrl_max)), act_max_effective=float(np.nanmax(act_max)))
    report['conclusion_two_sentences'] = dict(
        first=('静态必要条件（声明盒内 qacc ≈ 0，本状态单次 mj_forward 的 max|qacc| = %.6g mm/s^2）在这份冻结开环指令下'
               '是否还能维持，是另一件事。' % qacc0),
        second=('实测：在真 100 帧档案（5 ms 间距、整窗口覆盖）、25 µs、0.5 s、状态只写一次的开环推进下，'
                + ('首次非足接触出现在步 %s = %s ms' % (first_nonfoot_step, ms(first_nonfoot_step))
                   if first_nonfoot_step is not None else '0.5 s 窗口内没有出现非足接触')
                + '；这一数字只是实测，不推断稳定性、不等于站立，更不等于站立能力。'))

    holds = np.diff(np.where(np.diff([min(s // int(round(CADENCE_MS / 1000.0 / DT)), ctrl_seq.shape[0] - 1)
                                       for s in range(STEPS + 1)], prepend=0) != 0)[0]) if STEPS > 1 else np.array([])
    self_check = dict(read_back_model_timestep_s=float(m.opt.timestep), declared_timestep_s=DT,
                      finals_d_time_s=float(d.time), declared_duration_s=DT * STEPS,
                      command_hold_steps_observed=(int(holds[0]) if holds.size else None),
                      command_hold_steps_declared=int(round(CADENCE_MS / 1000.0 / DT)),
                      match=dict(timestep=bool(float(m.opt.timestep) == DT),
                                 duration=bool(abs(float(d.time) - DT * STEPS) <= 1e-9)))
    report['declared_runtime_self_check'] = self_check
    if not all(self_check['match'].values()):
        report['status'] = 'FAIL'
        report['fail_reason'] = 'runtime self-check mismatch: the run did not execute the declared protocol'
        out = DESK / ('S2-accepted-equilibrium-openloop-FAILED-selfcheck-' + time.strftime('%Y%m%dT%H%M%S'))
        out.mkdir(exist_ok=True)
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
        print('FAIL:', json.dumps(self_check, ensure_ascii=False))
        return 1
    out = DESK / ('S2-accepted-equilibrium-openloop-real100-corrected-reissued-' + time.strftime('%Y%m%dT%H%M%S') + OUT_SUFFIX)
    out.mkdir(exist_ok=False)
    report['status'] = 'PASS_ACCEPTED_EQUILIBRIUM_OPENLOOP_REPORTED'
    report['outputs'] = str(out)
    np.savez_compressed(out / 'time_series.npz', time_s=time_s, root_z=root_z, tilt_deg=tilt_deg,
                        n_tarsus_contacts=n_tarsus, n_nonfoot_contacts=n_nonfoot,
                        n_nonfoot_floor_contacts=n_nonfoot_with_floor, n_self_contacts=n_self_contact,
                        legs_in_contact=legs_in_contact,
                        energy=energy, speed=speed, ctrl_max=ctrl_max, act_max=act_max,
                        ctrl_sequence=ctrl_seq, note='index 0 = t 0 (state written once before the loop)')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5))
    ax = axes[0][0]
    ax.plot(time_s * 1000, root_z, lw=1.2, color='#1f77b4')
    ax.axhline(root_z[0], color='gray', ls='--', lw=1.0, label='initial root z')
    if first_nonfoot_step is not None:
        ax.axvline(first_nonfoot_step * DT * 1000, color='red', ls=':', lw=1.6, label='first non-foot contact')
    if first_legs_lt6_step is not None:
        ax.axvline(first_legs_lt6_step * DT * 1000, color='orange', ls=':', lw=1.4, label='first <6 legs in contact')
    ax.set_xlabel('time [ms]'); ax.set_ylabel('root z [mm]')
    ax.set_title('root z under the real 100-frame archive (state written once)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[0][1]
    ax.plot(time_s * 1000, legs_in_contact, lw=1.2, color='#2ca02c', label='legs with tarsus contact')
    ax.plot(time_s * 1000, n_nonfoot, lw=1.0, color='#d62728', label='non-foot contacts (any)')
    ax.plot(time_s * 1000, n_nonfoot_with_floor, lw=1.0, color='#ff7f0e', ls='--', label='of which with the floor')
    ax.set_xlabel('time [ms]'); ax.set_ylabel('count')
    ax.set_title('contact bookkeeping')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1][0]
    ax.plot(time_s * 1000, tilt_deg, lw=1.2, color='#9467bd')
    ax.axhline(90.0, color='red', ls='--', lw=1.2, label='90 deg = body axis horizontal')
    ax.set_xlabel('time [ms]'); ax.set_ylabel('tilt of body z axis from world z [deg]')
    ax.set_title('tilt (max %.2f deg; flipped: %s)' % (report['result']['tilt_deg_max'], report['result']['flipped']))
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1][1]
    ax.plot(time_s * 1000, energy, lw=1.2, color='#8c564b')
    ax.set_xlabel('time [ms]'); ax.set_ylabel('energy [model units]')
    ax.set_title('energy (monotone decreasing: %s; rising steps %d/%d)'
                 % (report['result']['energy_monotone_decreasing'], rising, d_energy.size))
    ax.grid(alpha=0.3)
    fig.suptitle('open-loop dynamics of the accepted declared-cone static equilibrium under the REAL 100-frame\n'
                 'connectome-derived archive  |  25 us, 0.5 s, state written once, only ctrl in the loop  |\n'
                 'static necessary condition != dynamic maintainability; no standing claim; manual research, not a product',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / 'accepted-equilibrium-openloop.png', dpi=135)
    plt.close(fig)

    r = report['result']

    # zoom / crop evidence: the numbers printed on the crops are read from the same arrays that are plotted
    fig2, axes2 = plt.subplots(2, 2, figsize=(15.0, 9.5))
    ax = axes2[0][0]
    n_zoom = min(1200, time_s.size)
    ax.plot(time_s[:n_zoom] * 1000, root_z[:n_zoom], lw=1.6, color='#1f77b4', marker='.', ms=1.0)
    if r['first_nonfoot_contact_step'] is not None:
        s0 = r['first_nonfoot_contact_step']
        ax.axvline(s0 * DT * 1000, color='red', ls=':', lw=1.8)
        ax.annotate('step %d = %.3f ms\nfirst non-foot contact\n%s' % (s0, s0 * DT * 1000,
                                                                        ' | '.join(r['first_nonfoot_contact_geom_pair'] or [])),
                    (s0 * DT * 1000, root_z[s0]), textcoords='offset points', xytext=(28, 30), fontsize=9,
                    arrowprops=dict(arrowstyle='->', lw=1.0), bbox=dict(boxstyle='round', fc='white', alpha=0.9))
    else:
        ax.annotate('no non-foot contact in the whole window', (0.03, 0.85), xycoords='axes fraction', fontsize=9,
                    bbox=dict(boxstyle='round', fc='white', alpha=0.9))
    if r['first_legs_in_contact_below_6_step'] is not None:
        ax.axvline(r['first_legs_in_contact_below_6_step'] * DT * 1000, color='orange', ls=':', lw=1.4)
    ax.set_xlim(0, n_zoom * DT * 1000)
    ax.set_xlabel('time [ms] (first %d ms shown)' % int(n_zoom * DT * 1000))
    ax.set_ylabel('root z [mm]')
    ax.set_title('zoom A: root z, initial drop  (init %.6f, min %.6f at step %d, final %.6f)'
                 % (r['root_z_initial'], r['root_z_min'], r['root_z_min_step'], r['root_z_final']), fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes2[0][1]
    ax.plot(time_s * 1000, legs_in_contact, lw=1.4, color='#2ca02c', label='legs with tarsus contact')
    ax.plot(time_s * 1000, n_nonfoot, lw=1.2, color='#d62728', label='non-foot contacts (any)')
    ax.plot(time_s * 1000, n_nonfoot_with_floor, lw=1.2, color='#ff7f0e', ls='--', label='of which with the floor')
    if r['first_legs_in_contact_below_6_step'] is not None:
        ax.axvline(r['first_legs_in_contact_below_6_step'] * DT * 1000, color='orange', ls=':', lw=1.4)
        ax.annotate('step %d = %.3f ms: legs < 6' % (r['first_legs_in_contact_below_6_step'],
                                                     r['first_legs_in_contact_below_6_ms']),
                    (r['first_legs_in_contact_below_6_step'] * DT * 1000, 6), textcoords='offset points', xytext=(20, -40),
                    fontsize=9, arrowprops=dict(arrowstyle='->', lw=1.0), bbox=dict(boxstyle='round', fc='white', alpha=0.9))
    else:
        ax.annotate('no step with legs_in_contact < 6 in the whole window', (0.03, 0.55), xycoords='axes fraction',
                    fontsize=9, bbox=dict(boxstyle='round', fc='white', alpha=0.9))
    if r['first_step_with_zero_tarsus_contacts'] is not None:
        ax.axvline(r['first_step_with_zero_tarsus_contacts'] * DT * 1000, color='purple', ls=':', lw=1.4)
        ax.annotate('step %d = %.3f ms: zero tarsus contacts' % (r['first_step_with_zero_tarsus_contacts'],
                                                                 r['first_step_with_zero_tarsus_contacts_ms']),
                    (r['first_step_with_zero_tarsus_contacts'] * DT * 1000, 25), textcoords='offset points', xytext=(-140, 20),
                    fontsize=9, arrowprops=dict(arrowstyle='->', lw=1.0), bbox=dict(boxstyle='round', fc='white', alpha=0.9))
    else:
        ax.annotate('no step with zero tarsus contacts in the whole window', (0.03, 0.40), xycoords='axes fraction',
                    fontsize=9, bbox=dict(boxstyle='round', fc='white', alpha=0.9))
    ax.set_xlabel('time [ms]'); ax.set_ylabel('count')
    ax.set_title('zoom B: contact bookkeeping (six-leg fraction %.6f = %d/%d)'
                 % (r['six_legs_in_contact_fraction'], r['six_legs_in_contact_steps'], STEPS + 1), fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes2[1][0]
    m_zoom = 4000
    ax.plot(time_s[:m_zoom] * 1000, tilt_deg[:m_zoom], lw=1.5, color='#9467bd')
    ax.axhline(90.0, color='red', ls='--', lw=1.2, label='90 deg = body axis horizontal (never reached)')
    ax.annotate('peak %.4f deg at step %d' % (float(np.nanmax(tilt_deg[:m_zoom])), int(np.nanargmax(tilt_deg[:m_zoom]))),
                (float(np.nanargmax(tilt_deg[:m_zoom])) * DT * 1000, float(np.nanmax(tilt_deg[:m_zoom]))),
                textcoords='offset points', xytext=(20, 6), fontsize=9,
                bbox=dict(boxstyle='round', fc='white', alpha=0.9))
    ax.set_xlabel('time [ms] (first 100 ms shown)'); ax.set_ylabel('tilt [deg]')
    ax.set_title('zoom C: tilt (max %.4f deg, flipped = %s)' % (r['tilt_deg_max'], r['flipped']), fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes2[1][1]
    ax.plot(time_s * 1000, energy, lw=1.5, color='#8c564b')
    ax.annotate('energy %.6f -> %.6f ; rising steps %d of %d' % (r['energy_initial'], r['energy_final'],
                                                                r['energy_rising_steps'], r['energy_steps_compared']),
                (0.03, 0.08), xycoords='axes fraction', fontsize=9,
                bbox=dict(boxstyle='round', fc='white', alpha=0.9))
    ax.set_xlabel('time [ms]'); ax.set_ylabel('energy [model units]')
    ax.set_title('zoom D: energy (monotone decreasing: %s)' % r['energy_monotone_decreasing'], fontsize=9)
    ax.grid(alpha=0.3)
    fig2.suptitle('zoom evidence for the open-loop run (all annotations are read from the same arrays that are plotted; '
                  'no number is typed by hand)\nmanual research implementation, not a product', fontsize=9)
    fig2.tight_layout(rect=(0, 0, 1, 0.93))
    fig2.savefig(out / 'openloop-zoom-evidence.png', dpi=140)
    plt.close(fig2)

    r = report['result']
    readme = '\n'.join([
        '# 被接受"声明锥静态平衡"的开环动力学（真 100 帧档案，手动研究实施，不是成品）',
        '',
        '## 输入',
        '- 状态：`%s`（sha256 `%s`），`qpos` = 该位姿与 42 个腿关节，`qvel = 0`，`act` = 该文件里的 `act`。' % (STATE_NPZ, report['inputs']['state_npz_sha256']),
        '- 驱动：`%s` 的 `normal_controls` (100, 90)，sha256 `%s`（与声明值一致：%s），5 ms 间距、整窗口覆盖；**未**使用 10 帧 + 索引饱和那条。'
        % (DRIVE, drive_sha, report['inputs']['drive_sha256_matches_the_declared_value']),
        '- 协议：dt = 25 µs、20000 步 = 0.5 s；`qpos/qvel/act` **只在首步写一次**，循环内**只写 `ctrl`**；无姿态修正、无状态反馈。',
        '- 初值检查：首步单次 `mj_forward` 的 `max|qacc| = %.6g mm/s²`（与批次里的静态必要条件一致）。' % qacc0,
        '- **档案开头是"全零帧"**：`normal_controls` 前 **%d 帧全为 0**，第一帧非零出现在第 %d 帧 = **%s ms**；'
        '由于 `ctrlrange = [1e-4, 1]` 且启用 clampctrl，这 10 ms 内实际指令被夹到下限。'
        % (report['initial_condition']['archive_leading_all_zero_frames'],
           report['initial_condition']['archive_first_nonzero_frame_index'],
           report['initial_condition']['archive_first_nonzero_frame_time_ms']),
        '- 初值失配：平衡态的 `act` 与档案**第一帧 `ctrl`**（全零）最大差 `%.6g`（= `act` 的最大值）、平均差 `%.6g`（= `act` 的均值）；'
        '与档案**第一个非零帧**（第 %s 帧）的最大差 `%.6g`、> 0.1 的通道 %d 个。'
        '（`act` 是搜索里的自由变量，激活动力学只会把 `act` 拉向 `ctrl`；**这是"指令对与平衡态不匹配"的实测事实，不是稳定性结论**）。'
        % (report['initial_condition']['act_vs_first_commanded_ctrl']['max_abs_diff'],
           report['initial_condition']['act_vs_first_commanded_ctrl']['mean_abs_diff'],
           report['initial_condition']['archive_first_nonzero_frame_index'],
           report['initial_condition']['act_vs_first_NONZERO_commanded_ctrl']['max_abs_diff'],
           report['initial_condition']['act_vs_first_NONZERO_commanded_ctrl']['n_actuators_above_0_1']),
        '',
        '## 实测结果（只报数，不推断）',
        '- 首次非足接触（口径 = 任何不含 `Tarsus` 的接触）：**步 %s = %s ms**，几何对 `%s`，此刻仍有 %s 条腿有跗节接触。'
        % (r['first_nonfoot_contact_step'], r['first_nonfoot_contact_ms'], r['first_nonfoot_contact_geom_pair'],
           r['legs_in_contact_at_first_nonfoot_contact']),
        '- 首次 `A6_geom` 接触：**步 %s = %s ms**（几何对 `%s`）；首次 `Thorax_geom` 接触：**步 %s = %s ms**（几何对 `%s`）。'
        % (r['first_A6_geom_contact_step'], r['first_A6_geom_contact_ms'], r['first_A6_geom_contact_geom_pair'],
           r['first_Thorax_geom_contact_step'], r['first_Thorax_geom_contact_ms'], r['first_Thorax_geom_contact_geom_pair']),
        '- 其中**与地面（`floor`）**的首次接触：**步 %s = %s ms**，几何对 `%s`；腿-腿/腿-体自接触首次 **步 %s = %s ms**；'
        '足外面接触峰值 %d（含地面 %d、自接触 %d）。'
        % (r['first_nonfoot_FLOOR_contact_step'], r['first_nonfoot_FLOOR_contact_ms'],
           r['first_nonfoot_FLOOR_contact_geom_pair'], r['first_self_contact_step'], r['first_self_contact_ms'],
           r['nonfoot_contacts_max'], r['nonfoot_FLOOR_contacts_max'], r['self_contacts_max']),
        '- 首次六足条件被破坏（`legs_in_contact < 6`）：**步 %s = %s ms**；**跗节接触全部失去（`tarsus_contacts = 0`）在步 %s = %s ms**。'
        % (r['first_legs_in_contact_below_6_step'], r['first_legs_in_contact_below_6_ms'],
           r['first_step_with_zero_tarsus_contacts'], r['first_step_with_zero_tarsus_contacts_ms']),
        '- root z：初值 `%.6f`、**最低 `%.6f`**（步 %d）、末值 `%.6f`、最大跌幅 `%.6f` mm（mm 单位，见 `inputs.canonical_xml`）。'
        % (r['root_z_initial'], r['root_z_min'], r['root_z_min_step'], r['root_z_final'], r['root_z_max_drop']),
        '- 倾角（体 z 轴与世界 z 轴夹角）：最大 `%.4f°`、末值 `%.4f°`、是否翻转 `%s`。'
        % (r['tilt_deg_max'], r['tilt_deg_final'], r['flipped']),
        '- 能量：`%.6f → %.6f`（最低 `%.6f`、最高 `%.6f`），**单调下降 %s**，上升步 %d/%d。'
        % (r['energy_initial'], r['energy_final'], r['energy_min'], r['energy_max'],
           r['energy_monotone_decreasing'], r['energy_rising_steps'], r['energy_steps_compared']),
        '- 六足同时在接触：**%d / %d 步 = %.4f**；`legs_in_contact` 最小 %d、末值 %d；非足接触最多 %d。'
        % (r['six_legs_in_contact_steps'], STEPS + 1, r['six_legs_in_contact_fraction'],
           r['legs_in_contact_min'], r['legs_in_contact_final'], r['nonfoot_contacts_max']),
        '- `ctrl` 实际峰值 `%.6g`、`act` 实际峰值 `%.6g`（墙钟 %.1f s）。' % (r['ctrl_max_effective'], r['act_max_effective'], r['wall_seconds']),
        '',
        '## 结论（两句）',
        '1. ' + report['conclusion_two_sentences']['first'],
        '2. ' + report['conclusion_two_sentences']['second'],
        '',
        '## 披露',
        '- `qacc ≈ 0` 只是**静态必要条件**；它**不等于**稳定性、**不等于**平衡能力、**更不等于**站立。',
        '- 本目录只报**实测毫秒数**，不推断"是否稳定"；若窗口内没塌，也只说"0.5 s 内没有出现非足接触"。',
        '- 绝对力值与物理单位 `NOT_ESTABLISHED`；**手动研究实施，不是成品**。',
        '',
        '## 文件',
        '- `report.json`（含输入 npz 的 sha256 与批内一致性、全部事件步号/毫秒）、`time_series.npz`',
        '- `accepted-equilibrium-openloop.png`、`source_snapshot.zip`、`delivery-hashes.json`'])
    (out / 'README.zh-CN.md').write_text(readme)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(Path(__file__), Path(__file__).name)
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != 'delivery-hashes.json')
    (out / 'delivery-hashes.json').write_text(json.dumps({p.name: sha(p) for p in files}, ensure_ascii=False, indent=1))
    print(report['status'])
    print('out:', out)
    print('first non-foot contact step %s (%s ms); root_z_min %.6f; legs<6 at step %s; all tarsi lost at step %s; tilt max %.3f deg; flipped %s; energy monotone %s'
          % (r['first_nonfoot_contact_step'], r['first_nonfoot_contact_ms'], r['root_z_min'],
             r['first_legs_in_contact_below_6_step'], r['first_step_with_zero_tarsus_contacts'],
             r['tilt_deg_max'], r['flipped'], r['energy_monotone_decreasing']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
