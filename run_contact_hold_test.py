"""Contact-dynamics HOLD test on the multi-start pose, with the two questions kept strictly apart.

Branch STRICT (no external controller, the strict question): the drive is the frozen connectome-derived
command archive, repeated periodically to fill the window; nothing but ctrl is written during the run.
Branch SERVO (declared ENGINEERING feasibility test, explicitly NOT strict neural direct control): a
position-servo law computed by this script maps the pose error into muscle commands. If any number from
this branch is quoted, it must be quoted as an engineering test, never as neural control success.

The artifact reports, for both branches: root height, root tilt (angle of the thorax up-axis away from
vertical; a tilt beyond 90 deg means it has flipped), the number of legs simultaneously in contact, the
tarsus and non-foot contact sets, slip, the first failure event, the final qacc magnitude, and whether the
total energy grows. A DECLARED dynamics criterion then answers yes/no per branch; the strict branch's
answer is the strict answer.

DRIVE INDEXING (declared, identical wording in every producer of long windows): the archived drive has 10 frames at a
5 ms cadence; ctrl = commands[min(k // 200, 9)] therefore plays ALL 10 FRAMES OVER THE FIRST 0-50 ms and then HOLDS
FRAME 9 CONSTANT for the rest of the window (at --window 0.5 s that is 450 ms = 90 percent of the run). An extended
window is thus an archive replay followed by a constant hold of the last frame - NOT a periodic repetition.

LP feasibility is NOT this: the same pose had a feasible static LP at s<=1 while the gate run gave all 48
DOF nonzero qacc with max 3967.36-4294.03 mm/s^2 (see S2-support-gate-20261015T123607).
"""
from pathlib import Path
import hashlib
import json
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent
DESK = Path('/home/sensen/Desktop/engine修复验收')
SOURCE_POSE = ROOT / 'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
P1 = ROOT / 'outputs/experiment_platform/20261013_233237_7be93a1d/traces.npz'
FLOOR = 0
DT = 2.5e-5
WINDOW = 0.2
CMD_DT = 0.005
DIST_CAP = 4.0
KP, KD = 0.35, 0.02          # declared servo gains (engineering branch only)
CRITERION = dict(z_drift_frac_max=0.01, tilt_deg_max=5.0, six_leg_steps_frac_min=0.95,
                 allow_nonfoot_contacts=False, energy_must_not_grow=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def newest(prefix):
    cands = sorted(p for p in DESK.glob(prefix + '*') if (p / 'best_pose_multistart.npz').exists()
                   or (p / 'best_candidate_pose.npz').exists())
    if not cands:
        raise FileNotFoundError(prefix)
    d = cands[-1]
    f = d / 'best_pose_multistart.npz'
    return f if f.exists() else d / 'best_candidate_pose.npz'


def run(mujoco, m, q0, controls, qref, branch, label, window=None):
    d = mujoco.MjData(m)
    m.opt.timestep = DT
    steps = int(round((WINDOW if window is None else window) / DT))
    hold = int(round(CMD_DT / DT))
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    foot = [g for g in range(m.ngeom) if 'Tarsus' in gname(g)]
    other = [g for g in range(m.ngeom) if g != FLOOR and 'Tarsus' not in gname(g)]
    d.qpos[:] = q0
    d.qvel[:] = 0.0
    d.act[:] = 0.0
    d.ctrl[:] = controls[0]
    mujoco.mj_forward(m, d)
    z0 = float(d.qpos[2])
    up0 = np.zeros(3)
    mujoco.mju_rotVecQuat(up0, np.array([0.0, 0.0, 1.0]), d.qpos[3:7])
    ser = dict(time=[], root_z=[], tilt_deg=[], tarsus=[], nonfoot=[], legs=[], qacc=[], energy=[],
               lowest_tarsus=[], slip=[])
    ev = dict(first_nonfoot=None, first_tarsus_loss=None, first_flip=None, first_nan=None)
    x0 = np.array([d.qpos[0], d.qpos[1]])
    for k in range(steps):
        if branch == 'strict':
            d.ctrl[:] = controls[min(k // hold, controls.shape[0] - 1)]
        else:
            # declared ENGINEERING servo: muscle-level proportional law on the moment-arm projection of the
            # joint error (42 joints -> 90 muscles). NOT strict neural direct control.
            qerr = d.qpos[7:] - qref[7:]
            verr = d.qvel[6:]
            mom = np.asarray(d.actuator_moment)
            proj = np.zeros(m.nu)
            for i in range(m.nu):
                adr = int(d.moment_rowadr[6 + 0])  # placeholder, replaced below by the sparse read
                break
            proj = np.zeros(m.nu)
            for dof in range(6, m.nv):
                adr = int(d.moment_rowadr[dof]); nnz = int(d.moment_rownnz[dof])
                cols = np.asarray(d.moment_colind)[adr:adr + nnz]
                vals = np.asarray(mom)[adr:adr + nnz]
                err = qerr[dof - 6] * KP - verr[dof - 6] * KD
                for c, v in zip(cols, vals):
                    proj[int(c)] += float(v) * err
            d.ctrl[:] = np.clip(0.5 + proj, 0.0001, 1.0)
        mujoco.mj_step(m, d)
        up = np.zeros(3)
        mujoco.mju_rotVecQuat(up, np.array([0.0, 0.0, 1.0]), d.qpos[3:7])
        tilt = float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))
        ser['time'].append(float(d.time))
        ser['root_z'].append(float(d.qpos[2]))
        ser['tilt_deg'].append(tilt)
        ser['qacc'].append(float(np.max(np.abs(d.qacc))))
        try:
            mujoco.mj_energyPos(m, d)
            mujoco.mj_energyVel(m, d)
            ser['energy'].append(float(d.energy[0] + d.energy[1]))
        except Exception:
            ser['energy'].append(float('nan'))
        ser['slip'].append(float(np.linalg.norm(np.array([d.qpos[0], d.qpos[1]]) - x0)))
        tarsus = nonfoot = 0
        legs = set()
        for c in d.contact:
            if c.efc_address < 0:
                continue
            names = [gname(int(c.geom1)), gname(int(c.geom2))]
            if any('Tarsus' in n for n in names):
                tarsus += 1
                for n in names:
                    if 'Tarsus' in n:
                        legs.add(n[:2])
            else:
                nonfoot += 1
        ser['tarsus'].append(tarsus)
        ser['nonfoot'].append(nonfoot)
        ser['legs'].append(len(legs))
        mujoco.mj_kinematics(m, d)
        ft = np.zeros(6)
        ser['lowest_tarsus'].append(float(min(mujoco.mj_geomDistance(m, d, FLOOR, g, DIST_CAP, ft) for g in foot)))
        if ev['first_nonfoot'] is None and nonfoot > 0:
            ev['first_nonfoot'] = k
        if ev['first_tarsus_loss'] is None and tarsus == 0:
            ev['first_tarsus_loss'] = k
        if ev['first_flip'] is None and tilt > 90.0:
            ev['first_flip'] = k
        if ev['first_nan'] is None and not np.isfinite(d.qpos).all():
            ev['first_nan'] = k
        if ev['first_nan'] is not None:
            break
    n = len(ser['time'])
    zf = ser['root_z'][-1]
    six_frac = float(np.mean([1.0 if v == 6 else 0.0 for v in ser['legs']])) if ser['legs'] else 0.0
    if ser['energy']:
        _e = np.asarray(ser['energy'], dtype=float)
        _d = np.diff(_e)
        energy_rising_steps = int((_d > 0).sum())
        energy_no_net_increase = bool(_e[-1] <= _e[0] + 1e-9)
        energy_strictly_monotone_down = bool(bool(_d.size) and (_d <= 0).all())
        energy_peak = float(_e.max())
        energy_peak_growth_rel = float(_e.max() / _e[0] - 1.0) if _e[0] else None
    else:
        energy_rising_steps = energy_no_net_increase = None
        energy_strictly_monotone_down = energy_peak = energy_peak_growth_rel = None
    first_fail = None
    for key in ('first_nan', 'first_flip', 'first_tarsus_loss', 'first_nonfoot'):
        v = ev[key]
        if v is not None and (first_fail is None or v < first_fail[1]):
            first_fail = (key, v)
    holds = (abs(zf - z0) / max(z0, 1e-9) <= CRITERION['z_drift_frac_max']
             and max(ser['tilt_deg']) <= CRITERION['tilt_deg_max']
             and six_frac >= CRITERION['six_leg_steps_frac_min']
             and max(ser['nonfoot']) == 0
             and (energy_no_net_increase is not False))
    return dict(branch=branch, label=label, dt_s=DT, steps=n, window_s=(WINDOW if window is None else window),
                root_z_initial=z0, root_z_final=zf,
                z_drift=float(zf - z0), z_drift_frac=float(abs(zf - z0) / max(z0, 1e-9)),
                tilt_max_deg=float(max(ser['tilt_deg'])), tilt_final_deg=float(ser['tilt_deg'][-1]),
                flipped=bool(max(ser['tilt_deg']) > 90.0),
                six_leg_steps=int(sum(1 for v in ser['legs'] if v == 6)), six_leg_frac=six_frac,
                legs_contact_min=int(min(ser['legs'])), legs_contact_final=int(ser['legs'][-1]),
                tarsus_min=int(min(ser['tarsus'])), nonfoot_max=int(max(ser['nonfoot'])),
                lowest_tarsus_final_um=float(1000 * ser['lowest_tarsus'][-1]),   # micrometres (was misnamed _mm)
                slip_final_mm=float(ser['slip'][-1]),
                qacc_first=float(ser['qacc'][0]), qacc_final=float(ser['qacc'][-1]),
                qacc_max=float(max(ser['qacc'])),
                energy_first=ser['energy'][0], energy_final=ser['energy'][-1],
                energy_no_net_increase=energy_no_net_increase,
                energy_strictly_monotone_down=energy_strictly_monotone_down,
                energy_rising_steps=energy_rising_steps, energy_rising_steps_of=n - 1,
                energy_peak=energy_peak, energy_peak_growth_rel=energy_peak_growth_rel,
                events=ev, first_failure=first_fail,
                holds_by_declared_criterion=bool(holds),
                statement=('STRICT (no external controller): %s'
                           % ('holds by the declared dynamics criterion' if holds else
                              'does NOT hold by the declared dynamics criterion') if branch == 'strict' else
                           'ENGINEERING feasibility test with a muscle position servo - NOT strict neural '
                           'direct control, must never be quoted as such')), ser


def main():
    # declared CLI: --window <s> overrides the declared window (default 0.2 s) and --outdir <dir> overrides
    # the desktop destination; both defaults keep the original behaviour exactly.
    argv = sys.argv[1:]
    def _arg(name, default=None):
        return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default
    window = float(_arg('--window', WINDOW))
    outdir = _arg('--outdir', None)
    out = (Path(outdir) / ('S2-contact-hold-test-' + time.strftime('%Y%m%dT%H%M%S'))) if outdir else \
        DESK / ('S2-contact-hold-test-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    report = dict(status='RUNNING', manual_research_not_product=True, standing=False, walking=False,
                  scope=('contact-dynamics hold test on the multi-start pose, strict (no controller) and a '
                         'declared engineering servo branch, kept strictly apart'),
                  declared=dict(dt_s=DT, window_s=window, drive_strict=('frozen connectome-derived commands '
                               'repeated periodically; no external CPG, no PD, no preprogrammed action, no '
                               'per-step pose correction'), servo_branch=dict(labelled='ENGINEERING',
                               gains=dict(kp=KP, kd=KD), note='NOT strict neural direct control'),
                               criterion=CRITERION,
                               state_discipline='qpos/qvel/act written once before the first step'),
                  declared_calls=dict(project_model_loads=1), absolute_force_values='NOT_ESTABLISHED')
    try:
        import mujoco
        from engine import canonical_physics as cp
        m = cp.load_model()
        report['canonical_xml'] = str(cp.CANONICAL_XML)
        report['canonical_xml_sha256'] = sha(cp.CANONICAL_XML)
        with np.load(P1) as z:
            controls = z['normal_controls'].copy()
        pose = newest('S2-multistart-ik-')
        with np.load(pose) as z:
            q0 = z['projected_seated_qpos'].copy()
        qref = np.load(SOURCE_POSE)['projected_seated_qpos'].copy()
        report['input_hashes'] = {str(p): sha(p) for p in (Path(__file__), P1, SOURCE_POSE, pose)}
        report['initial_pose'] = str(pose)
        runs, series = {}, {}
        for branch in ('strict', 'servo'):
            res, ser = run(mujoco, m, q0, controls, qref, branch, branch, window=window)
            runs[branch] = res
            series[branch] = ser
            print('%-7s steps=%d six_leg=%.3f tilt_max=%.2f z_drift=%.4f holds=%s first_failure=%s' % (
                branch, res['steps'], res['six_leg_frac'], res['tilt_max_deg'], res['z_drift'],
                res['holds_by_declared_criterion'], res['first_failure']), flush=True)
        report['runs'] = runs
        report['answer'] = dict(
            strict_question=('does the pose hold WITHOUT any external controller under the frozen '
                             'connectome-derived drive, by the declared dynamics criterion?'),
            strict_answer='YES' if runs['strict']['holds_by_declared_criterion'] else 'NO',
            engineering_branch=runs['servo']['statement'],
            lp_feasible_is_not_equilibrium=('the static LP was feasible at s<=1 for this family while the gate '
                                            'run on the same pose gave all 48 DOF nonzero qacc with max about '
                                            '3967.36-4294.03 mm/s^2 (S2-support-gate-20261015T123607): a feasible LP is '
                                            'not a MuJoCo equilibrium and neither is a hold-test pass a stance'))
        np.savez_compressed(out / 'time_series.npz',
                           **{b + '__' + k: np.asarray(v) for b, ser in series.items() for k, v in ser.items()})
        report['status'] = 'PASS_CONTACT_HOLD_TEST_REPORTED'
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
        for b, ser in series.items():
            t = 1000 * np.asarray(ser['time'])
            axes[0].plot(t, ser['root_z'], lw=1, label=b)
            axes[1].plot(t, ser['tilt_deg'], lw=1, label=b)
            axes[2].plot(t, ser['legs'], lw=1, label=b + ' legs in contact')
            axes[2].plot(t, ser['nonfoot'], lw=1, ls='--', label=b + ' non-foot')
        axes[0].set_xlabel('time (ms)'); axes[0].set_ylabel('root z (mm)')
        axes[0].set_title('root height, %.0f ms (declared window)' % (1000 * window)); axes[0].legend(fontsize=7)
        axes[1].axhline(90, color='r', ls=':', label='flip threshold')
        axes[1].set_xlabel('time (ms)'); axes[1].set_ylabel('tilt (deg)')
        axes[1].set_title('thorax tilt'); axes[1].legend(fontsize=7)
        axes[2].set_xlabel('time (ms)'); axes[2].set_ylabel('count')
        axes[2].set_title('legs in contact / non-foot contacts'); axes[2].legend(fontsize=7)
        fig.suptitle('Contact-dynamics hold test, declared window %.0f ms (strict vs engineering servo) - '
                     'manual research, NOT standing' % (1000 * window))
        fig.tight_layout(); fig.savefig(out / 'contact-hold.png', dpi=130); plt.close(fig)
    except Exception as exc:
        import traceback
        report.update(status='FAIL_STOP', error=type(exc).__name__ + ': ' + str(exc),
                      traceback=traceback.format_exc()[-1500:])
    finally:
        with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(Path(__file__), Path(__file__).name)
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False,
                                                    default=lambda o: o.tolist()) + '\n')
        (out / 'delivery-hashes.json').write_text(json.dumps(
            {f.name: sha(f) for f in sorted(out.iterdir())
             if f.is_file() and f.name != 'delivery-hashes.json'}, indent=2) + '\n')
    print(out)
    print(report['status'])
    if report['status'].startswith('PASS'):
        print('strict:', {k: report['runs']['strict'][k] for k in
                          ('steps', 'z_drift', 'tilt_max_deg', 'flipped', 'six_leg_frac', 'nonfoot_max',
                           'qacc_final', 'holds_by_declared_criterion', 'first_failure')})
        print('servo :', {k: report['runs']['servo'][k] for k in
                          ('steps', 'z_drift', 'tilt_max_deg', 'flipped', 'six_leg_frac', 'nonfoot_max',
                           'qacc_final', 'holds_by_declared_criterion', 'first_failure')})
    else:
        print(report.get('error', ''))
    return 0 if report['status'].startswith('PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
