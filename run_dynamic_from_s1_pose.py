"""First DYNAMIC run from the seated pose that is statically feasible at s=1 under the declared cone.

DRIVE (declared): the frozen connectome-derived motor command archive
  outputs/experiment_platform/20261013_233237_7be93a1d/traces.npz -> normal_controls (10 x 90, held 5 ms)
i.e. the commands the connectome model itself produced. This script adds NO external CPG, NO PD control,
NO preprogrammed action and NO per-step pose correction of any kind. Nothing but the command input is
written during the run: the state (qpos, qvel, act) is set ONCE before the first step and never touched
again; every step is a plain mj_step of the canonical model with its 90 native muscle actuators.

STARTS (declared, two by default; --start <label>=<path> adds more):
  A) S2-declared-cone-min-s-20261015T123603/best_candidate_pose.npz   (s = 0.9097466208040714 on the declared
     cone, i.e. feasible at s = 1 - this is the ONLY s value quoted for start A anywhere in this script)
  B) S2-seated-tarsus-only-20261015T114113/candidate_seated_pose.npz  (earlier pose, statically infeasible at
     s=1 on the declared cone: exact gaps 0.42162561898294587 native / 0.41982541733962897 union)

Timesteps (declared): the canonical XML timestep 1e-4 s as primary, and the previously qualified 25 us as a
secondary branch. The 1e-4 s branch is NUMERICALLY INVALID in the actuation channel (the native muscle
activation leaves its domain: act max 1.9993002399280217 > 1, and MuJoCo uses act UNCLAMPED - this model has
m.actuator_actlimited False for every actuator), so no number of that branch may be quoted as a physical
result; only the 25 us branch is plotted. Window: DEFAULT 50 ms = the length of the archived drive, i.e. the
10 archived frames at the 5 ms cadence; --window <s> extends it.

DRIVE INDEXING (declared, identical wording in every producer of long windows): the archived drive has 10 frames at a
5 ms cadence; ctrl = commands[min(k // 200, 9)] therefore plays ALL 10 FRAMES OVER THE FIRST 0-50 ms and then HOLDS
FRAME 9 CONSTANT for the rest of the window (at --window 0.5 s that is 450 ms = 90 percent of the run). An extended
window is thus an archive replay followed by a constant hold of the last frame - NOT a periodic repetition.

Reported: thorax (root) z time series, tarsus and non-foot contact sets, slip (root horizontal
displacement), loss of ground contact, survival in steps and seconds, first failure step and mode.
No completion percentage and no "standing" claim: only measured survival or failure.
"""
from pathlib import Path
import hashlib
import json
import sys
import time
import zipfile

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parent
DESK = Path('/home/sensen/Desktop/engine修复验收')
P1 = ROOT / 'outputs/experiment_platform/20261013_233237_7be93a1d/traces.npz'
STARTS = {
    'A_min_s_pose': DESK / 'S2-declared-cone-min-s-20261015T123603/best_candidate_pose.npz',
    'B_tarsus_only_pose': DESK / 'S2-seated-tarsus-only-20261015T114113/candidate_seated_pose.npz',
}
FLOOR_GEOM = 0
DURATION = 0.05
if '--window' in sys.argv:
    DURATION = float(sys.argv[sys.argv.index('--window') + 1])
# declared CLI: --start <label>=<path> adds a start pose in addition to the two declared starts (default: none,
# behaviour unchanged). The pose file must contain projected_seated_qpos, like the two declared starts.
_extra_starts = [a[len('--start='):].split('=', 1) for a in sys.argv[1:] if a.startswith('--start=')]
if not _extra_starts and '--start' in sys.argv:
    _i = sys.argv.index('--start')
    _extra_starts = [sys.argv[_i + 1].split('=', 1)]
for _label, _path in _extra_starts:
    STARTS[_label] = Path(_path)
# declared CLI: --only <label[,label]> restricts the run to those starts; --outdir <dir> redirects the output
# directory (used to stage a longer-window run before merging it into the 0.5 s artifact directory).
_only = None
if '--only' in sys.argv and sys.argv.index('--only') + 1 < len(sys.argv):
    _only = sys.argv[sys.argv.index('--only') + 1].split(',')
    STARTS = {k: v for k, v in STARTS.items() if k in _only}
_OUTDIR = None
if '--outdir' in sys.argv and sys.argv.index('--outdir') + 1 < len(sys.argv):
    _OUTDIR = Path(sys.argv[sys.argv.index('--outdir') + 1])
# declared CLI: --drive-mode hold|cycle. hold (default, unchanged behaviour): the archive is played over the first
# 50 ms and its LAST FRAME is then held constant (index saturates). cycle (declared control): the 10 archived frames
# are replayed cyclically for the whole window. Nothing else differs - same discipline, no model/gain/friction change.
# declared CLI: --drive-archive <npz> substitutes a DIFFERENT frozen command archive (e.g. a longer neural run);
# --gain <scalar> multiplies every command by a declared global scalar (no feedback, no per-step correction, no CPG/PD).
DRIVE_ARCHIVE = None
if '--drive-archive' in sys.argv and sys.argv.index('--drive-archive') + 1 < len(sys.argv):
    DRIVE_ARCHIVE = Path(sys.argv[sys.argv.index('--drive-archive') + 1])
DECLARED_GAIN = 1.0
if '--gain' in sys.argv and sys.argv.index('--gain') + 1 < len(sys.argv):
    DECLARED_GAIN = float(sys.argv[sys.argv.index('--gain') + 1])
DRIVE_MODE = 'hold'
if '--drive-mode' in sys.argv and sys.argv.index('--drive-mode') + 1 < len(sys.argv):
    DRIVE_MODE = sys.argv[sys.argv.index('--drive-mode') + 1]
assert DRIVE_MODE in ('hold', 'cycle'), DRIVE_MODE
CMD_DT = 0.005
DT_PRIMARY = 1e-4          # canonical XML value, unchanged
DT_SECONDARY = 2.5e-5      # previously qualified branch
DIST_CAP = 4.0
LEGS2 = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
JOINTS2 = ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll', 'Trochanter_yaw', 'Trochanter_pitch',
           'Trochanter_roll', 'Tibia_pitch')
SOURCE_POSE = ROOT / 'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


DIAMOND = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
S_CAP = 500.0


def cone_cols(contacts, dirs):
    cols = []
    for c in contacts:
        frame, mu, jac_t = c['frame'], c['mu'], c['J'].T
        normal, t1, t2 = frame[0], mu[0] * frame[1], mu[1] * frame[2]
        for p, q in dirs:
            cols.append(jac_t @ (normal + float(p) * t1 + float(q) * t2))
    return np.asarray(cols).T


def threshold(muscle_cols, cone, b, lo, hi):
    def ok(sc):
        A = np.hstack([muscle_cols * sc, cone])
        scale = np.maximum(1.0, np.maximum(np.abs(b), np.max(np.abs(A), axis=1)))
        res = linprog(np.zeros(A.shape[1]), A_eq=A / scale[:, None], b_eq=b / scale,
                      bounds=list(zip(lo, hi)) + [(0, None)] * cone.shape[1], method='highs',
                      options=dict(dual_feasibility_tolerance=1e-10, primal_feasibility_tolerance=1e-10))
        return res.status == 0
    f1 = ok(1.0)
    low, high = (0.5, 1.0) if f1 else (1.0, 2.0)
    while high <= S_CAP and not ok(high):
        high *= 1.25
    if high > S_CAP:
        return dict(feasible_at_s_1=f1, threshold_midpoint=None, cap_reached=True,
                    infeasible_up_to_s=high / 1.25)
    for _ in range(26):
        mid = 0.5 * (low + high)
        if ok(mid):
            high = mid
        else:
            low = mid
    return dict(feasible_at_s_1=f1, infeasible_below=low, feasible_above=high,
                threshold_midpoint=0.5 * (low + high), cap_reached=False)


def run_one(mujoco, m, qpos0, controls, dt, label):
    d = mujoco.MjData(m)
    m.opt.timestep = dt
    steps = int(round(DURATION / dt))
    hold = int(round(CMD_DT / dt))
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    foot = [g for g in range(m.ngeom) if 'Tarsus' in gname(g)]
    other = [g for g in range(m.ngeom) if g != FLOOR_GEOM and 'Tarsus' not in gname(g)]
    # ---- the ONLY state initialisation in this script
    d.qpos[:] = qpos0
    d.qvel[:] = 0.0
    d.act[:] = 0.0
    d.ctrl[:] = controls[0]
    mujoco.mj_forward(m, d)
    root_z0 = float(d.qpos[2])
    series = dict(time=[], root_z=[], root_x=[], root_y=[], speed=[], tarsus_contacts=[], nonfoot_contacts=[],
                  leg_tarsus_counts={L: [] for L in LEGS2},
                  lowest_tarsus=[], min_nonfoot_clearance=[], act_max=[], qacc_max=[], energy=[],
                  legs_in_contact=[], ctrl_max=[])
    events = dict(first_nonfoot_contact_step=None, first_tarsus_loss_step=None,
                  first_z_below_50pct_step=None, first_nan_step=None, nonfoot_contact_names=set())
    n = steps
    for k in range(n):
        frame = (min(k // hold, controls.shape[0] - 1) if DRIVE_MODE == 'hold'
                 else (k // hold) % controls.shape[0])
        cmd = controls[frame]
        d.ctrl[:] = cmd                      # command input from the frozen connectome archive only
        mujoco.mj_step(m, d)                 # no state write anywhere in this loop
        t = float(d.time)
        series['time'].append(t)
        series['root_z'].append(float(d.qpos[2]))
        series['root_x'].append(float(d.qpos[0]))
        series['root_y'].append(float(d.qpos[1]))
        series['speed'].append(float(np.linalg.norm(d.qvel[:3])))
        series['act_max'].append(float(np.max(d.act)))
        series['qacc_max'].append(float(np.max(np.abs(d.qacc))))
        try:
            mujoco.mj_energyPos(m, d)
            mujoco.mj_energyVel(m, d)
            e_now = float(d.energy[0] + d.energy[1])
        except Exception:
            e_now = float('nan')
        series['energy'].append(e_now)
        series['ctrl_max'].append(float(np.max(cmd)))
        tarsus = nonfoot = 0
        for c in d.contact:
            if c.efc_address < 0:
                continue
            names = [gname(int(c.geom1)), gname(int(c.geom2))]
            if any('Tarsus' in x for x in names):
                tarsus += 1
            else:
                nonfoot += 1
                events['nonfoot_contact_names'].add(tuple(sorted(names)))
        series['tarsus_contacts'].append(tarsus)
        _step_counts = {_L: 0 for _L in LEGS2}
        for _c in d.contact:
            if _c.efc_address < 0:
                continue
            for _nm in (gname(int(_c.geom1)), gname(int(_c.geom2))):
                if 'Tarsus' in _nm and _nm[:2] in _step_counts:
                    _step_counts[_nm[:2]] += 1
        for _L in LEGS2:
            series['leg_tarsus_counts'][_L].append(_step_counts[_L])
        legs = set()
        for c in d.contact:
            if c.efc_address < 0:
                continue
            for nm in (gname(int(c.geom1)), gname(int(c.geom2))):
                if 'Tarsus' in nm:
                    legs.add(nm[:2])
        series['legs_in_contact'].append(len(legs))
        series['nonfoot_contacts'].append(nonfoot)
        mujoco.mj_kinematics(m, d)
        ft = np.zeros(6)
        series['lowest_tarsus'].append(float(min(mujoco.mj_geomDistance(m, d, FLOOR_GEOM, g, DIST_CAP, ft)
                                                    for g in foot)))
        series['min_nonfoot_clearance'].append(float(min(mujoco.mj_geomDistance(m, d, FLOOR_GEOM, g, DIST_CAP, ft)
                                                         for g in other)))
        if events['first_nonfoot_contact_step'] is None and nonfoot > 0:
            events['first_nonfoot_contact_step'] = k
        if events['first_tarsus_loss_step'] is None and tarsus == 0:
            events['first_tarsus_loss_step'] = k
        if events['first_z_below_50pct_step'] is None and series['root_z'][-1] < 0.5 * root_z0:
            events['first_z_below_50pct_step'] = k
        if events['first_nan_step'] is None and not np.isfinite(d.qpos).all():
            events['first_nan_step'] = k
        if events['first_nan_step'] is not None:
            break
    events['nonfoot_contact_names'] = sorted(' | '.join(x) for x in events['nonfoot_contact_names'])
    first_fail = None
    for key in ('first_nan_step', 'first_tarsus_loss_step', 'first_nonfoot_contact_step',
                'first_z_below_50pct_step'):
        v = events[key]
        if v is not None and (first_fail is None or v < first_fail[1]):
            first_fail = (key, v)
    def _first_persistent(flags, start, persist=5):
        run_len = 0
        for k in range(start, len(flags)):
            run_len = run_len + 1 if flags[k] else 0
            if run_len >= persist:
                return k - persist + 1
        return None
    leg_loss = {}
    if events['first_nonfoot_contact_step'] is not None:
        t0k = events['first_nonfoot_contact_step']
        for _L in LEGS2:
            cnt = series['leg_tarsus_counts'].get(_L, [])
            if len(cnt) != n:
                continue
            carrying = cnt[t0k] > 0
            loss = _first_persistent([c == 0 for c in cnt], t0k + 1) if carrying else None
            leg_loss[_L] = dict(carrying_at_T0=bool(carrying), tarsus_contacts_at_T0=int(cnt[t0k]),
                                first_loss_all_tarsus_step=loss,
                                first_loss_all_tarsus_ms=(None if loss is None else 1000 * loss * dt))
        first_loss = min((v['first_loss_all_tarsus_ms'], L) for L, v in leg_loss.items()
                         if v['first_loss_all_tarsus_ms'] is not None) if any(
            v['first_loss_all_tarsus_ms'] is not None for v in leg_loss.values()) else None
        leg_loss['first_leg_to_lose_support_after_T0'] = (None if first_loss is None else
                                                          dict(leg=first_loss[1], ms=first_loss[0]))
    res = dict(label=label, drive_mode=DRIVE_MODE, dt_s=dt, steps=n, window_s=DURATION,
               leg_support_after_T0=leg_loss,
               root_z_initial=root_z0,
               root_z_final=series['root_z'][-1] if series['root_z'] else None,
               root_z_min=float(min(series['root_z'])) if series['root_z'] else None,
               root_horizontal_displacement=(float(np.hypot(series['root_x'][-1] - series['root_x'][0],
                                                            series['root_y'][-1] - series['root_y'][0]))
                                             if series['root_z'] else None),
               speed_max=float(max(series['speed'])) if series['speed'] else None,
               act_max_over_run=float(max(series['act_max'])) if series['act_max'] else None,
               ctrl_max_commanded_before_the_clamp=(float(max(series['ctrl_max'])) if series['ctrl_max'] else None),
               ctrl_max_effective=(None if not series['ctrl_max'] else float(min(max(series['ctrl_max']), 1.0))),
               qacc_max_over_run=float(max(series['qacc_max'])) if series['qacc_max'] else None,
               qacc_first_step=float(series['qacc_max'][0]) if series['qacc_max'] else None,
               energy_first=float(series['energy'][0]) if series['energy'] else None,
               energy_final=float(series['energy'][-1]) if series['energy'] else None,
               energy_max=float(max(series['energy'])) if series['energy'] else None,
               energy_rising_steps=int(sum(1 for a, b in zip(series['energy'], series['energy'][1:]) if b > a)),
               energy_max_relative_rise=float(max([(b - a) / abs(a) for a, b in
                                                   zip(series['energy'], series['energy'][1:]) if abs(a) > 0] or [0.0])),
               energy_is_strictly_monotone_down=bool(all(b <= a + 1e-12 for a, b in
                                                         zip(series['energy'], series['energy'][1:]))),
               energy_min=float(min(series['energy'])) if series['energy'] else None,
               steps_with_all_six_legs_in_contact=int(sum(1 for v in series['legs_in_contact'] if v == 6)),
               steps_total=int(len(series['legs_in_contact'])),
               legs_in_contact_min=int(min(series['legs_in_contact'])) if series['legs_in_contact'] else None,
               legs_in_contact_final=int(series['legs_in_contact'][-1]) if series['legs_in_contact'] else None,
               actuator_clamp_exceeded=bool(max(series['act_max']) > 1.0 + 1e-12) if series['act_max'] else None,
               events=events,
               first_failure_mode=(None if first_fail is None else first_fail[0]),
               first_failure_step=(None if first_fail is None else first_fail[1]),
               first_failure_time_s=(None if first_fail is None else first_fail[1] * dt),
               survived_whole_window=bool(first_fail is None),
               tarsus_contacts_min=int(min(series['tarsus_contacts'])) if series['tarsus_contacts'] else None,
               tarsus_contacts_final=int(series['tarsus_contacts'][-1]) if series['tarsus_contacts'] else None,
               nonfoot_contacts_max=int(max(series['nonfoot_contacts'])) if series['nonfoot_contacts'] else None,
               lowest_tarsus_final_um=(1000 * series['lowest_tarsus'][-1]) if series['lowest_tarsus'] else None,  # micrometres
               min_nonfoot_clearance_final_um=(1000 * series['min_nonfoot_clearance'][-1])
               if series['min_nonfoot_clearance'] else None,
               statement=(('measured survival for the whole %.0f ms window under the frozen connectome-derived '
                           'drive, with the failure-mode list empty' % (1000 * DURATION) if first_fail is None else
                           'measured FAILURE at step %d (t = %.3f ms, mode %s); nothing in this run is standing, '
                           'balance or biology' % (first_fail[1], 1000 * first_fail[1] * dt, first_fail[0]))
                          + (' [BRANCH INVALID: at dt=1e-4 s the native muscle activation leaves its domain '
                             '(act max %.10f > 1), so no number from this branch may be quoted as a physical '
                             'result; use the 25 us branch]' % max(series['act_max'])
                             if max(series['act_max']) > 1.0 + 1e-12 else
                             ' [branch valid: act stayed inside [0.0001,1]]')))
    return res, {k: np.asarray(v) for k, v in series.items()}


def main():
    out = (_OUTDIR / ('S2-dynamic-from-s1-pose-' + time.strftime('%Y%m%dT%H%M%S'))) if _OUTDIR is not None else \
        DESK / ('S2-dynamic-from-s1-pose-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(parents=True, exist_ok=(_OUTDIR is not None))
    report = dict(status='RUNNING', manual_research_not_product=True, standing=False, walking=False,
                  scope=('first dynamic mj_step run from the seated pose that is statically feasible at s=1 on '
                         'the declared pyramidal (axis-ray diamond) cone, driven only by the frozen '
                         'connectome-derived motor command archive'),
                  declared=dict(drive=('frozen connectome-derived commands normal_controls from the DECLARED archive recorded in '
                                        'drive.archive_path (no external CPG, no PD, no preprogrammed action, no per-step pose correction); '
                                        'the archived declaration below is completed at run time with the actual archive path, frame count, '
                                        'cadence, declared global gain and the fraction of the window that is a constant hold'),
                                state_writes=('qpos, qvel and act are written exactly once before the first step; the '
                                              'only input written inside the loop is ctrl (the archived command)'),
                                timesteps_s=[DT_PRIMARY, DT_SECONDARY], window_s=DURATION,
                                model='canonical XML, 90 native muscle actuators, unchanged',
                                starts={k: str(v) for k, v in STARTS.items()}),
                  declared_calls=dict(project_model_loads=1),
                  absolute_force_values='NOT_ESTABLISHED')
    inputs = [Path(__file__), (DRIVE_ARCHIVE if DRIVE_ARCHIVE is not None else P1), SOURCE_POSE,
              DESK / 'S2-seated-feasibility-search-20261015T122857/best_candidate_pose.npz'] + \
             [v for v in STARTS.values()]
    hashes = {str(p): sha(p) for p in inputs if Path(p).exists()}
    report['input_hashes'] = hashes
    try:
        import mujoco
        from engine import canonical_physics as cp
        m = cp.load_model()
        report['canonical_xml'] = str(cp.CANONICAL_XML)
        report['canonical_xml_sha256'] = sha(cp.CANONICAL_XML)
        _drive_path = DRIVE_ARCHIVE if DRIVE_ARCHIVE is not None else P1
        with np.load(_drive_path) as z:
            controls = z['normal_controls'].copy() * DECLARED_GAIN
        report['drive_mode'] = DRIVE_MODE   # the authoritative mode/rule text is written below, once the coverage is known
        _nf = int(controls.shape[0])
        if DRIVE_MODE == 'cycle':
            report['drive_indexing'] = ('ctrl = normal_controls[(k // %d) %% %d]: the %d archived frames are replayed CYCLICALLY over the '
                                        'whole %.3f s window, so there is no constant-hold phase in this run at all'
                                        % (int(round(CMD_DT / DT_SECONDARY)), _nf, _nf, DURATION))
        elif _nf * CMD_DT >= DURATION - 1e-12:
            report['drive_indexing'] = ('ctrl = normal_controls[min(k // %d, %d)]: the archive has %d frames covering %.3f s, i.e. the WHOLE '
                                        '%.3f s window, so no frame is held constant and there is no saturation inside this run'
                                        % (int(round(CMD_DT / DT_SECONDARY)), _nf - 1, _nf, _nf * CMD_DT, DURATION))
        else:
            report['drive_indexing'] = ('ctrl = normal_controls[min(k // %d, %d)]: the archive is played over the first %g ms and its last '
                                        'frame is then held constant for the rest of the window (at %.3f s that is %g ms = %g percent of the '
                                        'run), so an extended window is an archive replay followed by a hold of the last frame, NOT a periodic '
                                        'repetition'
                                        % (int(round(CMD_DT / DT_SECONDARY)), _nf - 1, 1000 * CMD_DT * _nf, DURATION,
                                           1000 * (DURATION - CMD_DT * _nf), 100 * max(0.0, 1 - CMD_DT * _nf / DURATION)))
        report['archive_covers_the_whole_window'] = bool(controls.shape[0] * CMD_DT >= DURATION - 1e-12)
        report['drive'] = dict(commands_shape=list(controls.shape),
                               nonzero_scalars=int(np.count_nonzero(controls)),
                               cadence_s=CMD_DT, sha256=hashes[str(_drive_path)],
                               archive_path=str(_drive_path), declared_global_gain=DECLARED_GAIN,
                               gain_note=('the declared global scalar multiplies every archived command; no feedback, no per-step correction, no '
                                          'external CPG or PD is added - a declared engineering scaling of the same frozen connectome-derived '
                                          'command archive'))
        report['drive']['ctrl_saturation'] = dict(
            actuator_ctrlrange=[float(m.actuator_ctrlrange[0, 0]), float(m.actuator_ctrlrange[0, 1])],
            actuator_ctrllimited_all=bool(np.all(m.actuator_ctrllimited)),
            mjDSBL_CLAMPCTRL_disabled=bool(not (m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL))),
            muscle_activation_input_clipped_to=[0.0, 1.0],
            n_scaled_entries_above_the_ceiling=int(np.sum(controls > 1.0)),
            n_scaled_entries_below_the_floor=int(np.sum(controls < 1e-4)),
            note=('with the declared global gain the scaled commands pass through the actuator ctrlrange [1e-4, 1] and the muscle activation '
                  'input is clipped to [0, 1], so a gain above 1 is a SATURATING command scaling: ctrl_max_commanded_before_the_clamp is what '
                  'was commanded, ctrl_max_effective is what the model could receive'))
        runs = {}
        series_store = {}
        for label, path in STARTS.items():
            with np.load(path) as z:
                q0 = z['projected_seated_qpos'].copy()
            report['declared_calls'][label + '_npz_sha256'] = hashes[str(path)]
            for dt, tag in ((DT_PRIMARY, 'xml_timestep_1e-4'), (DT_SECONDARY, 'qualified_25us')):
                res, series = run_one(mujoco, m, q0, controls, dt, label)
                runs[label + '@' + tag] = res
                series_store[label + '@' + tag] = series
                print('%-28s dt=%.1e survived=%s first_failure=%s @step %s' % (
                    label, dt, res['survived_whole_window'], res['first_failure_mode'],
                    res['first_failure_step']), flush=True)
        report['runs'] = runs
        report['conclusion'] = dict(
            per_run={k: dict(survived_whole_window=v['survived_whole_window'],
                             numerically_invalid=bool('xml_timestep_1e-4' in k),
                             invalid_reason=('at dt=1e-4 s the native muscle activation leaves its domain (act max 1.9993002399280217 > 1) '
                                             'and MuJoCo uses act UNCLAMPED, so the muscle force law is driven outside its parameterised '
                                             'domain; m.actuator_actlimited is False for every actuator of this model'
                                             if 'xml_timestep_1e-4' in k else None),
                             first_failure_mode=v['first_failure_mode'],
                             first_failure_step=v['first_failure_step'],
                             root_z_final=v['root_z_final'], nonfoot_contacts_max=v['nonfoot_contacts_max'])
                     for k, v in runs.items()},
            statement=('Measured outcomes only: see per_run. No run is called standing, balanced or successful; '
                       'survival of a 50 ms window under an archived drive is not standing and not a completion '
                       'percentage.'))
        # ---- independent recomputation of the reviewer's warm-start result (IK branch dependence)
        try:
            import math
            from scipy.optimize import least_squares
            warm = DESK / 'S2-seated-feasibility-search-20261015T122857/best_candidate_pose.npz'
            with np.load(warm) as zz:
                seed = zz['projected_seated_qpos'].copy()
            gname2 = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
            jname2 = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
            foot2 = [g for g in range(m.ngeom) if 'Tarsus' in gname2(g)]
            leg_geoms2 = {p2: [g for g in foot2 if gname2(g).startswith(p2)] for p2 in LEGS2}
            leg_joints2 = {}
            for p2 in LEGS2:
                js = [j for j in range(m.njnt) if jname2(j).startswith('joint_' + p2)]
                leg_joints2[p2] = [next(j for j in js if jname2(j).endswith(sfx)) for sfx in JOINTS2]
            d2 = mujoco.MjData(m)

            def gdist2(geom, points=False):
                ft = np.zeros(6)
                r = float(mujoco.mj_geomDistance(m, d2, 0, geom, 4.0, ft))
                return (r, ft) if points else r

            d2.qpos[:] = np.load(SOURCE_POSE)['projected_seated_qpos']
            mujoco.mj_kinematics(m, d2)
            src_xy = {p2: min((gdist2(g, True) for g in leg_geoms2[p2]), key=lambda t: t[0])[1][3:5] for p2 in LEGS2}
            d2.qpos[:] = seed
            mujoco.mj_kinematics(m, d2)
            seed_xy = {p2: min((gdist2(g, True) for g in leg_geoms2[p2]), key=lambda t: t[0])[1][3:5] for p2 in LEGS2}
            q = seed.copy()
            q[2] = 2.02
            for p2 in LEGS2:
                j = next(j for j in range(m.njnt) if jname2(j) == 'joint_' + p2 + 'Coxa_yaw')
                adr = m.jnt_qposadr[j]
                q[adr] = seed[adr] + math.radians(-10.0)
            for p2 in LEGS2:
                idx = [m.jnt_qposadr[j] for j in leg_joints2[p2]]
                lo = np.array([max(m.jnt_range[j, 0] + 5e-3, seed[idx[k]] - 0.40) for k, j in enumerate(leg_joints2[p2])])
                hi = np.array([min(m.jnt_range[j, 1] - 5e-3, seed[idx[k]] + 0.40) for k, j in enumerate(leg_joints2[p2])])
                x0 = seed[idx].copy()

                def residual(x):
                    qq = seed.copy()
                    qq[2] = 2.02
                    for p3 in LEGS2:
                        j3 = next(j for j in range(m.njnt) if jname2(j) == 'joint_' + p3 + 'Coxa_yaw')
                        adr3 = m.jnt_qposadr[j3]
                        qq[adr3] = seed[adr3] + math.radians(-10.0)
                    qq[idx] = x
                    d2.qpos[:] = qq
                    mujoco.mj_kinematics(m, d2)
                    best = min((gdist2(g, True) for g in leg_geoms2[p2]), key=lambda t: t[0])
                    return np.concatenate([[10.0 * (best[0] + 5e-3)],
                                           [0.5 * (best[1][3] - 0.95 * src_xy[p2][0]),
                                            0.5 * (best[1][4] - 0.95 * src_xy[p2][1])],
                                           0.05 * (x - x0)])
                sol = least_squares(residual, x0, bounds=(lo, hi), xtol=1e-13, ftol=1e-13, gtol=1e-13,
                                    max_nfev=3000)
                q[idx] = sol.x
            d2.qpos[:] = q
            d2.qvel[:] = 0
            d2.act[:] = 0
            d2.ctrl[:] = 0
            mujoco.mj_forward(m, d2)
            d2.qpos[:] = q
            mujoco.mj_kinematics(m, d2)
            cl = float(min(gdist2(g) for g in foot2))
            base2 = d2.qfrc_actuator.copy()
            tgt2 = d2.qfrc_bias - d2.qfrc_passive - base2
            R2 = np.zeros((m.nv, m.nu))
            for i in range(m.nu):
                d2.act[:] = 0
                d2.act[i] = 1
                mujoco.mj_forward(m, d2)
                R2[:, i] = d2.qfrc_actuator - base2
            contacts2 = []
            d2.act[:] = 0
            d2.qpos[:] = q
            d2.qvel[:] = 0
            mujoco.mj_forward(m, d2)
            for c in d2.contact:
                if c.efc_address < 0:
                    continue
                jac = np.zeros((3, m.nv))
                for geom, sign in ((c.geom2, 1), (c.geom1, -1)):
                    b2 = int(m.geom_bodyid[geom])
                    jp, jr = np.zeros_like(jac), np.zeros_like(jac)
                    mujoco.mj_jac(m, d2, jp, jr, c.pos, b2)
                    jac += sign * jp
                contacts2.append(dict(frame=np.asarray(c.frame).reshape(3, 3),
                                      mu=np.asarray(c.friction)[:2], J=jac))
            lo2 = m.actuator_ctrlrange[:, 0].copy()
            hi2 = m.actuator_ctrlrange[:, 1].copy()
            s_warm = threshold(R2, cone_cols(contacts2, DIAMOND), tgt2, lo2, hi2)
            report['ik_branch_upper_bound_check'] = dict(
                statement=('the s reported in the earlier artifacts is a function of the JOINT VECTOR produced by '
                           'that script IK, and the per-leg IK has several local minima that all reach zero touch '
                           'residual; the reported family minimum is therefore an UPPER BOUND, not a minimum'),
                reviewer_warm_start_value=0.9572098068892956,
                reviewer_parameters='z=2.02 mm, footprint 0.95, uniform coxa-yaw -10 deg',
                warm_start_seed=str(warm), warm_start_seed_sha256=hashes.get(str(warm)),
                recomputed_here=dict(s=s_warm, min_lowest_tarsus_um=1000 * cl,
                                     contacts=len(contacts2)),
                reproduced=bool(s_warm.get('threshold_midpoint') is not None and
                                abs(s_warm['threshold_midpoint'] - 0.9572098068892956) < 1e-6),
                verdict=('REPRODUCED: warm-starting the per-leg IK from the delivered joint vector gives s = %.16f, '
                         'matching the reviewer value to 1e-6' % s_warm['threshold_midpoint']
                         if (s_warm.get('threshold_midpoint') is not None and
                             abs(s_warm['threshold_midpoint'] - 0.9572098068892956) < 1e-6) else
                         'NOT REPRODUCED here: recomputed s = %s; the difference is not resolved by this run'
                         % (None if s_warm.get('threshold_midpoint') is None else s_warm['threshold_midpoint'])))
        except Exception as exc2:
            report['ik_branch_upper_bound_check'] = dict(error=type(exc2).__name__ + ': ' + str(exc2))
        np.savez_compressed(out / 'time_series.npz',
                            **{k.replace('@', '__') + '__' + name: arr
                               for k, series in series_store.items() for name, arr in series.items()})
        report['status'] = 'PASS_DYNAMIC_FROM_S1_POSE_REPORTED'

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
        for label, series in series_store.items():
            if 'qualified_25us' not in label:
                continue      # only the numerically valid branch is plotted
            t = 1000 * series['time']
            axes[0].plot(t, series['root_z'], lw=1, label=label.split('@')[0])
            axes[1].plot(t, series['tarsus_contacts'], lw=1, label=label.split('@')[0] + ' tarsus')
            axes[1].plot(t, series['nonfoot_contacts'], lw=1, ls='--',
                         label=label.split('@')[0] + ' non-foot')
            axes[2].plot(t, 1000 * series['lowest_tarsus'], lw=1, label=label.split('@')[0])
        axes[0].set_xlabel('time (ms)')
        axes[0].set_ylabel('root (thorax) z (mm)')
        axes[0].set_title('root height, 25 us valid branch')   # the invalid 1e-4 s branch is named in the suptitle
        axes[0].legend(fontsize=7)
        axes[1].set_xlabel('time (ms)')
        axes[1].set_ylabel('active contacts')
        axes[1].set_title('tarsus vs non-foot contacts')
        axes[1].legend(fontsize=7)
        axes[2].axhline(0, color='k', lw=0.6)
        axes[2].set_xlabel('time (ms)')
        axes[2].set_ylabel('lowest tarsus distance to floor (um)')
        axes[2].set_title('ground contact of the lowest tarsus')
        axes[2].legend(fontsize=7)
        _n_frames = int(controls.shape[0])
        _covers = bool(_n_frames * CMD_DT >= DURATION - 1e-12)
        if _covers and DRIVE_MODE == 'hold':
            _mode_txt = ('FULL ARCHIVE: %d frames x %g ms cover the whole %.3f s window (no hold, no cycling)'
                         % (_n_frames, 1000.0 * CMD_DT, DURATION))
        elif DRIVE_MODE == 'hold':
            _mode_txt = ('HOLD: archive of %d frames played 0-%g ms, then the last frame held constant'
                         % (_n_frames, 1000 * _n_frames * CMD_DT))
        else:
            _mode_txt = ('CYCLE: the same %d archived frames replayed for the whole window' % _n_frames)
        _gain_txt = '' if abs(DECLARED_GAIN - 1.0) < 1e-12 else ' | declared global gain %.2f on every command' % DECLARED_GAIN
        fig.suptitle('Connectome-drive run - %s%s\n25 us valid branch (the 1e-4 s branch is invalid, NOT plotted); '
                     'manual research, NOT standing' % (_mode_txt, _gain_txt), fontsize=9)
        fig.tight_layout()
        fig.savefig(out / 'dynamic-run.png', dpi=130)
        plt.close(fig)
    except Exception as exc:
        import traceback
        report.update(status='FAIL_STOP', error=type(exc).__name__ + ': ' + str(exc),
                      traceback=traceback.format_exc()[-1500:])
    finally:
        report['input_hashes_after'] = {p: sha(p) for p in hashes}
        report['hashes_unchanged'] = report['input_hashes_after'] == hashes
        if not report['hashes_unchanged']:
            report['status'] = 'FAIL_INPUT_MUTATED'
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
        print(json.dumps(report['conclusion']['per_run'], indent=1))
    else:
        print(report.get('error', ''))
    return 0 if report['status'].startswith('PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
