"""STAGE-2 sensory factorial candidate runner (HEAVY, serial, single job).

4 arms x 2 neural timesteps x 5 release phases = 40 main releases, plus at most 16
pre-registered conditional linkage controls.  Every (arm, dt) builds its OWN neural warm
state by replaying the recorded 1200x12000 float64 currents from rest at that dt; no arm
and no dt ever reuses another's neural cache, and no warm replay is repeated.  Connectivity,
body, readout, clocks and the frozen sources are untouched.  This diagnostic is declared
NONCONVERGED: no timestep is selected as truth and a gain at only one dt is not a stable
improvement.

Importing this module does nothing heavy.  Running it requires the parent-written heavy-GO
authorization file that record_teacher.py also consumes.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import importlib.util
import json
import pickle
import resource
import sys
import time
import numpy as np

SRC = Path(__file__).resolve().parent
REPO = Path('/run/media/sensen/Data2/cell_wound_prototype')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
OUT = Path('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/sensory')
NUM = SRC.parent / 'numerics/run.py'
ROOT = OUT / 'CANDIDATES'
TEACHER = OUT / 'TEACHER_dt_0p5ms'
STAGE1 = Path('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics')
PHASES = (1.225, 1.275, 1.3, 1.325, 1.375)
DTS = (.0005, .0000625)
CDT = .001
ARMS = ('baseline', 'A_positive_force', 'B_club_motion', 'C_local_amplitude')
CHANNELS = ('claw_position', 'hook_direction', 'hairplate_angle', 'campaniform_strain',
            'club_vibration', 'bristle_touch', 'multidendritic_noxious')
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
J_HISTORICAL = 1.5903704331878943
MAIN_CAP = len(PHASES) * len(DTS) * len(ARMS)
CONTROL_CAP = 2 * len(DTS) * len(ARMS)
TOTAL_CAP = MAIN_CAP + CONTROL_CAP


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def js(p):
    return json.loads(Path(p).read_text())


def arrays(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def as_json(v):
    return json.dumps(v, sort_keys=True, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))


def dtag(dt):
    return 'dt_' + str(dt * 1000).replace('.', 'p') + 'ms'


def rel(ph):
    return 'rel_' + str(int(round(ph * 1000))).zfill(4) + 'ms'


def label(arm, dt, ph, kind='main'):
    return arm + '__' + dtag(dt) + '__' + rel(ph) + ('' if kind == 'main' else '__' + kind)


def validate_decision(path):
    """The parent GO must freeze the same files record_teacher.py froze, plus this protocol."""
    decision = json.loads(Path(path).read_text())
    if decision.get('explicit_parent_heavy_GO') is not True:
        raise ValueError('requires parent heavy-process GO')
    if decision.get('operation') != 'record_sensory_raw_teacher_1p4s':
        raise ValueError('authorization is not the parent heavy GO for this serial job')
    if decision.get('numerics_source_sha256') != sha(NUM):
        raise ValueError('numerics runtime source not frozen at the supplied hash')
    if decision.get('numerical_status') != 'nonconverged_factorial_diagnostic':
        raise ValueError('this factorial diagnostic must be declared nonconverged')
    if decision.get('neural_dt_s') != .0005:
        raise ValueError('the single teacher recording must be at .0005 s')
    required = sorted(SRC.glob('*.py')) + [OUT / 'IDENTITY.json', OUT / 'PROTOCOL.json',
                                           OUT / 'OFFLINE_TESTS.json', ROOT / 'PREREGISTRATION.json']
    expected = decision.get('sensory_input_sha256', {})
    for p in required:
        if expected.get(str(p)) != sha(p):
            raise ValueError('sensory source/evidence not frozen: ' + str(p))
    if not js(OUT / 'OFFLINE_TESTS.json')['passed']:
        raise ValueError('offline gate failed')
    if not (ROOT / 'PREREGISTRATION.json').is_file():
        raise ValueError('preregistration must exist before construction')
    return decision


def check_historical_baseline(rows, num):
    """Mandatory: baseline at .0005 s must be bitwise the sealed stage-1 release."""
    checked = []
    for r in rows:
        if not (r['arm'] == 'baseline' and r['dt_s'] == DTS[0] and r['kind'] == 'main'):
            continue
        ph = r['release_time_s']
        old = STAGE1 / 'RUNS' / ('dt_0p5ms_paired_' + ('%g' % ph))
        if not old.is_dir():
            raise AssertionError('missing sealed stage-1 release ' + str(old))
        detail = dict(phase=ph, sealed=str(old))
        for name in ('BODY.npz', 'NATIVE.npz'):
            a, b = arrays(old / name), arrays(Path(r['folder']) / name)
            if set(a) != set(b):
                raise AssertionError('key mismatch ' + name)
            for k in a:
                if not np.array_equal(a[k], b[k]):
                    raise AssertionError(f'{name}:{k} differs from sealed stage-1 at {ph}')
            detail[name] = 'bitwise_exact:' + ','.join(sorted(a))
        a, b = arrays(old / 'NEURAL.npz'), arrays(Path(r['folder']) / 'NEURAL.npz')
        keys = ['spikes204', 'spikes12000_packed', 'rates', 'source_rates', 'applied', 'raw_command',
                'old_torque', 'features', 'sensory_current', 'sensory_current_norm', 'net_t_ms',
                'net_steps', 'time']
        for k in keys:
            if not np.array_equal(a[k], b[k]):
                raise AssertionError(f'NEURAL.npz:{k} differs from sealed stage-1 at {ph}')
        detail['NEURAL.npz'] = 'bitwise_exact:' + ','.join(keys)
        oldJ = js(old / 'RESULT.json')['J']
        if r['J'] != oldJ:
            raise AssertionError(f'J {r["J"]!r} != sealed stage-1 J {oldJ!r} at {ph}')
        if ph == 1.3 and r['J'] != J_HISTORICAL:
            raise AssertionError('1.3 s J is not the historical value')
        detail['J'] = r['J']
        checked.append(detail)
    if len(checked) != len(PHASES):
        raise AssertionError('baseline .0005 s releases incomplete')
    num.dump(ROOT / 'BASELINE_HISTORICAL_EXACT.json',
             dict(passed=True, releases=checked, J_1p3_s=J_HISTORICAL, importance='核心必需'))
    return dict(passed=True, releases=len(checked))


def check_warm_baseline(num, t, teacher, arm, dt, mats):
    """Baseline/.0005 s warm state must be bitwise the teacher's (== historical) neural cache."""
    if not (arm == 'baseline' and dt == DTS[0]):
        return dict(checked=False, reason='only baseline at .0005 s has a sealed neural reference')
    rows = []
    for ph in PHASES:
        c, o = t.cache[ph], teacher[ph]
        if not num.exact(vars(c['net']), vars(o['net'])):
            raise AssertionError(f'warm network not bitwise at {ph}')
        if not (num.exact(c['rates'], o['rates']) and num.exact(c['history'], o['history'])):
            raise AssertionError(f'warm rates/history not bitwise at {ph}')
        if not all(num.exact(m, getattr(c['net'], k)) for m, k in zip(mats, ('We', 'Wi'))):
            raise AssertionError(f'warm connectivity differs from ACTUAL matrices at {ph}')
        if not (num.exact(c['integration'], o['integration']) and num.exact(c['assist'], o['assist'])):
            raise AssertionError(f'warm physical integration/assist not bitwise at {ph}')
        rows.append(dict(phase=ph, network_exact=True, rates_exact=True, history_exact=True,
                         connectivity_exact=True, integration_exact=True, assist_exact=True))
    return dict(checked=True, arm=arm, neural_dt_s=dt, rows=rows,
                note='own replay from rest of the 1200 recorded float64 currents reproduces the sealed warm state exactly')


def eligibility(rows, dt):
    """Pre-registered rule for the conditional linkage controls (frozen in PROTOCOL.json)."""
    base = [r for r in rows if r['dt_s'] == dt and r['arm'] == 'baseline' and r['kind'] == 'main']
    if len(base) != len(PHASES):
        raise AssertionError('baseline incomplete for ' + dtag(dt))
    bJ = float(np.mean([r['J'] for r in base]))
    bf = int(sum(sum(1 for v in r['validation']['checks'].values() if not v) for r in base))
    out = {}
    for arm in ARMS:
        rs = [r for r in rows if r['dt_s'] == dt and r['arm'] == arm and r['kind'] == 'main']
        ok = len(rs) == len(PHASES) and all(r['warnings'] == 0 and r['release_aux_max'] == 0.
                                           and r['release_external_max'] == 0. for r in rs)
        if ok:
            J = float(np.mean([r['J'] for r in rs]))
            f = int(sum(sum(1 for v in r['validation']['checks'].values() if not v) for r in rs))
            notworse = bool(J <= bJ and f <= bf)
        else:
            J = f = None
            notworse = None
        out[arm] = dict(guard_feasible=bool(ok), mean_J=J, failed_D_checks=f,
                        baseline_mean_J=bJ, baseline_failed_D_checks=bf,
                        not_worse_than_baseline=notworse,
                        run_controls=bool(ok and notworse is True))
    return out


def reencode_diagnostics(arm, current, baseline_current, legacy, identity):
    diff = current - baseline_current
    changed = np.flatnonzero(np.any(diff != 0, axis=0))
    owner = {}
    for leg in LEGS:
        for ch in CHANNELS:
            for i in legacy.groups[leg][ch]:
                owner[int(i)] = (leg, ch)
    by_channel = {}
    for i in changed:
        leg, ch = owner[int(i)]
        by_channel.setdefault(leg + '/' + ch, []).append(int(i))
    consensus = {int(i) for g in identity['consensus_blocks'] for i in g['local_indices']}
    return dict(
        frames=int(current.shape[0]), neurons=int(current.shape[1]),
        reencoded_from='TEACHER_dt_0p5ms/RAW_STREAM.npz raw contacts/joints (dt-independent)',
        recorded_baseline_bitwise=True,
        n_cells_changed=int(len(changed)),
        n_cell_frames_changed=int(np.count_nonzero(diff)),
        n_frames_changed=int(np.count_nonzero(np.any(diff != 0, axis=1))),
        changed_cells_local=changed.tolist(),
        changed_cells_in_consensus_C=len(set(changed.tolist()) & consensus),
        changed_by_leg_channel={k: len(v) for k, v in sorted(by_channel.items())},
        max_abs_current_baseline=float(np.max(np.abs(baseline_current))),
        max_abs_current_arm=float(np.max(np.abs(current))),
        max_abs_diff=float(np.max(np.abs(diff))),
        arm_current_min=float(current.min()), arm_current_max=float(current.max()),
        saturation_cells_at_0p3nA=int(np.count_nonzero(np.isclose(current, .3))),
        input_changed=bool(len(changed) > 0),
        importance='核心必需')


def execution(decision, decision_path):
    from scipy import sparse
    import runtime_bridge
    from engine.embodied.sensory_channels import ChannelSensoryEncoder, ChannelConfig

    if (ROOT / 'SUMMARY.json').exists():
        raise ValueError('this candidates run is already finalised; start a new output root')
    record = js(TEACHER / 'RECORDING_RESULT.json')
    if not record.get('passed'):
        raise ValueError('teacher recording did not pass its own gates')
    if record.get('actual_old_observer_bitwise_frames') != 1200:
        raise ValueError('teacher bitwise frame count is not 1200')
    if record.get('neural_dt_s') != .0005 or record.get('numerical_status') != decision['numerical_status']:
        raise ValueError('teacher record disagrees with the authorization')

    module_name = 'sensory_frozen_numerics'
    spec = importlib.util.spec_from_file_location(module_name, NUM)
    num = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = num
    spec.loader.exec_module(num)
    num.O = ROOT; num.rt.O = ROOT; num.rt.b.O = ROOT
    frozen = js(OUT / 'PROTOCOL.json')['frozen_source_sha256']
    before = {p: sha(p) for p in frozen if Path(p).is_file()}
    num.dump(ROOT / 'INPUT_HASHES_BEFORE.json', before)
    bad = {p: [frozen[p], before[p]] for p in before if before[p] != frozen[p]}
    if bad:
        raise ValueError('frozen input hash mismatch: ' + json.dumps(bad))
    num.dump(ROOT / 'EXECUTION_DECISION.json',
             dict(decision=decision, source=str(decision_path), sha256=sha(decision_path),
                  importance='核心必需'))

    identity = js(OUT / 'IDENTITY.json')
    legacy = ChannelSensoryEncoder(identity['channel_groups'], 12000, ChannelConfig())
    teacher_layout = js(TEACHER / 'LAYOUT.json')
    mats = [sparse.load_npz(num.rt.b.AB / 'A' / ('ACTUAL_' + k + '.npz')) for k in ('We', 'Wi')]

    # --- 1) per-arm re-encoding of the ONE recorded raw stream (arrays only) -------------
    arm_dir = ROOT / 'ARMS'
    arm_dir.mkdir(parents=True, exist_ok=False)
    streams = {}
    baseline_current = arrays(TEACHER / 'RAW_STREAM.npz')['actual_current']
    for arm in ARMS:
        t0 = time.monotonic()
        times, current, layout = runtime_bridge.reencode_actual_stream(
            TEACHER / 'RAW_STREAM.npz', identity, legacy, arm)
        if as_json(layout.to_dict()) != as_json(teacher_layout):
            raise ValueError('recorded raw stream layout differs from the teacher layout')
        detail = reencode_diagnostics(arm, current, baseline_current, legacy, identity)
        detail.update(arm=arm, wall_s=time.monotonic() - t0)
        folder = arm_dir / arm
        folder.mkdir(parents=True, exist_ok=False)
        with (folder / 'REENCODED.npz').open('xb') as f:
            np.savez_compressed(f, time=times, current=current, baseline_current=baseline_current)
        num.dump(folder / 'REENCODE_DIAGNOSTIC.json', detail)
        streams[arm] = current
        print('REENCODED', arm, detail['n_cells_changed'], 'cells', flush=True)

    # --- 2) runtime with every legacy-constructor write redirected ----------------------
    Base = runtime_bridge.runtime_class(num, identity, ROOT, decision['numerical_status'])

    class Audited(Base):
        def episode(self, lab, ph, config, cut=False, sensory_cut=False):
            c = self.cache[ph]
            pre = (c['net'].We.data.copy(), c['net'].Wi.data.copy())
            rows = np.repeat(np.arange(c['net'].n), np.diff(c['net'].We.indptr))
            rh = np.isin(rows, self.rh)
            result = super().episode(lab, ph, config, cut=cut, sensory_cut=sensory_cut)
            post = self.cache[ph]['net']
            unmutated = bool(np.array_equal(pre[0], post.We.data) and np.array_equal(pre[1], post.Wi.data))
            pristine = bool(np.array_equal(pre[0], mats[0].data) and np.array_equal(pre[1], mats[1].data))
            if not unmutated:
                raise AssertionError('release mutated the cached warm connectivity')
            if not pristine:
                raise AssertionError('cached warm connectivity is not the frozen ACTUAL matrix')
            audit = dict(label=lab, arm=self.sensory_arm, phase=ph, neural_dt_s=self.neural_dt,
                         cut=bool(cut), sensory_cut=bool(sensory_cut),
                         warm_cache_unmutated_by_release=unmutated,
                         warm_cache_equals_ACTUAL_matrices=pristine,
                         exc_edges=int(c['net'].We.nnz), exc_sum=float(pre[0].sum()),
                         exc_rh_rows_sum=float(pre[0][rh].sum()),
                         exc_other_rows_sum=float(pre[0][~rh].sum()),
                         inh_edges=int(c['net'].Wi.nnz), inh_sum=float(pre[1].sum()),
                         rh_cells=int(len(self.rh)), rh_exc_gain=.375,
                         gain_applied_to_release_copy_once=True,
                         expected_release_exc_sum=float(pre[0].sum() - (1 - .375) * pre[0][rh].sum()),
                         importance='核心必需')
            num.dump(Path(result['folder']) / 'CONNECTIVITY_AUDIT.json', audit)
            return result

    t = Audited()
    if as_json(t.raw_layout.to_dict()) != as_json(teacher_layout):
        raise ValueError('runtime layout differs from the teacher recording layout')
    with (TEACHER / 'BASELINE_CACHE.pkl').open('rb') as f:
        teacher = pickle.load(f)
    if not set(PHASES) <= set(teacher):
        raise ValueError('teacher cache is missing a release phase: ' + str(sorted(teacher)))
    # The sealed recorder also captures the 1.4 s terminal state.  That is not a release phase,
    # so the release grid uses exactly the five pre-registered phases; the teacher recording
    # itself is left byte-identical and still holds the extra state.
    teacher_selection = dict(all_teacher_states=sorted(teacher), release_phases=list(PHASES),
                             unused_teacher_states=sorted(set(teacher) - set(PHASES)),
                             reason='only the 5 pre-registered release phases may seed a release; '
                                    'the 1.4 s state is a recorder-side terminal capture, not a phase',
                             importance='核心必需')
    num.dump(ROOT / 'TEACHER_CACHE_SELECTION.json', teacher_selection)
    teacher = {ph: teacher[ph] for ph in PHASES}

    # --- 3) 8 own warm replays + 40 main releases --------------------------------------
    results = []
    warm_log = []
    keep_cache = {}
    for dt in DTS:
        for arm in ARMS:
            out_dir = ROOT / 'WARM' / (arm + '__' + dtag(dt))
            cfg = runtime_bridge.replay_warm(num, t, teacher, streams[arm], dt, arm, out_dir)
            keep_cache[(arm, dt)] = t.cache
            params = js(cfg)
            params.update(numerical_status=decision['numerical_status'],
                          placeholder_replaced=('runtime_bridge wrote PARENT_DECISION_REQUIRED; replaced by the '
                                                'declared status of this job, no other field changed'),
                          importance='核心必需')
            num.dump(cfg, params)
            warm = dict(arm=arm, neural_dt_s=dt, folder=str(out_dir),
                        neural_steps=int(round(1.2 / dt)), commands=1200, own_state=True,
                        baseline_exact=check_warm_baseline(num, t, teacher, arm, dt, mats),
                        importance='核心必需')
            with (out_dir / 'WARM_CACHE.pkl').open('xb') as f:
                pickle.dump(keep_cache[(arm, dt)], f, protocol=5)
            warm['cache_sha256'] = sha(out_dir / 'WARM_CACHE.pkl')
            warm_log.append(warm)
            print('WARM_READY', arm, dtag(dt), flush=True)
            for ph in PHASES:
                r = t.episode(label(arm, dt, ph), ph, cfg)
                r.update(arm=arm, dt_s=dt, kind='main', warm_folder=str(out_dir),
                         warm_baseline_exact=bool(warm['baseline_exact'].get('checked')))
                results.append(r)
            gc.collect()
            num.dump(ROOT / 'PROGRESS.json',
                     dict(stage='main', releases=len(results), warm_replays=len(warm_log),
                          results=results, warm=warm_log,
                          rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                          importance='核心必需'))

    if len(results) != MAIN_CAP:
        raise AssertionError('main release budget violated')
    historical = check_historical_baseline(results, num)

    # --- 4) pre-registered conditional linkage controls (max 16) -----------------------
    selections = {dtag(dt): eligibility(results, dt) for dt in DTS}
    controls = []
    for dt in DTS:
        for arm in ARMS:
            if not selections[dtag(dt)][arm]['run_controls']:
                continue
            t.cache = keep_cache[(arm, dt)]
            cfg = ROOT / 'WARM' / (arm + '__' + dtag(dt)) / 'PARAMETERS.json'
            for kind, kw in (('sensory_cut', dict(sensory_cut=True)), ('motor_cut', dict(cut=True))):
                r = t.episode(label(arm, dt, 1.3, kind), 1.3, cfg, **kw)
                r.update(arm=arm, dt_s=dt, kind=kind,
                         warm_folder=str(ROOT / 'WARM' / (arm + '__' + dtag(dt))))
                controls.append(r)
            gc.collect()
    if len(controls) > CONTROL_CAP:
        raise AssertionError('conditional control budget violated')
    total = len(results) + len(controls)
    if total > TOTAL_CAP:
        raise AssertionError('total release budget violated')

    summary = dict(stage='STAGE-2 sensory factorial diagnostic',
                   numerical_status=decision['numerical_status'],
                   protocol=str(OUT / 'PROTOCOL.json'), protocol_sha256=sha(OUT / 'PROTOCOL.json'),
                   neural_dt_s=list(DTS), phases_s=list(PHASES), release_duration_s=.2,
                   arms=list(ARMS), main_releases=len(results), conditional_controls=len(controls),
                   total_releases=total,
                   caps=dict(main=MAIN_CAP, conditional=CONTROL_CAP, total=TOTAL_CAP),
                   teacher_recordings=1, warm_replays=len(warm_log),
                   warm=warm_log, results=results, controls=controls,
                   selections=selections, historical_baseline=historical,
                   selection_of_model='NONE; both timesteps reported; a single-dt gain is not a stable improvement',
                   walking_certified=False, autonomous_behavior='NOT_TESTED',
                   biological_validity='NOT_ESTABLISHED', absolute_units='NOT_ESTABLISHED',
                   connectivity='original ACTUAL_We/ACTUAL_Wi fixed; RH excitatory .375 once per release only',
                   importance='核心必需')
    num.dump(ROOT / 'SUMMARY.json', summary)
    after = {p: sha(p) for p in frozen if Path(p).is_file()}
    num.dump(ROOT / 'INPUT_HASHES_AFTER.json', after)
    if after != before:
        raise AssertionError('a frozen input changed during the run')
    print('CANDIDATES_READY', len(results), len(controls), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--authorization-file', type=Path, required=True)
    args = p.parse_args()
    decision = validate_decision(args.authorization_file)
    execution(decision, args.authorization_file)


if __name__ == '__main__':
    main()
