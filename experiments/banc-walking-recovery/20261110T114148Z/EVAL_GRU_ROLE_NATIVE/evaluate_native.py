"""Read-only S2/S5 native acceptance wrapper. Never steps physics or trains.
Run only after RESULT.json is committed by the managed recording job.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
from array import array
import numpy as np

OUT = Path(__file__).resolve().parent
RUN = OUT.parent
FROZEN = RUN.parent / '20261110T054510Z' / 'native_acceptance.py'
# Import snapshot physics from this run, never from the frozen evaluator's directory.
sys.path.insert(0, str(RUN))
spec = importlib.util.spec_from_file_location('_frozen_native_acceptance_role', FROZEN)
frozen = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = frozen
spec.loader.exec_module(frozen)


def evaluate_directory(input_dir, *, autonomous_window, auxiliary_window=None,
                       mass_native, gravity_native=9801., heading_world_xy=(1., 0.),
                       source_manifest=None, full_model=None, metadata=None,
                       max_scalar_samples=2_000_000):
    """Frozen public API, unchanged criteria, bounded extra D7 histograms."""
    base = frozen._Selected
    class DiagnosticSelected(base):
        def __init__(self, window, cap):
            super().__init__(window, cap)
            self.hist = np.zeros((4, 7), dtype=np.int64)
        def add(self, before, after, com_before, com_after, loads, nonfoot, speed,
                az_filtered, contact_filtered, original, accounted, external, weight, gravity, tol):
            previous = self.n
            super().add(before, after, com_before, com_after, loads, nonfoot, speed,
                        az_filtered, contact_filtered, original, accounted, external, weight, gravity, tol)
            if self.n != previous:
                for j, fraction in enumerate((.001, .01, .05, .1)):
                    self.hist[j, np.count_nonzero(loads >= fraction * weight)] += 1
        def summary(self, weight, heading, tol):
            result = super().summary(weight, heading, tol)
            rows = []
            for fraction, hist in zip((.001, .01, .05, .1), self.hist):
                cumulative = hist.cumsum()
                ranks = ((self.n - 1) // 2, self.n // 2)
                median = float(np.mean([np.searchsorted(cumulative, r, side='right') for r in ranks])) if self.n else None
                rows.append(dict(threshold_fraction_W=fraction, histogram_count_0_to_6=hist.tolist(),
                                 median=median, provisional_at_least_three=None if median is None else median >= 3))
            result['D7_threshold_sensitivity_diagnostic_only'] = rows
            return result
    frozen._Selected = DiagnosticSelected
    try:
        return frozen.evaluate_directory(input_dir, autonomous_window=autonomous_window,
            auxiliary_window=auxiliary_window, mass_native=mass_native, gravity_native=gravity_native,
            heading_world_xy=heading_world_xy, source_manifest=source_manifest, full_model=full_model,
            metadata=metadata, max_scalar_samples=max_scalar_samples)
    finally:
        frozen._Selected = base


def read_json(path):
    return json.loads(Path(path).read_text())


def compare_npz(left, right):
    """Decompress each array once per file; release before the next array."""
    rows = {}
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise ValueError('replay array key mismatch')
        for key in a.files:
            x, y = a[key], b[key]
            same_shape = x.shape == y.shape
            exact = same_shape and np.array_equal(x, y)
            delta = float(np.max(np.abs(x.astype(float) - y.astype(float)))) if same_shape and x.size else (0. if same_shape else None)
            rows[key] = dict(shape=list(x.shape), exact=bool(exact), max_abs_difference=delta,
                             within_1e_8=bool(same_shape and delta <= 1e-8))
            del x, y
    return dict(left=str(left), right=str(right), left_sha256=frozen.sha256(left),
                right_sha256=frozen.sha256(right), arrays=rows,
                all_exact=all(r['exact'] for r in rows.values()),
                all_within_1e_8=all(r['within_1e_8'] for r in rows.values()),
                archive_hash_note='ZIP container timestamps can differ; logical arrays determine replay equality')


def extra_native(manifest, windows, weight, cap=2_000_000):
    """Second bounded pass for raw force accounting, without retaining trajectory."""
    residual = {k: array('d') for k in windows}
    peaks = {k: 0. for k in ('external_body_wrench_native', 'external_generalized_force_native',
                             'adhesion_body_wrench_native', 'adhesion_enabled',
                             'adhesion_application_body_leg_point_force_native')}
    contact_events = samples = 0
    for entry in manifest['chunks']:
        path = Path(entry['path'])
        with np.load(path, allow_pickle=False) as z:
            before, after = z['time_before_s'], z['time_after_s']
            raw = z['force_balance_residual_native'][:, 2] / weight
            for key in peaks:
                v = z[key]
                if v.size:
                    peaks[key] = max(peaks[key], float(np.max(np.abs(v))))
                del v
            offsets = z['contact_offsets']
            if len(offsets) != len(before) + 1 or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
                raise ValueError('malformed contact event offsets')
            events = z['contact_geom_leg_ground']
            if offsets[-1] != len(events):
                raise ValueError('contact event count mismatch')
            contact_events += int(offsets[-1])
            samples += len(before)
            for key, (start, end) in windows.items():
                keep = (before >= start - 1e-10) & (after <= end + 1e-10)
                values = raw[keep]
                if len(residual[key]) + len(values) > cap:
                    raise ValueError('raw scalar cap exceeded')
                residual[key].extend(values)
    result = dict(samples=samples, contact_event_count=contact_events, actual_extra_force_absolute_peaks=peaks,
                  all_actual_extra_forces_zero=all(v == 0 for v in peaks.values()), windows={})
    for key, values in residual.items():
        v = np.frombuffer(values, dtype=np.float64)
        result['windows'][key] = dict(samples=len(v), raw_external_accounted_signed_p95_W=float(np.quantile(v, .95)),
            raw_external_accounted_absolute_p95_W=float(np.quantile(np.abs(v), .95)),
            note='native recorder raw FD residual; diagnostic, does not replace frozen filtered original D6')
    return result


def main():
    # RESULT is the completion barrier. Never inspect incomplete native chunks.
    result_path = OUT / 'RESULT.json'
    if not result_path.is_file():
        raise RuntimeError('recording incomplete: wait for parent completion message; RESULT.json absent')
    from physics import Core, PAD_XML
    from identity import verify_runtime
    identity = read_json(OUT / 'IDENTITY.json')
    original_identity = read_json(RUN / 'EVAL_GRU_ROLE_AUX' / 'IDENTITY.json')
    result = read_json(result_path)
    original = read_json(RUN / 'EVAL_GRU_ROLE_AUX' / 'RESULT.json')
    manifest = result['native']
    if manifest['pending_rows'] != 0 or result['control'] != 'student' or result['sampled']:
        raise ValueError('not a completed deterministic ANN recording')
    checkpoint = Path(identity['config']['cli']['checkpoint'])
    if checkpoint != Path(original_identity['config']['cli']['checkpoint']) or checkpoint != RUN / 'BC_GRU_ROLE_AUX_300' / 'final.pt':
        raise ValueError('checkpoint is not the original actual 300-update checkpoint')
    checkpoint_sha = frozen.sha256(checkpoint)
    if any(v['files'][str(checkpoint)]['sha256'] != checkpoint_sha for v in (identity, original_identity)):
        raise ValueError('checkpoint SHA mismatch')
    core = Core(adhesion=False)  # constructor/forward only; zero integration steps
    verify_runtime(identity, core)
    mass = float(core.plant.body_subtreemass[core.th])
    if not np.isclose(mass, .002494271478, rtol=0, atol=5e-13):
        raise ValueError('compiled mass differs from expected full-body mass')
    if (core.plant.nq, core.plant.nv, core.plant.nu) != (49, 48, 42):
        raise ValueError('full-body compiled dimensions mismatch')
    full_model = dict(path=str(PAD_XML), sha256=frozen.sha256(PAD_XML), nq=core.plant.nq,
        nv=core.plant.nv, nu=core.plant.nu, mesh_count=core.plant.nmesh,
        mass_native=mass, gravity_native=core.plant.opt.gravity.tolist(), adhesion=False,
        runtime_identity_verified=True, simulation_steps=0,
        scope='derived complete body, 42 leg actuators; biological validity NOT_ESTABLISHED')
    if not np.array_equal(core.plant.opt.gravity, [0., 0., -9801.]):
        raise ValueError('gravity changed')
    del core
    decisions_path = OUT / 'decisions_5ms.npz'
    trace_path = OUT / 'trace_1ms.npz'
    if frozen.sha256(decisions_path) != result['decisions']['sha256'] or frozen.sha256(trace_path) != result['trace']['sha256']:
        raise ValueError('actual output hash mismatch')
    with np.load(decisions_path, allow_pickle=False) as z:
        times, sources = z['time'], z['source']
        if not np.array_equal(sources, np.r_[np.full(960, 2), np.full(1920, 3)]):
            raise ValueError('decisions must be teacher prefix then ANN only')
        start, boundary = float(times[0]), float(times[np.flatnonzero(sources == 3)[0]])
    with np.load(trace_path, allow_pickle=False) as z:
        sources = z['control_source']
        if not np.array_equal(sources, np.r_[np.zeros(1000), np.full(4800, 2), np.full(9600, 3)]):
            raise ValueError('1ms sources must be exactly 0/2/3 in order')
    with np.load(Path(manifest['chunks'][-1]['path']), allow_pickle=False) as z:
        end = float(z['time_after_s'][-1])
    windows = dict(auxiliary=(start, boundary), autonomous=(boundary, end))
    trace_compare = compare_npz(trace_path, RUN / 'EVAL_GRU_ROLE_AUX' / 'trace_1ms.npz')
    decision_compare = compare_npz(decisions_path, RUN / 'EVAL_GRU_ROLE_AUX' / 'decisions_5ms.npz')
    if not trace_compare['all_within_1e_8'] or not decision_compare['all_within_1e_8']:
        raise ValueError('independent replay differs from actual original evaluation')
    # These source claims are narrow: same-run valid teacher and continuous handover.
    metadata = dict(run_id=str(OUT), evidence=[dict(path=str(result_path), sha256=frozen.sha256(result_path),
        claims=['auxiliary_control_valid', 'continuous_handover'] if result['transition']['all_exact'] else [])])
    report = evaluate_directory(OUT / 'native', autonomous_window=windows['autonomous'],
        auxiliary_window=windows['auxiliary'], mass_native=mass, gravity_native=9801.,
        source_manifest=manifest, full_model=PAD_XML, metadata=metadata, max_scalar_samples=2_000_000)
    report['actual_window_labels'] = dict(autonomous='actual substitute ANN after algorithm teacher',
        auxiliary='actual same-run teacher walking, excludes initial 0..1s settle')
    report['compiled_full_model'] = full_model
    report['logical_replay_comparison'] = dict(trace=trace_compare, decisions=decision_compare)
    report['legacy_original_1ms_engineering'] = result['autonomous']
    report['strict_actual_neural_cycles'] = result['autonomous_cycles']
    report['additional_native_accounting'] = extra_native(manifest, windows, mass * 9801.)
    report['provenance'] = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        source_manifest_path=str(result_path), source_manifest_sha256=frozen.sha256(result_path),
        teacher_reference=result['teacher_reference_evidence'],
        measured_reference={k: result['algorithm_prefix']['teacher_metadata']['reference'][k]
                            for k in ('label', 'source_path', 'source_sha256', 'source_trajectory_id')},
        measured_reference_hash_verified=(frozen.sha256(result['algorithm_prefix']['teacher_metadata']['reference']['source_path'])
                                         == result['algorithm_prefix']['teacher_metadata']['reference']['source_sha256']),
        source_hashes={str(RUN / name): frozen.sha256(RUN / name) for name in ('student_wave.py', 'models.py', 'physics.py', 'role_aux_bc.py')},
        control_closure='Actual X238/actions/qpos/qvel/ctrl compared with independent original; verified source uses model.act ANN branch only after teacher; role classifier is TRAIN-only and not called by models.act.',
        private_qdes_scope='StaticTeacher.action only in declared teacher phase; no teacher query in ANN branch',
        source_flag_check='exact 1ms 0/2/3 blocks and 5ms teacher-prefix/ANN-only blocks',
        full_goal_status='NOT_PASSED: legacy foot slip exceeds .5; substitute ANN and derived-body scope do not establish original biological goal',
        unresolved='proposed causal filter, provisional D7 threshold, original control/reference/video/dependency evidence requirements retained')
    output = OUT / 'NATIVE_CHECKS.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    a = report.get('windows', {}).get('autonomous', {})
    print(json.dumps(dict(output=str(output), status=report['status'],
        native_samples=report['integrity']['samples'], windows=windows,
        criteria={k:v['status'] for k,v in report['criteria'].items()},
        native_slip=a.get('slip_ratio_max_forward'), replay_exact=trace_compare['all_exact'] and decision_compare['all_exact']), ensure_ascii=False))
    return 2 if report['status'] == 'ERROR' else 0


if __name__ == '__main__':
    raise SystemExit(main())
