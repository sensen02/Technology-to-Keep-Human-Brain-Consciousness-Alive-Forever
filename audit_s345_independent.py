"""Independent S345 archive audit (S1/P3; no engine imports or simulation).

CLI: --output DIR [--verify]. Normal mode writes DIR/AUDIT_INDEPENDENT.json;
verification writes only DIR/verify/{AUDIT_INDEPENDENT,VERIFICATION}.json. A failing
instrument is a failing audit, not permission to relax a tolerance.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
ARCHIVES = Path('/home/sensen/Desktop/engine修复验收')
S290 = ARCHIVES / 's290-motor-mapping-coverage/20261026T174500Z'
S306 = ARCHIVES / 's306-cpg-source-audit/20261026T203904Z'
S344 = ARCHIVES / 's344-edge-trim-and-engagement-gate/20261027T032801Z'
BIN_MS = 20
FS = 50.0
BAND = (0.5, 4.0)
TRIM_S = 4.0
BOOTSTRAPS = 1000
SEED = 20261026
REQUIRED = ('PREREG.json', 'S345_PREREG_FROZEN', 'ARMS.json', 'CERT_V22.json',
            'TRIM_AUDIT.json', 'EXTENSION_GATE.json', 'DECODER_CRITERIA.json',
            'MOTOR_INTERFACE_V9.json', 'PHASE_DRIFT_V2.json', 'ENGAGEMENT_GATE_V3.json',
            'GAIT_SPEC_V5.json', 'CALL_LEDGER.json', 'OUTCOMES.json', 'AUDIT_INDEPENDENT.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def wrap(degrees):
    return (degrees + 180.0) % 360.0 - 180.0


def distance(a, b):
    return math.fabs(wrap(a - b))


def circular_span(degrees):
    """Length of the smallest covering arc, not max(angle)-min(angle)."""
    a = np.sort(np.mod(np.asarray(degrees, float), 360.0))
    if not len(a):
        return None
    gaps = np.diff(np.concatenate((a, a[:1] + 360.0)))
    return float(360.0 - gaps.max())


def circ(radians):
    z = np.mean(np.exp(1j * radians))
    return float(np.degrees(np.angle(z))), float(np.hypot(z.real, z.imag))


def slope_ci(unwrapped, fs=FS, seed=SEED):
    """OLS residual moving-block bootstrap; resample contiguous 2-second errors.

    Refitting trend + resampled residuals preserves the time design. Resampling
    phase levels themselves would manufacture discontinuities and biased drift.
    """
    y = np.asarray(unwrapped, float)
    t = np.arange(len(y), dtype=float) / fs
    tc = t - t.mean()
    den = float(tc @ tc)
    slope = float(tc @ (y - y.mean()) / den)
    fit = y.mean() + slope * tc
    residual = y - fit
    block = int(round(2.0 * fs))
    if len(y) < 5 * block:
        raise ValueError('fewer than five actual 2s drift blocks')
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(y) - block + 1,
                          size=(BOOTSTRAPS, int(math.ceil(len(y) / block))))
    idx = (starts[:, :, None] + np.arange(block)).reshape(BOOTSTRAPS, -1)[:, :len(y)]
    boot = slope + (residual[idx] @ tc) / den
    return float(np.degrees(slope)), np.degrees(np.quantile(boot, [0.025, 0.975])).tolist()


def band_trim(series, trim_s=TRIM_S):
    sos = signal.butter(3, BAND, btype='bandpass', fs=FS, output='sos')
    for section in sos:
        if np.max(np.abs(np.roots(section[3:]))) > 0.999:
            raise ValueError('unstable independent SOS')
    x = np.asarray(series, float)
    if not np.isfinite(x).all() or np.std(x) <= 0:
        raise ValueError('missing, nonfinite or constant count series')
    k = int(round(trim_s * FS))
    y = signal.sosfiltfilt(sos, x - x.mean())
    if len(y) <= 2 * k + 16:
        raise ValueError('insufficient retained data')
    return y[k:-k]


def metrics(a, b, trim_s=TRIM_S, bootstrap=True):
    ya, yb = band_trim(a, trim_s), band_trim(b, trim_s)
    if len(ya) != len(yb) or min(np.std(ya), np.std(yb)) <= 1e-12:
        raise ValueError('degenerate or unequal retained series')
    ha, hb = signal.hilbert(ya), signal.hilbert(yb)
    pa, pb = np.unwrap(np.angle(ha)), np.unwrap(np.angle(hb))
    delta = np.unwrap(np.angle(ha * np.conj(hb)))
    phase, resultant = circ(delta)
    duration = len(delta) / FS
    # Cycles belong to each oscillating channel, never to their phase DIFFERENCE.
    cycle_a, cycle_b = float((pa[-1] - pa[0]) / (2 * np.pi)), float((pb[-1] - pb[0]) / (2 * np.pi))
    blocks = [circ(delta[i:i + int(2 * FS)])[0]
              for i in range(0, len(delta) - int(2 * FS) + 1, int(2 * FS))]
    # Median endpoints suppress isolated Hilbert-edge samples. Count net half-turn
    # windings per preregistration (180 degrees), also expose physical 360 turns.
    edge = max(3, int(0.2 * FS))
    advance = float(np.degrees(np.median(delta[-edge:]) - np.median(delta[:edge])))
    slips = int(math.floor(math.fabs(advance) / 180.0))
    t = np.arange(len(delta)) / FS
    slope = float(np.degrees(np.polyfit(t, delta, 1)[0]))
    ci = None
    if bootstrap:
        slope, ci = slope_ci(delta)
    status = 'SLIPPING' if slips else ('LOCKED' if math.fabs(slope) * duration <= 45.0 else 'UNRESOLVED')
    return dict(phase_deg=phase, R=resultant, rho=float(np.corrcoef(ya, yb)[0, 1]),
                slope_deg_s=slope, slope_ci_deg_s=ci, bootstrap_replicates=BOOTSTRAPS if bootstrap else 0,
                bootstrap_block_s=2.0, duration_s=duration, retained=len(ya) / len(a),
                cycles=min(cycle_a, cycle_b), cycles_A=cycle_a, cycles_B=cycle_b,
                fa=cycle_a / ((len(ya) - 1) / FS), fb=cycle_b / ((len(yb) - 1) / FS),
                blocks=blocks, n_blocks=len(blocks), block_span=circular_span(blocks),
                total_advance_deg=advance, net_winding_turns=advance / 360.0,
                slip_count=slips, slip_period_s=duration / slips if slips else None,
                drift_status=status)


def pool_columns(pre):
    with np.load(S306 / 'TIER_ROOT_IDS.npz', allow_pickle=False) as z:
        roots = z['root_ids'].copy()
    lookup = {int(r): i for i, r in enumerate(roots)}
    by_leg = {}
    with (S290 / 'PER_CHANNEL.csv').open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            leg = (row.get('name') or '')[:2].lower()
            for root in json.loads(row.get('pool_root_ids') or '[]'):
                if int(root) in lookup:
                    by_leg.setdefault(leg, set()).add(lookup[int(root)])
    cols = {key: [i for leg in legs for i in sorted(by_leg.get(leg, set()))]
            for key, legs in pre['tripod'].items()}
    if set(cols) != {'A', 'B'} or not cols['A'] or not cols['B'] or set(cols['A']) & set(cols['B']):
        raise ValueError('invalid or overlapping tripod pool columns')
    return cols, len(roots)


def unpack(path):
    with np.load(path, allow_pickle=False) as z:
        shape = tuple(int(x) for x in z['shape'])
        packed = z['packed']
        if len(shape) != 2 or packed.dtype != np.uint8 or packed.shape != (shape[0], (shape[1] + 7) // 8):
            raise ValueError('invalid packed spike shape or dtype')
        return np.unpackbits(packed, axis=1)[:, :shape[1]]


def binned(spikes, columns):
    raw = spikes[:, columns].sum(axis=1, dtype=np.float64)
    if len(raw) % BIN_MS:
        raise ValueError('partial count bin')
    return raw.reshape(-1, BIN_MS).sum(axis=1)


def audit(output):
    checks = []
    def check(name, passed, detail=None):
        checks.append(dict(name=name, passed=bool(passed), detail=detail))
    pre = load_json(output / 'PREREG.json')
    specs = pre.get('arm_specs', {})
    check('ten_preregistered_arms', len(specs) == 10 and 'B25_s1' in specs, list(specs))
    columns, n_neurons = pool_columns(pre)
    arms, comparisons, errors = {}, {}, {}
    main = load_json(output / 'ARMS.json') if (output / 'ARMS.json').is_file() else {}
    for name, spec in specs.items():
        try:
            spikes = unpack(output / f'SPK_{name}.npz')
            check(f'{name}.archive_shape', spikes.shape == (spec['ticks'], n_neurons), list(spikes.shape))
            a, b = binned(spikes, columns['A']), binned(spikes, columns['B'])
            result = metrics(a, b)
            result.update(ticks=int(len(spikes)), archive_sha256=sha(output / f'SPK_{name}.npz'),
                          supplementary=(name == 'B25_s1'),
                          role_note=('Supplementary tenth budget arm; original table listed nine arms' if name == 'B25_s1' else 'preregistered table arm'),
                          rates={key: float(spikes[:, ix].sum() * 1000.0 / (len(spikes) * len(ix))) for key, ix in columns.items()})
            result['plateau'] = {str(tr): metrics(a, b, tr, bootstrap=False)['phase_deg'] for tr in (2.0, 3.0, 4.0)}
            result['plateau_span'] = max(distance(x, result['phase_deg']) for x in result['plateau'].values())
            result['eligible'] = result['cycles'] >= 8 and result['retained'] >= 0.4
            arms[name] = result
            check(f'{name}.finite_metrics', all(math.isfinite(result[k]) for k in ('phase_deg', 'rho', 'R', 'cycles', 'slope_deg_s')))
            check(f'{name}.bootstrap_interval', result['slope_ci_deg_s'][0] <= result['slope_ci_deg_s'][1])
            if name in main and main[name].get('phase'):
                ref = main[name]
                cmp = dict(phase_delta_deg=distance(result['phase_deg'], ref['phase']['phase_deg']),
                           rho_delta=math.fabs(result['rho'] - ref['rho']['value']))
                comparisons[name] = cmp
                check(f'{name}.signed_phase_tolerance', cmp['phase_delta_deg'] <= 8.0, cmp)
                check(f'{name}.rho_tolerance', cmp['rho_delta'] <= 0.05, cmp)
                check(f'{name}.eligibility_consistency', bool(ref.get('eligible')) == result['eligible'])
                # Different filters permit small frequency differences, not difference-frequency cycles.
                check(f'{name}.main_cycles_correct', isinstance(ref['phase'].get('cycles'), (float, int)) and
                      math.fabs(ref['phase']['cycles'] - result['cycles']) <= 1.0,
                      dict(main=ref['phase'].get('cycles'), independent=result['cycles']))
            else:
                check(f'{name}.main_result_present', False)
        except (OSError, KeyError, ValueError, TypeError) as exc:
            errors[name] = str(exc)
            check(f'{name}.archive_recomputed', False, str(exc))
    check('complete_arm_coverage', set(arms) == set(specs) and len(arms) == 10, errors)
    return dict(schema='s345.independent.v1', round='S345', scope='S1/P3 offline instrument audit; no physics',
                methods=dict(bin_ms=BIN_MS, filter='SOS Butterworth zero-phase', order=3, band_hz=list(BAND),
                             trim_each_side_s=TRIM_S, phase='signed circular mean of Hilbert A minus B',
                             drift='OLS with residual moving-block bootstrap', bootstrap_block_s=2.0,
                             bootstrap_replicates=BOOTSTRAPS, cycles='minimum channel phase advance / 2pi',
                             block_span='minimal circular covering arc', slip_count='robust net 180deg windings, not path length'),
                inputs=dict(prereg_sha256=sha(output / 'PREREG.json'), roots_sha256=sha(S306 / 'TIER_ROOT_IDS.npz'),
                            pools_sha256=sha(S290 / 'PER_CHANNEL.csv')),
                arms=arms, comparison=comparisons, errors=errors, checks=checks,
                passed=all(c['passed'] for c in checks))


def ast_checks(path):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    findings = dict(constant_branches=[], roll=[], arccos=[], unsigned_phase=[], main_import=[])
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp, ast.While)):
            test = node.test
            literal = isinstance(test, ast.Constant)
            try:
                ast.literal_eval(test)
                literal = True
            except (ValueError, TypeError, SyntaxError):
                pass
            if literal:
                findings['constant_branches'].append(node.lineno)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([node.module or ''] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names])
            if any('analyze_s345' in x for x in names):
                findings['main_import'].append(node.lineno)
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else (node.func.id if isinstance(node.func, ast.Name) else '')
            if name in ('roll', 'arccos', 'acos'):
                findings['roll' if name == 'roll' else 'arccos'].append(node.lineno)
            if name in ('abs', 'absolute', 'fabs') and node.args:
                text = ast.unparse(node.args[0]).lower()
                # Circular DISTANCE uses abs(wrap(a-b)); it must not be confused
                # with discarding the SIGN of an individual phase measurement.
                if any(s in text for s in ('phase_deg', "['phase']", 'circ(', 'angle(')) and not any(s in text for s in ('wrap(', 'distance(')) and not (isinstance(node.args[0], ast.BinOp) and isinstance(node.args[0].op, ast.Sub)):
                    findings['unsigned_phase'].append(node.lineno)
    return findings


def verify(output, report):
    checks = report['checks']
    def check(name, passed, detail=None):
        checks.append(dict(name=name, passed=bool(passed), detail=detail))
    def read(name):
        try:
            return load_json(output / name)
        except (OSError, ValueError) as exc:
            check('read.' + name, False, str(exc))
            return {}
    pre = read('PREREG.json')
    th, specs = pre.get('thresholds', {}), pre.get('arm_specs', {})
    for name in REQUIRED:
        check('required.' + name, (output / name).is_file())
    check('trim_rule_equals_four', th.get('trim_s') == max(1.0, 2.0 / BAND[0]) == 4.0)
    check('fixed_comparison_tolerances', th.get('phase_tol') == 8.0 and th.get('rho_tol') == 0.05)
    check('all_arm_ticks_20000', len(specs) == 10 and all(s.get('ticks') == 20000 for s in specs.values()))
    # Frozen copies are the verification authority, never mutable root source.
    manifest = read('MANIFEST_SHA256.json')
    manifest = manifest.get('files', manifest)
    manifest_ok = bool(manifest)
    hashes = {}
    for key, value in manifest.items():
        if not isinstance(value, str):
            manifest_ok = False
            continue
        path = Path(key)
        path = path if path.is_absolute() else output / path
        try:
            inside = path.resolve().is_relative_to(output.resolve())
            valid = inside and path.is_file() and sha(path) == value
        except OSError:
            valid = False
        hashes[str(path)] = valid
        manifest_ok = manifest_ok and valid
    check('manifest_all_hashes', manifest_ok, hashes)
    for name in REQUIRED:
        check('manifest_covers.' + name, str(output / name) in manifest or name in manifest)
    for name in specs:
        spike_name = f'SPK_{name}.npz'
        check('manifest_covers.' + spike_name, str(output / spike_name) in manifest or spike_name in manifest)
    scripts = sorted(p for p in output.rglob('*.py') if 'verify' not in p.relative_to(output).parts)
    check('frozen_main_and_auditor_present', {'analyze_s345.py', 'audit_s345_independent.py'} <= {p.name for p in scripts})
    for path in scripts:
        check('script_manifest.' + path.name, str(path) in manifest or path.name in manifest)
        try:
            findings = ast_checks(path)
            for kind, hits in findings.items():
                if kind != 'main_import' or path.name != 'analyze_s345.py':
                    check(f'AST.{path.name}.{kind}', not hits, hits)
        except (SyntaxError, OSError) as exc:
            check('AST.' + path.name, False, str(exc))
    try:
        marker = (output / 'S345_PREREG_FROZEN').read_text(encoding='utf-8')
        for name in ('PREREG.json', 'analyze_s345.py', 'audit_s345_independent.py'):
            path = output / name
            check('prereg_frozen_hash.' + name, path.is_file() and f'sha256({name})={sha(path)}' in marker)
        check('executing_frozen_auditor_identity', (output / 'audit_s345_independent.py').is_file() and sha(Path(__file__)) == sha(output / 'audit_s345_independent.py'))
    except OSError as exc:
        check('prereg_frozen_marker', False, str(exc))
    cert = read('CERT_V22.json')
    cells = cert.get('cells', {})
    check('certification_36_cells', len(cells) == 36)
    expected_grid = {(p, d, rates, df) for p in (150.0, 165.0, 180.0) for d in (0.3, 0.5, 0.7)
                     for rates in ((105.0, 59.0), (90.0, 80.0)) for df in (0.0, 0.02)}
    actual_grid = {(c.get('phase'), c.get('duty'), tuple(c.get('rates', [])), c.get('df')) for c in cells.values()}
    check('certification_preregistered_grid_coverage', actual_grid == expected_grid)
    check('certification_declared_n100_rows20000', cert.get('cert_seeds') == 100 and cert.get('rows') == 20000)
    check('certification_no_empty', bool(cells) and cert.get('empty_cells') == [])
    for key, cell in cells.items():
        check('cert.' + key + '.sample_count', cell.get('n') == 100 and cell.get('rows') == 20000)
        bias, resultant = cell.get('bias_deg'), cell.get('R')
        finite = (isinstance(bias, (int, float)) and math.isfinite(bias) and -180.0 <= bias <= 180.0 and
                  isinstance(resultant, (int, float)) and math.isfinite(resultant) and 0 < resultant <= 1.0)
        instrument_pass = finite and math.fabs(bias) <= 5.0 and resultant >= 0.9
        check('cert.' + key + '.not_degenerate', finite, cell)
        check('cert.' + key + '.declared_result_truthful', finite and cell.get('passes') == instrument_pass,
              dict(instrument_pass=instrument_pass, declared=cell.get('passes')))
    certificate_failed = not cells or any(c.get('passes') is not True for c in cells.values())
    report['certification'] = dict(passed=not certificate_failed,
                                   failed_cells=[k for k, c in cells.items() if c.get('passes') is not True],
                                   note='A truthful failed certificate is valid archival evidence, never engagement certification.')
    ledger = read('CALL_LEDGER.json')
    ticks = sum(s.get('ticks', 0) for s in specs.values())
    expected = dict(tier_build=1, select_touch_tier=2, edges_build=1, banc_load=1,
                    network_init=len(specs), control_tick=ticks, tier_advance=0,
                    neural_substep=2 * ticks, physics_step=0, model_load=0, render=0, decoder_eval=4)
    for key, value in expected.items():
        check('actual_counter.' + key, ledger.get(key) == value, dict(expected=value, actual=ledger.get(key)))
    check('prereg_counter_contract', pre.get('counters') == expected)
    frozen_audit = read('AUDIT_INDEPENDENT.json')
    check('frozen_audit_schema', frozen_audit.get('schema') == report['schema'])
    check('frozen_audit_recompute_identity', frozen_audit.get('arms') == report['arms'] and frozen_audit.get('inputs') == report['inputs'])
    outcomes, main = read('OUTCOMES.json'), read('ARMS.json')
    check('outcome_counter_consistency', outcomes.get('counters') == ledger)
    check('archive_tick_total', sum(r['ticks'] for r in report['arms'].values()) == ticks)
    check('main_arm_coverage', set(main) == set(specs))
    for name, arm in main.items():
        ph = arm.get('phase') or {}
        ci = ph.get('slope_ci_deg_s', ph.get('slope_ci'))
        check(name + '.main_bootstrap_ci', isinstance(ci, list) and len(ci) == 2 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in ci) and ci[0] <= ci[1], ci)
        if name in report['arms']:
            check(name + '.main_circular_block_span', ph.get('block_span') is not None and math.fabs(ph['block_span'] - circular_span(ph.get('blocks', []))) <= 1e-7)
            independent = report['arms'][name]
            check(name + '.main_drift_classification', arm.get('drift_status') == independent['drift_status'])
            check(name + '.main_slip_count_consistency', ph.get('slip_count') == independent['slip_count'],
                  dict(main=ph.get('slip_count'), independent=independent['slip_count']))
            check(name + '.main_tick_consistency', arm.get('ticks') == independent['ticks'])
            check(name + '.main_rates_exact', all(isinstance(arm.get('rates', {}).get(key), (int, float)) and
                  math.fabs(arm['rates'][key] - value) <= 1e-9 for key, value in independent['rates'].items()))
        nr = arm.get('rho', {}).get('null', {})
        check(name + '.null_actual_1000', nr.get('n') == 1000 and nr.get('nulls') == 1000)
        value, p1 = arm.get('rho', {}).get('value'), nr.get('p1')
        criterion = isinstance(value, (int, float)) and isinstance(p1, (int, float)) and value <= p1 and math.fabs(value - p1) > 1e-9
        check(name + '.rho_criterion_consistency', arm.get('rho', {}).get('criterion') == criterion)
    extension = read('EXTENSION_GATE.json').get('pairs', {})
    pairs = pre.get('extension_pairs', {})
    check('extension_pair_coverage', bool(pairs) and set(extension) == set(pairs))
    for name, refname in pairs.items():
        try:
            ref, cur = unpack(S344 / refname), unpack(output / f'SPK_{name}.npz')
            equal = cur.shape[1] == ref.shape[1] and len(cur) > len(ref) and np.array_equal(cur[:len(ref)], ref)
            saved = extension.get(name, {})
            check('extension.' + name + '.bitwise_complete_prefix', equal, dict(reference_rows=len(ref)))
            check('extension.' + name + '.record_consistent', saved.get('tested') is True and saved.get('equal') == equal and saved.get('rows_compared') == len(ref))
        except (OSError, ValueError, KeyError) as exc:
            check('extension.' + name, False, str(exc))
    decoder = read('DECODER_CRITERIA.json')
    check('decoder_threshold_trim_four', decoder.get('thresholds', {}).get('trim_s') == 4.0)
    decoder_arms = decoder.get('arms', {})
    check('decoder_arm_coverage', set(pre.get('gate_arms', {}).get('c2', [])) <= set(decoder_arms) and len(decoder_arms) == 4)
    for name, dec in decoder_arms.items():
        signed_pairs = dec.get('leg_pairs', {})
        cross = []
        for key, value in signed_pairs.items():
            legs = key.split('|')
            if len(legs) == 2 and value is not None and not any(all(leg in ls for leg in legs) for ls in pre['tripod'].values()):
                cross.append(value)
        spans = dec.get('spans', {})
        count = sum(distance(p, 180.0) <= 30.0 for p in cross)
        common = dec.get('common_sat')
        measured = len(cross) == 9 and len(spans) == 6 and isinstance(common, (int, float))
        measured_pass = measured and count == 9 and min(spans.values()) >= 0.4 and common <= 0.05
        check('decoder.' + name + '.actual_nine_cross_pairs', measured, dict(cross=len(cross), spans=len(spans)))
        check('decoder.' + name + '.derived_gate_consistency', measured and dec.get('pass_gate') == measured_pass and dec.get('n_cross_ge150') == count)
        check('decoder.' + name + '.span_consistency', bool(spans) and math.fabs(min(spans.values()) - dec.get('min_leg_span', -1.0)) <= 1e-9)
        check('decoder.' + name + '.finite_common_mode', isinstance(common, (int, float)) and math.isfinite(common) and 0 <= common <= 1)
    # Check actual source call, not a decorative threshold copied to the output.
    try:
        tree = ast.parse((output / 'analyze_s345.py').read_text(encoding='utf-8'))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'trimmed' and n.args and 'ser[' in ast.unparse(n.args[0])]
        check('decoder_actual_filter_trim_four', bool(calls) and all(len(n.args) >= 3 and (ast.unparse(n.args[2]) in ('TRIM', '4.0', '4') or ast.unparse(n.args[2]) == "TH['trim_s']") for n in calls), [ast.unparse(n) for n in calls])
    except (OSError, SyntaxError) as exc:
        check('decoder_actual_filter_trim_four', False, str(exc))
    gates = read('ENGAGEMENT_GATE_V3.json')
    status = gates.get('status', {})
    check('six_four_state_gates', set(status) == {'c1', 'c2', 'c3', 'c4', 'c5', 'c6'} and all(v in ('PASS', 'FAIL', 'INELIGIBLE', 'UNTESTED') for v in status.values()))
    gate_arms = pre.get('gate_arms', {})
    recomputed = {}
    decarms = decoder.get('arms', {})
    for gate, names in gate_arms.items():
        selected = [report['arms'].get(a) for a in names]
        if gate == 'c2':
            missing = any(a not in decarms for a in names)
            ok = all(decarms.get(a, {}).get('pass_gate') is True for a in names)
            recomputed[gate] = 'UNTESTED' if missing else ('PASS' if ok else 'FAIL')
        elif any(a is None for a in selected):
            recomputed[gate] = 'UNTESTED'
        elif any(not a['eligible'] for a in selected):
            recomputed[gate] = 'INELIGIBLE'
        else:
            locked = all(a['drift_status'] == 'LOCKED' for a in selected)
            if gate in ('c1', 'c3'):
                ok = locked and all(a['rho'] < main.get(n, {}).get('rho', {}).get('null', {}).get('p1', -1.0) for n, a in zip(names, selected))
                if gate == 'c1':
                    ok = ok and all(distance(a['phase_deg'], 180.0) <= 30.0 for a in selected)
            elif gate == 'c4':
                periods = [1.0 / a['fa'] for a in selected]
                ok = float(np.std(periods) / np.mean(periods)) <= 0.10
            elif gate == 'c5':
                ok = locked and selected[0]['n_blocks'] >= 5 and selected[0]['block_span'] <= 45.0
            else:
                ok = locked
            recomputed[gate] = 'PASS' if ok else 'FAIL'
        check('gate_state.' + gate, status.get(gate) == recomputed[gate], dict(saved=status.get(gate), independent=recomputed[gate]))
    check('outcomes_gate_consistency', outcomes.get('status') == status and outcomes.get('branch') == gates.get('branch'))
    invalid = any(not c['passed'] for c in checks)
    check('failed_certificate_blocks_handoff', not certificate_failed or gates.get('branch') == 'NOT_JUDGED')
    if invalid or certificate_failed or any(s in ('INELIGIBLE', 'UNTESTED') for s in recomputed.values()):
        expected_branch = 'NOT_JUDGED'
    elif len(recomputed) == 6 and all(s == 'PASS' for s in recomputed.values()):
        expected_branch = 'e1'
    elif recomputed.get('c1') == 'PASS' and recomputed.get('c3') == 'PASS':
        expected_branch = 'e2'
    else:
        expected_branch = 'e3'
    check('branch_instrument_failure_blocks_engagement', gates.get('branch') == expected_branch,
          dict(saved=gates.get('branch'), expected=expected_branch))
    check('at_least_twenty_actual_checks', len(checks) >= 20, len(checks))
    report['gate_recomputed'] = recomputed
    report['passed'] = all(c['passed'] for c in checks)
    return dict(schema='s345.verify.v1', passed=report['passed'], n_checks=len(checks),
                n_failed=sum(not c['passed'] for c in checks), checks=checks,
                independent_report='AUDIT_INDEPENDENT.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    target = output / 'verify' if args.verify else output
    # Normal invocation must never mutate a frozen evidence directory.
    if not args.verify and (output / 'MANIFEST_SHA256.json').exists():
        parser.error('directory is frozen; use --verify (writes only DIR/verify/)')
    target.mkdir(parents=True, exist_ok=True)
    try:
        report = audit(output)
        summary = verify(output, report) if args.verify else None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report = dict(schema='s345.independent.v1', passed=False, arms={},
                      errors={'fatal': str(exc)}, checks=[dict(name='audit_inputs', passed=False, detail=str(exc))])
        summary = dict(schema='s345.verify.v1', passed=False, n_checks=1, n_failed=1, checks=report['checks']) if args.verify else None
    save_json(target / 'AUDIT_INDEPENDENT.json', report)
    if summary is not None:
        save_json(target / 'VERIFICATION.json', summary)
    print(json.dumps(dict(passed=report['passed'], arms=len(report['arms']),
                          checks=len(report['checks']), failed=sum(not x['passed'] for x in report['checks']),
                          report=str(target / 'AUDIT_INDEPENDENT.json')), ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
