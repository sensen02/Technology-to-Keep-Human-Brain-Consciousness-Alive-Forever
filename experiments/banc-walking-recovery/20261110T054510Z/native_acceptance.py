"""Pure, streaming NativeRecorder analysis; no simulation or policy imports.

D1--D8/D4b refer to GOAL_ROADMAP lines 47--77. D6 signed p95 is
reported literally AND absolute/external-accounted diagnostics are separate.
The causal 1ms EMA and D7 .1W load threshold are proposed processing choices,
not a retroactive claim that the roadmap fully froze these choices.
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROADMAP = HERE.parents[2] / 'GOAL_ROADMAP.zh-CN.md'
NATIVE_DT = 12.5e-6
H0 = 1.2072528224791905
REQUIRED = {
    'time_before_s': (), 'time_after_s': (), 'com_before': (3,),
    'com_after': (3,), 'com_velocity_before': (3,), 'com_velocity_after': (3,),
    'loads_native': (6,), 'nonfoot_force_native': (),
    'contact_speed_native_per_s': (6,), 'ground_force_native': (3,),
    'external_total_force_native': (3,),
}
EXTERNAL_CLAIMS = ('real_motion_reference', 'full_body', 'continuous_handover',
                   'source_control_closure', 'same_run_video', 'neural_dependency')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def criterion(value=None, threshold=None, passed=None, reason=None, **extra):
    return dict(status='UNKNOWN' if passed is None else ('PASS' if passed else 'FAIL'),
                value=value, threshold=threshold, reason=reason, **extra)


def _window(value, name):
    if value is None:
        return None
    v = np.asarray(value, dtype=float)
    if v.shape != (2,) or not np.isfinite(v).all() or v[1] <= v[0]:
        raise ValueError(f'{name} must be finite [start,end), end > start')
    return tuple(map(float, v))


def _evidence(metadata):
    """Hash validation is provenance, not semantic verification of a claim."""
    valid, rows, errors = set(), [], []
    for item in metadata.get('evidence', []):
        try:
            p = Path(item['path'])
            if not p.is_absolute() or not p.is_file():
                raise ValueError('evidence needs existing absolute file path')
            actual = sha256(p)
            if actual != item['sha256']:
                raise ValueError('evidence SHA-256 mismatch')
            claims = item.get('claims', [])
            if not isinstance(claims, list) or not all(isinstance(x, str) for x in claims):
                raise ValueError('claims must be a list of strings')
            rows.append(dict(path=str(p.resolve()), sha256=actual, claims=claims))
            valid.update(claims)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            errors.append(str(exc))
    return valid, rows, errors


class _Selected:
    """Only selected scalar channels retained, bounded by an explicit sample cap."""
    def __init__(self, window, cap):
        self.window, self.cap = window, cap
        self.data = {k: array('d') for k in ('support', 'height', 'nonfoot', 'nf_fraction',
                     'loaded_count', 'original', 'external_accounted')}
        self.n = 0
        self.duration = self.freefall_duration = 0.
        self.first = self.last = self.com_start = self.com_end = None
        self.weighted_speed = np.zeros(6)
        self.load_integral = np.zeros(6)
        self.external_peak = 0.
        self.external_abs_integral = np.zeros(3)
        self.boundary_partial = False

    def add(self, before, after, com_before, com_after, loads, nonfoot, speed,
            az_filtered, contact_filtered, original, accounted, external, weight, gravity, tol):
        if self.window is None:
            return
        start, end = self.window
        if after <= start + tol or before >= end - tol:
            return
        if before < start - tol or after > end + tol:
            self.boundary_partial = True
            return
        if self.n >= self.cap:
            raise ValueError('selected scalar sample cap exceeded; choose smaller explicit window')
        dt = after - before
        if self.first is None:
            self.first, self.com_start = before, com_before.copy()
        self.last, self.com_end = after, com_after.copy()
        self.n += 1
        self.duration += dt
        foot = float(loads.sum())
        total = foot + nonfoot
        vals = (foot / weight, min(com_before[2], com_after[2]), nonfoot,
                nonfoot / total if total > 0 else 0.,
                float(np.count_nonzero(loads >= .1 * weight)), original / weight, accounted / weight)
        for key, val in zip(self.data, vals):
            self.data[key].append(float(val))
        if contact_filtered < .2 * weight and az_filtered <= -.5 * gravity:
            self.freefall_duration += dt
        self.load_integral += loads * dt
        self.weighted_speed += loads * speed * dt
        self.external_peak = max(self.external_peak, float(np.linalg.norm(external)))
        self.external_abs_integral += np.abs(external) * dt

    def summary(self, weight, heading, tol):
        if not self.n:
            return dict(valid=False, reason='window has no complete recorded native steps',
                        requested_window_s=self.window, samples=0)
        start, end = self.window
        complete = (not self.boundary_partial and abs(self.first - start) <= tol
                    and abs(self.last - end) <= tol and abs(self.duration - (end - start)) <= tol)
        a = {k: np.frombuffer(v, dtype=np.float64) for k, v in self.data.items()}
        delta = self.com_end[:2] - self.com_start[:2]
        fwd = float(delta @ heading / self.duration)
        lateral = float(delta @ np.array([-heading[1], heading[0]]) / self.duration)
        travel_speed = float(np.linalg.norm(delta) / self.duration)
        per_leg = np.divide(self.weighted_speed, self.load_integral,
                            out=np.full(6, np.nan), where=self.load_integral > 0)
        # Forward denominator avoids lowering slip through sideways drift. Legacy is separate.
        ratios = per_leg / fwd if fwd > 1e-9 else np.full(6, np.nan)
        legacy = per_leg / travel_speed if travel_speed > 1e-9 else np.full(6, np.nan)
        clean = lambda v: [float(x) if np.isfinite(x) else None for x in v]
        return dict(valid=complete, reason=None if complete else 'window boundary not native-aligned or incompletely covered',
                    requested_window_s=list(self.window), observed_window_s=[self.first, self.last],
                    samples=self.n, duration_s=self.duration, freefall_duration_s=self.freefall_duration,
                    foot_support_mean=float(a['support'].mean()), foot_support_p10=float(np.quantile(a['support'], .1)),
                    nonfoot_peak_native=float(a['nonfoot'].max()), nonfoot_fraction_mean=float(a['nf_fraction'].mean()),
                    zero_total_contact_convention='nonfoot/total=0 only when both are zero; D2/D4 detect unloading',
                    com_height_min=float(a['height'].min()),
                    # Auxiliary reference is endpoint mean COM height, not mean of minima.
                    loaded_leg_count_median=float(np.median(a['loaded_count'])),
                    original_D6_signed_p95_W=float(np.quantile(a['original'], .95)),
                    original_D6_absolute_p95_W=float(np.quantile(np.abs(a['original']), .95)),
                    external_accounted_signed_p95_W=float(np.quantile(a['external_accounted'], .95)),
                    external_accounted_absolute_p95_W=float(np.quantile(np.abs(a['external_accounted']), .95)),
                    external_force_peak_native=self.external_peak,
                    external_force_abs_integral_native_s=self.external_abs_integral.tolist(),
                    forward_signed_native_per_s=fwd, lateral_signed_native_per_s=lateral,
                    endpoint_displacement_world_xy=delta.tolist(),
                    per_leg_load_integral_native_s=self.load_integral.tolist(),
                    per_leg_load_weighted_speed_native_per_s=clean(per_leg),
                    per_leg_slip_ratio_forward=clean(ratios),
                    per_leg_slip_ratio_legacy_displacement_norm=clean(legacy),
                    slip_ratio_max_forward=float(np.max(ratios)) if np.isfinite(ratios).all() else None,
                    slip_ratio_max_legacy=float(np.max(legacy)) if np.isfinite(legacy).all() else None)


def evaluate_chunks(chunks, *, autonomous_window, mass_native, gravity_native=9801.,
                    auxiliary_window=None, heading_world_xy=(1., 0.), metadata=None,
                    expected_native_dt=NATIVE_DT, max_scalar_samples=2_000_000,
                    source=None, recording_manifest=None):
    """Evaluate iterable of chunk dictionaries, retaining scalar quantile buffers only.

    Rows use integration intervals [time_before,time_after). Whole steps only.
    The filter is continuous across chunks/windows, initialized at first sample.
    Errors invalidate all criteria instead of partially passing malformed data.
    """
    report = dict(schema='native-acceptance-v1', status='ERROR', errors=[], criteria={},
                  final_acceptance=criterion(reason='physical metrics alone cannot establish the full goal'))
    source = dict(source or {})
    report['source'] = source
    report['integrity'] = dict(samples=0, chunks=0, time_unique=None, gap_count=0, overlap_count=0)
    metadata = dict(metadata or {})
    try:
        auto = _window(autonomous_window, 'autonomous_window')
        if auto is None:
            raise ValueError('explicit autonomous_window is required')
        aux = _window(auxiliary_window, 'auxiliary_window')
        if aux and aux[1] > auto[0] + 1e-10:
            raise ValueError('auxiliary must precede autonomous and must not overlap')
        mass, gravity, dt = map(float, (mass_native, gravity_native, expected_native_dt))
        if not np.isfinite([mass, gravity, dt]).all() or min(mass, gravity, dt) <= 0:
            raise ValueError('positive finite mass, gravity and native dt required')
        if not np.isclose(dt, NATIVE_DT, rtol=0, atol=1e-12):
            raise ValueError('original roadmap requires 12.5us native samples')
        if int(max_scalar_samples) != max_scalar_samples or max_scalar_samples < 1:
            raise ValueError('max_scalar_samples must be positive integer')
        heading = np.asarray(heading_world_xy, float)
        if heading.shape != (2,) or not np.isfinite(heading).all() or np.linalg.norm(heading) <= 0:
            raise ValueError('heading_world_xy must be finite nonzero 2-vector')
        heading = heading / np.linalg.norm(heading)
        weight = mass * gravity
        tol = max(1e-10, dt * 1e-5)
        windows = dict(autonomous=_Selected(auto, max_scalar_samples), auxiliary=_Selected(aux, max_scalar_samples))
        # Mean height for D5 auxiliary reference: midpoint of actual before/after COM.
        height_sum = dict(autonomous=0., auxiliary=0.)
        filtered = None
        alpha = float(-np.expm1(-dt / .001))
        last_time = last_com = last_velocity = None
        first_time, rows, chunk_count, gaps, overlaps = None, 0, 0, 0, 0
        gap_duration, overlap_duration = 0., 0.
        for chunk in chunks:
            chunk_count += 1
            if not isinstance(chunk, dict):
                raise ValueError('chunk must be a field dictionary')
            missing = sorted(set(REQUIRED) - set(chunk))
            if missing:
                raise ValueError('missing required fields: ' + ', '.join(missing))
            arrays = {k: np.asarray(chunk[k]) for k in REQUIRED}
            n = len(arrays['time_before_s'])
            if n == 0:
                raise ValueError('empty native chunk')
            for key, shape in REQUIRED.items():
                v = arrays[key]
                if v.shape != (n,) + shape or v.dtype.kind not in 'fiu' or not np.isfinite(v).all():
                    raise ValueError(f'malformed/nonfinite {key}: {v.shape}, expected {(n,) + shape}')
            if (arrays['loads_native'] < 0).any() or (arrays['nonfoot_force_native'] < 0).any() or (arrays['contact_speed_native_per_s'] < 0).any():
                raise ValueError('loads, nonfoot force and contact speed must be nonnegative')
            for i in range(n):
                row = {k: v[i] for k, v in arrays.items()}
                before, after = float(row['time_before_s']), float(row['time_after_s'])
                if abs(after - before - dt) > tol:
                    raise ValueError('native interval duration differs from expected dt')
                if last_time is not None:
                    difference = before - last_time
                    if difference > tol:
                        gaps += 1
                        gap_duration += difference
                        raise ValueError('native time gap detected')
                    if difference < -tol:
                        overlaps += 1
                        overlap_duration -= difference
                        raise ValueError('duplicate/nonunique or overlapping native time detected')
                    if not np.allclose(row['com_before'], last_com, rtol=1e-9, atol=1e-9):
                        raise ValueError('COM position discontinuity across native rows')
                    if not np.allclose(row['com_velocity_before'], last_velocity, rtol=1e-9, atol=1e-8):
                        raise ValueError('COM velocity discontinuity across native rows')
                if first_time is None:
                    first_time = before
                foot, nf = float(row['loads_native'].sum()), float(row['nonfoot_force_native'])
                contact = foot + nf
                if not np.isclose(row['ground_force_native'][2], contact, rtol=1e-7, atol=max(1e-9, 1e-8 * weight)):
                    raise ValueError('ground vertical force inconsistent with foot+nonfoot contact')
                az = float((row['com_velocity_after'][2] - row['com_velocity_before'][2]) / (after - before))
                external = row['external_total_force_native']
                vector = np.array([az, contact, external[2]], float)
                filtered = vector.copy() if filtered is None else filtered + alpha * (vector - filtered)
                original = mass * filtered[0] - (filtered[1] - weight)
                accounted = original - filtered[2]
                for key, selected in windows.items():
                    old_n = selected.n
                    selected.add(before, after, row['com_before'], row['com_after'], row['loads_native'], nf,
                                 row['contact_speed_native_per_s'], filtered[0], filtered[1], original, accounted,
                                 external, weight, gravity, tol)
                    if selected.n != old_n:
                        height_sum[key] += .5 * float(row['com_before'][2] + row['com_after'][2])
                last_time, last_com, last_velocity = after, row['com_after'].copy(), row['com_velocity_after'].copy()
                rows += 1
        if not rows:
            raise ValueError('no native chunks/rows')
        if recording_manifest is not None:
            if recording_manifest.get('pending_rows', 0) != 0:
                raise ValueError('recorder manifest contains unwritten pending rows')
            if recording_manifest.get('native_samples') != rows:
                raise ValueError('recorder manifest sample count differs from chunks')
        summaries = {k: v.summary(weight, heading, tol) for k, v in windows.items()}
        for key, summary in summaries.items():
            summary['com_height_mean_endpoint_midpoint'] = height_sum[key] / windows[key].n if windows[key].n else None
        claims, evidence, evidence_errors = _evidence(metadata)
        if evidence_errors:
            raise ValueError('invalid external evidence: ' + '; '.join(evidence_errors))
        a, b = summaries['autonomous'], summaries['auxiliary']
        aux_valid = bool(aux and b['valid'] and 'auxiliary_control_valid' in claims and metadata.get('run_id')
                         and b['nonfoot_peak_native'] == 0 and b['freefall_duration_s'] == 0
                         and b['foot_support_mean'] >= .9 and b['foot_support_p10'] >= .5
                         and b['com_height_min'] >= .8 * H0)
        report.update(status='OK', source=source, mass_native=mass, gravity_native=gravity,
                      weight_native=weight, mass_origin='caller supplies compiled whole-body mass; no model compilation here',
                      units='native; SI NOT_ESTABLISHED', windows=summaries,
                      integrity=dict(samples=rows, chunks=chunk_count, recorded_window_s=[first_time, last_time],
                                     time_unique=True, gap_count=gaps, overlap_count=overlaps,
                                     gap_duration_s=gap_duration, overlap_duration_s=overlap_duration,
                                     recorded_duration_s=last_time-first_time, integrated_duration_s=rows*dt,
                                     native_dt_s=dt, manifest_count_verified=recording_manifest is not None,
                                     limitation='without completed recorder manifest, missing trailing chunks cannot be inferred'),
                      processing=dict(filter='causal first-order EMA, initialized at first recorded sample',
                                      time_constant_s=.001, alpha=alpha,
                                      applied_to=['actual COM velocity-difference acceleration', 'total contact force', 'applied external force'],
                                      residual_equivalence='same linear filter on acceleration and all forces; residual formed afterward',
                                      D2_uses='filtered total contact and filtered az; raw duration of matching native intervals',
                                      D6_original='signed residual m*az-(contact-W), p95/W <= .15; no absolute-value substitution',
                                      D6_diagnostic='absolute p95 additionally reported; external-accounted includes physical applied force',
                                      filter_status='PROPOSED: roadmap specifies low-pass but does not freeze exact filter',
                                      D7_load_threshold_W=.1, D7_status='PROVISIONAL: roadmap does not specify loaded-leg threshold',
                                      D5_aux_height='mean of before/after endpoint midpoints; explicit proposed sampling convention',
                                      quantiles='exact numpy linear quantile on retained selected scalar arrays, not all 3D trajectory',
                                      max_scalar_samples_per_window=int(max_scalar_samples), heading_world_xy=heading.tolist()),
                      auxiliary_reference_valid=aux_valid, evidence=dict(run_id=metadata.get('run_id'),
                      hash_verified_claim_files=evidence, semantic_verification='not inferred by evaluator'))
        if not a['valid']:
            for key in ('D1', 'D2', 'D3', 'D4', 'D4b', 'D5', 'D6', 'D7', 'D8'):
                report['criteria'][key] = criterion(reason=a['reason'])
        else:
            c = report['criteria']
            c['D1'] = criterion(a['nonfoot_peak_native'], 0., a['nonfoot_peak_native'] == 0)
            c['D2'] = criterion(a['freefall_duration_s'], 0., a['freefall_duration_s'] == 0,
                                processing='proposed causal filter; not fully frozen')
            c['D3'] = criterion(a['foot_support_mean'], b['foot_support_mean'] - .10 if aux_valid else None,
                                a['foot_support_mean'] >= b['foot_support_mean'] - .10 if aux_valid else None,
                                reason=None if aux_valid else 'no same-run physically valid and externally evidenced auxiliary reference')
            c['D4'] = criterion(a['foot_support_p10'], .5, a['foot_support_p10'] >= .5)
            c['D4b'] = criterion(a['nonfoot_fraction_mean'], .20, a['nonfoot_fraction_mean'] <= .20)
            c['D5'] = criterion(a['com_height_min'], b['com_height_mean_endpoint_midpoint'] - .10 if aux_valid else None,
                                a['com_height_min'] >= b['com_height_mean_endpoint_midpoint'] - .10 if aux_valid else None,
                                reason=None if aux_valid else 'no same-run physically valid and externally evidenced auxiliary reference')
            c['D6'] = criterion(a['original_D6_signed_p95_W'], .15, a['original_D6_signed_p95_W'] <= .15,
                                processing='proposed causal filter; signed literal roadmap formula, extras NOT subtracted')
            provisional_pass = a['loaded_leg_count_median'] >= 3
            c['D7'] = criterion(a['loaded_leg_count_median'], 3., False if not provisional_pass else None,
                                reason='roadmap loaded-leg threshold missing; .1W indicator provisional',
                                provisional_indicator_pass=bool(provisional_pass), loaded_leg_threshold_W=.1)
            c['D8'] = criterion(a['com_height_min'], .8 * H0, a['com_height_min'] >= .8 * H0)
        c = report['criteria']
        report['physical_metric_conjunction'] = criterion(passed=False if any(x['status'] == 'FAIL' for x in c.values()) else None,
            reason='D7 provisional and D2/D6 filter proposed; no fully frozen physical certification')
        external_present = a.get('external_force_peak_native', 0.) > max(1e-12, 1e-12 * weight)
        report['external_force_review'] = dict(present=external_present,
            declared='external_force_declaration' in claims,
            status='REQUIRES_REVIEW' if external_present else 'ZERO_RECORDED',
            note='external-accounted D6 is diagnostic only; original D6 remains unchanged; applied forces do not silently authorize assistance')
        if a['valid']:
            slip = a['slip_ratio_max_forward']
            report['engineering_only'] = dict(label='legacy .2W engineering branch; NOT full goal',
                support=criterion(a['foot_support_mean'], .90, a['foot_support_mean'] >= .90),
                nonfoot=criterion(a['nonfoot_peak_native'], .20 * weight, a['nonfoot_peak_native'] <= .20 * weight),
                com=criterion(a['com_height_min'], 1.20, a['com_height_min'] >= 1.20),
                forward=criterion(a['forward_signed_native_per_s'], .05, a['forward_signed_native_per_s'] >= .05),
                slip=criterion(slip, .5, None if slip is None else slip <= .5,
                               reason='native load-weighted contact speed / signed forward endpoint speed; no-load legs UNKNOWN'),
                legacy_displacement_norm_slip=a['slip_ratio_max_legacy'])
        missing_claims = sorted(set(EXTERNAL_CLAIMS) - claims)
        report['final_acceptance'] = criterion(passed=False if report['physical_metric_conjunction']['status'] == 'FAIL' else None,
            reason='full goal requires external control/closure/video/reference evidence and frozen processing review; never inferred from physical metrics',
            missing_external_claims=missing_claims, external_forces_require_review=external_present)
    except (ValueError, TypeError, KeyError, OSError, zipfile.BadZipFile, EOFError) as exc:
        report['errors'].append(str(exc))
        report['integrity'].update(samples=locals().get('rows', 0), chunks=locals().get('chunk_count', 0),
            gap_count=locals().get('gaps', 0), overlap_count=locals().get('overlaps', 0),
            gap_duration_s=locals().get('gap_duration', 0.), overlap_duration_s=locals().get('overlap_duration', 0.),
            time_unique=False if locals().get('overlaps', 0) else None,
            validation_complete=False)
        report['status'] = 'ERROR'
        report['criteria'] = {k: criterion(reason='malformed/incomplete data: ' + str(exc))
                              for k in ('D1', 'D2', 'D3', 'D4', 'D4b', 'D5', 'D6', 'D7', 'D8')}
        report['final_acceptance'] = criterion(reason='input error; no acceptance possible')
    return report


def evaluate_directory(input_dir, *, source_manifest=None, full_model=None, **kwargs):
    """Hash each file and load required NPZ arrays one chunk at a time.

    Optional source_manifest is the NativeRecorder summary dict or JSON path.
    Missing manifest does not imply a complete/trailing-chunk-verified recording.
    """
    root = Path(input_dir).expanduser().resolve()
    source = dict(input_dir=str(root), chunks=[], evaluator=dict(path=str(Path(__file__).resolve()),
                  sha256=sha256(__file__)), roadmap=dict(path=str(ROADMAP), sha256=sha256(ROADMAP), lines=[47, 77]))
    manifest = source_manifest
    try:
        if isinstance(manifest, (str, Path)):
            p = Path(manifest).resolve()
            source['manifest'] = dict(path=str(p), sha256=sha256(p))
            manifest = json.loads(p.read_text())
        if full_model is not None:
            p = Path(full_model).expanduser().resolve(strict=True)
            source['full_model'] = dict(path=str(p), sha256=sha256(p), note='identity only; mass/gravity caller-supplied')
        paths = sorted(root.glob('native_*.npz'))
        if not root.is_dir() or not paths:
            raise ValueError('input directory has no native_*.npz chunks')
        if [p.name for p in paths] != [f'native_{i:06d}.npz' for i in range(len(paths))]:
            raise ValueError('chunk filenames not a contiguous zero-based sequence')
        expected = None
        if manifest is not None:
            expected = manifest.get('chunks')
            if not isinstance(expected, list) or len(expected) != len(paths):
                raise ValueError('manifest chunk count mismatch')
        def chunks():
            for i, p in enumerate(paths):
                digest = sha256(p)
                if expected is not None:
                    if Path(expected[i]['path']).name != p.name or digest != expected[i]['sha256']:
                        raise ValueError('manifest chunk filename/hash mismatch')
                # Guard even one malicious/unbounded compressed field before allocation.
                with zipfile.ZipFile(p) as archive:
                    sizes = {item.filename: item.file_size for item in archive.infolist()}
                    if any(sizes.get(k + '.npy', 0) > 64 * 1024 * 1024 for k in REQUIRED):
                        raise ValueError('individual required chunk field exceeds 64MiB allocation cap')
                with np.load(p, allow_pickle=False) as z:
                    data = {k: z[k] for k in REQUIRED if k in z.files}
                n = len(data.get('time_before_s', []))
                if expected is not None and expected[i].get('rows') != n:
                    raise ValueError('manifest per-chunk row count mismatch')
                source['chunks'].append(dict(path=str(p.resolve()), sha256=digest, rows=n))
                yield data
                if sha256(p) != digest:
                    raise ValueError('chunk changed during evaluation')
        return evaluate_chunks(chunks(), source=source, recording_manifest=manifest, **kwargs)
    except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as exc:
        return dict(schema='native-acceptance-v1', status='ERROR', source=source, errors=[str(exc)],
                    criteria={k: criterion(reason='source error') for k in ('D1','D2','D3','D4','D4b','D5','D6','D7','D8')},
                    final_acceptance=criterion(reason='source error'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--out', required=True, help='new output path; refuses overwrite')
    parser.add_argument('--autonomous', nargs=2, type=float, required=True, metavar=('START', 'END'))
    parser.add_argument('--auxiliary', nargs=2, type=float, metavar=('START', 'END'))
    parser.add_argument('--mass', type=float, help='compiled whole-body subtree mass, native units')
    parser.add_argument('--weight', type=float, help='native body weight; mass=weight/g')
    parser.add_argument('--g', type=float, default=9801.)
    parser.add_argument('--full-model', '--model', dest='full_model', help='XML identity, read/hash only')
    parser.add_argument('--heading', nargs=2, type=float, default=(1., 0.))
    parser.add_argument('--metadata', help='external evidence JSON, paths+hashes+claims')
    parser.add_argument('--manifest', help='NativeRecorder completed summary JSON')
    parser.add_argument('--max-scalar-samples', type=int, default=2_000_000)
    args = parser.parse_args(argv)
    if args.mass is None and args.weight is None:
        parser.error('--mass or --weight is required; model is not compiled by this pure evaluator')
    if args.g <= 0 or (args.mass is not None and args.weight is not None and not np.isclose(args.mass * args.g, args.weight)):
        parser.error('mass/weight/g inconsistent')
    out = Path(args.out).expanduser().resolve()
    if out.exists():
        parser.error('refusing to overwrite an existing output')
    metadata = json.loads(Path(args.metadata).read_text()) if args.metadata else None
    report = evaluate_directory(args.input_dir, autonomous_window=args.autonomous,
        auxiliary_window=args.auxiliary, mass_native=args.mass if args.mass is not None else args.weight / args.g,
        gravity_native=args.g, heading_world_xy=args.heading, metadata=metadata,
        source_manifest=args.manifest, full_model=args.full_model, max_scalar_samples=args.max_scalar_samples)
    if args.metadata:
        p = Path(args.metadata).resolve()
        report.setdefault('source', {})['evidence_metadata'] = dict(path=str(p), sha256=sha256(p))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], final_acceptance=report['final_acceptance'], out=str(out))))
    return 2 if report['status'] == 'ERROR' else 0


if __name__ == '__main__':
    sys.exit(main())
