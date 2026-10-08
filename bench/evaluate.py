"""Recording-only BENCH measurements; no live acquisition or body decoder imports."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / 'outputs' / 'bench'
VERSION = 'bench-1.0'
MAX_BYTES = 16 * 1024 * 1024
MAX_ROWS = 20000
UNITS = {
    'impedance_magnitude': 'ohm', 'impedance_phase': 'deg',
    'input_noise_asd': 'V/sqrt(Hz)', 'gain': 'V/V', 'phase': 'deg',
    'crosstalk': 'dB', 'alias_rejection': 'dB',
    'saturation_input': 'V_peak', 'reference_noise_asd': 'V/sqrt(Hz)',
}
FIELDS = ['schema_version', 'config_version', 'evidence_id', 'provenance',
          'metric', 'channel', 'source_channel', 'frequency_hz', 'value', 'unit',
          'bandwidth_low_hz', 'bandwidth_high_hz', 'sample_rate_hz',
          'injected_amplitude_v_peak', 'injected_frequency_hz',
          'reference_id', 'setup_id', 'instrument_id', 'calibration_evidence_id',
          'method', 'duration_s', 'timestamp_utc']
NUMBERS = ['frequency_hz', 'value', 'bandwidth_low_hz', 'bandwidth_high_hz',
           'sample_rate_hz', 'injected_amplitude_v_peak', 'injected_frequency_hz', 'duration_s']


class ValidationError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise ValidationError(message)


def number(value, name):
    require(not isinstance(value, bool), name + ': boolean is not numeric')
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValidationError(name + ': expected finite number')
    require(math.isfinite(result), name + ': nonfinite number')
    return result


def text(value):
    return isinstance(value, str) and bool(value.strip())


def bounded(path):
    path = Path(path)
    require(path.stat().st_size <= MAX_BYTES, 'input exceeds 16 MiB limit')
    return path


def digest(path):
    h = hashlib.sha256()
    with bounded(path).open('rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def load_config(path):
    with bounded(path).open(encoding='utf-8') as f:
        c = json.load(f, object_pairs_hook=unique_object,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValidationError('nonfinite JSON: ' + x)))
    require(isinstance(c, dict), 'config must be object')
    require(c.get('schema_version') == VERSION, 'config schema_version mismatch')
    require(text(c.get('config_version')), 'missing config_version')
    require(c.get('scope') == 'recording_only_bench', 'scope must be recording_only_bench')
    g = c.get('geometry', {})
    require(isinstance(g, dict), 'geometry must be object')
    require(g.get('diameter_um') == 7 and g.get('pitch_um') == 20, 'fixed geometry must be 7um diameter / 20um pitch')
    require(g.get('diameter_interpretation') in (None, 'shaft', 'exposed_contact'), 'diameter_interpretation must distinguish shaft/exposed_contact')
    if g.get('exposed_active_area_um2') is not None:
        require(number(g['exposed_active_area_um2'], 'active area') > 0, 'active area must be positive')
    require(isinstance(c.get('hardware'), dict), 'missing hardware metadata object')
    channels = c.get('channels')
    require(isinstance(channels, list) and 1 <= len(channels) <= 128 and all(text(x) for x in channels), 'channels must be 1..128 nonempty labels')
    require(len(set(channels)) == len(channels), 'duplicate channels')
    fs = c.get('test_frequencies_hz')
    require(isinstance(fs, list) and 1 <= len(fs) <= 100, 'frequency plan must have 1..100 frequencies')
    fs = [number(x, 'planned frequency') for x in fs]
    require(all(x > 0 for x in fs) and len(set(fs)) == len(fs), 'frequencies must be positive and unique')
    c['test_frequencies_hz'] = fs
    planned_count = len(fs) * (8 * len(channels) + len(channels) * max(1, len(channels) - 1))
    require(planned_count <= MAX_ROWS, 'measurement plan exceeds 20000 checks; split into approved bounded plans')
    thresholds = c.get('planned_engineering_thresholds', {})
    require(isinstance(thresholds, dict), 'thresholds must be object')
    require(set(thresholds) == set(UNITS), 'thresholds must cover exactly the nine defined metrics')
    for metric, unit in UNITS.items():
        t = thresholds[metric]
        require(isinstance(t, dict) and set(t) == {'unit', 'min', 'max'} and t['unit'] == unit, 'invalid threshold/unit: ' + metric)
        for key in ('min', 'max'):
            if t[key] is not None:
                t[key] = number(t[key], metric + ' threshold ' + key)
        require(t['min'] is None or t['max'] is None or t['min'] <= t['max'], 'reversed threshold: ' + metric)
    return c


def config_gaps(c):
    gaps = []
    for key in ('diameter_interpretation', 'exposed_active_area_um2', 'active_area_evidence_id'):
        value = c['geometry'].get(key)
        if (key == 'exposed_active_area_um2' and value is None) or (key != 'exposed_active_area_um2' and not text(value)):
            gaps.append('geometry.' + key)
    for key in ('electrode_id', 'frontend_id', 'reference_description', 'reference_evidence_id'):
        if not text(c['hardware'].get(key)):
            gaps.append('hardware.' + key)
    for key in ('threshold_rationale_evidence_id', 'measurement_plan_evidence_id'):
        if not text(c.get(key)):
            gaps.append(key)
    if len(c['channels']) < 2:
        gaps.append('at least two channels required to evaluate crosstalk')
    return gaps


def import_rows(path, c):
    rows, seen = [], set()
    with bounded(path).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames == FIELDS, 'CSV header must exactly match documented schema/order (no duplicates or extra columns)')
        for line, r in enumerate(reader, 2):
            require(len(rows) < MAX_ROWS, 'CSV exceeds 20000 rows')
            prefix = 'CSV line %d: ' % line
            require(None not in r and all(v is not None and v == v.strip() for v in r.values()), prefix + 'extra/missing fields or surrounding whitespace')
            require(r['schema_version'] == VERSION and r['config_version'] == c['config_version'], prefix + 'version mismatch')
            require(r['provenance'] in ('measured', 'synthetic'), prefix + 'provenance must be measured/synthetic')
            metric = r['metric']
            require(metric in UNITS and r['unit'] == UNITS[metric], prefix + 'unknown metric or wrong unit')
            require(r['channel'] in c['channels'], prefix + 'unknown channel')
            require(r['source_channel'] in c['channels'] and r['source_channel'] != r['channel'] if metric == 'crosstalk' else r['source_channel'] == '', prefix + 'invalid source_channel')
            for key in ('evidence_id', 'reference_id', 'setup_id', 'instrument_id', 'calibration_evidence_id', 'method', 'timestamp_utc'):
                require(text(r[key]), prefix + 'missing ' + key)
            from datetime import datetime
            try:
                stamp = datetime.fromisoformat(r['timestamp_utc'].replace('Z', '+00:00'))
                require(stamp.utcoffset() is not None and stamp.utcoffset().total_seconds() == 0, prefix + 'timestamp must be UTC')
            except ValueError:
                raise ValidationError(prefix + 'invalid UTC timestamp')
            for key in NUMBERS:
                r[key] = number(r[key], prefix + key) if r[key] != '' else None
            freq, value = r['frequency_hz'], r['value']
            require(freq in c['test_frequencies_hz'] and value is not None, prefix + 'missing value or unplanned frequency')
            require(r['duration_s'] is not None and r['duration_s'] > 0, prefix + 'duration_s required positive')
            for key in ('sample_rate_hz', 'injected_amplitude_v_peak', 'injected_frequency_hz'):
                require(r[key] is None or r[key] > 0, prefix + key + ' must be positive')
            if metric in ('impedance_magnitude', 'input_noise_asd', 'reference_noise_asd', 'gain', 'saturation_input'):
                require(value >= 0, prefix + 'negative magnitude/noise/gain/saturation')
            if metric in ('impedance_phase', 'phase'):
                require(-180 <= value <= 180, prefix + 'phase outside [-180,180]')
            if metric in ('input_noise_asd', 'reference_noise_asd'):
                low, high, rate = r['bandwidth_low_hz'], r['bandwidth_high_hz'], r['sample_rate_hz']
                require(low is not None and high is not None and rate is not None and 0 <= low < high <= rate / 2 and low <= freq <= high, prefix + 'ASD requires valid measured bandwidth and sample rate')
            if metric in ('gain', 'phase', 'crosstalk', 'alias_rejection', 'saturation_input'):
                require(r['injected_amplitude_v_peak'] is not None and r['injected_frequency_hz'] is not None and r['sample_rate_hz'] is not None, prefix + 'known injection and sample rate required')
                rate, injected = r['sample_rate_hz'], r['injected_frequency_hz']
                if metric == 'alias_rejection':
                    folded = abs((injected + rate / 2) % rate - rate / 2)
                    require(injected > rate / 2 and math.isclose(folded, freq, rel_tol=1e-8, abs_tol=1e-8), prefix + 'alias injection must exceed Nyquist and fold to measurement frequency')
                else:
                    require(injected == freq and freq < rate / 2, prefix + 'injection must match test frequency below Nyquist')
                if metric == 'saturation_input':
                    require(value > 0 and math.isclose(value, r['injected_amplitude_v_peak'], rel_tol=1e-6), prefix + 'saturation value must match known onset amplitude')
            key = (metric, r['channel'], r['source_channel'], freq)
            require(key not in seen, prefix + 'duplicate measurement key; aggregate replicates explicitly before importing')
            seen.add(key)
            rows.append(r)
    require(len({r['provenance'] for r in rows}) <= 1, 'mixed synthetic/measured CSV forbidden')
    return rows


def evaluate(config_path, csv_path=None):
    result = {'schema_version': VERSION, 'scope': 'recording_only_bench',
              'readiness_claim': False, 'live_or_body_validation': 'NOT_EVALUATED',
              'status': 'INVALID_INPUT', 'errors': [], 'checks': []}
    try:
        c = load_config(config_path)
        result.update(config_version=c['config_version'], config_sha256=digest(config_path),
                      config_path=str(Path(config_path).resolve()), geometry=c['geometry'],
                      planned_engineering_thresholds=c['planned_engineering_thresholds'],
                      configuration_gaps=config_gaps(c))
        rows = import_rows(csv_path, c) if csv_path else []
        result.update(measurement_path=str(Path(csv_path).resolve()) if csv_path else None,
                      measurement_sha256=digest(csv_path) if csv_path else None,
                      measurement_count=len(rows), provenance=rows[0]['provenance'] if rows else 'no_measurements')
        indexed = {(r['metric'], r['channel'], r['source_channel'], r['frequency_hz']): r for r in rows}
        for metric in UNITS:
            for channel in c['channels']:
                sources = [s for s in c['channels'] if s != channel] if metric == 'crosstalk' else ['']
                if not sources:
                    result['checks'].append({'metric': metric, 'channel': channel, 'status': 'NOT_RUN', 'reason': 'no second channel for crosstalk'})
                for source in sources:
                    for freq in c['test_frequencies_hz']:
                        check = dict(metric=metric, channel=channel, source_channel=source, frequency_hz=freq)
                        r = indexed.get((metric, channel, source, freq))
                        t = c['planned_engineering_thresholds'][metric]
                        if r is None:
                            check.update(status='NOT_RUN', reason='missing measurement', measured_value=None)
                        else:
                            check.update(measured_value=r['value'], unit=r['unit'], evidence_id=r['evidence_id'], measurement=r)
                            if t['min'] is None and t['max'] is None:
                                check.update(status='UNCONFIGURED', reason='no planned engineering acceptance threshold')
                            else:
                                passed = (t['min'] is None or r['value'] >= t['min']) and (t['max'] is None or r['value'] <= t['max'])
                                check['status'] = 'PASS' if passed else 'FAIL'
                        result['checks'].append(check)
        statuses = {x['status'] for x in result['checks']}
        engineering = 'FAIL' if 'FAIL' in statuses else 'NOT_RUN' if 'NOT_RUN' in statuses else 'UNCONFIGURED' if result['configuration_gaps'] or 'UNCONFIGURED' in statuses else 'BENCH_PASS'
        result['engineering_evaluation'] = engineering
        result['status'] = 'SYNTHETIC_ONLY' if result['provenance'] == 'synthetic' else engineering
        result['summary_counts'] = {s: sum(x['status'] == s for x in result['checks']) for s in sorted(statuses)}
    except (ValidationError, OSError, ValueError, TypeError, KeyError, csv.Error) as exc:
        result['status'] = 'INVALID_INPUT'
        result['errors'].append(str(exc))
    return result


def main(argv=None):
    # Linux hard address-space cap; no numpy, BANC, engine or waveform allocations.
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = 480 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (min(cap, hard) if hard != resource.RLIM_INFINITY else cap, hard))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--measurements')
    parser.add_argument('--output', default=str(OUTPUT_ROOT / 'status.json'))
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    require(output.is_relative_to(OUTPUT_ROOT.resolve()), 'output must be inside ' + str(OUTPUT_ROOT))
    result = evaluate(args.config, args.measurements)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: result.get(k) for k in ('status', 'measurement_count', 'summary_counts', 'errors')}))
    return 2 if result['status'] == 'INVALID_INPUT' else 1 if result['status'] == 'FAIL' else 0


if __name__ == '__main__':
    sys.exit(main())
