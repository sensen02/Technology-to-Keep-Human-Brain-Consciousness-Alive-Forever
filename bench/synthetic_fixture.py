"""PURELY SYNTHETIC software test vectors, never hardware measurements/specifications."""
import copy
import csv
import json
from pathlib import Path
from bench.evaluate import FIELDS, ROOT, UNITS


def fixture(template):
    c = copy.deepcopy(template)
    c['config_version'] = 'SYNTHETIC-software-fixture-1'
    c['channels'] = ['CH1', 'CH2']
    c['geometry'].update(diameter_interpretation='shaft', exposed_active_area_um2=1,
                         active_area_evidence_id='SYNTHETIC-area')
    c['hardware'] = {k: 'SYNTHETIC-' + k for k in c['hardware']}
    c['threshold_rationale_evidence_id'] = 'SYNTHETIC-test-assertions-not-hardware-targets'
    c['measurement_plan_evidence_id'] = 'SYNTHETIC-test-grid'
    values = dict(impedance_magnitude=100000, impedance_phase=-45,
                  input_noise_asd=1e-8, gain=10, phase=-5,
                  crosstalk=-60, alias_rejection=60,
                  saturation_input=0.01, reference_noise_asd=1e-8)
    for metric, value in values.items():
        c['planned_engineering_thresholds'][metric].update(min=value - max(abs(value) * 0.1, 1e-10), max=value + max(abs(value) * 0.1, 1e-10))
    rows = []
    for channel in c['channels']:
        for metric, value in values.items():
            row = dict.fromkeys(FIELDS, '')
            row.update(schema_version=c['schema_version'], config_version=c['config_version'],
                       evidence_id='SYNTHETIC-' + channel + '-' + metric, provenance='synthetic',
                       metric=metric, channel=channel, frequency_hz=1000, value=value,
                       unit=UNITS[metric], sample_rate_hz=10000, reference_id='SYNTHETIC-reference',
                       setup_id='SYNTHETIC-setup', instrument_id='SYNTHETIC-no-instrument',
                       calibration_evidence_id='SYNTHETIC-no-calibration', method='SYNTHETIC-test-vector',
                       duration_s=1, timestamp_utc='2026-01-01T00:00:00Z')
            if metric == 'crosstalk':
                row['source_channel'] = 'CH2' if channel == 'CH1' else 'CH1'
            if metric in ('input_noise_asd', 'reference_noise_asd'):
                row.update(bandwidth_low_hz=100, bandwidth_high_hz=3000)
            if metric in ('gain', 'phase', 'crosstalk', 'alias_rejection', 'saturation_input'):
                row.update(injected_amplitude_v_peak=0.01 if metric == 'saturation_input' else 0.001,
                           injected_frequency_hz=9000 if metric == 'alias_rejection' else 1000)
            rows.append(row)
    return c, rows


def save(config, rows, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cp, mp = directory / 'config.json', directory / 'measurements.csv'
    cp.write_text(json.dumps(config, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with mp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return cp, mp


if __name__ == '__main__':
    c = json.loads((ROOT / 'bench' / 'config.template.json').read_text(encoding='utf-8'))
    paths = save(*fixture(c), ROOT / 'outputs' / 'bench' / 'synthetic')
    print('\n'.join(str(p) for p in paths))
