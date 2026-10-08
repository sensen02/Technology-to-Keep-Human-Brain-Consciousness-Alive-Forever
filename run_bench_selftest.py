"""Bounded stdlib bench tests. Synthetic data verifies SOFTWARE ONLY."""
import copy
import csv
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import unittest
from bench.evaluate import ROOT, FIELDS, evaluate
from bench.synthetic_fixture import fixture, save

cap = 480 * 1024 * 1024
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (min(cap, hard) if hard != resource.RLIM_INFINITY else cap, hard))
OUT = ROOT / 'outputs' / 'bench'
OUT.mkdir(parents=True, exist_ok=True)


class BenchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=OUT)
        self.addCleanup(self.tmp.cleanup)
        template = json.loads((ROOT / 'bench' / 'config.template.json').read_text())
        self.config, self.rows = fixture(template)

    def run_case(self, config=None, rows=None):
        cp, mp = save(config or self.config, self.rows if rows is None else rows, self.tmp.name)
        return evaluate(cp, mp)

    def test_template_no_data(self):
        r = evaluate(ROOT / 'bench' / 'config.template.json')
        self.assertEqual(r['status'], 'NOT_RUN')
        self.assertFalse(r['readiness_claim'])
        self.assertTrue(all(x['status'] == 'NOT_RUN' for x in r['checks']))

    def test_synthetic_never_readiness(self):
        r = self.run_case()
        self.assertEqual(r['status'], 'SYNTHETIC_ONLY')
        self.assertEqual(r['engineering_evaluation'], 'BENCH_PASS')
        self.assertFalse(r['readiness_claim'])
        self.assertEqual(len(r['checks']), 18)

    def test_missing_measurement(self):
        r = self.run_case(rows=self.rows[:-1])
        self.assertEqual(r['engineering_evaluation'], 'NOT_RUN')

    def test_measured_bench_status_only(self):
        # Temporary software simulation of provenance branch, never an emitted real fixture.
        for row in self.rows:
            row['provenance'] = 'measured'
        r = self.run_case()
        self.assertEqual(r['status'], 'BENCH_PASS')
        self.assertFalse(r['readiness_claim'])
        self.assertEqual(r['live_or_body_validation'], 'NOT_EVALUATED')

    def test_bad_rows(self):
        changes = [('value', 'NaN'), ('value', 'inf'), ('value', ''), ('unit', 'kohm'),
                   ('channel', 'UNKNOWN'), ('frequency_hz', 123), ('schema_version', 'bad'),
                   ('config_version', 'bad'), ('evidence_id', ''), ('reference_id', ''),
                   ('timestamp_utc', 'yesterday'), ('timestamp_utc', '2026-01-01T00:00:00'),
                   ('instrument_id', ''), ('calibration_evidence_id', ''), ('value', -1),
                   ('duration_s', 0)]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                rows = copy.deepcopy(self.rows)
                rows[0][key] = value
                self.assertEqual(self.run_case(rows=rows)['status'], 'INVALID_INPUT')

    def test_metadata_and_numeric_ranges(self):
        changes = [('input_noise_asd', 'bandwidth_high_hz', ''),
                   ('input_noise_asd', 'bandwidth_high_hz', 6000),
                   ('input_noise_asd', 'sample_rate_hz', ''),
                   ('gain', 'injected_amplitude_v_peak', ''),
                   ('phase', 'value', 181), ('crosstalk', 'source_channel', 'CH1'),
                   ('alias_rejection', 'injected_frequency_hz', 1000),
                   ('alias_rejection', 'injected_frequency_hz', 8500),
                   ('saturation_input', 'value', 0.02)]
        for metric, key, value in changes:
            with self.subTest(metric=metric, key=key):
                rows = copy.deepcopy(self.rows)
                next(r for r in rows if r['metric'] == metric)[key] = value
                self.assertEqual(self.run_case(rows=rows)['status'], 'INVALID_INPUT')

    def test_duplicates_mixed_and_empty(self):
        self.assertEqual(self.run_case(rows=self.rows + [self.rows[0]])['status'], 'INVALID_INPUT')
        rows = copy.deepcopy(self.rows)
        rows[0]['provenance'] = 'measured'
        self.assertEqual(self.run_case(rows=rows)['status'], 'INVALID_INPUT')
        self.assertEqual(self.run_case(rows=[])['status'], 'NOT_RUN')

    def test_config_errors(self):
        for mutate in (lambda c: c['geometry'].update(diameter_um=8),
                       lambda c: c.update(channels=['CH1', 'CH1']),
                       lambda c: c.update(test_frequencies_hz=[1000, 1000]),
                       lambda c: c['planned_engineering_thresholds']['gain'].update(min=11, max=9),
                       lambda c: c['planned_engineering_thresholds']['gain'].update(unit='dB'),
                       lambda c: c['planned_engineering_thresholds']['gain'].update(max=float('inf'))):
            c = copy.deepcopy(self.config)
            mutate(c)
            # Nonfinite config is written deliberately to test parser rather than save().
            cp, mp = save(self.config, self.rows, self.tmp.name)
            cp.write_text(json.dumps(c))
            self.assertEqual(evaluate(cp, mp)['status'], 'INVALID_INPUT')

    def test_headers_and_json_duplicate(self):
        cp, mp = save(self.config, self.rows, self.tmp.name)
        original = mp.read_text()
        for header in (','.join(FIELDS[:-1]), ','.join(FIELDS + ['value']), ','.join(reversed(FIELDS))):
            mp.write_text(header + '\n' + original.split('\n', 1)[1])
            self.assertEqual(evaluate(cp, mp)['status'], 'INVALID_INPUT')
        cp.write_text('{"schema_version":"bench-1.0","schema_version":"bench-1.0"}')
        self.assertEqual(evaluate(cp)['status'], 'INVALID_INPUT')

    def test_threshold_failure_and_unconfigured(self):
        self.rows[0]['value'] = 999999
        self.assertEqual(self.run_case()['engineering_evaluation'], 'FAIL')
        self.rows[0]['value'] = 100000
        self.config['planned_engineering_thresholds']['gain'].update(min=None, max=None)
        self.assertEqual(self.run_case()['engineering_evaluation'], 'UNCONFIGURED')

    def test_cli_exit_codes_and_output(self):
        cp, mp = save(self.config, self.rows, self.tmp.name)
        output = Path(self.tmp.name) / 'cli.json'
        cmd = [sys.executable, '-m', 'bench.evaluate', '--config', str(cp), '--measurements', str(mp), '--output', str(output)]
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads(output.read_text())['status'], 'SYNTHETIC_ONLY')
        self.rows[0]['unit'] = 'bad'
        save(self.config, self.rows, self.tmp.name)
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertEqual(json.loads(output.read_text())['status'], 'INVALID_INPUT')


if __name__ == '__main__':
    from datetime import datetime, timezone
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BenchTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'scope': 'bench_software_tests_only', 'synthetic': True,
              'hardware_validation': 'NOT_RUN', 'tests_run': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors),
              'passed': result.wasSuccessful(), 'address_space_cap_bytes': cap,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'test_file': str(Path(__file__).resolve())}
    (OUT / 'selftest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    sys.exit(0 if result.wasSuccessful() else 1)
