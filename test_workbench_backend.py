"""P2 stdlib tests: temporary state, fake tiny workers, no scientific simulation."""
import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from serve_workbench import make_server
from workbench import ArtifactStore, DatasetRegistry, RunManager, BenchService
from workbench.artifacts import digest
from workbench.runs import Conflict, validate_recording

ROOT = Path(__file__).resolve().parent


def report():
    return {'recording_only': True, 'status': 'fake test fixture; not scientific result',
            'fixed_geometry': {'diameter_um': 7, 'pitch_um': 20},
            'chain': {'sampling_Hz': 30000},
            'bands': {name: {'total_V2': 1e-12, 'noise_rms_uV': 1,
                            'baseband_V2': 9e-13, 'alias_V2': 1e-13} for name in ('spike','lfp','wide')},
            'tests': {'passed': 1, 'checks': ['tiny fake worker']}}


def command(delay=.02, raw=None, code=0):
    def build(job, folder):
        text = raw if raw is not None else json.dumps(report())
        script = "import time,pathlib,sys; time.sleep(%r); pathlib.Path(%r).write_text(%r); sys.exit(%r)" % (
            delay, str(folder / 'electrode_recording.json'), text, code)
        return [sys.executable, '-c', script]
    return build


def wait(manager, jid, states=None):
    states = states or {'completed','failed','timed_out','cancelled','interrupted'}
    deadline = time.monotonic() + 5
    with manager.condition:
        while manager.jobs[jid]['state'] not in states:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError('job did not reach expected state: ' + repr(manager.jobs[jid]))
            manager.condition.wait(min(.05, remaining))
    return manager.get(jid)


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = ArtifactStore(self.root / 'state')
        self.managers = []

    def tearDown(self):
        for manager in self.managers:
            if not manager.owner.closed:
                manager.close()
        self.temp.cleanup()

    def manager(self, **kwargs):
        manager = RunManager(self.root, self.store, command_factory=kwargs.pop('command_factory', command()), **kwargs)
        self.managers.append(manager)
        return manager

    def test_atomic_write_failure_preserves_original(self):
        self.store.write_json('result.json', {'old':1})
        with patch('workbench.artifacts.os.replace', side_effect=OSError('injected')):
            with self.assertRaises(OSError):
                self.store.write_json('result.json', {'new':2})
        self.assertEqual(self.store.read_json('result.json'), {'old':1})
        self.assertFalse(list(self.store.root.glob('.pending-*')))
        with self.assertRaises(ValueError):
            self.store.write_json('../escape.json', {})

    def test_cached_manifest_and_independent_clocks(self):
        path = self.root / 'data.js'
        data = {'body': {'times_s':[.05,8.0]}, 'neural':{'times_s':[.01,1.0], 'units':'Hz'}, 'calcium':{'times_s':[0,25]}}
        path.write_text('window.VIEWER_DATA=' + json.dumps(data) + ';')
        registry = DatasetRegistry(path)
        first = registry.manifest()
        a = registry.snapshot(); a['body']['times_s'].clear()
        self.assertEqual(registry.snapshot()['body']['times_s'], [.05,8.0])
        self.assertEqual(registry.parse_count, 1)
        section = registry.section('body', first['source_hash'])
        section['times_s'].clear()
        self.assertEqual(registry.section('body')['times_s'], [.05,8.0])
        self.assertNotIn('neural', registry.section('body'))
        with self.assertRaises(KeyError): registry.section('../private')
        clocks = {d['dataset_id']:d['clock'] for d in first['datasets']}
        self.assertFalse(clocks['body']['synchronized'])
        self.assertNotEqual(clocks['body']['clock_id'], clocks['neural']['clock_id'])
        path.write_text('window.VIEWER_DATA=' + json.dumps(dict(data, extra=1)) + ';')
        self.assertNotEqual(first['source_hash'], registry.manifest()['source_hash'])
        self.assertEqual(registry.parse_count, 2)
        with self.assertRaises(Conflict): registry.section('body', first['source_hash'])

    def test_serial_queue_bound_and_duplicates(self):
        manager = self.manager(autostart=False, max_pending=2)
        first, duplicate = manager.submit({})
        self.assertFalse(duplicate)
        with self.assertRaises(Conflict): manager.submit({})
        second, _ = manager.submit({'idempotency_key':'two'})
        self.assertEqual(manager.submit({'idempotency_key':'two'})[0]['id'], second['id'])
        with self.assertRaises(Conflict): manager.submit({'idempotency_key':'three'})
        manager.start()
        a, b = wait(manager, first['id']), wait(manager, second['id'])
        self.assertEqual((a['state'],b['state']), ('completed','completed'))
        self.assertGreaterEqual(b['started_at'], a['finished_at'])
        self.assertEqual(manager.result(a['id'])['run_id'], a['id'])

    def test_queued_cancel_and_idempotency(self):
        manager = self.manager(autostart=False)
        job, _ = manager.submit({'idempotency_key':'one'})
        self.assertEqual(manager.cancel(job['id'])['state'], 'cancelled')
        self.assertEqual(manager.cancel(job['id'])['state'], 'cancelled')
        self.assertTrue(manager.submit({'idempotency_key':'one'})[1])
        with self.assertRaises(Conflict): manager.result(job['id'])

    def test_running_cancel_and_queue_continues(self):
        manager = self.manager(command_factory=command(.4))
        first, _ = manager.submit({'idempotency_key':'one'})
        second, _ = manager.submit({'idempotency_key':'two'})
        wait(manager, first['id'], {'running'})
        manager.cancel(first['id'])
        self.assertEqual(wait(manager, first['id'])['state'], 'cancelled')
        self.assertEqual(wait(manager, second['id'])['state'], 'completed')
        self.assertFalse(self.store.path('runs/' + first['id'] + '/result.json').exists())

    def test_timeout(self):
        manager = self.manager(command_factory=command(2), timeout=.1)
        job, _ = manager.submit({})
        self.assertEqual(wait(manager, job['id'])['state'], 'timed_out')
        self.assertFalse(manager.recording()['result_available'])

    def test_corrupt_results(self):
        for raw in ('{broken', '{"recording_only":true}', json.dumps(report()).replace('30000', 'NaN')):
            with self.subTest(raw=raw):
                manager = self.manager(command_factory=command(raw=raw))
                job, _ = manager.submit({})
                self.assertEqual(wait(manager, job['id'])['state'], 'failed')
                self.assertIn('error', manager.get(job['id']))
                manager.close()

    def test_units_validation(self):
        bad = report(); bad['bands']['spike']['noise_rms_uV'] = 1e6
        with self.assertRaises(ValueError): validate_recording(bad)
        bad = report(); bad['fixed_geometry']['pitch_um'] = 21
        with self.assertRaises(ValueError): validate_recording(bad)

    def test_nonzero_exit_never_publishes(self):
        manager = self.manager(command_factory=command(code=3))
        job, _ = manager.submit({})
        self.assertEqual(wait(manager, job['id'])['exit_code'], 3)
        with self.assertRaises(Conflict): manager.result(job['id'])

    def test_submission_persistence_failure_rolls_back(self):
        manager = self.manager(autostart=False)
        with patch.object(self.store, 'write_json', side_effect=OSError('disk failure')):
            with self.assertRaises(OSError): manager.submit({})
        self.assertEqual(manager.listing()['jobs'], [])
        self.assertIsNone(manager.latest_id)

    def test_source_change_rejected_before_execution(self):
        manager = self.manager(autostart=False)
        job, _ = manager.submit({})
        (self.root / 'run_electrode_recording.py').write_text('# source changed')
        manager.start()
        self.assertEqual(wait(manager, job['id'])['state'], 'failed')
        self.assertIn('source changed', manager.get(job['id'])['error'])

    def test_publication_failure_marks_dispatcher_unhealthy(self):
        manager = self.manager(autostart=False)
        job, _ = manager.submit({})
        original = self.store.write_json
        def failure(name, value):
            if name.endswith('/result.json'):
                raise OSError('injected publication disk failure')
            return original(name, value)
        with patch.object(self.store, 'write_json', side_effect=failure):
            manager.start()
            self.assertEqual(wait(manager, job['id'])['state'], 'failed')
            self.assertIn('disk failure', manager.worker_error)
            with self.assertRaises(Conflict): manager.submit({})
        self.assertNotIn('bands', manager.recording())

    def test_production_child_limits_and_isolated_output(self):
        # Exercise the real launcher against a tiny script, not the scientific workload.
        script = '''import json, os, resource
from pathlib import Path
OUT = Path(__file__).parent / 'legacy-output'
def main():
    assert resource.getrlimit(resource.RLIMIT_AS)[0] == 2 * 1024**3
    assert all(os.environ[k] == '1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'electrode_recording.json').write_text(PAYLOAD)
'''.replace('PAYLOAD', repr(json.dumps(report())))
        (self.root / 'run_electrode_recording.py').write_text(script)
        manager = RunManager(self.root, self.store)
        self.managers.append(manager)
        job, _ = manager.submit({})
        self.assertEqual(wait(manager, job['id'])['state'], 'completed')
        self.assertFalse((self.root / 'legacy-output').exists())
        self.assertEqual(manager.recording()['run_id'], job['id'])

    def test_missing_result(self):
        manager = self.manager(command_factory=lambda job, folder:[sys.executable, '-c', 'pass'])
        job, _ = manager.submit({})
        self.assertEqual(wait(manager, job['id'])['state'], 'failed')

    def test_old_success_not_latest_while_new_pending_or_failed(self):
        manager = self.manager()
        old, _ = manager.submit({}); wait(manager, old['id'])
        manager.command_factory = command(.2, raw='{}')
        new, _ = manager.submit({})
        current = manager.recording()
        self.assertEqual(current['run_id'], new['id'])
        self.assertEqual(current['previous_run_id'], old['id'])
        self.assertTrue(current['stale_result'])
        self.assertNotIn('bands', current)
        wait(manager, new['id'])
        self.assertEqual(manager.recording()['status'], 'FAILED')
        self.assertEqual(manager.result(old['id'])['run_id'], old['id'])

    def test_restart_recovers_queue_interrupts_running(self):
        manager = self.manager(autostart=False)
        first, _ = manager.submit({'idempotency_key':'first'})
        second, _ = manager.submit({'idempotency_key':'second'})
        manager.jobs[first['id']]['state'] = 'running'; manager._persist()
        manager.close()
        restarted = self.manager()
        self.assertEqual(restarted.get(first['id'])['state'], 'interrupted')
        self.assertEqual(wait(restarted, second['id'])['state'], 'completed')
        self.assertTrue(restarted.submit({'idempotency_key':'second'})[1])

    def test_shutdown_preserves_pending_queue(self):
        manager = self.manager(command_factory=command(2))
        first, _ = manager.submit({'idempotency_key':'first'})
        second, _ = manager.submit({'idempotency_key':'second'})
        wait(manager, first['id'], {'running'})
        manager.close()
        restarted = self.manager()
        self.assertEqual(restarted.get(first['id'])['state'], 'interrupted')
        self.assertEqual(wait(restarted, second['id'])['state'], 'completed')

    def test_corrupt_persistence_quarantined(self):
        self.store.write_bytes('jobs.json', b'{broken')
        manager = self.manager(autostart=False)
        self.assertIn('quarantined', manager.recovery_warning)
        self.assertEqual(manager.latest()['state'], 'idle')
        self.assertEqual(len(list(self.store.root.glob('jobs.corrupt-*.json'))), 1)

    def test_single_manager_owns_state(self):
        self.manager(autostart=False)
        with self.assertRaises(RuntimeError): self.manager(autostart=False)

    def test_tampered_result_rejected(self):
        manager = self.manager(); job, _ = manager.submit({}); wait(manager, job['id'])
        value = manager.result(job['id']); value['run_id'] = 'different'
        self.store.write_json('runs/' + job['id'] + '/result.json', value)
        with self.assertRaises(ValueError): manager.result(job['id'])
        self.assertEqual(manager.recording()['status'], 'INVALID_RESULT')

    def test_bench_invalid_cannot_replace_latest(self):
        bench = BenchService(self.store)
        valid = {'config': json.loads((ROOT / 'bench/config.template.json').read_text()),
                 'csv':(ROOT / 'bench/measurements.template.csv').read_text()}
        result = bench.evaluate(valid)
        self.assertEqual(result['status'], 'NOT_RUN')
        self.assertEqual(bench.evaluate({'config':{},'csv':''})['status'], 'INVALID_INPUT')
        self.assertEqual(bench.latest()['run_id'], result['run_id'])
        with self.assertRaises(ValueError): bench.evaluate({'report':result})


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        (root / 'viewer/modules').mkdir(parents=True)
        (root / 'viewer/data.js').write_text('window.VIEWER_DATA={"bench":{"chain":["legacy"]}};')
        (root / 'viewer/index.html').write_text('test')
        (root / 'viewer/modules/service.js').write_text('export const test=1;')
        (root / 'private.txt').write_text('SECRET')
        (root / 'viewer/modules/layers.js').symlink_to(root / 'private.txt')
        self.server = make_server(root, port=0, command_factory=command(.3))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown(); self.thread.join()
        self.server.server_close(); self.server.workbench.close(); self.temp.cleanup()

    def req(self, path, data=None, headers=None, raw=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        hs = dict(headers or {})
        if data is not None or raw is not None:
            hs.setdefault('Content-Type', 'application/json')
            body = raw if raw is not None else json.dumps(data)
            method = 'POST'
        else:
            body, method = None, 'GET'
        conn.request(method, path, body, hs)
        response = conn.getresponse(); status, result = response.status, response.read()
        conn.close()
        try: result = json.loads(result)
        except ValueError: pass
        return status, result

    def test_security_limits_and_static_allowlist(self):
        self.assertEqual(self.req('/api/status', headers={'Host':'evil.invalid'})[0],403)
        self.assertEqual(self.req('/api/status', headers={'Origin':'http://evil.invalid'})[0],403)
        self.assertEqual(self.req('/api/jobs/recording', {}, {'Origin':'null'})[0],403)
        self.assertEqual(self.req('/api/jobs/recording', {}, {'Content-Type':'text/plain'})[0],415)
        self.assertEqual(self.req('/api/jobs/recording', {}, {'Content-Length':str(2*1024**2+1)})[0],413)
        for path in ('/data.js','/../private.txt','/%2e%2e/private.txt','/modules/private.js','/modules/layers.js'):
            self.assertEqual(self.req(path)[0],404,path)
        self.assertEqual(self.req('/modules/service.js')[0],200)
        self.assertEqual(self.req('/api/jobs/recording', raw='{"idempotency_key":"a","idempotency_key":"b"}')[0],400)
        self.assertEqual(self.req('/api/jobs/recording', raw='[]')[0],400)
        self.assertEqual(self.req('/api/jobs/recording', raw='{"x":NaN}')[0],400)
        self.assertEqual(self.req('/api/jobs/recording', raw='{"x":1e999}')[0],400)

    def test_section_api_snapshot_consistency(self):
        manifest = self.req('/api/manifest')[1]
        bench_url = next(item['data_url'] for item in manifest['datasets'] if item['dataset_id'] == 'bench')
        self.assertIn('?snapshot=' + manifest['source_hash'], bench_url)
        code, bench = self.req(bench_url)
        self.assertEqual(code, 200)
        self.assertEqual(bench, {'chain':['legacy']})
        self.assertEqual(self.req('/api/datasets/../private.txt')[0], 404)
        self.assertEqual(self.req('/api/datasets/unknown')[0], 404)
        self.assertEqual(self.req('/api/datasets/body')[0], 404)
        path = self.server.workbench.view / 'data.js'
        path.write_text('window.VIEWER_DATA={"bench":{"chain":["new"]}};')
        self.assertEqual(self.req(bench_url)[0], 409)
        self.assertEqual(self.req('/api/datasets/bench')[1]['chain'], ['new'])
        self.assertEqual(self.req('/api/datasets/bench?snapshot=x&snapshot=y')[0], 400)

    def test_api_reconnect_idempotency_and_cancel(self):
        self.assertEqual(self.req('/api/status')[1]['service'], 'cell-workbench')
        self.assertEqual(self.req('/api/data')[1]['bench']['chain'], ['legacy'])
        self.assertEqual(self.req('/api/manifest')[1]['schema_version'], 'workbench-manifest-1')
        status, job = self.req('/api/jobs/recording', {'idempotency_key':'browser-reconnect'})
        self.assertEqual(status, 202)
        jid = job['id']
        duplicate = self.req('/api/jobs/recording', {'idempotency_key':'browser-reconnect'})
        self.assertEqual(duplicate[0], 200); self.assertEqual(duplicate[1]['id'], jid)
        self.assertEqual(self.req('/api/jobs/' + jid)[1]['run_id'], jid)
        self.assertEqual(self.req('/api/jobs')[1]['latest_id'], jid)
        self.assertEqual(self.req('/api/jobs/' + jid + '/result')[0],409)
        self.assertEqual(self.req('/api/jobs/' + jid + '/cancel', {})[0],200)
        wait(self.server.workbench.runs, jid)
        self.assertEqual(self.req('/api/recording')[1]['status'], 'CANCELLED')
        self.assertEqual(self.req('/api/jobs/missing')[0],404)


if __name__ == '__main__':
    unittest.main(verbosity=2)
