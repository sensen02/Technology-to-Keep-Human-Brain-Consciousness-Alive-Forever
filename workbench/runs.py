"""Persistent bounded serial recording queue with explicit result ownership."""
import copy
import fcntl
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid
from .artifacts import strict_loads, digest

ACTIVE = {'queued', 'running', 'cancelling'}


class Conflict(ValueError):
    pass


def validate_recording(report):
    if not isinstance(report, dict) or report.get('recording_only') is not True:
        raise ValueError('result must be recording-only report')
    geometry = report.get('fixed_geometry', {})
    if geometry.get('diameter_um') != 7 or geometry.get('pitch_um') != 20:
        raise ValueError('result fixed geometry must be 7um / 20um')
    if report.get('chain', {}).get('sampling_Hz') != 30000:
        raise ValueError('result sampling_Hz must be 30000')
    bands = report.get('bands', {})
    for band in ('spike', 'lfp', 'wide'):
        budget = bands.get(band, {})
        for key in ('total_V2', 'noise_rms_uV', 'baseband_V2', 'alias_V2'):
            value = budget.get(key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError('missing/nonfinite/negative result unit field: ' + band + '.' + key)
        if not math.isclose(budget['noise_rms_uV'] ** 2 * 1e-12, budget['total_V2'], rel_tol=1e-5, abs_tol=1e-30):
            raise ValueError('noise RMS uV / variance V2 inconsistency')
        if not math.isclose(budget['baseband_V2'] + budget['alias_V2'], budget['total_V2'], rel_tol=1e-5, abs_tol=1e-30):
            raise ValueError('baseband / alias variance inconsistency')
    tests = report.get('tests', {})
    if not isinstance(tests.get('checks'), list) or not tests['checks'] or tests.get('passed') != len(tests['checks']):
        raise ValueError('result acceptance checks missing/inconsistent')
    return report


class RunManager:
    def __init__(self, root, store, command_factory=None, timeout=120, max_pending=4, autostart=True):
        self.root, self.store = Path(root), store
        self.timeout, self.max_pending = timeout, max_pending
        self.command_factory = command_factory or self._command
        self.condition = threading.Condition(threading.RLock())
        self.stopping = False
        self.thread = None
        self.process = None
        self.owner = self.store.path('manager.lock').open('a+')
        try:
            fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.owner.close()
            raise RuntimeError('workbench state already owned by another service')
        self.recovery_warning = None
        self.worker_error = None
        try:
            saved = store.read_json('jobs.json', {'jobs': [], 'latest_id': None})
            if not isinstance(saved, dict) or not isinstance(saved.get('jobs'), list):
                raise ValueError('invalid job state schema')
            if any(not isinstance(j, dict) or not isinstance(j.get('id'), str) for j in saved['jobs']):
                raise ValueError('invalid persisted jobs')
            self.jobs = {j['id']: j for j in saved['jobs']}
            if len(self.jobs) != len(saved['jobs']):
                raise ValueError('duplicate persisted job ids')
            self.latest_id = saved.get('latest_id')
            if self.latest_id is not None and self.latest_id not in self.jobs:
                raise ValueError('latest id not present in persisted jobs')
            for jid, job in self.jobs.items():
                if not isinstance(jid, str) or len(jid) != 32 or any(c not in '0123456789abcdef' for c in jid):
                    raise ValueError('invalid persisted job id')
                if job.get('state') not in ACTIVE | {'completed', 'failed', 'timed_out', 'cancelled', 'interrupted'}:
                    raise ValueError('unknown persisted job state')
                if job['state'] in ('running', 'cancelling'):
                    job.update(state='interrupted', error='service restarted during execution; resubmit explicitly', finished_at=time.time())
        except (ValueError, KeyError, TypeError) as exc:
            self.recovery_warning = 'corrupt queue quarantined: ' + str(exc)
            source = store.path('jobs.json')
            if source.exists():
                os.replace(source, store.path('jobs.corrupt-' + uuid.uuid4().hex + '.json'))
            self.jobs, self.latest_id = {}, None
        self._persist()
        if autostart:
            self.start()

    def _command(self, job, folder):
        python = self.root / 'venv/bin/python'
        return [str(python if python.exists() else Path(sys.executable)),
                str(Path(__file__).with_name('recording_child.py')), str(self.root), str(folder), str(os.getpid())]

    def _persist(self):
        self.store.write_json('jobs.json', {'schema_version': 1, 'latest_id': self.latest_id, 'jobs': list(self.jobs.values())})

    def start(self):
        with self.condition:
            if self.thread is None:
                self.thread = threading.Thread(target=self._loop, name='workbench-serial', daemon=True)
                self.thread.start()

    def submit(self, data):
        if not isinstance(data, dict) or set(data) - {'idempotency_key'}:
            raise ValueError('recording accepts only optional idempotency_key; parameters are fixed')
        key = data.get('idempotency_key')
        if key is not None and (not isinstance(key, str) or not 1 <= len(key) <= 128):
            raise ValueError('idempotency_key must be 1..128 characters')
        with self.condition:
            if self.stopping or self.worker_error:
                raise Conflict('service stopping or dispatcher unhealthy: ' + str(self.worker_error))
            for job in self.jobs.values():
                if key is not None and job.get('idempotency_key') == key:
                    return copy.deepcopy(job), True
                if key is None and job['state'] in ACTIVE and job.get('idempotency_key') is None:
                    raise Conflict('identical recording already active: ' + job['id'])
            if sum(j['state'] == 'queued' for j in self.jobs.values()) >= self.max_pending:
                raise Conflict('recording queue full')
            jid = uuid.uuid4().hex
            source_hash = self._source_hash()
            job = {'id': jid, 'run_id': jid, 'kind': 'recording', 'state': 'queued', 'submitted_at': time.time(),
                   'idempotency_key': key, 'input_hash': digest({'kind':'recording', 'parameters':'fixed',
                   'source_hash': source_hash}), 'source_hash': source_hash, 'memory_limit_GiB': 2, 'threads': 1, 'timeout_s': self.timeout}
            previous_latest = self.latest_id
            self.jobs[jid] = job
            self.latest_id = jid
            try:
                self._persist()
            except Exception:
                self.jobs.pop(jid)
                self.latest_id = previous_latest
                raise
            self.condition.notify_all()
            return copy.deepcopy(job), False

    def _source_hash(self):
        import hashlib
        h = hashlib.sha256()
        for path in [self.root / 'run_electrode_recording.py', *sorted((self.root / 'engine').glob('electrode_*.py'))]:
            if path.exists():
                h.update(path.name.encode()); h.update(path.read_bytes())
        return h.hexdigest()

    def get(self, jid):
        with self.condition:
            if jid not in self.jobs:
                raise KeyError('unknown job')
            return copy.deepcopy(self.jobs[jid])

    def listing(self):
        with self.condition:
            return {'jobs': copy.deepcopy(list(self.jobs.values())), 'latest_id': self.latest_id,
                    'max_pending': self.max_pending, 'recovery_warning': self.recovery_warning,
                    'worker_error': self.worker_error}

    def latest(self):
        with self.condition:
            return copy.deepcopy(self.jobs.get(self.latest_id, {'state': 'idle'}))

    def cancel(self, jid):
        with self.condition:
            job = self.jobs.get(jid)
            if job is None:
                raise KeyError('unknown job')
            if job['state'] == 'queued':
                job.update(state='cancelled', finished_at=time.time())
            elif job['state'] in ('running', 'cancelling'):
                job['state'] = 'cancelling'
            self._persist()
            self.condition.notify_all()
            return copy.deepcopy(job)

    @staticmethod
    def _kill(process):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()

    def _loop(self):
        try:
            self._dispatch()
        except Exception as exc:
            # An unhealthy disk must not leave a silently dead dispatcher accepting jobs.
            with self.condition:
                self.worker_error = type(exc).__name__ + ': ' + str(exc)
                for job in self.jobs.values():
                    if job['state'] in ('running', 'cancelling'):
                        job.update(state='failed', error='dispatcher/storage failure: ' + self.worker_error,
                                   finished_at=time.time())
                        job.pop('result_url', None)
                        job.pop('result_hash', None)
                try:
                    self._persist()
                except Exception:
                    pass  # State remains visibly unhealthy in memory even when disk is unavailable.
                self.condition.notify_all()

    def _dispatch(self):
        while True:
            with self.condition:
                queued = next((j for j in self.jobs.values() if j['state'] == 'queued'), None)
                if self.stopping:
                    return
                if queued is None:
                    self.condition.wait()
                    continue
                queued.update(state='running', started_at=time.time())
                self._persist()
                jid = queued['id']
            self._execute(jid)

    def _execute(self, jid):
        folder = self.store.path('runs/' + jid)
        log = folder / 'worker.log'
        state, error, code, result = 'failed', None, None, None
        try:
            folder.mkdir(parents=True, exist_ok=False)
            expected_hash = self.get(jid).get('source_hash')
            if expected_hash is None or self._source_hash() != expected_hash:
                raise ValueError('recording source changed since submission; submit a new run')
            env = dict(os.environ)
            for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
                env[name] = '1'
            with log.open('wb') as stream:
                process = subprocess.Popen(self.command_factory(self.get(jid), folder), cwd=self.root, env=env,
                                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                self.process = process
                deadline = time.monotonic() + self.timeout
                while True:
                    with self.condition:
                        cancelled = self.jobs[jid]['state'] == 'cancelling'
                        stopping = self.stopping
                    if cancelled or stopping:
                        self._kill(process)
                        state = 'cancelled' if cancelled else 'interrupted'
                        break
                    if time.monotonic() >= deadline:
                        self._kill(process)
                        state, error = 'timed_out', 'recording timeout'
                        break
                    try:
                        code = process.wait(timeout=.05)
                        break
                    except subprocess.TimeoutExpired:
                        continue
                code = process.returncode
            if state == 'failed' and code == 0:
                path = folder / 'electrode_recording.json'
                if path.is_symlink() or path.stat().st_size > 16 * 1024**2:
                    raise ValueError('result exceeds size limit or is symlink')
                result = validate_recording(strict_loads(path.read_bytes()))
                result.update(run_id=jid, schema_version='workbench-recording-1', result_validated=True,
                              result_units={'noise_rms_uV':'uV','total_V2':'V^2','sampling_Hz':'Hz','diameter_um':'um','pitch_um':'um'})
                state = 'completed'
            elif state == 'failed':
                error = 'worker exited with code ' + str(code)
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        finally:
            self.process = None
        with self.condition:
            job = self.jobs[jid]
            if job['state'] == 'cancelling':
                state, result = 'cancelled', None
            if self.stopping and state == 'completed':
                state, result = 'interrupted', None
            if state == 'completed':
                self.store.write_json('runs/' + jid + '/result.json', result)
                job['result_url'] = '/api/jobs/' + jid + '/result'
                job['result_hash'] = digest(result)
            tail = ''
            if log.exists():
                with log.open('rb') as stream:
                    stream.seek(max(0, log.stat().st_size - 4000))
                    tail = stream.read().decode('utf-8', errors='replace')
            job.update(state=state, finished_at=time.time(), exit_code=code, log_tail=tail)
            if error:
                job['error'] = error
            try:
                self._persist()
            except Exception:
                job.update(state='failed', error='terminal state persistence failed')
                job.pop('result_url', None)
                job.pop('result_hash', None)
                raise
            self.condition.notify_all()

    def result(self, jid):
        job = self.get(jid)
        if job['state'] != 'completed':
            raise Conflict('result unavailable for ' + job['state'])
        report = self.store.read_json('runs/' + jid + '/result.json')
        validate_recording(report)
        if report.get('run_id') != jid or digest(report) != job.get('result_hash'):
            raise ValueError('stored result identity/hash mismatch')
        return report

    def recording(self):
        with self.condition:
            latest = self.latest()
            previous = next((j['id'] for j in reversed(list(self.jobs.values())) if j['state'] == 'completed'), None)
            if latest['state'] == 'completed':
                try:
                    return dict(self.result(latest['id']), stale_result=False)
                except (ValueError, OSError, TypeError) as exc:
                    return {'status':'INVALID_RESULT', 'run_id':latest['id'], 'stale_result':True, 'error':str(exc)}
            return {'status': 'NOT_RUN' if latest['state'] == 'idle' else latest['state'].upper(),
                    'run_id': latest.get('id'), 'job_state': latest['state'], 'stale_result': previous is not None,
                    'previous_run_id': previous, 'result_available': False,
                    'note':'No legacy/global recording is presented as a new run result.'}

    def close(self):
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        if self.thread:
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                raise RuntimeError('recording worker did not stop')
        fcntl.flock(self.owner, fcntl.LOCK_UN)
        self.owner.close()
