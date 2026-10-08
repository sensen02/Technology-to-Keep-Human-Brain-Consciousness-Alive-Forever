"""Only raw config and CSV are evaluated; uploaded reports are never trusted."""
import threading
import uuid
from .artifacts import digest


class BenchService:
    def __init__(self, store, fallback=None):
        self.store, self.fallback = store, fallback
        self.lock = threading.RLock()

    def latest(self):
        with self.lock:
            result = self.store.read_json('bench_latest.json')
            if result is not None:
                return result
            if self.fallback and self.fallback.exists():
                from .artifacts import strict_loads
                return strict_loads(self.fallback.read_bytes())
            return {'status': 'NOT_RUN', 'readiness_claim': False}

    def evaluate(self, data):
        if not isinstance(data, dict) or set(data) != {'config', 'csv'} or not isinstance(data.get('config'), dict) or not isinstance(data.get('csv'), str):
            raise ValueError('config object and csv string required; reports not accepted')
        from bench.evaluate import evaluate
        with self.lock:
            run_id = uuid.uuid4().hex
            prefix = 'bench/' + run_id + '/'
            config = self.store.write_json(prefix + 'config.json', data['config'])
            csv = self.store.write_bytes(prefix + 'measurements.csv', data['csv'].encode('utf-8'))
            result = evaluate(config, csv)
            result.update(run_id=run_id, input_hash=digest(data))
            self.store.write_json(prefix + 'result.json', result)
            if result['status'] != 'INVALID_INPUT':
                self.store.write_json('bench_latest.json', result)
            return result
