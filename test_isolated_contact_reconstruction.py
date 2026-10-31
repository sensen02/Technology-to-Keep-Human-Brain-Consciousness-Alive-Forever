"""Pure software fail-closed tests; an import hook forbids MuJoCo."""
import builtins
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import implement_isolated_contact_reconstruction as target


class FakeBackend:
    def __init__(self, out, fail=None):
        self.out = Path(out)
        self.fail = fail
        self.calls = []
        self.data = []

    def observed(self, kind):
        # Inspect DURABLE reservation at the actual backend boundary.
        l = json.loads((self.out/'LEDGER.json').read_text())
        assert l['status'] == 'RUNNING'
        assert l['calls'][-1]['kind'] == kind
        assert l['calls'][-1]['status'] == 'UNKNOWN'
        assert l['attempted'][kind] == sum(k == kind for k in self.calls) + 1
        self.calls.append(kind)
        if self.fail == kind:
            raise RuntimeError('injected backend failure')

    def load_model(self, source):
        self.observed('model_load')
        return 'FAKE'

    def new_data(self, model):
        d = {}
        self.data.append(d)
        return d

    def restore(self, model, data, state):
        assert not data
        data['state'] = state

    def forward(self, model, data):
        self.observed('forward')
        data['done'] = True

    def export(self, model, data):
        return {'done': data['done']}


def states(n=3):
    return [dict(identity={'index': i}) for i in range(n)]


class ProtectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)/'run'
        self.original_import = builtins.__import__
        def guarded(name, *a, **kw):
            if name.split('.')[0] == 'mujoco':
                raise AssertionError('MuJoCo import forbidden in tests')
            return self.original_import(name, *a, **kw)
        self.hook = patch('builtins.__import__', guarded)
        self.hook.start()

    def tearDown(self):
        self.hook.stop()
        self.tmp.cleanup()

    def execute(self, backend, cap=3, persist=target.atomic_json):
        return target.run(self.out, {'model_load':1,'forward':cap}, {'test':'fake'}, 'NO MODEL', states(), backend, persist)

    def test_success_fresh_data_and_reservations(self):
        b = FakeBackend(self.out)
        l = self.execute(b)
        self.assertEqual(l['status'], 'COMPLETE')
        self.assertEqual(len({id(d) for d in b.data}), 3)
        self.assertEqual(b.calls, ['model_load','forward','forward','forward'])
        self.assertEqual(len(list(self.out.glob('STATE_*.json'))), 3)

    def test_first_state_failure_stops_no_refund(self):
        b = FakeBackend(self.out, 'forward')
        with self.assertRaises(RuntimeError): self.execute(b)
        l = json.loads((self.out/'LEDGER.json').read_text())
        self.assertEqual(b.calls,['model_load','forward'])
        self.assertEqual(l['failed']['forward'],1)
        self.assertEqual(l['attempted']['forward'],1)
        self.assertFalse(list(self.out.glob('STATE_*.json')))
        with self.assertRaises(target.Refused): self.execute(b)
        self.assertEqual(len(b.calls),2)

    def test_planned_count_does_not_block_calls(self):
        b = FakeBackend(self.out)
        ledger = self.execute(b,cap=1)
        self.assertEqual(b.calls,['model_load','forward','forward','forward'])
        self.assertEqual(ledger['completed']['forward'],3)
        self.assertEqual(len(list(self.out.glob('STATE_*.json'))),3)

    def test_zero_planned_counts_are_not_permissions(self):
        b = FakeBackend(self.out)
        ledger = target.run(self.out,{'model_load':0,'forward':0},{},'',states(),b)
        self.assertEqual(b.calls,['model_load','forward','forward','forward'])
        self.assertEqual(ledger['status'],'COMPLETE')

    def test_duplicate_complete_start(self):
        b = FakeBackend(self.out);self.execute(b)
        with self.assertRaises(target.Refused): self.execute(b)
        self.assertEqual(len(b.calls),4)

    def test_existing_unknown_ledger(self):
        self.out.mkdir();(self.out/'LEDGER.json').write_text('{"status":"RUNNING"}')
        b=FakeBackend(self.out)
        with self.assertRaises(target.Refused):self.execute(b)
        self.assertEqual(b.calls,[])

    def test_initial_write_failure_claim_retained(self):
        b=FakeBackend(self.out)
        def fail(p,v):raise OSError('disk failure')
        with self.assertRaises(OSError):self.execute(b,persist=fail)
        self.assertEqual(b.calls,[])
        with self.assertRaises(target.Refused):self.execute(b)

    def test_reservation_write_failure_no_unreserved_call(self):
        b=FakeBackend(self.out)
        def fail(p,v):
            if v.get('attempted',{}).get('forward',0):raise OSError('reserve write failure')
            target.atomic_json(p,v)
        with self.assertRaises(OSError):self.execute(b,persist=fail)
        self.assertEqual(b.calls,['model_load'])
        with self.assertRaises(target.Refused):self.execute(b)

    def test_shard_write_failure_stops_without_repeat(self):
        b=FakeBackend(self.out)
        def fail(p,v):
            if p.name.startswith('STATE_'):raise OSError('shard write failure')
            target.atomic_json(p,v)
        with self.assertRaises(OSError):self.execute(b,persist=fail)
        self.assertEqual(b.calls,['model_load','forward'])
        self.assertEqual(json.loads((self.out/'LEDGER.json').read_text())['status'],'STOPPED')

    def test_completion_write_failure_stays_unknown(self):
        b=FakeBackend(self.out)
        def fail(p,v):
            if v.get('completed',{}).get('forward',0):raise OSError('completion write failure')
            target.atomic_json(p,v)
        with self.assertRaises(OSError):self.execute(b,persist=fail)
        l=json.loads((self.out/'LEDGER.json').read_text())
        self.assertEqual(l['calls'][-1]['status'],'UNKNOWN')
        self.assertEqual(b.calls,['model_load','forward'])
        with self.assertRaises(target.Refused):self.execute(b)

    def test_model_failure_charged(self):
        b=FakeBackend(self.out,'model_load')
        with self.assertRaises(RuntimeError):self.execute(b)
        l=json.loads((self.out/'LEDGER.json').read_text())
        self.assertEqual(b.calls,['model_load'])
        self.assertEqual(l['failed']['model_load'],1)
        self.assertEqual(l['attempted']['model_load'],1)

    def test_second_state_failure_no_third(self):
        b=FakeBackend(self.out)
        original=b.forward
        def fail_second(m,d):
            if b.calls.count('forward')==1:b.fail='forward'
            original(m,d)
        b.forward=fail_second
        with self.assertRaises(RuntimeError):self.execute(b)
        self.assertEqual(b.calls,['model_load','forward','forward'])
        self.assertEqual(len(list(self.out.glob('STATE_*.json'))),1)

    def test_plot_failure_is_read_only(self):
        b=FakeBackend(self.out);self.execute(b);before=list(b.calls)
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.out.iterdir()}
        def plot_only(path):
            import summarize_isolated_contact_results as post
            summary=post.summarize(path)
            assert summary['saved_endpoints']==3
            raise RuntimeError('plot failed')
        with self.assertRaises(RuntimeError):plot_only(self.out)
        self.assertEqual(before,b.calls)
        self.assertEqual(hashes,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.out.iterdir()})

if __name__=='__main__':unittest.main(verbosity=2)
