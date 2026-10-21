import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from engine.embodied.research_report import atomic_report, guard_snapshot

class ReportTests(unittest.TestCase):
    def test_numpy_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'report.json'
            atomic_report(p,dict(a=np.arange(3),b=np.int64(9)))
            self.assertEqual(json.loads(p.read_text()),dict(a=[0,1,2],b=9))
    def test_bad_encoding_keeps_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'report.json';atomic_report(p,dict(status='RUNNING'))
            for bad in (lambda: None,float('nan')):
                with self.assertRaises((TypeError,ValueError)):
                    atomic_report(p,dict(bad=bad))
                self.assertEqual(json.loads(p.read_text()),dict(status='RUNNING'))
    def test_replace_error_keeps_checkpoint_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'report.json';atomic_report(p,dict(status='RUNNING'))
            with patch('engine.embodied.research_report.os.replace',side_effect=OSError('injected')):
                with self.assertRaises(OSError):atomic_report(p,dict(status='FINAL'))
            self.assertEqual(json.loads(p.read_text()),dict(status='RUNNING'))
            self.assertEqual(list(Path(d).iterdir()),[p])
    def test_guard_callable_status(self):
        for status in ('PASS','FAIL_STOP'):
            calls=[]
            def report(result_status):calls.append(result_status);return dict(result_status=result_status,peak_VmPeak_mb=123.)
            r=guard_snapshot(report,status)
            self.assertEqual(calls,[status]);self.assertEqual(r['result_status'],status)
    def test_bad_guard_interface_rejected(self):
        for value in ({},None,lambda **kwargs:0):
            with self.assertRaises(TypeError):guard_snapshot(value,'FAIL')

if __name__=='__main__':unittest.main()
