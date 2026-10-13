"""Unit + persisted source consistency tests; no network or model build."""
import gzip
import json
import tempfile
import unittest
from pathlib import Path
import run_touch_transmitter_source_audit as a


class AuditTests(unittest.TestCase):
    def test_root_id_exactness(self):
        root = '720575941550977671'
        self.assertEqual(a.root_id(int(root)), root)
        self.assertEqual(a.root_id(' '+root+' '), root)
        for bad in (float(root), '7.20575941550977671e17', True):
            with self.assertRaises(ValueError):
                a.root_id(bad)
        self.assertIsNone(a.root_id(None))
        self.assertIsNone(a.root_id(''))
        self.assertIsNone(a.root_id(float('nan')))

    def test_null_and_label_mapping(self):
        for value in (None, '', ' ', float('nan')):
            self.assertTrue(a.missing(value))
            self.assertIsNone(a.normalized_nt(value))
        self.assertFalse(a.missing(0.0))
        self.assertEqual(a.normalized_nt('ACH'), 'acetylcholine')
        self.assertEqual(a.normalized_nt('GABA'), 'gaba')
        self.assertEqual(a.normalized_nt('GLUT'), 'glutamate')
        self.assertEqual(a.normalized_nt('novel_label'), 'novel_label')
        # Literal strings are preserved, not confused with actual nulls.
        self.assertFalse(a.missing('NA'))

    def test_stream_header_mapping_and_duplicates(self):
        rid = '720575941550977671'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'fixture.csv.gz'
            with gzip.open(path, 'wt') as f:
                f.write('Predicted NT type,Root ID,Verified NT type,Predicted NT confidence\n')
                f.write('ACH,'+rid+',,0.8\nGABA,'+rid+',,0.6\n')
            rows, ids, summary = a.stream_neurons(path, {rid, '123'})
            self.assertEqual(len(rows[rid]), 2)
            self.assertEqual(rows['123'], [])
            self.assertEqual(ids, {rid})
            self.assertEqual(rows[rid][0]['values']['Verified NT type'], '')
            self.assertEqual(rows[rid][0]['values']['Predicted NT type'], 'ACH')
            self.assertEqual(summary['rows'], 2)

    def test_persisted_consistency(self):
        path = a.OUT/'touch_transmitter_source_audit.json'
        if not path.exists():
            self.skipTest('Run guarded audit first')
        data = json.loads(path.read_text())
        self.assertEqual(data['roots_count'], 450)
        self.assertEqual(len(data['per_root']), 450)
        self.assertEqual(data['source_sha256_before'], data['source_sha256_after'])
        self.assertTrue(data['all_sources_unchanged'])
        self.assertEqual(data['csv']['exact450_histograms']['Verified NT type'], {'<missing>': 450})
        self.assertEqual(data['csv']['exact450_histograms']['Predicted NT type'], {'<missing>': 5, 'ACH': 424, 'GABA': 19, 'GLUT': 2})
        self.assertEqual(data['consistency']['csv_cached_verified_mismatches'], [])
        for row in data['per_root']:
            self.assertEqual(len(row['raw_csv_matches']), 1)
            self.assertTrue(row['raw_csv_verified_equals_cached'])
        for path, digest in data['source_sha256_after'].items():
            self.assertEqual(a.sha(path), digest, path)


if __name__ == '__main__':
    unittest.main()
