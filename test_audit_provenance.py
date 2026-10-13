"""Historical artifact integrity is not live-source execution equivalence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from engine.embodied.audit_provenance import (
    AUDITS, captured_hash, freshness_report, inspect_audit, live_source_freshness,
    sha256, validate_saved_bundle, verify_artifact,
)
from engine.embodied.closure_bundle import publish_bundle

ROOT = Path(__file__).resolve().parent


class ProvenanceUnitTests(unittest.TestCase):
    def test_hash_format_not_authentication(self):
        self.assertEqual(captured_hash('a' * 64), 'a' * 64)
        for invalid in (None, 1, '', 'a' * 63, 'g' * 64, 'A' * 64, 'a' * 64 + '\n'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                captured_hash(invalid)

    def test_live_matched_drift_missing_and_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.py'
            source.write_text('old code')
            hashes = {str(source): sha256(source)}
            self.assertEqual(live_source_freshness(hashes, strict=True)[str(source)]['status'], 'matched')
            source.write_text('new code')
            row = live_source_freshness(hashes)[str(source)]
            self.assertEqual(row['status'], 'drift')
            self.assertFalse(row['current_match'])
            with self.assertRaises(ValueError):
                live_source_freshness(hashes, strict=True)
            source.unlink()
            row = live_source_freshness(hashes)[str(source)]
            self.assertEqual(row['status'], 'missing')
            self.assertIsNone(row['current_sha256'])
            with self.assertRaises(ValueError):
                live_source_freshness(hashes, strict=True)

    def test_saved_evidence_survives_live_change_not_artifact_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.py'
            source.write_text('old code')
            hashes = {str(source): sha256(source)}
            artifact = root / 'traces.npz'
            artifact.write_bytes(b'opaque artifact bytes')
            expected = sha256(artifact)
            bundle = publish_bundle(root / 'output', {'traces.npz': artifact})
            reference = copy.deepcopy(bundle)
            reference['manifest_sha256'] = sha256(Path(bundle['path']) / 'manifest.json')
            # Move the live index too: saved reference must still select old run.
            publish_bundle(root / 'output', {'different.npz': artifact})
            source.write_text('new code')
            self.assertEqual(validate_saved_bundle(reference), bundle)
            verify_artifact(artifact, expected)
            self.assertEqual(live_source_freshness(hashes)[str(source)]['status'], 'drift')
            artifact.write_bytes(b'tampered')
            with self.assertRaises(ValueError):
                verify_artifact(artifact, expected)
            saved = Path(bundle['path']) / 'traces.npz'
            saved.write_bytes(b'tampered')
            with self.assertRaises(ValueError):
                validate_saved_bundle(reference)

    def test_manifest_anchors_reject_tampering_and_current_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / 'data'
            artifact.write_bytes(b'original')
            bundle = publish_bundle(root / 'output', {'data': artifact})
            manifest = Path(bundle['path']) / 'manifest.json'
            digest_ref = dict(path=bundle['path'], run_id=bundle['run_id'],
                              files=bundle['manifest']['files'], manifest_sha256=sha256(manifest))
            validate_saved_bundle(digest_ref)
            wrong = dict(digest_ref, manifest_sha256='0' * 64)
            with self.assertRaises(ValueError):
                validate_saved_bundle(wrong)
            with self.assertRaises(ValueError):
                validate_saved_bundle(dict(bundle, path=str(root / 'output')))
            changed = copy.deepcopy(bundle['manifest'])
            changed['extra'] = 'tampering'
            manifest.write_text(json.dumps(changed))
            for reference in (bundle, digest_ref):
                with self.subTest(reference=reference), self.assertRaises(ValueError):
                    validate_saved_bundle(reference)


class SavedAuditContractTests(unittest.TestCase):
    def test_reports_distinguish_stale_sources_and_unavailable_snapshots(self):
        report = freshness_report(ROOT)
        self.assertEqual(set(report['audits']), set(AUDITS))
        self.assertFalse(report['replay_performed'])
        for name, row in report['audits'].items():
            with self.subTest(audit=name):
                self.assertEqual(row['historical_artifact_integrity'], 'validated')
                self.assertEqual(row['immutable_inputs'], 'matched')
                self.assertEqual(row['source_snapshot'], 'unavailable')
                self.assertFalse(row['original_source_authenticated'])
                self.assertFalse(row['execution_equivalence_claimed'])
                if name != 'neural_duration_probe':
                    self.assertEqual(row['live_source_status'], 'drift')
                    self.assertEqual(row['run_sources'][str(ROOT/'run_neural_muscle_closure.py')]['status'], 'drift')
                for group in ('own_source', 'audit_time_sources', 'run_sources'):
                    for path, entry in row[group].items():
                        self.assertEqual(entry['status'], 'matched' if sha256(path) == entry['captured_sha256'] else 'drift')
                        self.assertEqual(entry['current_match'], entry['status'] == 'matched')
                if row['live_source_status'] != 'matched':
                    with self.assertRaises(ValueError):
                        inspect_audit(ROOT, name, strict=True)
                else:
                    self.assertEqual(inspect_audit(ROOT, name, strict=True), row)


if __name__ == '__main__':
    unittest.main()
