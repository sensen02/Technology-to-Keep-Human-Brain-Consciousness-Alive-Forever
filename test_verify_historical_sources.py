"""Synthetic fixtures only: never import or execute archived project modules."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import warnings
import zipfile

import tools_verify_historical_sources as verify


class HistoricalSourcesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / 'project'
        self.root.mkdir()
        self.source = self.root / 'engine/a.py'
        self.source.parent.mkdir()
        self.payload = b'raise RuntimeError("must never execute")\n'
        self.source.write_bytes(self.payload)
        self.expected = {str(self.source): verify.digest(self.payload)}
        self.branch = self.root / 'outputs/branch'
        self.branch.mkdir(parents=True)
        self.result = self.branch / 'results.json'

    def directory(self, after=False):
        archived = self.branch / 'source_archive/engine/a.py'
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_bytes(self.payload)
        key = 'source_hashes_after' if after else 'source_hashes'
        self.result.write_text(json.dumps({'provenance': {key: self.expected}}))
        return archived

    def closure(self, payload=None, records=None, extras=()):
        path = self.branch / 'source_snapshot.zip'
        if records is None:
            records = [{'id': 'sources/000000', 'path': str(self.source),
                        'sha256': self.expected[str(self.source)], 'size': len(self.payload)}]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('manifest.json', json.dumps({'version': 1, 'files': records}))
                if payload is not False:
                    archive.writestr('sources/000000', self.payload if payload is None else payload)
                for name, data in extras:
                    archive.writestr(name, data)
        self.result.write_text(json.dumps({'provenance': {'code_hashes': self.expected,
             'source_snapshot': {'file': 'source_snapshot.zip', 'sha256': verify.digest(path.read_bytes())}}}))
        return path

    def report(self):
        return verify.verify_result(self.result, self.root)

    def test_directory_matches_but_live_stale(self):
        self.directory(after=True)
        self.source.write_bytes(b'new code')
        report = self.report()
        self.assertTrue(report['archive_integrity_ok'])
        self.assertEqual(report['live_sources'][0]['status'], 'live_stale')
        self.assertEqual(report['sources'][0]['source_path'], str(self.source))

    def test_directory_missing_and_mismatch(self):
        archived = self.directory()
        archived.write_bytes(b'wrong')
        self.assertEqual(self.report()['sources'][0]['status'], 'archive_hash_mismatch')
        archived.unlink()
        self.assertEqual(self.report()['sources'][0]['status'], 'archive_missing')

    def test_source_paths_reject_outside_and_traversal(self):
        for key in (str(self.base / 'outside.py'), str(self.root) + '/engine/../a.py'):
            self.result.write_text(json.dumps({'provenance': {'source_hashes': {key: '0' * 64}}}))
            with self.assertRaises(ValueError):
                self.report()

    def test_directory_symlink_rejected(self):
        archived = self.directory()
        archived.unlink()
        archived.symlink_to(self.source)
        self.assertEqual(self.report()['sources'][0]['status'], 'archive_unsafe_or_unreadable')

    def test_non_python_inputs_explicitly_excluded(self):
        self.directory()
        mapping = dict(self.expected, **{str(self.root / 'data.npz'): '0' * 64})
        self.result.write_text(json.dumps({'provenance': {'source_hashes': mapping}}))
        report = self.report()
        self.assertTrue(report['archive_integrity_ok'])
        self.assertEqual(report['excluded_recorded_inputs'], [str(self.root / 'data.npz')])

    def test_archive_hash_map_cannot_override_historical_hash(self):
        archived = self.directory()
        self.result.write_text(json.dumps({'provenance': {'source_hashes': self.expected,
                       'source_archive_hashes': {str(archived): '0' * 64}}}))
        self.assertFalse(self.report()['archive_integrity_ok'])

    def test_zip_matches_without_live_source(self):
        self.closure()
        self.source.unlink()
        report = self.report()
        self.assertTrue(report['archive_integrity_ok'])
        self.assertEqual(report['live_sources'][0]['status'], 'live_missing')

    def test_zip_payload_mismatch_and_missing(self):
        self.closure(payload=b'incorrect')
        self.assertEqual(self.report()['sources'][0]['status'], 'archive_hash_mismatch')
        self.closure(payload=False)
        self.assertEqual(self.report()['sources'][0]['status'], 'archive_missing')

    def test_zip_duplicate_unsafe_and_symlink_members(self):
        for name in ('sources/000000', '../escape', '/absolute', 'sources/../escape'):
            self.closure(extras=[(name, b'bad')])
            self.assertFalse(self.report()['archive_integrity_ok'])
            self.assertTrue(self.report()['errors'])
        path = self.closure()
        with zipfile.ZipFile(path, 'a') as archive:
            info = zipfile.ZipInfo('sources/000001')
            info.create_system = 3
            info.external_attr = 0o120777 << 16
            archive.writestr(info, b'target')
        self.assertFalse(self.report()['archive_integrity_ok'])

    def test_zip_duplicate_manifest_paths_and_ids(self):
        entry = {'id': 'sources/000000', 'path': str(self.source),
                 'sha256': self.expected[str(self.source)], 'size': len(self.payload)}
        self.closure(records=[entry, entry])
        self.assertIn('duplicate ZIP manifest', self.report()['errors'][0])

    def test_duplicate_json_keys_rejected(self):
        self.result.write_text('{"provenance": {}, "provenance": {}}')
        with self.assertRaisesRegex(ValueError, 'duplicate JSON'):
            self.report()

    def test_zip_manifest_hash_is_not_trust_anchor(self):
        entry = {'id': 'sources/000000', 'path': str(self.source),
                 'sha256': verify.digest(b'wrong'), 'size': 5}
        self.closure(payload=b'wrong', records=[entry])
        report = self.report()
        self.assertFalse(report['archive_integrity_ok'])
        self.assertEqual(report['sources'][0]['status'], 'archive_hash_mismatch')
        self.assertIn('historical hash mismatch', report['errors'][0])

    def test_limits_reject_zip_manifest_size(self):
        entry = {'id': 'sources/000000', 'path': str(self.source),
                 'sha256': self.expected[str(self.source)], 'size': verify.MAX_SOURCE + 1}
        self.closure(records=[entry])
        self.assertFalse(self.report()['archive_integrity_ok'])

    def test_output_never_overwrites_or_writes_history(self):
        self.directory()
        for target in (self.result, self.source, self.branch / 'new.json', self.root / 'report.json'):
            with self.assertRaises(ValueError):
                verify.write_report(target, {}, self.root, [self.result])
        output = self.base / 'report.json'
        verify.write_report(output, {'ok': True}, self.root, [self.result])
        with self.assertRaises(FileExistsError):
            verify.write_report(output, {}, self.root, [self.result])
        link = self.base / 'linked'
        link.symlink_to(self.branch, target_is_directory=True)
        with self.assertRaises(ValueError):
            verify.write_report(link / 'new.json', {}, self.root, [self.result])

    def test_cli_success_stale_and_failure_no_write(self):
        archived = self.directory()
        self.source.write_bytes(b'new')
        output = self.base / 'cli.json'
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            code = verify.main([str(self.result), '--project-root', str(self.root), '--output', str(output)])
        self.assertEqual(code, 0)
        report = json.loads(output.read_text())
        self.assertFalse(report['historical_reproduced'])
        self.assertFalse(report['historical_code_executed'])
        self.assertIn(str(self.result), stdout.getvalue())
        archived.unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(verify.main([str(self.result), '--project-root', str(self.root)]), 1)
            self.assertEqual(verify.main([str(self.result), '--project-root', str(self.root),
                                         '--output', str(self.result)]), 2)

    def test_real_format_external_mesh_and_archived_xml_excluded(self):
        archived = self.directory()
        external_mesh = str(self.base / 'cache/Thorax.stl')
        xml = str(self.root / 'model.xml')
        mapping = dict(self.expected)
        mapping.update({external_mesh: '0' * 64, xml: '1' * 64})
        declared = {str(archived): self.expected[str(self.source)],
                    str(self.branch / 'source_archive/model.xml'): '1' * 64}
        self.result.write_text(json.dumps({'provenance': {
            'source_hashes_after': mapping, 'source_archive_hashes': declared}}))
        report = self.report()
        self.assertTrue(report['archive_integrity_ok'])
        self.assertEqual(report['excluded_recorded_inputs'], [external_mesh, xml])

    def test_directory_aggregate_limit(self):
        self.directory()
        original = verify.MAX_TOTAL
        verify.MAX_TOTAL = len(self.payload) - 1
        try:
            self.assertFalse(self.report()['archive_integrity_ok'])
        finally:
            verify.MAX_TOTAL = original

    def test_zip_boolean_version_rejected(self):
        path = self.closure()
        with zipfile.ZipFile(path) as archive:
            payload = archive.read('sources/000000')
            manifest = json.loads(archive.read('manifest.json'))
        manifest['version'] = True
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('manifest.json', json.dumps(manifest))
            archive.writestr('sources/000000', payload)
        self.assertFalse(self.report()['archive_integrity_ok'])

    def test_default_pointer_traversal_rejected(self):
        base = self.root / 'outputs/neural_muscle_closure'
        base.mkdir()
        (base / 'current_run.json').write_text(json.dumps({'path': '../escape'}))
        with self.assertRaisesRegex(ValueError, 'unsafe current pointer'):
            verify.default_results(self.root)


if __name__ == '__main__':
    unittest.main()
