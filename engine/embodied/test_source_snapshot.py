"""Standalone stdlib tests: python3 engine/embodied/test_source_snapshot.py."""
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile

# Avoid importing the embodied package or any optional simulation dependencies.
_SPEC = importlib.util.spec_from_file_location(
    'source_snapshot_under_test', Path(__file__).with_name('source_snapshot.py'))
snapshot = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(snapshot)


class SourceSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'code.py'
        self.raw = b'# raw source bytes\nvalue = 1\n\xff\x00'
        self.source.write_bytes(self.raw)
        self.expected = {str(self.source): hashlib.sha256(self.raw).hexdigest()}
        self.dest = self.root / 'source_snapshot.zip'

    def build(self):
        return snapshot.build_source_snapshot(self.expected, self.dest)

    def rewrite(self, change):
        with zipfile.ZipFile(self.dest) as archive:
            entries = [(info, archive.read(info)) for info in archive.infolist()]
        entries = change(entries)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(self.dest, 'w') as archive:
                for info, data in entries:
                    archive.writestr(info, data)

    def edit_manifest(self, mutate):
        def change(entries):
            manifest = json.loads(entries[0][1])
            mutate(manifest)
            entries[0] = (entries[0][0], json.dumps(manifest).encode())
            return entries
        self.rewrite(change)

    def test_retains_raw_bytes_after_live_source_changed_or_removed(self):
        manifest = self.build()
        self.source.write_bytes(b'changed')
        self.assertEqual(snapshot.verify_source_snapshot(self.dest, self.expected), manifest)
        self.source.unlink()
        self.assertEqual(snapshot.verify_source_snapshot(self.dest, self.expected), manifest)
        with zipfile.ZipFile(self.dest) as archive:
            self.assertEqual(archive.read('sources/000000'), self.raw)
        self.assertEqual(manifest['files'][0]['size'], len(self.raw))

    def test_deterministic_across_input_order_and_file_metadata(self):
        other = self.root / 'another.py'
        other.write_bytes(b'pass\n')
        self.expected[str(other)] = hashlib.sha256(other.read_bytes()).hexdigest()
        self.build()
        first = self.dest.read_bytes()
        self.source.chmod(0o600)
        snapshot.build_source_snapshot(dict(reversed(list(self.expected.items()))), self.dest)
        self.assertEqual(first, self.dest.read_bytes())

    def test_drift_preserves_existing_destination_and_leaves_no_temp(self):
        self.dest.write_bytes(b'previous archive')
        self.source.write_bytes(b'drift')
        with self.assertRaises(snapshot.SourceSnapshotError):
            self.build()
        self.assertEqual(self.dest.read_bytes(), b'previous archive')
        self.assertEqual(list(self.root.glob('.source-snapshot-*')), [])

    def test_drift_does_not_create_destination(self):
        self.source.write_bytes(b'drift')
        with self.assertRaises(snapshot.SourceSnapshotError):
            self.build()
        self.assertFalse(self.dest.exists())

    def test_replace_failure_preserves_destination_and_cleans_temp(self):
        self.dest.write_bytes(b'previous archive')
        with mock.patch.object(snapshot.os, 'replace', side_effect=OSError('failure')):
            with self.assertRaises(OSError):
                self.build()
        self.assertEqual(self.dest.read_bytes(), b'previous archive')
        self.assertEqual(list(self.root.glob('.source-snapshot-*')), [])

    def test_tampered_payload_with_valid_crc_fails(self):
        self.build()
        self.rewrite(lambda entries: [entries[0], (entries[1][0], b'X' * len(self.raw))])
        with self.assertRaisesRegex(snapshot.SourceSnapshotError, 'hash mismatch'):
            snapshot.verify_source_snapshot(self.dest, self.expected)

    def test_tampered_size_fails(self):
        self.build()
        self.edit_manifest(lambda manifest: manifest['files'][0].update(size=1))
        with self.assertRaises(snapshot.SourceSnapshotError):
            snapshot.verify_source_snapshot(self.dest, self.expected)

    def test_traversal_id_fails(self):
        self.build()
        self.rewrite(lambda entries: [entries[0], ('../escape.py', self.raw)])
        with self.assertRaises(snapshot.SourceSnapshotError):
            snapshot.verify_source_snapshot(self.dest, self.expected)
        self.assertFalse((self.root.parent / 'escape.py').exists())

    def test_duplicate_zip_id_fails(self):
        self.build()
        self.rewrite(lambda entries: [entries[0], entries[0]])
        with self.assertRaisesRegex(snapshot.SourceSnapshotError, 'duplicate ZIP'):
            snapshot.verify_source_snapshot(self.dest, self.expected)

    def test_extra_entry_fails(self):
        self.build()
        self.rewrite(lambda entries: entries + [('unexpected', b'')])
        with self.assertRaises(snapshot.SourceSnapshotError):
            snapshot.verify_source_snapshot(self.dest, self.expected)

    def test_duplicate_json_key_fails(self):
        self.build()
        self.rewrite(lambda entries: [(entries[0][0],
                     entries[0][1].replace(b'"version":1', b'"version":1,"version":1')),
                     entries[1]])
        with self.assertRaisesRegex(snapshot.SourceSnapshotError, 'duplicate JSON'):
            snapshot.verify_source_snapshot(self.dest, self.expected)

    def test_expected_keys_and_hashes_are_external_trust_anchor(self):
        self.build()
        for expected in ({}, {str(self.root / 'wrong.py'): next(iter(self.expected.values()))},
                         {str(self.source): '0' * 64}):
            with self.subTest(expected=expected), self.assertRaises(snapshot.SourceSnapshotError):
                snapshot.verify_source_snapshot(self.dest, expected)

    def test_bad_expected_mapping_fails_before_publish(self):
        for mapping in ({'relative.py': '0' * 64}, {str(self.source): 'a' * 63},
                        {str(self.source): 'A' * 64}, {str(self.source): 123},
                        {str(self.root) + '/x/../code.py': '0' * 64}, []):
            with self.subTest(mapping=mapping), self.assertRaises(snapshot.SourceSnapshotError):
                snapshot.build_source_snapshot(mapping, self.dest)
        self.assertFalse(self.dest.exists())

    def test_duplicate_manifest_source_fails(self):
        other = self.root / 'other.py'
        other.write_bytes(self.raw)
        self.expected[str(other)] = self.expected[str(self.source)]
        self.build()
        self.edit_manifest(lambda manifest: manifest['files'][1].update(path=str(self.source)))
        with self.assertRaises(snapshot.SourceSnapshotError):
            snapshot.verify_source_snapshot(self.dest, self.expected)

    def test_malformed_manifest_hash_fails(self):
        self.build()
        self.edit_manifest(lambda manifest: manifest['files'][0].update(sha256='invalid'))
        with self.assertRaises(snapshot.SourceSnapshotError):
            snapshot.verify_source_snapshot(self.dest, self.expected)

    def test_symlink_source_and_parent_are_rejected(self):
        alias = self.root / 'alias.py'
        alias.symlink_to(self.source)
        linked_directory = self.root / 'linked'
        linked_directory.symlink_to(self.root, target_is_directory=True)
        for path in (alias, linked_directory / self.source.name):
            with self.subTest(path=path), self.assertRaises((OSError, snapshot.SourceSnapshotError)):
                snapshot.build_source_snapshot({str(path): self.expected[str(self.source)]}, self.dest)
        self.assertFalse(self.dest.exists())

    def test_zip_symlink_metadata_is_rejected(self):
        self.build()
        def change(entries):
            entries[1][0].external_attr = (stat.S_IFLNK | 0o777) << 16
            return entries
        self.rewrite(change)
        with self.assertRaises(snapshot.SourceSnapshotError):
            snapshot.verify_source_snapshot(self.dest, self.expected)

    def test_snapshot_symlink_is_rejected(self):
        self.build()
        link = self.root / 'link.zip'
        link.symlink_to(self.dest)
        with self.assertRaises((OSError, snapshot.SourceSnapshotError)):
            snapshot.verify_source_snapshot(link, self.expected)

    def test_individual_and_aggregate_limits_build_and_verify(self):
        self.build()
        original = self.dest.read_bytes()
        for constant in ('MAX_FILE_BYTES', 'MAX_TOTAL_BYTES'):
            with self.subTest(constant=constant), mock.patch.object(snapshot, constant, 1):
                with self.assertRaises(snapshot.SourceSnapshotError):
                    self.build()
                with self.assertRaises(snapshot.SourceSnapshotError):
                    snapshot.verify_source_snapshot(self.dest, self.expected)
            self.assertEqual(self.dest.read_bytes(), original)

    def test_manifest_archive_and_count_limits(self):
        self.build()
        for constant, value in (('MAX_MANIFEST_BYTES', 1), ('MAX_FILES', 0)):
            with self.subTest(constant=constant), mock.patch.object(snapshot, constant, value):
                with self.assertRaises(snapshot.SourceSnapshotError):
                    self.build()
                with self.assertRaises(snapshot.SourceSnapshotError):
                    snapshot.verify_source_snapshot(self.dest, self.expected)
        with mock.patch.object(snapshot, 'MAX_ARCHIVE_BYTES', 1):
            with self.assertRaises(snapshot.SourceSnapshotError):
                snapshot.verify_source_snapshot(self.dest, self.expected)

    def test_source_destination_collision_fails(self):
        with self.assertRaises(snapshot.SourceSnapshotError):
            snapshot.build_source_snapshot(self.expected, self.source)
        self.assertEqual(self.source.read_bytes(), self.raw)

    def test_empty_allowlist(self):
        manifest = snapshot.build_source_snapshot({}, self.dest)
        self.assertEqual(manifest, {'version': 1, 'files': []})
        self.assertEqual(snapshot.verify_source_snapshot(self.dest, {}), manifest)

    def test_invalid_zip_fails(self):
        self.dest.write_bytes(b'not a zip')
        with self.assertRaises(snapshot.SourceSnapshotError):
            snapshot.verify_source_snapshot(self.dest, self.expected)


if __name__ == '__main__':
    unittest.main()
