"""Data-free engineering regressions for RunRecord content hashing."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from engine.record import RunRecord


class RunRecordHashTests(unittest.TestCase):
    def record(self):
        with patch('engine.record._versions', return_value={}), patch(
                'engine.record._git_commit', return_value={}):
            return RunRecord('engineering regression')

    def test_write_preserves_content_hash(self):
        record = self.record().set_metric('count', 1)
        expected = record.hash()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'record.json'
            record.write(path)
            self.assertEqual(record.hash(), expected)
            record.write(path)
            self.assertEqual(json.loads(path.read_text())['record_hash'], expected)

    def test_semantic_change_after_write_changes_hash(self):
        record = self.record().set_metric('count', 1)
        with tempfile.TemporaryDirectory() as tmp:
            record.write(Path(tmp) / 'record.json')
            expected = record.data['record_hash']
            record.set_metric('count', 2)
            self.assertNotEqual(record.hash(), expected)

    def test_serialization_failure_preserves_previous_file_and_hash(self):
        class Unserializable:
            def __str__(self):
                raise ValueError('serialization failure')
        record = self.record()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'record.json'
            record.write(path)
            previous = path.read_bytes()
            old_hash = record.data['record_hash']
            record.set_metric('count', 2)
            record.data['outputs'] = [Unserializable()]
            with self.assertRaisesRegex(ValueError, 'serialization failure'):
                record.write(path)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(record.data['record_hash'], old_hash)
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_replace_failure_preserves_file_and_cleans_temporary(self):
        record = self.record()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'record.json'
            path.write_text('previous record', encoding='utf-8')
            with patch('engine.record.os.replace', side_effect=OSError('replace denied')):
                with self.assertRaisesRegex(OSError, 'replace denied'):
                    record.write(path)
            self.assertEqual(path.read_text(encoding='utf-8'), 'previous record')
            self.assertNotIn('record_hash', record.data)
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_utf8_and_nested_destination(self):
        record = self.record().add_note('中文记录')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'nested' / 'record.json'
            self.assertEqual(record.write(path), path)
            saved = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(saved['notes'], ['中文记录'])
            self.assertEqual(saved['record_hash'], record.hash())

    def test_timestamp_and_outputs_are_not_content(self):
        record = self.record()
        expected = record.hash()
        record.data['created_utc'] = 'different time'
        record.add_output('different-output.json')
        self.assertEqual(record.hash(), expected)


if __name__ == '__main__':
    unittest.main()
