"""Pure filesystem tests: no neural simulation or external dependencies."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

# Load the stdlib utility without executing the embodied simulation package init.
_MODULE_PATH = Path(__file__).parent / "engine" / "embodied" / "closure_bundle.py"
_SPEC = importlib.util.spec_from_file_location("closure_bundle", _MODULE_PATH)
bundle = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bundle)


class ClosureBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.output = self.base / "output"
        self.a = self.base / "a.source"
        self.b = self.base / "b.source"
        self.a.write_bytes(b"first artifact\x00\xff")
        self.b.write_bytes(b"second artifact")
        self.files = {"a.bin": self.a, "b.txt": self.b}

    def publish(self):
        return bundle.publish_bundle(self.output, self.files)

    def test_valid_unique_snapshot_and_return_api(self):
        first = self.publish()
        self.assertEqual(set(first), {"run_id", "path", "manifest"})
        self.assertEqual(first, bundle.validate_bundle(self.output))
        index = json.loads((self.output / "current_run.json").read_text())
        run = Path(first["path"])
        self.assertEqual(index["path"], f"runs/{first['run_id']}")
        self.assertEqual(index["manifest_sha256"],
                         hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest())
        self.assertTrue(first["manifest"]["complete"])
        for name, source in self.files.items():
            content = source.read_bytes()
            self.assertEqual((run / name).read_bytes(), content)
            self.assertEqual(first["manifest"]["files"][name],
                             {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        self.a.write_bytes(b"new source")
        second = self.publish()
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertEqual(bundle.validate_bundle(self.output / "current_run.json"), second)
        self.assertEqual(bundle.validate_bundle(run, manifest_sha256=index["manifest_sha256"]), first)
        self.assertFalse(any(p.name.startswith(".staging-") for p in (self.output / "runs").iterdir()))

    def test_tamper_same_size_and_mixed_file_rejected(self):
        first = self.publish()
        self.a.write_bytes(b"X" * len(self.a.read_bytes()))
        second = self.publish()
        shutil.copyfile(Path(first["path"]) / "a.bin", Path(second["path"]) / "a.bin")
        with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
            bundle.validate_bundle(self.output)

    def test_size_change_and_missing_file_rejected(self):
        for mode in ("size", "missing"):
            with self.subTest(mode=mode):
                run = Path(self.publish()["path"])
                if mode == "size":
                    (run / "a.bin").write_bytes(b"")
                else:
                    (run / "a.bin").unlink()
                with self.assertRaises(ValueError):
                    bundle.validate_bundle(self.output)

    def test_interrupted_copy_preserves_previous_index(self):
        first = self.publish()
        index = (self.output / "current_run.json").read_bytes()
        real_copy = shutil.copyfile
        calls = []

        def fail_second(source, destination):
            calls.append(destination)
            self.assertFalse((Path(destination).parent / "manifest.json").exists())
            if len(calls) == 2:
                Path(destination).write_bytes(b"partial")
                raise OSError("simulated interrupted copy")
            return real_copy(source, destination)

        with mock.patch.object(bundle.shutil, "copyfile", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "interrupted copy"):
                self.publish()
        self.assertEqual(len(calls), 2)
        self.assertEqual((self.output / "current_run.json").read_bytes(), index)
        self.assertEqual(bundle.validate_bundle(self.output), first)
        self.assertEqual(len(list((self.output / "runs").iterdir())), 1)

    def test_failed_index_replace_preserves_previous_index(self):
        first = self.publish()
        old_index = (self.output / "current_run.json").read_bytes()
        real_replace = bundle.os.replace

        def fail_index(source, destination):
            if Path(destination).name == "current_run.json":
                raise OSError("index replace failure")
            return real_replace(source, destination)

        with mock.patch.object(bundle.os, "replace", side_effect=fail_index):
            with self.assertRaises(OSError):
                self.publish()
        self.assertEqual((self.output / "current_run.json").read_bytes(), old_index)
        self.assertEqual(bundle.validate_bundle(self.output), first)
        self.assertFalse(list(self.output.glob(".current-*.tmp")))
        for run in (self.output / "runs").iterdir():
            bundle.validate_bundle(run)

    def test_bad_publish_basenames_rejected(self):
        first = self.publish()
        for name in ("../escape", "/tmp/escape", "a/b", "a\\b", "..", ".", "", "C:escape", "manifest.json", "MANIFEST.JSON", "bad\x00name"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                bundle.publish_bundle(self.output, {name: self.a})
        self.assertEqual(bundle.validate_bundle(self.output), first)
        self.assertFalse((self.base / "escape").exists())

    def test_index_traversal_rejected(self):
        self.publish()
        index_path = self.output / "current_run.json"
        index = json.loads(index_path.read_text())
        for path in ("../outside", "/tmp/outside", "runs/../outside", "runs\\outside"):
            index["path"] = path
            index_path.write_text(json.dumps(index))
            with self.subTest(path=path), self.assertRaises(ValueError):
                bundle.validate_bundle(self.output)

    def test_manifest_tamper_and_traversal_rejected(self):
        first = self.publish()
        run = Path(first["path"])
        manifest = first["manifest"]
        metadata = manifest["files"].pop("a.bin")
        manifest["files"]["../a.source"] = metadata
        (run / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Manifest SHA-256 mismatch"):
            bundle.validate_bundle(self.output)
        with self.assertRaisesRegex(ValueError, "basename"):
            bundle.validate_bundle(run)

    def test_incomplete_manifest_rejected(self):
        first = self.publish()
        run = Path(first["path"])
        manifest = first["manifest"]
        manifest["complete"] = False
        (run / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            bundle.validate_bundle(run)

    def test_symlink_artifact_and_runs_rejected(self):
        first = self.publish()
        artifact = Path(first["path"]) / "a.bin"
        artifact.unlink()
        artifact.symlink_to(self.a)
        with self.assertRaises(ValueError):
            bundle.validate_bundle(self.output)
        other = self.base / "other"
        other.mkdir()
        (other / "runs").symlink_to(self.output / "runs", target_is_directory=True)
        with self.assertRaises(ValueError):
            bundle.publish_bundle(other, self.files)

    def test_mutation_during_copy_rejected(self):
        first = self.publish()
        real_copy = shutil.copyfile

        def mutate(source, destination):
            Path(source).write_bytes(b"changed during copy")
            return real_copy(source, destination)

        with mock.patch.object(bundle.shutil, "copyfile", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "Source changed"):
                self.publish()
        self.assertEqual(bundle.validate_bundle(self.output), first)

    def test_index_named_artifact_is_not_mistaken_for_pointer(self):
        result = bundle.publish_bundle(self.output, {"current_run.json": self.a})
        self.assertEqual(bundle.validate_bundle(self.output), result)
        self.assertEqual(bundle.validate_bundle(result["path"]), result)

    def test_empty_bundle(self):
        result = bundle.publish_bundle(self.output, {})
        self.assertEqual(result["manifest"]["files"], {})
        self.assertEqual(bundle.validate_bundle(self.output), result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
