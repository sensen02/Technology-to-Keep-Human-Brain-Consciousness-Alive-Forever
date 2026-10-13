"""Atomic, stdlib-only snapshots of already completed artifact files.

The publisher never edits a published run. Hashes detect subsequent changes;
this is not an access-control mechanism or a signature. Callers must finish and
stop writing ALL inputs before publishing: this helper cannot establish that
arbitrary source artifacts came from the same scientific run.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

__all__ = ["publish_bundle", "validate_bundle"]
_MANIFEST = "manifest.json"
_INDEX = "current_run.json"


def _basename(name: str) -> str:
    # Check both separator conventions even when running on POSIX.
    if (not isinstance(name, str) or not name or name in (".", "..")
            or any(c in name for c in ("/", "\\", ":", "\x00"))
            or name.casefold() == _MANIFEST):
        raise ValueError(f"Not an artifact basename: {name!r}")
    return name


def _run_id(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Invalid run_id")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError("Invalid run_id") from exc
    if str(parsed) != value:
        raise ValueError("run_id must be a canonical UUID")
    return value


def _digest(path: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "size": size}


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _regular(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular, non-symlink file: {path}")


def _json_bytes(path: Path) -> tuple[dict, bytes]:
    _regular(path)
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value, raw


def _validate_run(path: Path, expected_id: str, manifest_sha256=None) -> dict:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"Expected non-symlink run directory: {path}")
    manifest, raw = _json_bytes(path / _MANIFEST)
    if manifest_sha256 is not None:
        if hashlib.sha256(raw).hexdigest() != manifest_sha256:
            raise ValueError("Manifest SHA-256 mismatch")
    if (manifest.get("schema_version") != 1
            or manifest.get("complete") is not True
            or manifest.get("run_id") != expected_id):
        raise ValueError("Incomplete or inconsistent bundle manifest")
    entries = manifest.get("files")
    if not isinstance(entries, dict):
        raise ValueError("Manifest files must be an object")
    for name, expected in entries.items():
        _basename(name)
        if (not isinstance(expected, dict)
                or type(expected.get("size")) is not int
                or expected["size"] < 0
                or not isinstance(expected.get("sha256"), str)):
            raise ValueError(f"Invalid metadata for {name!r}")
        artifact = path / name
        _regular(artifact)
        if _digest(artifact) != expected:
            raise ValueError(f"Artifact hash/size mismatch: {name}")
    return {"run_id": expected_id, "path": str(path.resolve()), "manifest": manifest}


def validate_bundle(output_dir: str | Path, *, manifest_sha256: str | None = None) -> dict:
    """Validate current index and every listed file, returning run_id/path/manifest.

    Pass the publication root (or its current_run.json) to verify the indexed
    manifest digest as well. A retained runs/<uuid> directory is also accepted;
    supply manifest_sha256 to anchor its manifest to a previously saved digest.
    Without that anchor direct-run validation checks internal consistency only.
    Malformed, incomplete, missing, tampered or traversing bundles raise ValueError
    (underlying filesystem I/O failures may raise OSError).
    """
    target = Path(output_dir).absolute()
    if target.name == _INDEX:
        root = target.parent
    elif target.parent.name == "runs" and (target / _MANIFEST).exists():
        # A snapshot may itself contain an artifact named current_run.json.
        return _validate_run(target, _run_id(target.name), manifest_sha256)
    elif (target / _INDEX).exists() or not (target / _MANIFEST).exists():
        root = target
    else:
        return _validate_run(target, _run_id(target.name), manifest_sha256)
    index, _ = _json_bytes(root / _INDEX)
    run_id = _run_id(index.get("run_id"))
    relative = f"runs/{run_id}"
    if index.get("path") != relative:
        raise ValueError("Index path must be runs/<run_id>; traversal is forbidden")
    digest = index.get("manifest_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Missing or invalid manifest SHA-256")
    if manifest_sha256 is not None and digest != manifest_sha256:
        raise ValueError("Index manifest SHA-256 mismatch")
    runs = root / "runs"
    if runs.is_symlink() or not runs.is_dir():
        raise ValueError("Expected a non-symlink runs directory")
    return _validate_run(runs / run_id, run_id, digest)


def publish_bundle(output_dir: str | Path, files: dict[str, Path]) -> dict:
    """Copy completed inputs, then atomically publish a new immutable-by-contract run.

    Result is {run_id: str, path: absolute str, manifest: dict}. The manifest has
    schema_version, run_id, complete, and files={basename: {sha256, size}}.
    current_run.json stores run_id, relative path, and manifest_sha256.
    Each copy is checked against the source's pre-copy hash. Basenames only;
    manifest.json is reserved. Empty bundles are allowed. Source paths may be
    anywhere, but must be regular non-symlink files.

    A failure before the final index os.replace leaves the old index unchanged.
    An index-write failure may leave an unreferenced but complete run; readers
    never select it. Concurrent publishers are last-successful-index-replace wins.
    Atomic visibility is guaranteed on a filesystem supporting atomic rename;
    this is not a cross-file source transaction or power-loss durability promise.
    """
    if not isinstance(files, dict):
        raise TypeError("files must map basenames to source Paths")
    sources = {}
    for name, source in files.items():
        _basename(name)
        source = Path(source)
        _regular(source)
        sources[name] = source
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    runs = root / "runs"
    if runs.is_symlink():
        raise ValueError("runs must not be a symlink")
    runs.mkdir(exist_ok=True)
    run_id = str(uuid.uuid4())
    destination = runs / run_id
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=runs))
    index_temp = root / f".current-{run_id}.tmp"
    try:
        entries = {}
        for name, source in sorted(sources.items()):
            expected = _digest(source)
            copied = staging / name
            shutil.copyfile(source, copied)
            if _digest(copied) != expected:
                raise ValueError(f"Source changed while copying: {name}")
            with copied.open("rb") as stream:
                os.fsync(stream.fileno())
            entries[name] = expected
        manifest = {"schema_version": 1, "run_id": run_id,
                    "complete": True, "files": entries}
        # The completion marker is created only after ALL copies succeed.
        _write_json(staging / _MANIFEST, manifest)
        digest = _digest(staging / _MANIFEST)["sha256"]
        _validate_run(staging, run_id, digest)
        os.replace(staging, destination)
        result = {"run_id": run_id, "path": str(destination), "manifest": manifest}
        _write_json(index_temp, {"run_id": run_id, "path": f"runs/{run_id}",
                                 "manifest_sha256": digest})
        os.replace(index_temp, root / _INDEX)
        return result
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if index_temp.exists():
            index_temp.unlink()
