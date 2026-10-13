"""Bounded, deterministic archives of explicitly allowlisted source bytes.

No discovery, extraction, imports, execution or unpickling occurs. The caller
must supply an explicit nonsecret source/provenance allowlist (code or required
input JSON, for example): this utility cannot recognize secrets inside a file. Hash integrity is NOT signed authenticity.

Both APIs return the validated JSON-compatible manifest. Validation failures
raise SourceSnapshotError; ordinary filesystem failures may raise OSError.
Verification is independent of live source files (which need not still exist).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import zipfile

MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_FILES = 4096
MAX_ARCHIVE_BYTES = MAX_TOTAL_BYTES + MAX_MANIFEST_BYTES + 2 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class SourceSnapshotError(ValueError):
    """The source inputs or snapshot violate the archive contract."""


def _fail(message):
    raise SourceSnapshotError(message)


def _expected(mapping):
    if not isinstance(mapping, dict) or len(mapping) > MAX_FILES:
        _fail("expected a bounded dict of absolute paths to SHA-256 hashes")
    result = {}
    for path, digest in mapping.items():
        if (not isinstance(path, str) or not os.path.isabs(path)
                or '\x00' in path or os.path.normpath(path) != path
                or path.startswith('//')):
            _fail("source paths must be normalized absolute paths")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            _fail("SHA-256 must be 64 lowercase hexadecimal characters")
        result[path] = digest
    return result


def _open_regular(path):
    """Walk each path component without following symlinks, including parents.

    Requires POSIX dir_fd/O_NOFOLLOW support; never silently weakens this rule.
    """
    if not hasattr(os, 'O_NOFOLLOW') or os.open not in os.supports_dir_fd:
        _fail("secure no-symlink file opening is unsupported on this platform")
    directory = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        components = path.split('/')[1:]
        for component in components[:-1]:
            next_directory = os.open(component, os.O_RDONLY | os.O_DIRECTORY
                                     | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = next_directory
        fd = os.open(components[-1], os.O_RDONLY | os.O_NOFOLLOW
                     | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            _fail("input must be a regular file")
        return os.fdopen(fd, 'rb')
    except BaseException:
        os.close(fd)
        raise


def _json_bytes(manifest):
    return (json.dumps(manifest, ensure_ascii=True, sort_keys=True,
                       separators=(',', ':')) + '\n').encode('ascii')


def _entry(name):
    entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    entry.compress_type = zipfile.ZIP_STORED
    entry.create_system = 3
    entry.external_attr = (stat.S_IFREG | 0o600) << 16
    return entry


def build_source_snapshot(source_hashes: dict, destination_path):
    """Hash-check raw source bytes, then atomically publish a deterministic ZIP.

    No destination is replaced unless all inputs and the complete temporary ZIP
    pass validation. Existing destinations survive any pre-publication failure.
    An explicit source path may not also be the destination. Source file suffix
    and encoding are unrestricted; bytes are never decoded. Returns manifest.
    """
    expected = _expected(source_hashes)
    destination = os.path.abspath(os.fspath(destination_path))
    if destination in expected:
        _fail("destination may not overwrite a source")
    payloads = []
    records = []
    total = 0
    for index, path in enumerate(sorted(expected)):
        with _open_regular(path) as source:
            if os.fstat(source.fileno()).st_size > MAX_FILE_BYTES:
                _fail("individual source size limit exceeded")
            payload = source.read(MAX_FILE_BYTES + 1)
        if len(payload) > MAX_FILE_BYTES:
            _fail("individual source size limit exceeded")
        total += len(payload)
        if total > MAX_TOTAL_BYTES:
            _fail("aggregate source size limit exceeded")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected[path]:
            _fail("source hash drift: " + path)
        identifier = f'sources/{index:06d}'
        records.append({'id': identifier, 'path': path, 'sha256': digest,
                        'size': len(payload)})
        payloads.append(payload)
    manifest = {'version': 1, 'files': records}
    encoded = _json_bytes(manifest)
    if len(encoded) > MAX_MANIFEST_BYTES:
        _fail("manifest size limit exceeded")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w+b', prefix='.source-snapshot-',
                                         suffix='.tmp', dir=os.path.dirname(destination),
                                         delete=False) as stream:
            temporary = stream.name
            with zipfile.ZipFile(stream, 'w', allowZip64=False) as archive:
                archive.writestr(_entry('manifest.json'), encoded)
                for record, payload in zip(records, payloads):
                    archive.writestr(_entry(record['id']), payload)
            stream.flush()
            os.fsync(stream.fileno())
        verify_source_snapshot(temporary, expected)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)
    return manifest


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate JSON object key")
        result[key] = value
    return result


def _read_member(archive, info, limit):
    if info.file_size > limit:
        _fail("ZIP member size limit exceeded")
    with archive.open(info) as stream:
        payload = stream.read(limit + 1)
    if len(payload) > limit or len(payload) != info.file_size:
        _fail("ZIP member size mismatch")
    return payload


def verify_source_snapshot(path, expected_hashes: dict):
    """Validate an archive against external expected hashes; never read sources.

    Only numbered payload IDs and manifest.json are accepted. ZIP entries must
    be unique, unencrypted, stored regular files, without extra metadata. All
    manifest paths, hashes, sizes and all payload bytes are checked. The external
    expected mapping is the trust anchor, not the archive's self-reported hashes.
    """
    expected = _expected(expected_hashes)
    try:
        with _open_regular(os.path.abspath(os.fspath(path))) as stream:
            if os.fstat(stream.fileno()).st_size > MAX_ARCHIVE_BYTES:
                _fail("archive size limit exceeded")
            with zipfile.ZipFile(stream, 'r') as archive:
                infos = archive.infolist()
                if len(infos) != len(expected) + 1:
                    _fail("ZIP entry count mismatch")
                by_name = {}
                for info in infos:
                    if info.filename in by_name:
                        _fail("duplicate ZIP entry")
                    if (info.orig_filename != info.filename or info.extra
                            or info.comment or info.flag_bits & 1
                            or info.compress_type != zipfile.ZIP_STORED
                            or info.compress_size != info.file_size
                            or stat.S_IFMT(info.external_attr >> 16)
                            not in (0, stat.S_IFREG)):
                        _fail("unsupported ZIP entry metadata")
                    by_name[info.filename] = info
                allowed = {'manifest.json'} | {
                    f'sources/{i:06d}' for i in range(len(expected))}
                if set(by_name) != allowed:
                    _fail("unexpected ZIP ID (no traversal or extra members allowed)")
                raw = _read_member(archive, by_name['manifest.json'], MAX_MANIFEST_BYTES)
                manifest = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_object)
                if (not isinstance(manifest, dict)
                        or set(manifest) != {'version', 'files'}
                        or type(manifest['version']) is not int
                        or manifest['version'] != 1
                        or not isinstance(manifest['files'], list)
                        or len(manifest['files']) != len(expected)):
                    _fail("invalid manifest schema")
                total = 0
                for index, (record, original) in enumerate(
                        zip(manifest['files'], sorted(expected))):
                    if (not isinstance(record, dict)
                            or set(record) != {'id', 'path', 'sha256', 'size'}
                            or record['id'] != f'sources/{index:06d}'
                            or record['path'] != original
                            or record['sha256'] != expected[original]
                            or type(record['size']) is not int
                            or not 0 <= record['size'] <= MAX_FILE_BYTES):
                        _fail("invalid manifest record or expected source mismatch")
                    total += record['size']
                    if total > MAX_TOTAL_BYTES:
                        _fail("aggregate source size limit exceeded")
                    info = by_name[record['id']]
                    if info.file_size != record['size']:
                        _fail("archived payload size mismatch")
                    payload = _read_member(archive, info, MAX_FILE_BYTES)
                    if hashlib.sha256(payload).hexdigest() != record['sha256']:
                        _fail("archived payload hash mismatch")
                return manifest
    except (zipfile.BadZipFile, UnicodeError, json.JSONDecodeError,
            RecursionError, RuntimeError, NotImplementedError) as error:
        raise SourceSnapshotError("malformed source snapshot: " + str(error)) from error
