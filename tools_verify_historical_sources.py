"""Scoped byte verification, not historical execution or full artifact validation.

Only explicitly recorded sources are read; no recursive discovery, extraction,
project imports, simulations or historical writes. Exit 0 permits live drift;
exit 1 means archived integrity failed; exit 2 means invalid input/output.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent
KNOWN_BRANCHES = ('activation_timestep', 'clock_refinement', 'fine_clock',
                  'fixed_replay_refinement', 'zero_floor')
MAX_JSON = 128 * 1024 * 1024
MAX_SOURCE = 4 * 1024 * 1024
MAX_TOTAL = 16 * 1024 * 1024
MAX_ZIP = 19 * 1024 * 1024
MAX_FILES = 4096
SHA = re.compile(r'[0-9a-f]{64}\Z')


def unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('duplicate JSON key: ' + key)
        out[key] = value
    return out


def normalized(path):
    if (not isinstance(path, str) or not path.startswith('/') or
            path.startswith('//') or '\x00' in path or '\\' in path or
            os.path.normpath(path) != path):
        raise ValueError('expected normalized absolute path: ' + repr(path))
    return Path(path)


def anchored(path, root):
    path = normalized(str(path))
    path.relative_to(root)
    return path


def open_regular(path):
    """Refuse symlinks in every component, and special files (POSIX)."""
    path = normalized(str(path))
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        parts = path.parts[1:]
        for part in parts[:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                          dir_fd=fd)
            os.close(fd)
            fd = nxt
        source = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=fd)
    finally:
        os.close(fd)
    if not stat.S_ISREG(os.fstat(source).st_mode):
        os.close(source)
        raise ValueError('not a regular file: ' + str(path))
    return os.fdopen(source, 'rb')


def read_bytes(path, limit):
    with open_regular(path) as stream:
        if os.fstat(stream.fileno()).st_size > limit:
            raise ValueError('size limit exceeded: ' + str(path))
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError('size limit exceeded: ' + str(path))
    return data


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_json(path):
    return json.loads(read_bytes(path, MAX_JSON), object_pairs_hook=unique_object)


def hash_map(mapping, root):
    if not isinstance(mapping, dict) or not mapping or len(mapping) > MAX_FILES:
        raise ValueError('expected nonempty bounded historical hash map')
    for path, expected in mapping.items():
        anchored(path, root)
        if not isinstance(expected, str) or not SHA.fullmatch(expected):
            raise ValueError('invalid SHA-256 for ' + path)
    return mapping


def file_check(path, expected, prefix, budget=None):
    record = {'path': str(path), 'expected_sha256': expected}
    try:
        limit = MAX_ZIP if str(path).endswith('.zip') else MAX_SOURCE
        if budget is not None:
            limit = min(limit, MAX_TOTAL - budget[0])
        data = read_bytes(path, limit)
        if budget is not None:
            budget[0] += len(data)
            if budget[0] > MAX_TOTAL:
                raise ValueError('aggregate source size limit exceeded: ' + str(path))
        actual = digest(data)
        record.update(actual_sha256=actual,
                      status=prefix + ('_match' if actual == expected else
                                       '_stale' if prefix == 'live' else '_hash_mismatch'))
    except FileNotFoundError:
        record['status'] = prefix + '_missing'
    except (OSError, ValueError) as error:
        record.update(status=prefix + '_unsafe_or_unreadable', error=str(error))
    return record


def check_zip(path, expected, errors):
    """Check external recorded hashes, never trust self-reported ZIP hashes alone."""
    records = []
    with open_regular(path) as stream:
        if os.fstat(stream.fileno()).st_size > MAX_ZIP:
            raise ValueError('archive size limit exceeded: ' + str(path))
        with zipfile.ZipFile(stream) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_FILES + 1:
                raise ValueError('ZIP member count limit exceeded: ' + str(path))
            by_name = {}
            for info in infos:
                name = info.filename
                if name in by_name:
                    raise ValueError('duplicate ZIP member: ' + str(path) + '::' + name)
                if (name != 'manifest.json' and not re.fullmatch(r'sources/[0-9]{6}', name)):
                    raise ValueError('unsafe/unexpected ZIP member: ' + str(path) + '::' + name)
                if (info.orig_filename != name or info.flag_bits & 1 or info.extra or
                        info.comment or info.compress_type != zipfile.ZIP_STORED or
                        stat.S_IFMT(info.external_attr >> 16) not in (0, stat.S_IFREG)):
                    raise ValueError('unsafe ZIP metadata: ' + str(path) + '::' + name)
                by_name[name] = info
            def member(name, limit):
                info = by_name[name]
                if info.file_size > limit:
                    raise ValueError('ZIP size limit: ' + str(path) + '::' + name)
                with archive.open(info) as source:
                    data = source.read(limit + 1)
                if len(data) > limit or len(data) != info.file_size:
                    raise ValueError('ZIP member size mismatch: ' + str(path) + '::' + name)
                return data
            if 'manifest.json' not in by_name:
                raise ValueError('ZIP manifest missing: ' + str(path))
            manifest = json.loads(member('manifest.json', 1024 * 1024),
                                  object_pairs_hook=unique_object)
            if (not isinstance(manifest, dict) or set(manifest) != {'version', 'files'} or
                    type(manifest.get('version')) is not int or manifest.get('version') != 1 or
                    not isinstance(manifest.get('files'), list) or
                    len(manifest['files']) > MAX_FILES):
                raise ValueError('invalid ZIP manifest schema: ' + str(path))
            indexed, ids, total = {}, set(), 0
            for entry in manifest['files']:
                if not isinstance(entry, dict) or set(entry) != {'id', 'path', 'sha256', 'size'}:
                    raise ValueError('invalid ZIP manifest record: ' + str(path))
                key, name = entry['path'], entry['id']
                normalized(key)
                if key in indexed or name in ids:
                    raise ValueError('duplicate ZIP manifest path/id: ' + str(path) + '::' + str(key))
                if not isinstance(name, str) or not re.fullmatch(r'sources/[0-9]{6}', name):
                    raise ValueError('unsafe ZIP manifest id: ' + str(path) + '::' + str(name))
                if (not isinstance(entry['sha256'], str) or not SHA.fullmatch(entry['sha256']) or
                        type(entry['size']) is not int or not 0 <= entry['size'] <= MAX_SOURCE):
                    raise ValueError('invalid ZIP manifest hash/size: ' + str(path) + '::' + key)
                total += entry['size']
                if total > MAX_TOTAL:
                    raise ValueError('ZIP aggregate size limit: ' + str(path))
                indexed[key] = entry
                ids.add(name)
            if set(by_name) - ids - {'manifest.json'}:
                errors.append('unreferenced ZIP members: ' + str(path))
            if set(indexed) - set(expected):
                errors.append('unrecorded ZIP source paths: ' + str(path))
            for key, sha in expected.items():
                entry = indexed.get(key)
                record = {'source_path': key, 'path': str(path), 'expected_sha256': sha}
                if entry is None or entry['id'] not in by_name:
                    record['status'] = 'archive_missing'
                else:
                    record['member'] = str(path) + '::' + entry['id']
                    actual = digest(member(entry['id'], MAX_SOURCE))
                    record.update(actual_sha256=actual)
                    record['status'] = ('archive_match' if actual == sha else 'archive_hash_mismatch')
                    if entry['sha256'] != sha:
                        errors.append('ZIP manifest historical hash mismatch: ' + key)
                    if by_name[entry['id']].file_size != entry['size']:
                        errors.append('ZIP manifest size mismatch: ' + record['member'])
                records.append(record)
    return records


def verify_result(result_path, root=PROJECT_ROOT):
    result_path = Path(os.path.abspath(result_path))
    root = normalized(str(root))
    data = load_json(result_path)
    prov = data.get('provenance', {})
    report = {'result_path': str(result_path), 'errors': [], 'sources': [],
              'excluded_recorded_inputs': [], 'archive_files': []}
    if 'code_hashes' in prov:
        expected = hash_map(prov['code_hashes'], root)
        snapshot = prov.get('source_snapshot', {})
        if snapshot.get('file') != 'source_snapshot.zip':
            raise ValueError('missing/unsafe closure snapshot descriptor: ' + str(result_path))
        archive = result_path.parent / 'source_snapshot.zip'
        sha = snapshot.get('sha256')
        if not isinstance(sha, str) or not SHA.fullmatch(sha):
            raise ValueError('invalid snapshot digest: ' + str(result_path))
        report['archive_files'].append(file_check(archive, sha, 'archive'))
        try:
            report['sources'] = check_zip(archive, expected, report['errors'])
        except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as error:
            report['errors'].append(str(error))
    else:
        mapping = prov.get('source_hashes_after', prov.get('source_hashes'))
        if not isinstance(mapping, dict) or not mapping or len(mapping) > MAX_FILES:
            raise ValueError('expected nonempty bounded historical hash map')
        # Non-Python historical mesh/data inputs can lie outside the project;
        # they are excluded, not opened or mapped into the source archive.
        for path, sha in mapping.items():
            normalized(path)
            if not isinstance(sha, str) or not SHA.fullmatch(sha):
                raise ValueError('invalid recorded SHA-256: ' + path)
        expected = hash_map({p: sha for p, sha in mapping.items() if p.endswith('.py')}, root)
        if not expected:
            raise ValueError('no recorded Python sources: ' + str(result_path))
        report['excluded_recorded_inputs'] = [p for p in mapping if p not in expected]
        archive_root = result_path.parent / 'source_archive'
        budget = [0]
        for key, sha in expected.items():
            archived = archive_root / Path(key).relative_to(root)
            record = file_check(archived, sha, 'archive', budget)
            record['source_path'] = key
            report['sources'].append(record)
        declared = prov.get('source_archive_hashes', {})
        if declared:
            hash_map(declared, root)
            mapped = {str(archive_root / Path(p).relative_to(root)): sha
                      for p, sha in expected.items()}
            declared_sources = {p: sha for p, sha in declared.items() if p.endswith('.py')}
            report['excluded_recorded_archive_inputs'] = [p for p in declared
                                                        if p not in declared_sources]
            if declared_sources != mapped:
                report['errors'].append('archive hash map disagrees with historical source mapping: ' + str(result_path))
    live_budget = [0]
    report['live_sources'] = [file_check(Path(key), sha, 'live', live_budget)
                              for key, sha in expected.items()]
    report['archive_integrity_ok'] = (not report['errors'] and
        bool(report['sources']) and all(r['status'] == 'archive_match'
        for r in report['sources'] + report['archive_files']))
    return report


def default_results(root):
    paths = [root / 'outputs/model_branches' / name / (name + '_results.json')
             for name in KNOWN_BRANCHES]
    base = root / 'outputs/neural_muscle_closure'
    pointer_path = base / 'current_run.json'
    pointer = load_json(pointer_path)
    relative = pointer.get('path')
    if not isinstance(relative, str) or not re.fullmatch(r'runs/[A-Za-z0-9-]+', relative):
        raise ValueError('unsafe current pointer: ' + str(pointer_path))
    bundle = base / relative
    manifest_path = bundle / 'manifest.json'
    manifest_raw = read_bytes(manifest_path, MAX_JSON)
    if digest(manifest_raw) != pointer.get('manifest_sha256'):
        raise ValueError('current pointer manifest hash mismatch: ' + str(manifest_path))
    manifest = json.loads(manifest_raw, object_pairs_hook=unique_object)
    result = bundle / 'closure_results.json'
    if digest(read_bytes(result, MAX_JSON)) != manifest['files']['closure_results.json']['sha256']:
        raise ValueError('bundle result hash mismatch: ' + str(result))
    snapshot = bundle / 'source_snapshot.zip'
    if digest(read_bytes(snapshot, MAX_ZIP)) != manifest['files']['source_snapshot.zip']['sha256']:
        raise ValueError('bundle snapshot hash mismatch: ' + str(snapshot))
    return paths + [result]


def write_report(output, report, root, results):
    target = Path(os.path.abspath(output))
    resolved = target.resolve()
    forbidden = [root.resolve()] + [p.parent.resolve() for p in results]
    if any(resolved == base or base in resolved.parents for base in forbidden):
        raise ValueError('output must be outside project/historical input directories: ' + str(target))
    # Exclusive create: never truncate existing files, even through hardlinks.
    with target.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs='*', help='Explicit result JSON paths; defaults to known five branches + current closure')
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT,
                        help='Absolute recorded project root used to anchor source mappings')
    parser.add_argument('--output', help='New caller-chosen JSON file outside project/history directories')
    args = parser.parse_args(argv)
    try:
        root = normalized(str(args.project_root))
        results = ([Path(os.path.abspath(p)) for p in args.results] if args.results
                   else default_results(root))
        reports = []
        for path in results:
            try:
                reports.append(verify_result(path, root))
            except (OSError, ValueError, KeyError, TypeError, RecursionError) as error:
                reports.append({'result_path': str(path), 'archive_integrity_ok': False,
                                'errors': [str(error)], 'sources': [], 'live_sources': []})
        report = {'scope': 'Recorded directory Python sources and closure ZIP code/pose inputs only; excluded inputs, other artifacts and environment NOT validated; unsigned integrity only.',
                  'scope_not_validated': ['NPZ/XML and excluded large input artifacts',
                                          'full dependency/environment closure',
                                          'signed authenticity/authorship',
                                          'historical numerical reproduction'],
                  'historical_reproduced': False, 'historical_code_executed': False,
                  'results': reports,
                  'archive_integrity_ok': all(r['archive_integrity_ok'] for r in reports)}
        if args.output:
            write_report(args.output, report, root, results)
        for item in reports:
            counts = dict(Counter(r['status'] for r in item['sources']))
            live = dict(Counter(r['status'] for r in item['live_sources']))
            print(item['result_path'] + ': ' + json.dumps({'archived': counts, 'live': live,
                  'archive_integrity_ok': item['archive_integrity_ok'], 'errors': item['errors']}))
        print(report['scope'] + ' Historical reproduction: false.')
        if args.output:
            print('Report: ' + os.path.abspath(args.output))
        return 0 if report['archive_integrity_ok'] else 1
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as error:
        print('Invalid input/output: ' + str(error))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
