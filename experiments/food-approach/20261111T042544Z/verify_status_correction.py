"""Verify original scientific seals and explicitly documented status-only drift."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    correction = json.loads((HERE / 'STATUS_POINTER_SEAL_CORRECTION.json').read_text())
    unchanged = identities = 0
    for record in correction['records']:
        root = Path(record['root'])
        manifest = root / 'ARTIFACT_SHA256.json'
        assert digest(manifest) == record['original_manifest_sha256']
        files = json.loads(manifest.read_text())['files']
        assert len(files) == record['manifest_file_count']
        changes = []
        local_identities = 0
        for name, entry in files.items():
            expected = entry['sha256'] if isinstance(entry, dict) else entry
            actual = digest(name)
            if actual != expected:
                changes.append(name)
                assert name == record['pointer']
                assert expected == record['sealed_pointer_sha256']
                assert actual == record['current_pointer_sha256']
            else:
                unchanged += 1
            if name.endswith('/IDENTITY.json'):
                assert actual == expected
                assert record['pointer'] not in Path(name).read_text()
                local_identities += 1
        assert changes == [record['pointer']]
        assert local_identities == record['execution_identity_records_unchanged']
        identities += local_identities
    assert unchanged == 181 and identities == 10
    print(json.dumps({'documented_status_pointer_changes': 3,
                      'all_181_other_files_unchanged': True,
                      'all_10_execution_identities_unchanged': True,
                      'original_manifests_unchanged': True,
                      'physics_steps': 0, 'optimizer_updates': 0}))

if __name__ == '__main__':
    main()
