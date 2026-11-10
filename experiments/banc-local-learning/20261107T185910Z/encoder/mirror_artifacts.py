"""Mirror only delegated encoder outputs and verify every delivered byte hash."""
from pathlib import Path
import hashlib
import json
import shutil

SOURCE = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/encoder')
DESKTOP = Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T185910Z/encoder')
NAMES = ('PREREG.json', 'encoder.py', 'CONTRACT.json', 'EXCLUSIONS.json',
         'verify_encoder.py', 'VERIFICATION.json', 'REPORT.md',
         'EXECUTION_RECORD.json', 'mirror_artifacts.py')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    DESKTOP.mkdir(parents=True, exist_ok=True)
    records = []
    for name in NAMES:
        source, dest = SOURCE / name, DESKTOP / name
        shutil.copy2(source, dest)
        sh, dh = digest(source), digest(dest)
        assert sh == dh, name
        records.append({'source': str(source), 'desktop': str(dest),
                        'bytes': source.stat().st_size, 'sha256': sh, 'byte_equal': True})
    result = {'status': 'PASS_BYTE_IDENTICAL_DELIVERY', 'artifacts': records,
              'artifact_count_excluding_this_record': len(records),
              'record_self_hash_note': 'This delivery record is mirrored identically and directly byte-compared; its own hash is not embedded recursively.',
              'neural_steps': 0, 'physical_steps': 0}
    record = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    (SOURCE / 'DELIVERY_VERIFICATION.json').write_text(record)
    (DESKTOP / 'DELIVERY_VERIFICATION.json').write_text(record)
    assert digest(SOURCE / 'DELIVERY_VERIFICATION.json') == digest(DESKTOP / 'DELIVERY_VERIFICATION.json')
    print(json.dumps({'status': result['status'], 'files_including_record': len(records)+1,
                      'desktop': str(DESKTOP), 'delivery_record_sha256': digest(SOURCE/'DELIVERY_VERIFICATION.json')}))


if __name__ == '__main__':
    main()
