"""Read-only independent hash check of referenced historical evidence."""
from pathlib import Path
import hashlib
import json

OUT = Path('/home/sensen/Desktop/engine修复验收/s5-robust-readout/20261106T031010Z')
ROOT = Path('/home/sensen/Desktop/engine修复验收')
PACKAGES = [
    ROOT / 's5-neural-translator/20261105T200021Z',
    ROOT / 's5-neural-parameter-audit/20261106T022431Z',
    ROOT / 's5-readout-gain/20261106T024500Z',
]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    rows = []
    for folder in PACKAGES:
        manifest = folder / 'MANIFEST_SHA256.json'
        data = json.loads(manifest.read_text())
        bad = []
        for path, expected in data.items():
            p = Path(path)
            if not p.exists() or sha(p) != expected:
                bad.append(path)
        rows.append(dict(manifest=str(manifest), manifest_sha256=sha(manifest), entries=len(data), mismatches=bad))
    output = dict(importance='重要辅助', purpose='Independent verification that historical evidence remains intact; historical statements are not endorsed by hashes.', archives=rows, passed=all(not r['mismatches'] for r in rows))
    destination = OUT / 'HISTORICAL_INTEGRITY.json'
    assert not destination.exists(), 'Do not overwrite previous verification'
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(output, ensure_ascii=False))
    assert output['passed']

if __name__ == '__main__':
    main()
