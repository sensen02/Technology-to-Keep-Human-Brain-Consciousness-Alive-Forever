"""Verify allowlisted desktop delivery hashes and no credential/archive filenames.
Checks filenames only for secrets, never reads credentials. Manifest itself is
local build metadata, not a cryptographic trust root.
"""
from pathlib import Path
import hashlib,json
D=Path('/home/sensen/Desktop/cell_wound_prototype/brain_isolation')

def main():
    m=json.loads((D/'MANIFEST.json').read_text())
    checked=0
    for item in m['files']:
        p=(D/item['path']).resolve()
        if not p.is_relative_to(D.resolve()):raise AssertionError('manifest escapes delivery')
        assert p.is_file(),p
        assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],p
        checked+=1
    prohibited=[]
    for p in D.rglob('*'):
        if not p.is_file():continue
        name=p.name.lower()
        if any(s in name for s in ('cookie','api_token','credential')) or name.endswith('.zip'):
            prohibited.append(str(p))
    assert not prohibited,prohibited
    print(json.dumps({'manifest_files_verified':checked,'prohibited_filenames':prohibited,
                      'directory':str(D),'scope':'hashes and filename exclusion, not formal security audit'},indent=2))

if __name__=='__main__':main()
