"""One-time strict freeze, stdlib only. No model imports or reanchor option."""
import json
import time
from pathlib import Path
from probe_identity import HERE,EXPECTED_ROOT,FREEZE,sha256,cpu1


def main():
    if HERE!=EXPECTED_ROOT:raise RuntimeError('ROOT mismatch')
    if FREEZE.exists():raise RuntimeError('Strict freeze exists; never overwrite')
    if list(HERE.glob('RESULT_STATIC_PROBE_*.json')):raise RuntimeError('Cannot freeze after execution')
    cpu1()
    old=HERE.parent/'left_wider_radius'/'left_yaw_p060';ident=old/'IDENTITY.json'
    manifest=json.loads(ident.read_text())
    bad=[p for p,h in manifest['files'].items() if not Path(p).is_file() or sha256(p)!=h['sha256']]
    if bad:raise RuntimeError(f'Inherited actual source/input identity drift: {bad[:8]}')
    paths={Path(p) for p in manifest['files']};paths.update(HERE.glob('*.py'));paths.add(HERE/'PREREG.json');paths.add(ident)
    previous=HERE.parent/'left_wider_radius'/'SOURCEHASH_WIDER_RADIUS.json';paths.add(previous)
    paths.update(Path(p) for p in json.loads(previous.read_text())['files'])
    for name in ('TRACE_1MS.npz','RESULT.json','NATIVE_MANIFEST.json','REFERENCE_WIDER_RADIUS.json'):
        paths.add(old/name)
    value=dict(status='STRICT_FROZEN_BEFORE_STATIC_EXECUTION',root=str(HERE),inherited_actual_identity=str(ident),
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),files={str(p.resolve()):sha256(p) for p in sorted(paths)},
        excludes='New exclusive generated static/audit reports/logs and mutable tracker/head-training files. No old output writers invoked.')
    with FREEZE.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=value['status'],files=len(value['files']),path=str(FREEZE),sha256=sha256(FREEZE))))


if __name__=='__main__':main()
