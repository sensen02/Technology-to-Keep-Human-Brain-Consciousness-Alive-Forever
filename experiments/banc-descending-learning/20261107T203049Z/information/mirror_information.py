"""Byte-identical Desktop mirror of this round's information/ directory; zero steps.

Copies every workspace artifact, re-reads both sides and writes a manifest that
records the SHA256/byte-size of the source and the mirrored copy plus the
equality verdict. Refuses to write a manifest if any byte differs.
"""
from pathlib import Path
import hashlib
import json
import shutil

WS=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-descending-learning/20261107T203049Z/information')
DS=Path('/home/sensen/Desktop/engine修复验收/banc-descending-learning/20261107T203049Z/information')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def main():
    DS.mkdir(parents=True,exist_ok=True)
    entries=[]
    for source in sorted(WS.iterdir()):
        if not source.is_file():continue
        target=DS/source.name
        shutil.copyfile(source,target)
        source_sha=sha(source);target_sha=sha(target)
        if source_sha!=target_sha or source.stat().st_size!=target.stat().st_size:
            raise RuntimeError('Mirror mismatch: '+str(source))
        entries.append(dict(name=source.name,bytes=int(source.stat().st_size),sha256=source_sha,mirrored_sha256=target_sha,identical=True))
    manifest=dict(schema='banc-local-information-mirror/1',source_directory=str(WS),mirror_directory=str(DS),
                  file_count=len(entries),total_bytes=int(sum(e['bytes'] for e in entries)),
                  verification='every file re-read on both sides; SHA256 and byte size equal for all files',
                  physics_steps=0,neural_steps=0,files=entries)
    (DS/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(file_count=manifest['file_count'],total_bytes=manifest['total_bytes'],
                          all_identical=all(e['identical'] for e in entries),manifest=str(DS/'MANIFEST.json'))),flush=True)


if __name__=='__main__':main()
