"""Move only our superseded audit generations; preserve sealed bytes and old paths."""
from pathlib import Path
import hashlib,json,shutil
SRC=Path('/home/sensen/Desktop/engine修复验收/s4-initial-pose-version-audit')
OUT=Path('/run/media/sensen/Data2/cell_wound_prototype_legacy_archive/20261028T020000Z/initial-pose-audit')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inventory(p):return {str(f.relative_to(p)):digest(f) for f in sorted(p.rglob('*')) if f.is_file()}
def main():
    assert not OUT.exists(), 'Archive must be new; do not overwrite'
    old=[SRC/n for n in ('20261028T013000Z','20261028T014000Z','20261028T015000Z')]
    assert all(p.is_dir() and not p.is_symlink() for p in old)
    final=SRC/'20261028T016000Z/MANIFEST_SHA256.json'
    seal=json.loads(final.read_text())
    assert all(digest(Path(k))==v for k,v in seal.items())
    before={str(p):inventory(p) for p in old}
    OUT.mkdir(parents=True)
    rows=[]
    for p in old:
        target=OUT/p.name
        shutil.move(str(p),str(target))
        p.symlink_to(target,target_is_directory=True)
        assert inventory(target)==before[str(p)]
        assert inventory(p)==before[str(p)]
        rows.append(dict(original=str(p),archived=str(target),importance='重要辅助',purpose='Superseded or invalid read-only audit evidence; not a runnable approved version',files_sha256=before[str(p)],compatibility_symlink=True))
    assert all(digest(Path(k))==v for k,v in seal.items())
    record=dict(scope='Only this task superseded initial-pose audits; no original models, sessions, fixed stances or active runners moved',moves=rows,final_016_manifest_verified=True,warning='Earlier audit manifests reference a mutable desktop report whose hash was already superseded; preserved internal evidence unchanged',importance='重要辅助')
    (OUT/'ARCHIVE_MAP.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(archive=str(OUT),directories_moved=len(rows),files_preserved=sum(len(r['files_sha256']) for r in rows),final_manifest_verified=True),ensure_ascii=False))
if __name__=='__main__':main()
