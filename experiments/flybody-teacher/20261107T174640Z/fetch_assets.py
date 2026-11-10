"""Fetch verified pretrained policies; freeze source identity without running models."""
from pathlib import Path
import hashlib,json,urllib.request,zipfile,io,subprocess
ROOT=Path(__file__).resolve().parent
OUT=Path('/home/sensen/Desktop/engine修复验收/flybody-teacher/20261107T174640Z')
URL='https://ndownloader.figshare.com/files/44815195'
EXPECTED='12934d5a1c60631a710bc2b6d297d3ce'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    dest=ROOT/'policies'
    if dest.exists():raise FileExistsError(dest)
    with urllib.request.urlopen(URL,timeout=60) as r:data=r.read()
    assert hashlib.md5(data).hexdigest()==EXPECTED
    files=[]
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert z.testzip() is None
        for item in z.infolist():
            p=Path(item.filename)
            assert not p.is_absolute() and '..' not in p.parts
            if item.is_dir():continue
            blob=z.read(item)
            f=dest/p;f.parent.mkdir(parents=True,exist_ok=True)
            with f.open('xb') as out:out.write(blob)
            files.append(dict(path=str(f),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()))
    source=ROOT/'official-flybody'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    manifest=dict(url=URL,archive_bytes=len(data),md5=EXPECTED,sha256=hashlib.sha256(data).hexdigest(),commit=commit,source=str(source),files=files,physics_steps=0,training_steps=0)
    with (OUT/'ASSETS.json').open('x') as f:json.dump(manifest,f,indent=2)
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
