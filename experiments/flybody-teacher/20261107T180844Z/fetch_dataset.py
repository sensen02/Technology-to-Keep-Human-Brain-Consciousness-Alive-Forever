"""Stream verified official archive; no policy execution, preserve source identity."""
from pathlib import Path
import urllib.request,hashlib,json,time,zipfile,shutil
ROOT=Path(__file__).resolve().parent
OUT=Path('/home/sensen/Desktop/engine修复验收/flybody-teacher/20261107T180844Z')
URL='https://ndownloader.figshare.com/files/51196868'
EXPECTED='de7d5367850430adbe03a3783967c712'
SIZE=3019999672

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    p=ROOT/'walking_dataset.zip';temp=ROOT/'walking_dataset.zip.part'
    assert not p.exists() and not temp.exists()
    m=hashlib.md5();h=hashlib.sha256();count=0;t=time.monotonic();last=t
    with urllib.request.urlopen(URL,timeout=120) as response,temp.open('xb') as f:
        header=dict(response.headers.items());resolved=response.url
        while True:
            b=response.read(4*1024*1024)
            if not b:break
            f.write(b);m.update(b);h.update(b);count+=len(b)
            if time.monotonic()-last>20:
                print(json.dumps(dict(download_bytes=count,expected_bytes=SIZE,wall_s=time.monotonic()-t)),flush=True);last=time.monotonic()
    assert count==SIZE,(count,SIZE)
    assert m.hexdigest()==EXPECTED,m.hexdigest()
    temp.rename(p)
    with zipfile.ZipFile(p) as z:
        entries=[dict(name=i.filename,bytes=i.file_size,compressed=i.compress_size) for i in z.infolist()]
        assert sum(i.file_size for i in z.infolist())<20*1024**3
        for i in z.infolist():
            rel=Path(i.filename)
            assert not rel.is_absolute() and '..' not in rel.parts
            if i.is_dir():continue
            target=ROOT/'dataset'/rel;target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(i) as src,target.open('xb') as dest:shutil.copyfileobj(src,dest,4*1024*1024)
    report=dict(url=URL,resolved_url=resolved,bytes=count,md5=m.hexdigest(),sha256=h.hexdigest(),archive_entries=entries,header=header,wall_s=time.monotonic()-t,archive=str(p))
    with (OUT/'DATASET_ASSET.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
