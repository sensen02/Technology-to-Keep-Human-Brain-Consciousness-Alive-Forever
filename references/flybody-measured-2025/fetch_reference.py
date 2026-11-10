"""Download data only, pin public source identity; never execute archive contents."""
from pathlib import Path
from urllib.request import Request,urlopen
import hashlib
import json
import os
import time
import zipfile

HERE=Path(__file__).resolve().parent
URL='https://ndownloader.figshare.com/files/51196868'
SIZE=3019999672
MD5='de7d5367850430adbe03a3783967c712'


def main():
    archive=HERE/'datasets_walking-imitation.zip'
    partial=HERE/'datasets_walking-imitation.zip.part'
    if archive.exists() or partial.exists():raise FileExistsError('do not overwrite existing reference download')
    metadata=dict(source_api='https://api.figshare.com/v2/articles/25309105',article_doi='10.25378/janelia.25309105.v4',
      paper='https://www.nature.com/articles/s41586-025-09029-4',official_code='https://github.com/TuragaLab/flybody',
      filename=archive.name,download_url=URL,expected_bytes=SIZE,expected_md5=MD5,
      license='GPL3.0+ per Figshare metadata; preserve original authors/attribution',
      use='measured/reference imitation data provenance and kinematic inspection; no direct59-to42actiontransfer, no physical model replacement',
      caveat='derived full-body trajectories fitted to measured keypoints; do not describe fitted unmeasured joints as directly observed',status='downloading')
    (HERE/'SOURCE.json').write_text(json.dumps(metadata,indent=2)+'\n')
    started=time.monotonic();md5=hashlib.md5();sha=hashlib.sha256();count=0
    try:
        with urlopen(Request(URL,headers={'User-Agent':'scientific-reference-verification/1.0'}),timeout=60) as response, partial.open('xb') as output:
            while True:
                data=response.read(4*1024*1024)
                if not data:break
                output.write(data);md5.update(data);sha.update(data);count+=len(data)
            output.flush();os.fsync(output.fileno())
        if count!=SIZE or md5.hexdigest()!=MD5:raise ValueError('reference archive size/MD5 mismatch')
        os.replace(partial,archive)
        with zipfile.ZipFile(archive) as z:
            members=[dict(name=i.filename,uncompressed=i.file_size,compressed=i.compress_size,crc32=i.CRC) for i in z.infolist()]
            # Listing only. Do not extract arbitrary paths or execute downloaded code.
        metadata.update(status='verified_download',actual_bytes=count,actual_md5=md5.hexdigest(),
                        sha256=sha.hexdigest(),wall_s=time.monotonic()-started,members=members)
        (HERE/'SOURCE.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(json.dumps(dict(status=metadata['status'],bytes=count,sha256=metadata['sha256'],members=len(members),wall_s=metadata['wall_s'])),flush=True)
    except BaseException as exc:
        if partial.exists():partial.unlink()
        metadata.update(status='failed',error=repr(exc),received_bytes=count,wall_s=time.monotonic()-started)
        (HERE/'SOURCE.json').write_text(json.dumps(metadata,indent=2)+'\n')
        raise

if __name__=='__main__':main()
