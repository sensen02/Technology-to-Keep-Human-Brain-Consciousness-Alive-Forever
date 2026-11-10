"""Extract one checksum-verified small measured reference member, inspect metadata only."""
from pathlib import Path
import hashlib
import json
import os
import sys
import zipfile
import h5py
import numpy as np

HERE=Path(__file__).resolve().parent
MEMBER='walking-dataset-small_female-only_snippets-100_min-len-0.5s_trk-files-0-9.hdf5'


def main():
    meta=json.loads((HERE/'SOURCE.json').read_text())
    assert meta['status']=='verified_download'
    archive=HERE/meta['filename']
    assert archive.stat().st_size==meta['actual_bytes']
    output=HERE/'walking_reference_small.hdf5';partial=HERE/'walking_reference_small.hdf5.part'
    if output.exists() or partial.exists():raise FileExistsError('refuse overwriting reference extraction')
    sha=hashlib.sha256();size=0
    try:
        with zipfile.ZipFile(archive) as z:
            info=z.getinfo(MEMBER)
            assert info.file_size==94818544 and info.CRC==1918851071
            with z.open(info) as stream,partial.open('xb') as target:
                while True:
                    block=stream.read(1024*1024)
                    if not block:break
                    target.write(block);sha.update(block);size+=len(block)
                target.flush();os.fsync(target.fileno())
        assert size==94818544
        os.replace(partial,output)
    finally:
        if partial.exists():partial.unlink()
    schema=[]
    with h5py.File(output,'r') as h5:
        def visit(name,obj):
            row=dict(name=name,type='dataset' if isinstance(obj,h5py.Dataset) else 'group')
            if isinstance(obj,h5py.Dataset):row.update(shape=list(obj.shape),dtype=str(obj.dtype))
            if obj.attrs:
                row['attributes']={k:repr(v)[:500] for k,v in obj.attrs.items()}
            schema.append(row)
        h5.visititems(visit)
        root_attrs={k:repr(v)[:500] for k,v in h5.attrs.items()}
        root_keys=list(h5.keys())
    report=dict(status='small_member_verified_and_schema_inspected',archive_sha256=meta['sha256'],
                member=MEMBER,member_crc32=1918851071,bytes=size,sha256=sha.hexdigest(),
                output=str(output),reader_python=sys.executable,h5py=h5py.__version__,numpy=np.__version__,
                root_keys=root_keys,root_attributes=root_attrs,schema=schema,
                limitations=['schema inspection only; joint/frame/unit mapping to42actuator derived body not established',
                             'no direct action transfer, body replacement or behavioral success from this dataset'])
    (HERE/'REFERENCE_SCHEMA.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],bytes=size,root_keys=root_keys,schema_entries=len(schema),sha256=report['sha256'])),flush=True)

if __name__=='__main__':main()
