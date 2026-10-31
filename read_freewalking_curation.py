"""Bounded curation-only followup. Never enumerates root or downloads whole HDF5."""
from pathlib import Path
import argparse,json,hashlib,urllib.request
import h5py
from audit_freewalking_scale import Cached,D
URL='https://drive.usercontent.google.com/download?id=1jTAdpwAaAJprKIBgn17PVJseQlFqYrSn&export=download&confirm=t'
class Reader(Cached):
 def __init__(self,o):super().__init__();self.o=o;self.fetched=[]
 def read(self,n=-1):
  if n<0:raise OSError('unbounded read prohibited')
  parts=[];end=self.pos+n
  while self.pos<end:
   k=self.pos//262144;old=D/f'block_{k:05}.bin';p=old if old.exists() else self.o/f'block_{k:05}.bin'
   if not p.exists():
    if len(self.fetched)>=12:raise OSError('curation-only new block cap 12 exceeded')
    lo=k*262144;hi=min(lo+262144,3303870746)-1
    with urllib.request.urlopen(urllib.request.Request(URL,headers={'Range':f'bytes={lo}-{hi}'}),timeout=45) as r:
     assert r.status==206 and r.headers['Content-Range']==f'bytes {lo}-{hi}/3303870746'
     data=r.read();assert len(data)==hi-lo+1
    p.write_bytes(data);self.fetched.append(dict(block=k,range=[lo,hi],sha256=hashlib.sha256(data).hexdigest()));print('FETCH',lo,hi,flush=True)
   data=p.read_bytes();offset=self.pos%262144;take=min(end-self.pos,len(data)-offset)
   if take<=0:raise OSError('invalid block')
   parts.append(data[offset:offset+take]);self.pos+=take
  return b''.join(parts)
def value(v):
 if isinstance(v,bytes):return v.decode()
 if hasattr(v,'tolist'):return value(v.tolist())
 if isinstance(v,list):return [value(x) for x in v]
 return v
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output;o.mkdir(parents=True,exist_ok=False);r=Reader(o);out={};error=None
 try:
  with h5py.File(r,'r') as f:
   g=f['info/curation'];out['attrs']={k:value(v) for k,v in g.attrs.items()}
   names=[g.id.get_objname_by_idx(j).decode() for j in range(g.id.get_num_objs())];out['fields']={};print('FIELDS',names,flush=True)
   for name in names:
    d=g[name];record=dict(type=type(d).__name__,attrs={k:value(v) for k,v in d.attrs.items()})
    if isinstance(d,h5py.Dataset):
     record.update(shape=list(d.shape),dtype=str(d.dtype))
     # Only first row belonging to first curated bout; scalar metadata complete.
     record['value']=value(d[()] if d.shape==() else d[0])
    else:
     record['children']=[d.id.get_objname_by_idx(j).decode() for j in range(min(20,d.id.get_num_objs()))]
     if '0' in d and isinstance(d['0'],h5py.Dataset):record['first_value']=value(d['0'][()])
     if 'bout_0000' in d and isinstance(d['bout_0000'],h5py.Dataset):record['bout_0000_value']=value(d['bout_0000'][()])
    out['fields'][name]=record
 except Exception as e:error=type(e).__name__+': '+str(e)
 finally:
  (o/'CURATION_METADATA.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
  (o/'READ_RECEIPT.json').write_text(json.dumps(dict(url=URL,fetched=r.fetched,error=error,exit_code=1 if error else 0),indent=2))
 print(json.dumps(dict(metadata=out,error=error),ensure_ascii=False))
 if error:raise SystemExit(1)
if __name__=='__main__':main()
