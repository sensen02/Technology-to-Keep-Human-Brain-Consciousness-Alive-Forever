"""Bounded public HDF5 range reader; preserves fetched raw blocks and hashes."""
import io,json,urllib.request,hashlib,argparse
from pathlib import Path
import h5py
URL='https://drive.usercontent.google.com/download?id=1jTAdpwAaAJprKIBgn17PVJseQlFqYrSn&export=download&confirm=t'
class Remote(io.RawIOBase):
 def __init__(self,out):self.pos=0;self.size=3303870746;self.cache={};self.out=out;self.block=262144
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  return self.pos
 def readinto(self,buf):
  data=self.read(len(buf));buf[:len(data)]=data;return len(data)
 def read(self,n=-1):
  if n<0:n=self.size-self.pos
  end=min(self.pos+n,self.size);parts=[]
  while self.pos<end:
   k=self.pos//self.block;start=k*self.block;stop=min(start+self.block,self.size)-1
   if k not in self.cache and (self.out/f'block_{k:05}.bin').exists():self.cache[k]=(self.out/f'block_{k:05}.bin').read_bytes()
   if k not in self.cache:
    if len(self.cache)>=160:raise RuntimeError('bounded fetch budget 40MiB exceeded')
    req=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{stop}'})
    with urllib.request.urlopen(req,timeout=45) as r:
     assert r.status==206 and r.headers['Content-Range'].startswith(f'bytes {start}-')
     data=r.read();assert len(data)==stop-start+1
    self.cache[k]=data;(self.out/f'block_{k:05}.bin').write_bytes(data)
    print('range',k,flush=True)
   local=self.pos-start;take=min(end-self.pos,len(self.cache[k])-local);parts.append(self.cache[k][local:local+take]);self.pos+=take
  return b''.join(parts)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 r=Remote(a.output)
 def extract(g):
  result={}
  for name in g:
   d=g[name]
   if isinstance(d,h5py.Group):
    if name in ('fly_ids','source_flies'):
     children=[d.id.get_objname_by_idx(0).decode()];result[name]={n:str(d[n][()]) for n in children}
    else:result[name]=extract(d)
   else:
    v=d[()]
    if getattr(v,'dtype',None) is not None and v.dtype.kind not in 'OSU':
     import numpy as np
     np.save(a.output/(d.name.strip('/').replace('/','__')+'.npy'),v,allow_pickle=False)
    result[name]=str(v) if getattr(v,'shape',())==() else dict(shape=list(v.shape),values=v.tolist() if v.size<200 else None)
  return result
 try:
  with h5py.File(r,'r') as f:
   print('ROOT_ATTRS',dict(f.attrs),flush=True)
   keys=[f.id.get_objname_by_idx(j).decode() for j in range(3)]
   print('FIRST_KEYS',keys,flush=True)
   for group in ('info/kp_names','info/names_qpos','info/names_xpos','info/site_names_egocentric','bout_0000/geometric_angles'):
    g=f[group];print('GROUP',group,'COUNT',g.id.get_num_objs(),'FIRST',g.id.get_objname_by_idx(0),flush=True)
    if group!='bout_0000/geometric_angles':
     values=[g[str(j)][()].decode() for j in range(g.id.get_num_objs())];(a.output/(group.replace('/','__')+'.json')).write_text(json.dumps(values,indent=2));print('NAMES',values,flush=True)
    else:
     meta=extract(g);(a.output/'geometric_angles.json').write_text(json.dumps(meta,default=str,indent=2))
   for group in ('info/fly_ids','info/source_flies'):
    d=f[group];first=d.id.get_objname_by_idx(0).decode();print(group,first,d[first][()],flush=True)
   for k in []:
    v=f[k];print(k,type(v).__name__,v.shape if isinstance(v,h5py.Dataset) else list(v.keys())[:60],dict(v.attrs),flush=True)
    for name in v:
     d=v[name];print('FIELD',k,name,d.shape if isinstance(d,h5py.Dataset) else 'group',dict(d.attrs),flush=True)
     if isinstance(d,h5py.Dataset):
      data=d[()];print('VALUE',name,str(data)[:1800],flush=True)
      import numpy as np
      np.save(a.output/(k+'__'+name+'.npy'),data,allow_pickle=False)
 finally:
  (a.output/'RANGE_MANIFEST.json').write_text(json.dumps(dict(url=URL,total_remote_bytes=r.size,blocks={f'block_{k:05}.bin':hashlib.sha256(v).hexdigest() for k,v in r.cache.items()}),indent=2))
if __name__=='__main__':main()
