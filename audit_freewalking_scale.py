"""Offline-only cached HDF5 metadata probe: never requests additional bytes."""
from pathlib import Path
import io,json,hashlib
import h5py
D=Path('/home/sensen/Desktop/engine修复验收/real-freewalking-3d/20261027T150000Z')
O=Path('/home/sensen/Desktop/engine修复验收/real-freewalking-scale/20261027T151500Z')
class Cached(io.RawIOBase):
 def __init__(self):self.pos=0
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,p,w=0):self.pos=p if w==0 else self.pos+p if w==1 else 3303870746+p;return self.pos
 def readinto(self,b):v=self.read(len(b));b[:len(v)]=v;return len(v)
 def read(self,n=-1):
  if n<0:raise OSError('unbounded read forbidden')
  parts=[];end=self.pos+n
  while self.pos<end:
   k=self.pos//262144;p=D/f'block_{k:05}.bin'
   if not p.exists():raise OSError('uncached block '+str(k))
   v=p.read_bytes();offset=self.pos%262144;take=min(end-self.pos,len(v)-offset)
   if take<=0:raise OSError('invalid cached block')
   parts.append(v[offset:offset+take]);self.pos+=take
  return b''.join(parts)
def main():
 result={}
 with h5py.File(Cached(),'r') as f:
  for path in ('/','bout_0000','bout_0000/kp_data','info','info/curation'):
   try:
    g=f[path];v=dict(attrs={k:str(x) for k,x in g.attrs.items()})
    if isinstance(g,h5py.Group):v['children']=([g.id.get_objname_by_idx(j).decode() for j in range(min(g.id.get_num_objs(),30))] if path!='/' else 'not enumerated: avoid full metadata traversal')
    result[path]=v
   except (OSError,KeyError) as e:result[path]=dict(unavailable=str(e))
 (O/'OFFLINE_METADATA.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
