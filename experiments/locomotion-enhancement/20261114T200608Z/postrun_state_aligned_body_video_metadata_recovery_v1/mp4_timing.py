"""Stdlib MP4 final sample shortening; no invented source frames or padded window.
Encoder must use no B frames, video timescale1000, moov after mdat. Full20ms tail no-op.
"""
from pathlib import Path
import struct

def boxes(data):
 pos=0
 while pos<len(data):
  if pos+8>len(data):raise ValueError('Truncated box')
  size,tag=struct.unpack('>I4s',data[pos:pos+8])
  if size<8 or pos+size>len(data):raise ValueError('Unsupported box size')
  yield tag,data[pos+8:pos+size]
  pos+=size

def box(tag,payload):return struct.pack('>I4s',len(payload)+8,tag)+payload

def shorten(path,duration_ms,last_frame_ms):
 if last_frame_ms==20:return dict(adjusted=False,last_sample_ms=20)
 if not 0<last_frame_ms<20:raise ValueError('Tail duration')
 path=Path(path);data=path.read_bytes();top=list(boxes(data));tags=[k for k,v in top]
 if tags.count(b'moov')!=1 or tags.index(b'moov')<tags.index(b'mdat'):raise ValueError('moov must follow mdat; do not use faststart')
 counts={k:0 for k in ('stts','mvhd','tkhd','mdhd','elst')}
 def rewrite(tag,payload):
  if tag in (b'moov',b'trak',b'mdia',b'minf',b'stbl',b'edts'):
   return b''.join(box(k,rewrite(k,v)) for k,v in boxes(payload))
  p=bytearray(payload)
  if tag==b'stts':
   if p[0]!=0:raise ValueError('stts version')
   count=struct.unpack('>I',p[4:8])[0];entries=[list(struct.unpack('>II',p[8+i*8:16+i*8])) for i in range(count)]
   if entries[-1][1]!=20:raise ValueError('Unexpected encoded50fps tail')
   if entries[-1][0]>1:entries[-1][0]-=1;entries.append([1,last_frame_ms])
   else:entries[-1][1]=last_frame_ms
   if sum(n*dt for n,dt in entries)!=duration_ms:raise ValueError('Actual MP4 duration disagreement')
   counts['stts']+=1;return bytes(p[:4])+struct.pack('>I',len(entries))+b''.join(struct.pack('>II',*x) for x in entries)
  if tag in (b'mvhd',b'mdhd'):
   if p[0]!=0 or struct.unpack('>I',p[12:16])[0]!=1000:raise ValueError('Expected version0 timescale1000')
   p[16:20]=struct.pack('>I',duration_ms);counts[tag.decode()]+=1
  elif tag==b'tkhd':
   if p[0]!=0:raise ValueError('tkhd version')
   p[20:24]=struct.pack('>I',duration_ms);counts['tkhd']+=1
  elif tag==b'elst':
   if p[0]!=0 or struct.unpack('>I',p[4:8])[0]!=1:raise ValueError('Single non-delayed edit required')
   p[8:12]=struct.pack('>I',duration_ms);counts['elst']+=1
  return bytes(p)
 new=b''.join(box(tag,rewrite(tag,payload)) for tag,payload in top)
 if any(counts[k]!=1 for k in ('stts','mvhd','tkhd','mdhd')):raise ValueError('Single-track timing required')
 temp=path.with_suffix('.timing.tmp');temp.write_bytes(new);temp.replace(path)
 return dict(adjusted=True,last_sample_ms=last_frame_ms,physical_duration_ms=duration_ms,boxes_updated=counts,media_bytes_unchanged=True)
