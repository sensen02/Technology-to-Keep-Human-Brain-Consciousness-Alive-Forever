"""Mirror only this authorized audit directory and verify every copied byte."""
from pathlib import Path
import hashlib,json,shutil
SRC=Path(__file__).resolve().parent
DST=Path('/home/sensen/Desktop/engine修复验收/flybody-retarget/20261107T182940Z/neural-interface')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 DST.mkdir(parents=True,exist_ok=True)
 entries=[]
 for p in sorted(SRC.iterdir()):
  if not p.is_file() or p.name=='DELIVERY_VERIFICATION.json':continue
  q=DST/p.name
  if q.exists():
   assert sha(q)==sha(p), 'Existing Desktop artifact differs; refusing overwrite: '+str(q)
  else:shutil.copy2(p,q)
  assert sha(p)==sha(q)
  entries.append({'workspace_path':str(p),'desktop_path':str(q),'sha256':sha(p),'bytes':p.stat().st_size})
 v={'status':'PASS','file_count':len(entries),'total_bytes_each_copy':sum(e['bytes'] for e in entries),'all_mirror_sha256_equal':True,'files':entries,'note':'Manifest artifact paths identify original workspace files; same-named Desktop copies verified byte-for-byte. Source data not duplicated.'}
 text=json.dumps(v,indent=2,ensure_ascii=False)+'\n'
 (SRC/'DELIVERY_VERIFICATION.json').write_text(text);(DST/'DELIVERY_VERIFICATION.json').write_text(text)
 assert sha(SRC/'DELIVERY_VERIFICATION.json')==sha(DST/'DELIVERY_VERIFICATION.json')
 print(json.dumps({k:x for k,x in v.items() if k!='files'},indent=2))
if __name__=='__main__':main()
