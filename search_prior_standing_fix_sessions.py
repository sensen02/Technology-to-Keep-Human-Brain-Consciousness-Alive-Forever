"""Read-only bounded DSH conversation search; no credentials/settings or simulation."""
from pathlib import Path
import subprocess,json,re,hashlib
BASE=Path('/home/sensen/.dsh/sessions/--run-media-sensen-Data2--')
LIST=Path('/tmp/dsh-spill-3HEZa4/session-940f68ccff6a/3266538a8882-glob-results.txt')
OUT=Path('/home/sensen/Desktop/engine修复验收/s4-critical-fix-regression/20261028T020000Z')
OUT.mkdir(parents=True,exist_ok=True)
paths=[Path(s) for s in LIST.read_text().splitlines() if s.startswith(str(BASE)) and s.endswith('/session.jsonl.zstd')]
meta=[]
for p in paths:
 proc=subprocess.Popen(['zstd','-dc',str(p)],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
 try:
  x=json.loads(proc.stdout.readline())
 finally:
  proc.stdout.close();proc.wait()
 if x.get('delegationDepth',99)==0:meta.append(dict(path=str(p),meta=x))
(OUT/'MAIN_SESSIONS.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
print('Files',len(paths),'main',len(meta),flush=True)
rx=re.compile(r'(站不起来|站不住|无法站立|不能站立|不能站稳|关键问题|根本问题|根因|致命|springref|chirality|镜像|力臂|关节轴|被动弹簧)')
res=[]
for m in meta:
 p=Path(m['path']);proc=subprocess.Popen(['zstd','-dc',str(p)],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
 for ln,line in enumerate(proc.stdout,1):
  try:x=json.loads(line)
  except ValueError:continue
  typ=x.get('type','');data=x.get('data',{})
  if typ=='assistant/chunk':
   c=data.get('chunk',{})
   if c.get('type')!='block-end' or c.get('block',{}).get('type')!='text':continue
   text=c['block'].get('text','')
  elif isinstance(data,dict) and data.get('role') in ('assistant','user'):
   text=json.dumps(data.get('content',[]),ensure_ascii=False)
  else:continue
  if rx.search(text):res.append(dict(session=str(p),line=ln,type=typ,time=x.get('time'),text=text))
 proc.stdout.close();rc=proc.wait();assert rc==0
 print(p.parent.name,'matches',sum(v['session']==str(p) for v in res),flush=True)
(OUT/'MATCHED_RECORDS.json').write_text(json.dumps(res,ensure_ascii=False,indent=2))
print('RESULT',len(res),str(OUT),flush=True)
