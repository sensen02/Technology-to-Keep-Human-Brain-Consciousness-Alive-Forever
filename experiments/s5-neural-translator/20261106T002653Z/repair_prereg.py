from pathlib import Path
import json,hashlib
import regression as r
p=r.O/'INPUT_HASHES_BEFORE.json'
old=json.loads(p.read_text());new={str(x):r.sha(x) for x in r.protected()};p.write_text(json.dumps(new,ensure_ascii=False,indent=2)+'\n');print('updated',len(new))
