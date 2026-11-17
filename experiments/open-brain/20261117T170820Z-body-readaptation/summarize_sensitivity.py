"""Compare the original and retrained body under the same uncertain fixed parameters."""
import json
from context import HERE,NECK,first,hash_file

old=json.loads((NECK/'SENSITIVITY.json').read_text())['cases']
rows={}
for path in sorted((HERE/'SENSITIVITY').glob('*/RESULT.json')):
    result=json.loads(path.read_text())
    assert result['trace_sha256']==hash_file(path.parent/'TRACE.npz')
    before=old[path.parent.name]
    rows[path.parent.name]=dict(pass_=result['pass_'],baseline_pass=before['pass_'],gates=result['gates'],
          baseline_halt=before.get('braking'),retrained_halt=result.get('braking'))
assert len(rows)==30
first.write(HERE/'SENSITIVITY.json',dict(pass_=all(r['pass_'] for r in rows.values()),cases=rows,
          baseline_pass_count=sum(r['baseline_pass'] for r in rows.values()),
          retrained_pass_count=sum(r['pass_'] for r in rows.values())))
print('sensitivity old/new',sum(r['baseline_pass'] for r in rows.values()),sum(r['pass_'] for r in rows.values()),'of',len(rows))
