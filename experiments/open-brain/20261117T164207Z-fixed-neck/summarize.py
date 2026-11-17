import json
from pathlib import Path
from fixed_neck import HERE, hash_file, write

records={}
for path in sorted((HERE/'SENSITIVITY').glob('*/RESULT.json')):
 result=json.loads(path.read_text())
 identity=json.loads((path.parent/'SENSITIVITY.json').read_text())
 runtime=json.loads((path.parent/'RUNTIME.json').read_text())
 assert runtime['original_result_sha256']==hash_file(path)
 assert result['trace_sha256']==hash_file(path.parent/'TRACE.npz')
 records[path.parent.name]={'variant':identity,'pass_':result['pass_'],'gates':result['gates'],
    'active_speed_mm_s':result.get('active_speed_mm_s'), 'yaw_deg':result.get('yaw_deg',result.get('yaw_degrees')),
    'braking':result.get('braking')}
assert len(records)==30,len(records)
write(HERE/'SENSITIVITY.json',dict(pass_=all(r['pass_'] for r in records.values()),cases=records))
print('Sensitivity',sum(r['pass_'] for r in records.values()),'of',len(records))
for name,row in records.items():
 if not row['pass_']:print(name,[k for k,v in row['gates'].items() if not v])
