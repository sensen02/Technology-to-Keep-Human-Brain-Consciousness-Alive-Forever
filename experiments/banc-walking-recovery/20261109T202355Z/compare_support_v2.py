from pathlib import Path
import json
import support_probe as P
from core_v2 import Core
HERE=Path(__file__).resolve().parent
P.rc.Core=Core
runs=[]
for h,v,s in [(0,10,1),(10,30,1),(3,10,.5)]:
    summary,trace=P.run(h,v,s);runs.append(dict(summary=summary,trace=trace));print(json.dumps(summary),flush=True)
    (HERE/'SUPPORT_PROBE_V2.json').write_text(json.dumps(dict(label='COM-wrench corrected whole-body teacher diagnosis; not walking acceptance',runs=runs),indent=2)+'\n')
