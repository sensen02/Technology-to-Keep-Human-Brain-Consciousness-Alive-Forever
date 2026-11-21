from scope import *
from run_body_v3 import run
selected=json.loads((HERE/'SELECTION.json').read_text());checkpoint=Path(selected['checkpoint']);rows=[]
for family in ('arc_right','a01_right','stride_right','back','speed'):
 rows.append(run(family,2,'student',HERE/'REPLAY'/f'{family}_2_student',checkpoint))
write(HERE/'REPLAY_RESULT.json',rows)
