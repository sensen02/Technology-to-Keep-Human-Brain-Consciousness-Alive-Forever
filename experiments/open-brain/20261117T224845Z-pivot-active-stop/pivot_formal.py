from scope import *
from run_body_v3 import run
from metrics_v2 import compare
rows=[]
for f in ('pivot_left','pivot_right'):
 t=run(f,2,'teacher',HERE/'HELD'/f'{f}_2_teacher',WEIGHT);s=run(f,2,'student',HERE/'HELD'/f'{f}_2_student',WEIGHT);rows.append(compare(t,s))
write(HERE/'HELD_PIVOT_RESULT.json',rows)
