from scope import *
from run_body_v2 import run
from metrics import compare
rows=[]
for f in ('stop60','stop74a','stop74b'):
 t=run(f,2,'teacher',HERE/'HELD'/f'{f}_2_teacher',WEIGHT);s=run(f,2,'student',HERE/'HELD'/f'{f}_2_student',WEIGHT)
 rows.append(compare(t,s))
write(HERE/'HELD_STOP_RESULT.json',rows)
