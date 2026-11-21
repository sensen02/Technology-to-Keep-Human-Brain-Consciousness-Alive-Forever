from scope import *
from run_body_v3 import run
from metrics_v2 import compare
rows=[]
for f in ('stop60','stop74a','stop74b'):
 t=run(f,7,'teacher',HERE/'SEQUENCED'/f'{f}_7_teacher',WEIGHT);s=run(f,7,'student',HERE/'SEQUENCED'/f'{f}_7_student',WEIGHT)
 rows.append(compare(t,s))
write(HERE/'SEQUENCED_RESULT.json',rows)
