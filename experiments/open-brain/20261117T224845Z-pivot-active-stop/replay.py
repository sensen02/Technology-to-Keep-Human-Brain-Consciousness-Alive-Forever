from scope import *
from run_body_v3 import run
rows=[]
for f in FAMILIES:rows.append(run(f,2,'student',HERE/'REPLAY'/f'{f}_2_student',WEIGHT))
write(HERE/'REPLAY_RESULT.json',rows)
