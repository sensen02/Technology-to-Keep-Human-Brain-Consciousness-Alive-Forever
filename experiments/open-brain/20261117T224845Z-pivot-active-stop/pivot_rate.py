from scope import *
from run_body_v2 import run
from metrics import compare
import argparse
p=argparse.ArgumentParser();p.add_argument('variant',type=int);a=p.parse_args();rows=[]
for f in ('pivot_left','pivot_right'):
 t=run(f,a.variant,'teacher',HERE/'RATE'/f'{f}_{a.variant}_teacher',WEIGHT)
 s=run(f,a.variant,'student',HERE/'RATE'/f'{f}_{a.variant}_student',WEIGHT)
 rows.append(compare(t,s))
write(HERE/f'RATE_{a.variant}_RESULT.json',rows)
