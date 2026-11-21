from common import *
from rollout_repair import run
results=[]
for height,seed in ((.15,1723),(.30,263),(.30,181)):
 results.append(run(height,seed,'student',HERE/'FIT_PROBE'/f'{height}_{seed}',HERE/'FIT_REPAIR/epoch030.pt'))
write(HERE/'FIT_PROBE_RESULT.json',results)
