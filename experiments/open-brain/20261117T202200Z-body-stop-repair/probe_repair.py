from common import *
from rollout_repair import run
results=[]
for height,seed in ((.15,1723),(.30,263)):
 results.append(run(height,seed,'student',HERE/'ROUTING_PROBE'/f'{height}_{seed}',HERE/'ROUTING_FIXED.pt'))
write(HERE/'ROUTING_PROBE_RESULT.json',results)
