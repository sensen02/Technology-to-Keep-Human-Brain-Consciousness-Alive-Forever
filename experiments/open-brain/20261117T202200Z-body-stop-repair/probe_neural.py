from common import *
from rollout_neural import run
write(HERE/'NEURAL_ONLY_PREREG.json',dict(reason='Phase projection fixes original lift but creates matched high-block regressions and new-phase stop failures. Test already trained sensory weights with original readout to separate learning from gating.',
 checkpoint=str(HERE/'FIT_REPAIR/epoch030.pt'),sha256=sha(HERE/'FIT_REPAIR/epoch030.pt'),new_training=0,
 development='181/263/1151/1723 and now-used2039/2711/3253',new_held=[4313,5903],
 selection='full physical gates, then pass count; no reused-held claims',new_held_not_yet_run=True))
results=[]
for height,seed in ((.15,1723),(.30,181),(.30,263),(0.,2711)):
 results.append(run(height,seed,'student',HERE/'NEURAL_PROBE'/f'{height}_{seed}',HERE/'FIT_REPAIR/epoch030.pt'))
write(HERE/'NEURAL_PROBE_RESULT.json',results)
