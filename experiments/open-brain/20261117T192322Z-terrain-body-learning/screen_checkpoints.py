"""Development-only screening of the already saved second-round checkpoints."""
from shared import *
from rollout_plus import run
results=[]
for epoch in (10,20,40):
    checkpoint=HERE/'FIT2'/f'epoch{epoch:03d}.pt'
    result=run(.30,263,'student',HERE/'DEV_SCREEN'/f'epoch{epoch:03d}_block030_263',checkpoint)
    results.append(dict(epoch=epoch,checkpoint=str(checkpoint),result=result))
    write(HERE/'DEV_SCREEN_RESULT.json',results)
