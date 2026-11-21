import json
import numpy as np
from context import HERE,PREV,tr
from reflex import controller
import body_search as bench
def main():
    full=json.loads((HERE/'SELECTED.json').read_text())['params']
    bench.make=controller
    rows={}
    for site,record in [('CLEAR','R2_CLEAR'),('DEBRIS','R1_DEBRIS')]:
        crop=tr.Crop(site);env,bank=tr.build(crop);env._time_limit=bench.DURATION+.02
        env.task.check_termination=lambda _:False
        with np.load(PREV/record/'NECK.npz') as f:drive=f['drive']
        rows[site]={}
        for name,params in [('pace_only',[0,0,0,full[-1]]),('reflex_only',full[:3]+[1.]),('repeat_full',full)]:
            rows[site][name]=bench.run(crop,env,bank,drive,params)
        env.close()
        (HERE/'ABLATION.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
