"""Rank local corrective reflexes on two actual orchard drive sequences."""
import json
import numpy as np
from context import HERE,PREV,tr
import body_search as bench
from reflex import controller
def main():
    scenes=[]
    for site,record in [('CLEAR','R2_CLEAR'),('DEBRIS','R1_DEBRIS')]:
        crop=tr.Crop(site);env,bank=tr.build(crop)
        env._time_limit=bench.DURATION+.02
        env.task.check_termination=lambda _:False
        with np.load(PREV/record/'NECK.npz') as z:
            for i,d in enumerate(z['delays']):
                assert np.array_equal(z['emitted'][:-d,i],z['arrived'][d:,i])
            drive=z['drive']
        assert len(drive)>=int(bench.DURATION*1e4)
        scenes.append((crop,env,bank,drive))
    bench.make=controller
    candidates=[(0,0,0,1),(.55,.86,.72,1),(.9,.35,1,1),(.3,.95,.95,.75),(.8,.8,.85,1.25),(.18,.38,.9,1.35),(.95,.95,1,.65),(.55,.62,.6,1.55)]
    history=[]
    for i,params in enumerate(candidates):
        rows=[bench.run(*s,params) for s in scenes]
        score=(sum(r['gate'] for r in rows),min(r['return_value'] for r in rows),sum(r['return_value'] for r in rows))
        item=dict(index=i,params=params,sites=rows,score=score)
        history.append(item)
        (HERE/'BODY_TRAIN.json').write_text(json.dumps(history,indent=2)+'\n')
        print(i,score,flush=True)
    chosen=max(history[1:],key=lambda r:r['score'])
    (HERE/'SELECTED.json').write_text(json.dumps(chosen,indent=2)+'\n')
    for _,env,_,_ in scenes:env.close()
if __name__=='__main__':main()
