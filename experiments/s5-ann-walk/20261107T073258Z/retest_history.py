"""Matched old-model retest: prime causal history during original teacher warmup.
Do not retrain or change existing weights/body. Teacher calls end at handover;
only label placeholders are zero thereafter in inherited logger.
"""
import sys
from pathlib import Path
import json
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
OLD=ROOT/'experiments/s5-ann-walk/20261107T024636Z'
BASE=ROOT/'experiments/s5-ann-incremental/20261107T010000Z'
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T073258Z')
sys.path[:0]=[str(OLD),str(BASE),str(ROOT),str(ROOT/'tools')]
import walk_full as wf
import loop_walk as lw
import learn_stand as ls
from translator_th import NumpyTranslatorTH, THWrapper
from plant_probe import sha


def main():
    out=OUT/'history_fix';out.mkdir(parents=True,exist_ok=False)
    old=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T024636Z')
    specs=[('th_primed',wf.WalkRuntime,old/'th/learning/TRANSLATOR_TH.npz'),
           ('loop_primed',lw.LoopRuntime,old/'loop/learning/TRANSLATOR_LOOP.npz')]
    (out/'PREREGISTRATION.json').write_text(json.dumps(dict(
        purpose='fix genuine history reset at handover; matched old weights/body/gait1Hz',
        changes='fresh wrapper per episode; one observation per5ms including all warmup; no teacher evaluation after1s',
        caveat='TH metadata20 matches these existing models; postrelease teacher labels in inherited DATA are zeros, not ground truth',
        source=str(Path(__file__)),source_hash=sha(__file__),
        models={str(s[2]):sha(s[2]) for s in specs}),indent=2)+'\n')
    ls.O=out; (out/'learning').mkdir()
    for name,base,path in specs:
        wrapper=THWrapper(NumpyTranslatorTH(path),20)
        class Primed(base):
            def teacher(self,s):
                if float(s['d'].time)<1.-1e-8:
                    state=(s['a'].target_rad-self.p.q0[self.p.qa]).astype(np.float32)
                    wrapper(s['history'][None],state[None])
                    return super().teacher(s)
                return np.zeros(42)
        r=Primed().run(name,'ann',55,wrapper,seconds=3.,disturb=False)
        r['walk']=wf.walk_metrics(Path(r['folder']),1.)
        (Path(r['folder'])/'PRIMED_RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps(dict(tag=name,walk=r['walk'])),flush=True)


if __name__=='__main__':main()
