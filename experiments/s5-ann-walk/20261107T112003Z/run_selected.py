"""Strict evaluation and neural learning of the single physically selected candidate.
Loads one measured parameter vector only; never combines independent search gains.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T102842Z/run_student.py')
spec=importlib.util.spec_from_file_location('original_broadlift_neural_runtime',SOURCE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
n=b.n
SEARCH=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T112003Z')
ORIGINAL_REFERENCE=n.Reference
SELECTION=None


class SelectedReference:
    def __init__(self,p):
        assert SELECTION is not None
        self.base=ORIGINAL_REFERENCE(p)
        self.lo=self.base.lo;self.hi=self.base.hi;self.q0=self.base.q0
        self.offset=np.zeros(42)
        names=[p.m.joint(int(j)).name for j in p.jids]
        for name,value in zip(SELECTION['joints'],SELECTION['parameters_rad']):
            self.offset[names.index(name)]=value
        grid=self.at(np.linspace(0,2*np.pi,4096,endpoint=False))
        self.center=(grid.max(0)+grid.min(0))/2
        self.scale=np.maximum((grid.max(0)-grid.min(0))/2,.025)

    def at(self,phase):
        return np.clip(self.base.at(phase)+self.offset,self.lo,self.hi)

    def warm(self,t):
        u=np.clip(t/.5,0.,1.);ramp=u**3*(10-15*u+6*u*u)
        return np.clip(self.base.warm(t)+ramp*self.offset,self.lo,self.hi)


def prereg():
    n.OUT.mkdir(parents=True,exist_ok=True)
    path=n.OUT/'PREREGISTRATION.json'
    if path.exists():return
    n.dump(path,dict(importance='核心必需',stage='S5 strict independent validation of one measured parameter candidate',
        selection=SELECTION,search=str(SEARCH/'BEST_PARAMETERS.json'),search_sha256=n.sha(SEARCH/'BEST_PARAMETERS.json'),
        reference=str(n.GAIT),reference_sha256=n.sha(n.GAIT),
        changes='Exactlyone premeasuredsinglejointoffset from13trialsearch; sameS5offsetrampfirst.5s, nativeangleclipexplicit. Noaddingindependentcoordinateeffects.',
        physical='unchangedfull71mesh49/48/42neq0/free root/gravity/contact/friction/nativeforcecaps; finitePD128/30rad/s targetrate, physics12.5us/command1ms/policy5ms',
        control='teacher modecontinuousalgorithm only; ANNdeployment1ssamephysicalwarmupthenfrozenlearnedweights+causalownhistory, no clock/phase/lookup/teacher',
        assessment='fullrelease unchangednativeDgates/actualsixlegsteps/heading/materialpointslip<=.5 perleg, independentphysicsreplay at0.5ms, heldoutduration. Search5msscores notacceptance.',
        failure='No studentorfinalsuccessinferredfromsearchbest orstatisticalranking; strictphysicspassneeded.',
        sources={str(p):n.sha(p) for p in [Path(__file__),SOURCE,Path(n.__file__),SEARCH/'BEST_PARAMETERS.json',n.GAIT]},
        units='native; originalbrainnecessity/absoluteSIforce/biologicalvalidity NOT_ESTABLISHED'))


def main():
    global SELECTION
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['teacher','train','ann','hold','motor_cut','offline'])
    ap.add_argument('--seconds',type=float,default=4.);ap.add_argument('--tag',default=None)
    ap.add_argument('--steps',type=int,default=3000)
    ap.add_argument('--name',default='MODEL_SELECTED')
    args=ap.parse_args()
    record=json.loads((SEARCH/'BEST_PARAMETERS.json').read_text());SELECTION=record['selected']
    assert SELECTION is not None and not SELECTION['score']['failed']
    assert SELECTION['score']['completed_full_duration']
    n.OUT=SEARCH/'selected';n.Reference=SelectedReference;n.prereg=prereg
    prereg()
    model=n.OUT/(args.name+'.npz')
    if args.mode=='train':
        model=n.train_model(args.steps,args.name);n.offline(model)
    elif args.mode=='offline':n.offline(model)
    else:
        tag=args.tag or (args.mode+'_selected_'+format(args.seconds,'g')+'s')
        if args.mode=='teacher':model=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T102842Z/MODEL.npz')
        assert model.exists(),model
        n.simulate(model,tag,args.seconds,args.mode,False)


if __name__=='__main__':main()
