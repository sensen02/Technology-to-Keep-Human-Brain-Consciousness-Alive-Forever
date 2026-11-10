"""One cadence control on the broad-lift, measured single-offset reference.
Old narrow-lift slow-cadence result is not silently reused for this new geometry.
"""
import importlib.util
from pathlib import Path
import sys
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T112003Z/run_selected.py')
spec=importlib.util.spec_from_file_location('measured_broadlift_reference',SOURCE)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
n=r.n
n.FREQ=.5
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T114023Z')
original_prereg=r.prereg


def prereg():
    # r.main() alwayssetsouttoSEARCH/selected; redirectjustbeforewriting/running.
    n.OUT=OUT
    p=OUT/'PREREGISTRATION.json'
    OUT.mkdir(parents=True,exist_ok=True)
    if p.exists():return
    n.dump(p,dict(importance='核心必需',stage='S5 isolatedcadencecheck withnewbroad-lift measuredreference',
        parameters=r.SELECTION['parameters_rad'],joints=r.SELECTION['joints'],
        baseline='same RHtrochanteryaw+.02 measuredcandidate at1Hz allD/steps/headingpass buthindlegs slip>.5',
        change='frequency1->.5Hz ONLY, whole6s total gives5srelease; same broadlift/stance geometry/offset/gain128/nativeforce/ratecaps',
        hypothesis='Existingnarrow-lift .5Hz failure cannotsettle inertiaquestion afterswing-drag reduction. RH middlestancedynamiccapsjustifyonecadencecontrol; no frequencysearch.',
        measurement='FullnativeDgates,actualsixlegstep/heading/materialpointslip<=.5 perleg,independentreplay0.5ms; old5mssearchscoresnotusedforacceptance',
        control='continuousalgorithmteacheronly, currentoldMLP inertlogger; NOTneuralgoal',
        search_budget_relation='Outside26parameter-searchtrialscope, oneindependentlypreregisteredcadencecontrol; actualphysicscallsreported',
        physical='originalfull71mesh49/48/42neq0/freeroot/gravity/contact/friction/nativecaps, PD128 and30rad/s; no actualrootwrites/forces',
        sources={str(q):n.sha(q) for q in [Path(__file__),SOURCE,Path(n.__file__),r.SEARCH/'BEST_PARAMETERS.json',n.GAIT]},
        units='native; absoluteSIforce/originalbrainnecessity/biologicalvalidity NOT_ESTABLISHED'))

r.prereg=prereg

if __name__=='__main__':r.main()
