"""Single lower-air-excursion physical control on the measured reference offset.
The selected RH+.02 offset, finitePD128 and everyoriginalphysicalbound are held.
"""
import importlib.util
from pathlib import Path
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T112003Z/run_selected.py')
spec=importlib.util.spec_from_file_location('singleoffset_fullphysics_runtime',SOURCE)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
n=r.n
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T114444Z')
n.GAIT=ROOT/'GAIT.npz'
OUT=ROOT/'neural'


def prereg():
    n.OUT=OUT;OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'PREREGISTRATION.json'
    if p.exists():return
    n.dump(p,dict(importance='核心必需',stage='S5 one lower-air-leg-excursion physical reference control',
        reference=str(n.GAIT),reference_sha256=n.sha(n.GAIT),parameters=r.SELECTION['parameters_rad'],joints=r.SELECTION['joints'],
        baseline='112003strictselected+.02RHtrochanteryaw,1Hz,gain128,oldlift.25; D/steps/headingpass, hindmaterialslipfails',
        change='Broadliftpeak.25->.18 ONLYinreferenceZ, originalXY/duty/sourceSTANCEq/offsets andmeasured+.02 targetoffset held; actualwholefootplateauguard>=.1native unchanged',
        physical='samefull71mesh49/48/42neq0/free root/gravity/contact/friction/nativecaps,finitePD128,targetrate30rad/s,dt12.5us,nomuscle/rootforceorstatewrites',
        control='continuousalgorithmteacher for initialscreen; ANNmode onlywithnewtrainedmodel andgenuine1sphysicalwarmupthenpureweights/causalhistory. Nevercountsasoriginalconnectome.',
        assessment='Wholepost1s originalDgates/actualalllegunloading/heading/independentreplay andmaterialslip<=.5 perlegat0.5ms. Offlinereferenceacceptanceisnotphysicalgoal.',
        sources={str(p):n.sha(p) for p in [Path(__file__),SOURCE,Path(n.__file__),n.GAIT,r.SEARCH/'BEST_PARAMETERS.json']},
        units='native; absoluteSIforce/biologicalvalidity/originalbrainnecessity NOT_ESTABLISHED'))

r.prereg=prereg

if __name__=='__main__':r.main()
