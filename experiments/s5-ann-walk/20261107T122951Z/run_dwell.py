"""One support-transition dwell timing control; original contact/model untouched."""
import importlib.util
from pathlib import Path
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T112003Z/run_selected.py')
spec=importlib.util.spec_from_file_location('single_offset_reference_runtime',SOURCE)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
n=r.n
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T122951Z')
n.GAIT=ROOT/'GAIT.npz';n.FREQ=.8
OUT=ROOT/'neural'


def prereg():
    n.OUT=OUT;OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'PREREGISTRATION.json'
    if p.exists():return
    n.dump(p,dict(importance='核心必需',stage='S5 one supporttransitiondwell physicaltimingcontrol',
        reference=str(n.GAIT),reference_sha256=n.sha(n.GAIT),parameters=r.SELECTION['parameters_rad'],joints=r.SELECTION['joints'],
        baseline='101114fullbroad-lift+.02RHtrochyaw,1Hz,duty5/6,gain128, D/steps/headingpass buthindmaterialslipfails',
        change='Duty.9 addsnominalall6stanceinterval.0667cycle betweenunloadings; frequency.8Hz maintainsoriginalexactjoint30rad/scapaftershorterswing. Perlegsourcefootposepath unchanged.',
        hypothesis='Measuredswitchesamplify25–32%hindslip butarenotdominant. Testoneboundedloadhandoffcontrol, notassumecure.',
        physical='samefull71mesh49/48/42neq0/free-root/gravity/contact/friction/solref/solimp/noslip0/nativeτcaps,finitePD128and30rad/s; no activecontactfeedback orstaticFF',
        control='Continuousdeclaredalgorithmteacher forscreen; notneuralgoal. NewANNwouldneednewtrainedweightsandstrictwithdrawaltest.',
        assessment='Originalwholepost1s D/actualsustainedsixlegsteps/heading/matpointslip<=.5 perleg at0.5ms/independentphysicsreplay; sourceFKacceptance notphysicalacceptance',
        sources={str(p):n.sha(p) for p in [Path(__file__),SOURCE,Path(n.__file__),n.GAIT,r.SEARCH/'BEST_PARAMETERS.json']},
        units='native; absoluteSIforce/biologicalvalidity/originalbrainnecessity NOT_ESTABLISHED'))

r.prereg=prereg

if __name__=='__main__':r.main()
