"""Full physics/neural execution of the independently accepted constrained path.
No execution is allowed here before the new asset has been exported by all gates.
"""
import importlib.util
from pathlib import Path
import sys
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T095519Z/run_lift.py')
spec=importlib.util.spec_from_file_location('bounded_lift_execution',SOURCE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
n=b.n
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T101114Z')
n.GAIT=ROOT/'GAIT.npz';n.OUT=ROOT/'neural';n.FREQ=1.
OriginalPrereg=n.prereg


def prereg():
    OriginalPrereg()
    p=n.OUT/'RATE_PATH_IDENTITY.json'
    if not p.exists():
        n.dump(p,dict(stage='S5 explicit-rate constrained lift-first reference execution',
            reference=str(n.GAIT),reference_sha256=n.sha(n.GAIT),
            driver=str(Path(__file__)),driver_sha256=n.sha(__file__),
            inherited_driver=str(SOURCE),inherited_sha256=n.sha(SOURCE),
            geometry='same095519swingpositions/clearance, originalstancefixed. Entireairjointpath under nativeposition+ratebounds, no posthoc clipping.',
            numerical_change='Nearlyconverged inner-QP steps admitted onlyif finite/primalbounds/trustbound and actualnonlinearobjective descent allverified. PhysicalD/slip/ratecriteria unchanged.',
            unchanged='full71mesh49/48/42neq0/free root/originalgravity/contact/friction/nativecaps, finitePD32 and30rad/s incrementalratecap',
            assessment='whole-release originalDgates/allactualsixfootsteps/bodyforward/heading/independentreplay/materialcontactslip<=.5body,0.5msmaterialsampling; teachernotneuralgoal',
            neural='samelearnedcausalMLP onlyafter1sphysicalwarmup in ann mode, no teacher/time/phase/lookup; originalconnectome notrun'))

n.prereg=prereg

if __name__=='__main__':
    assert n.GAIT.exists(),n.GAIT
    prereg()
    if len(sys.argv)>1 and sys.argv[1]=='teacher':
        seconds=float(sys.argv[2]) if len(sys.argv)>2 else 4.
        tag=sys.argv[3] if len(sys.argv)>3 else 'teacher_lift_4s'
        inert=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T073258Z/MODEL.npz')
        n.simulate(inert,tag,seconds,'teacher',False)
    else:n.main()
