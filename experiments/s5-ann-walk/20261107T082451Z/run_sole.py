"""Physical and learned-ANN execution of contact-height-aligned stance reference.
The source FK asset must be reviewed before invoking. No root correction is added.
"""
import importlib.util
from pathlib import Path
import sys
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T082037Z/run_gain32.py')
spec=importlib.util.spec_from_file_location('finite_gain_wave_runtime',SOURCE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
n=b.n
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T082451Z')
RUNTIME_FILES=[Path('/run/media/sensen/Data2/cell_wound_prototype')/rel for rel in [
    'experiments/s5-ann-walk/20261107T081009Z/run_wave.py',
    'experiments/s5-ann-walk/20261107T080652Z/compare_tracking.py',
    'experiments/s5-ann-walk/20261107T080221Z/run_stance.py',
    'experiments/s5-ann-walk/20261107T075536Z/run_stance.py',
    'experiments/s5-ann-incremental/20261107T010000Z/incremental_actuator.py',
    'experiments/s5-ann-incremental/20261107T010000Z/plant_probe.py',
    'experiments/s5-neural-joint-torque/20261105T062800Z/run_trial.py',
    'tools/dynamic_support_metrics.py']]
n.GAIT=ROOT/'GAIT.npz';n.OUT=ROOT/'neural';n.FREQ=1.
assert hasattr(n.IncrementalActuator,'scale')
n.IncrementalActuator.scale=32.


def prereg():
    n.OUT.mkdir(parents=True,exist_ok=True)
    p=n.OUT/'PREREGISTRATION.json'
    if p.exists():return
    n.dump(p,dict(importance='核心必需',stage='S5 true-sole-height alignment physical screen and ANN',
        reference=str(n.GAIT),reference_sha256=n.sha(n.GAIT),frequency_hz=1.,
        change='fixed small perleg vertical path translation, preserving original x/y/fullperphasefootrotation; reference aligns actualtarsalcollisionsole levels. No physicalrootstate edits.',
        actuator=dict(kp='native_limit/.2*32',kd='native_limit*.005*sqrt32',
                      native_torque_caps='unchanged',max_rate_rad_s=30.),
        control='continuous declaredalgorithm in teacher mode only; ann mode1s samephysicalteacherwarmup then learnedcausal40frameMLP, withoutphase/clock/lookup afterhandover',
        original_brain='notrun in thisreplacementANN arm',
        physics='original49/48/42/71/free root/neq0, gravity[0,0,-9801], contact/friction andtimestep12.5us unchanged',
        assessment='whole-release originalDgates, allsixrealstepping, bodyforwardmotion, declaredheadinglimits, independentreplay andrealmaterialpointslip; teacher cannotfulfilneuralgoal',
        sources={str(q):n.sha(q) for q in [Path(__file__),SOURCE,Path(n.__file__),n.GAIT,*RUNTIME_FILES]},
        units='native; absoluteSIforce NOT_ESTABLISHED'))


n.prereg=prereg

if __name__=='__main__':
    assert n.GAIT.exists(),n.GAIT
    prereg()
    if len(sys.argv)>1 and sys.argv[1]=='teacher':
        seconds=float(sys.argv[2]) if len(sys.argv)>2 else 4.
        tag=sys.argv[3] if len(sys.argv)>3 else 'teacher_sole_4s'
        inert=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T073258Z/MODEL.npz')
        n.simulate(inert,tag,seconds,'teacher',False)
    else:n.main()
