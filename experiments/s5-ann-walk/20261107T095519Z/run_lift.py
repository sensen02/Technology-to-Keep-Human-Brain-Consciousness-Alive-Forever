"""Teacher and replacement-ANN execution for separately verified lift-first asset.
Actual physical integration uses unchanged original full body and finite motors.
"""
import importlib.util
from pathlib import Path
import sys
BASE=Path('/run/media/sensen/Data2/cell_wound_prototype')
SOURCE=BASE/'experiments/s5-ann-walk/20261107T083316Z/run_aligned.py'
spec=importlib.util.spec_from_file_location('aligned_motor_runtime',SOURCE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
n=b.n
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T095519Z')
n.GAIT=ROOT/'GAIT.npz';n.OUT=ROOT/'neural';n.FREQ=1.
n.IncrementalActuator.scale=32.
DEPENDENCIES=[BASE/rel for rel in [
    'experiments/s5-ann-walk/20261107T082451Z/run_sole.py',
    'experiments/s5-ann-walk/20261107T082037Z/run_gain32.py',
    'experiments/s5-ann-walk/20261107T081009Z/run_wave.py',
    'experiments/s5-ann-walk/20261107T080652Z/compare_tracking.py',
    'experiments/s5-ann-walk/20261107T080221Z/run_stance.py',
    'experiments/s5-ann-walk/20261107T075536Z/run_stance.py',
    'experiments/s5-ann-incremental/20261107T010000Z/incremental_actuator.py',
    'experiments/s5-ann-incremental/20261107T010000Z/plant_probe.py',
    'experiments/s5-neural-joint-torque/20261105T062800Z/run_trial.py',
    'tools/dynamic_support_metrics.py']]


def prereg():
    n.OUT.mkdir(parents=True,exist_ok=True)
    path=n.OUT/'PREREGISTRATION.json'
    if path.exists():return
    n.dump(path,dict(importance='核心必需',stage='S5 lift-before-return teacher physical screen and neural imitation',
        reference=str(n.GAIT),reference_sha256=n.sha(n.GAIT),frequency_hz=1.,
        changes='Swing foot orientation is free in AIR; source STANCE geometry/sole alignment and nominal5/6 duty preserved. Liftfrontquarter, returnmiddle, lowerlastquarter; explicit engineering retiming based on measured draggedswing.',
        actuator=dict(kp='native_limit/.2*32',kd='native_limit*.005*sqrt32',
                      torque_caps='original unchanged',target_rate_cap_rad_s=30.,derivative_target=0.),
        physics=dict(nq=49,nv=48,nu=42,nmesh=71,neq=0,free_root=True,
                     gravity=[0,0,-9801],dt_s=.0000125,actuator_s=.001,policy_s=.005,
                     auxiliary_forces=False,physical_state_writes='initial keyframe only; afterwards physicsintegrates'),
        control='teacher mode is continuous algorithm assistance; ann mode samefirst1s teacher thenlearnedMLP only, no teacher/time/phase/lookup afterwards',
        neural='replacement ANN; originalconnectome notrun in thisarm',
        assessment='Unchanged fullreleaseDgates, actual sixleg sustained unloading, independentphysicsreplay withmaterialpointspeed at0.5ms, perlegslip<=.5bodyforward, declaredheadingcriteria. Teacherdoesnotcountasneuralgoal.',
        rollout_contract='Causal40frame targethistory primedthroughout warmup; allnativecontact/command/torque/body records kept; no handoverreset',
        sources={str(q):n.sha(q) for q in [Path(__file__),SOURCE,Path(n.__file__),n.GAIT,*DEPENDENCIES]},
        units='native; absoluteSIforce/originalbrainnecessity/biologicalvalidity NOT_ESTABLISHED'))


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
