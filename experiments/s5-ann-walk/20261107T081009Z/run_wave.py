"""Physical/ANN execution of the verified wave reference at finite gain8.
Uses the same original torque caps and no added body/root feedback controller.
"""
import importlib.util
from pathlib import Path
import sys
PARENT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T080652Z/compare_tracking.py')
spec=importlib.util.spec_from_file_location('bounded_tracking_actuator',PARENT)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
n=b.n
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T081009Z')
n.GAIT=ROOT/'GAIT.npz'
n.OUT=ROOT/'neural'
n.FREQ=1.
b.DiagnosticActuator.scale=8.


def prereg():
    n.OUT.mkdir(parents=True,exist_ok=True)
    p=n.OUT/'PREREGISTRATION.json'
    if p.exists():return
    n.dump(p,dict(importance='核心必需',stage='S5 finite-gain wave physical screen and replacement ANN',
        reference=str(n.GAIT),reference_sha256=n.sha(n.GAIT),reference_fk='4097pose geometry/bounds/rate checks pass',
        motivation='Tripod gain8 still left LHloaded and bodypose canceled intended worldfootlift; distribute swing acrosssixlegs withfivelegs in nominalstance.',
        control='teacher mode is continuous algorithm diagnostic; ann mode uses teacher onlyfirst1s, thereafter trained causal-history MLP withoutclock/phase/lookup/bodyrootassist',
        neural='replacement artificial motor network; originalconnectome not run in thisarm',
        physics=dict(fullmesh=71,nq=49,nv=48,nu=42,neq=0,gravity=[0,0,-9801],native_dt=.0000125,
                     command_dt=.001,policy_dt=.005,frequency_hz=1.,max_rate_rad_s=30.,
                     kp='native_limit/.2*8',kd='native_limit*.005*sqrt8',torque_cap='unchanged original limits'),
        assessment='unchanged nativeDgates, actual sustained allsixfootunloadings, independentreplay and materialpointslip; headinglimits and lateraltravel required separately; teacher cannotcountas neuralgoal',
        sources={str(q):n.sha(q) for q in [Path(__file__),PARENT,Path(n.__file__),n.GAIT]},
        units='native; absoluteSIforce NOT_ESTABLISHED'))


n.prereg=prereg

if __name__=='__main__':
    assert n.GAIT.exists(),n.GAIT
    prereg()
    if len(sys.argv)>1 and sys.argv[1]=='teacher':
        seconds=float(sys.argv[2]) if len(sys.argv)>2 else 4.
        tag=sys.argv[3] if len(sys.argv)>3 else 'teacher_4s'
        inert=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T073258Z/MODEL.npz')
        n.simulate(inert,tag,seconds,'teacher',False)
    else:n.main()
