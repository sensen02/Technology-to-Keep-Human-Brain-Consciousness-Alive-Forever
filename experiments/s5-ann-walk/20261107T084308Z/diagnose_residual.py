"""Two bounded followups to the verified sole-aligned gait's residual stance slip.
A changes only local servo stiffness; B changes only reference cadence. Neither
changes native torque/rate caps, body, contact/friction, root or gravity.
"""
import importlib.util
from pathlib import Path
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T083316Z/run_aligned.py')
spec=importlib.util.spec_from_file_location('sole_aligned_execution',SOURCE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
n=b.n
n.OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T084308Z')
ARMS=[('teacher_gain128_f1_4s',128.,1.,4.),('teacher_gain32_f05_6s',32.,.5,6.)]

if __name__=='__main__':
    n.OUT.mkdir(parents=True,exist_ok=True)
    path=n.OUT/'PREREGISTRATION.json'
    if path.exists():raise FileExistsError('preserve prior two-arm diagnostic')
    n.dump(path,dict(importance='核心必需',stage='S5 two focused residual-slip physical teacher diagnostics',
        baseline=dict(gain_scale=32.,frequency_hz=1.,native_D='allpass',real_six_leg_steps=True,
                      slip_to_body_speed_range=[.64625,1.16545],speed_native_per_s=.23515),
        arms=[dict(tag=t,gain_scale=g,frequency_hz=f,total_seconds=s) for t,g,f,s in ARMS],
        hypotheses='A tests remaining finite jointtrackingcompliance; B tests cadence-dependent inertial load and cyclicbodymotion. No arbitrarylargersweep.',
        unchanged='accepted sole-alignedgeometry, original71mesh/freebody, originalgravity/contact/friction/torquecaps/ratecap30rad/s, dt12.5us, actuator1ms, policy5ms; no rootwrites/appliedforces',
        control='continuous algorithmteacher only; inert oldneuralmodel usedonlyforlogger, neveractuatorcommands',
        assessment='whole post1swindow originalDgates, actualsixlegunloading, independentreplay andmaterialslip<=.5bodyfor eachleg, headingcriteria unchanged. No teacherpasscountsasneuralgoal.',
        sources={str(p):n.sha(p) for p in [Path(__file__),SOURCE,Path(n.__file__),n.GAIT]},
        units='native; absoluteSIforce NOT_ESTABLISHED'))
    inert=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T073258Z/MODEL.npz')
    for tag,gain,freq,seconds in ARMS:
        n.FREQ=freq
        n.IncrementalActuator.scale=gain
        n.simulate(inert,tag,seconds,'teacher',False)
