"""Matched derivative-reference diagnostic, continuous algorithm teacher only."""
import importlib.util
from pathlib import Path
from rate_servo import RateTrackingServo
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T083316Z/run_aligned.py')
spec=importlib.util.spec_from_file_location('aligned_reference_runtime',SOURCE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
n=b.n
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T084945Z')
n.OUT=OUT;n.FREQ=1.;n.IncrementalActuator=RateTrackingServo
ORIGINAL_SIMULATE=n.simulate


def simulate(*args,**kwargs):
    result=ORIGINAL_SIMULATE(*args,**kwargs)
    result.update(actuator_derivative_target='integrator_effective_rate',
                  actuator_damping_mode=RateTrackingServo.damping_mode,
                  actuator_gain_scale=RateTrackingServo.gain_scale,
                  actuator_damping_ratio=RateTrackingServo.damping_ratio if RateTrackingServo.damping_mode=='critical' else None)
    n.dump(Path(result['folder'])/'RESULT.json',result)
    return result


n.simulate=simulate

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'PREREGISTRATION.json'
    if p.exists():raise FileExistsError('preserve prior derivative diagnostic')
    n.dump(p,dict(importance='核心必需',stage='S5 incremental finite-PD derivative-reference comparison',
        baseline='same alignedwave gait at1Hz, kp native_limit/.2*32, kd native_limit*.005*sqrt32; damping reference0; allD/steps/headingpass but materialslipfails',
        unchanged='original full71mesh/free root/neq0, gravity/contact/friction/timestep/native symmetric torque caps, joint range and targetratecap30; no physicalstate/rootwrites or externalforces',
        input='same bounded42jointincrement rates integratedonceper1ms, policy5ms',
        variants=[dict(tag='teacher_rate_samekd_4s',derivative_target='actualclippedintegratorrate',kp='samegain32',kd='samebaselinekd'),
                  dict(tag='teacher_rate_critical_4s',derivative_target='actualclippedintegratorrate',kp='samegain32',kd='2*.7*sqrt(kp*initial_model_Mjj)')],
        purpose='Separate damping-target tracking from additionaldamping; one matchedpair, not torqueauthorityincrease or bodycontroller',
        new_formula='tau=clip(kp*(integrated_target-q)+kd*(integrator_effective_rate-qvel), +/-original_limit)',
        inertial_damping='engineering gain hypothesis from original model inertia atkeyframe, not biologicalcalibration; inertia/mass untouched',
        control='continuous algorithmteacher for each4s feasibilityscreen; no neuralwalkingclaim',
        verification='independentreplay MUST implement declared derivative target; frozen zero-targetPD replay is inapplicable. OriginalnativeD/slip/headinglimitsunchanged.',
        sources={str(q):n.sha(q) for q in [Path(__file__),Path(__file__).with_name('rate_servo.py'),SOURCE,Path(n.__file__),n.GAIT]},
        units='native; absoluteSIforce NOT_ESTABLISHED'))
    inert=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T073258Z/MODEL.npz')
    for mode,tag in [('same','teacher_rate_samekd_4s'),('critical','teacher_rate_critical_4s')]:
        RateTrackingServo.damping_mode=mode
        simulate(inert,tag,4.,'teacher',False)
