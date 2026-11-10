"""One matched tracking-gain arm on the new accepted broad-lift path.
Original torque/rate/contact/body limits remain; geometry is unchanged from101114.
"""
import importlib.util
from pathlib import Path
import sys
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T101114Z/run_rate_path.py')
spec=importlib.util.spec_from_file_location('lift_reference_runtime',SOURCE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
n=b.n
n.OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T101753Z')
n.IncrementalActuator.scale=128.


def prereg():
    n.OUT.mkdir(parents=True,exist_ok=True)
    p=n.OUT/'PREREGISTRATION.json'
    if p.exists():return
    n.dump(p,dict(importance='核心必需',stage='S5 matched tracking gain under accepted broad-lift geometry',
        previous='101114gain32 passedD/steps/heading, body .30696native/s, LF/LH/RF perlegslip<=.5 butLM/RM/RH failed',
        reference=str(n.GAIT),reference_sha256=n.sha(n.GAIT),frequency_hz=1.,
        only_change='finitePD gain32->128, kp=limit/.2*128,kd=limit*.005*sqrt128. No newgeometry/gait/torqueauthority; original nativecaps and30rad/s unchanged.',
        control='continuous declaredalgorithmteacher; ifANNmode later, first1steacherthenneuralonly, notoriginalconnectome',
        assessment='samewhole3sreleaseDgates/realsteps/heading/independentreplay/0.5msmaterialslip<=.5 perleg; failure retained without broadgainsearch',
        sources={str(q):n.sha(q) for q in [Path(__file__),SOURCE,Path(n.__file__),n.GAIT]},
        units='native; SIforce/biologicalvalidity NOT_ESTABLISHED'))

n.prereg=prereg

if __name__=='__main__':
    prereg()
    if len(sys.argv)>1 and sys.argv[1]=='teacher':
        seconds=float(sys.argv[2]) if len(sys.argv)>2 else 4.
        tag=sys.argv[3] if len(sys.argv)>3 else 'teacher_gain128_4s'
        inert=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T073258Z/MODEL.npz')
        n.simulate(inert,tag,seconds,'teacher',False)
    else:n.main()
