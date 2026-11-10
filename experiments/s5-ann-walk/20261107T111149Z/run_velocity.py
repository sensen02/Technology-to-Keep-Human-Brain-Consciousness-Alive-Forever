"""Physical screen of velocity-aware trained model; teacher and body unchanged."""
import importlib.util
from pathlib import Path
import sys
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T102842Z/run_student.py')
spec=importlib.util.spec_from_file_location('fixed_broadlift_student_runtime',SOURCE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
n=b.n
n.OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T111149Z')

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--seconds',type=float,default=4.)
    ap.add_argument('--tag',default='ann_velocity_4s');ap.add_argument('--mode',default='ann',choices=['ann','hold','motor_cut','teacher'])
    a=ap.parse_args()
    weights=n.OUT/'MODEL_VELOCITY.npz';assert weights.exists(),weights
    identity=n.OUT/(a.tag+'_PHYSICAL_PROTOCOL.json')
    if identity.exists():raise FileExistsError('preserve prior velocity-model physicalscreen')
    n.dump(identity,dict(stage='S5 velocity-aware ANN physicalscreen, unchangedphysicalteacher/model/actuator',
        model=str(weights),model_sha256=n.sha(weights),reference=str(n.GAIT),reference_sha256=n.sha(n.GAIT),
        source=str(Path(__file__)),source_sha256=n.sha(__file__),physical_runtime=str(Path(n.__file__)),physical_runtime_sha256=n.sha(n.__file__),
        neural='replacementlearnedMLP, genuine40framewarmtargethistory, first1steacherthenneuralweights only; no originalconnectome inarm',
        physics='same full71mesh49/48/42neq0/free root/gravity/contact/nativecaps,30rad/s,finitePD128. No conditionalintegration, rootwrites, teacherafthandover.',
        duration=a.seconds,mode=a.mode,
        acceptance='OriginalD/realsteps/heading/materialpointslip<=.5 perleg at0.5ms/independentreplay allunchanged; teacherstillnotfulllow-slippass, exploratorytrainingexplicit'))
    n.simulate(weights,a.tag,a.seconds,a.mode,False)
