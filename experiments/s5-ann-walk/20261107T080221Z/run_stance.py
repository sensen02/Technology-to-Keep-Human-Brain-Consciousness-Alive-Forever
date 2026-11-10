"""Execute v4 workspace-corrected reference using the reviewed bounded-PCHIP runtime.
The imported module is not executed as a script and never opens its old asset.
"""
import importlib.util
from pathlib import Path
import sys
WRAPPER=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T075536Z/run_stance.py')
spec=importlib.util.spec_from_file_location('reviewed_stance_runtime',WRAPPER)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
n=b.n
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T080221Z')
n.GAIT=ROOT/'GAIT.npz'
n.OUT=ROOT/'neural'
n.FREQ=1.

if __name__=='__main__':
    assert n.GAIT.exists(),n.GAIT
    n.prereg()
    proto=n.OUT/'REFERENCE_PROTOCOL.json'
    if not proto.exists():
        n.dump(proto,dict(reference=str(n.GAIT),reference_sha256=n.sha(n.GAIT),
            interpolation='exact periodic two-knot-extended PCHIP; unclipped values with native-bound assertions',
            frequency_hz=n.FREQ,unchanged_physics='full71mesh/free root/originalgravity/contact/native torque cap',
            driver=str(Path(__file__)),driver_sha256=n.sha(__file__),
            interpolation_runtime=str(WRAPPER),interpolation_runtime_sha256=n.sha(WRAPPER),
            physical_runtime=str(Path(n.__file__)),physical_runtime_sha256=n.sha(n.__file__),
            readiness='FK reviewed before physical screen, no FK walking claim; teacher mode is algorithm diagnostic only'))
    if len(sys.argv)>1 and sys.argv[1]=='teacher':
        seconds=float(sys.argv[2]) if len(sys.argv)>2 else 4.
        tag=sys.argv[3] if len(sys.argv)>3 else 'teacher_4s'
        old=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T073258Z/MODEL.npz')
        n.simulate(old,tag,seconds,'teacher',False)
    else:n.main()
