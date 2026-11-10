"""Execute smoothed whole-foot-aligned reference with original bounded physical motors."""
import importlib.util
from pathlib import Path
import sys
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T082451Z/run_sole.py')
spec=importlib.util.spec_from_file_location('sole_aligned_runtime',SOURCE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
n=b.n
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T083316Z')
n.GAIT=ROOT/'GAIT.npz';n.OUT=ROOT/'neural';n.FREQ=1.

if __name__=='__main__':
    assert n.GAIT.exists(),n.GAIT
    n.prereg()
    identity=n.OUT/'DRIVER_IDENTITY.json'
    if not identity.exists():
        n.dump(identity,dict(source=str(Path(__file__)),sha256=n.sha(__file__),
            inherited_runtime=str(SOURCE),inherited_sha256=n.sha(SOURCE),
            gait=str(n.GAIT),gait_sha256=n.sha(n.GAIT),
            change='boundedlambda100 source-relative correction smoothing plus onefilteredprior localIKreprojection; preserves originalphasegeometry/wholefootsolealignment, no source clipping',
            actuator='same finite gain32 and originaltorque/ratecaps as baseline wave comparison',
            controller='teacher when explicitlyrequested, otherwise1swarmup thenreplacementANN with genuinecausaltargethistory'))
    if len(sys.argv)>1 and sys.argv[1]=='teacher':
        seconds=float(sys.argv[2]) if len(sys.argv)>2 else 4.
        tag=sys.argv[3] if len(sys.argv)>3 else 'teacher_aligned_4s'
        inert=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T073258Z/MODEL.npz')
        n.simulate(inert,tag,seconds,'teacher',False)
    else:n.main()
