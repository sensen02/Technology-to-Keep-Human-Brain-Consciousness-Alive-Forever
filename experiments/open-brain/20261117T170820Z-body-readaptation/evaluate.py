"""Saved-body inference through the fixed neck and archived evaluation entry points."""
import argparse
from pathlib import Path
import sys

from context import HERE,NECK,OLD_WEIGHT,phase,study,ground,overlap,first,hash_file
from fixed_neck import write


def archived(suite,name,checkpoint,out):
    if suite=='body':
        import phase_rollout
        import cases
        caseset={**study.CASES,**{case.name:case for case in cases.HELD}}
        phase_rollout.rollout(caseset[name],'evaluate',checkpoint,out)
    elif suite=='ground':
        target,variant,mode=name.split('__')
        case=ground.case_for(target,variant)
        calibration=None
        if target.startswith('stride'):
            sys.path.insert(0,str(ground.HERE/'REFINE'))
            import refine
            case=refine.trial(target,variant)
            calibration=ground.HERE/'REFINE/CALIBRATION.json'
        ground.WEIGHT=checkpoint
        ground.run(case,mode,out,calibration)
    else:
        key,mode=name.rsplit('__',1)
        overlap.WEIGHT=checkpoint
        overlap.rollout(overlap.HERE/'BRAIN'/key,overlap.HERE/'CALIBRATION.json',mode,out)
    write(out/'BODY_MODEL.json',dict(checkpoint=str(checkpoint.resolve()),sha256=hash_file(checkpoint),
           dynamics_sha256=hash_file(NECK/'fixed_neck.py'),parameters_sha256=hash_file(NECK/'PARAMETERS.json'),
           readout_sha256=hash_file(NECK/'READOUT.json'),body_source_sha256=hash_file(Path(phase.__file__)),
           evaluator_sha256=hash_file(HERE/'evaluate.py')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite',choices=('body','ground','eon'))
    parser.add_argument('name')
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('out',type=Path)
    a=parser.parse_args()
    archived(a.suite,a.name,a.checkpoint,a.out)
