"""Run archived body tasks through the new fixed cervical circuit."""
import argparse
import json
from pathlib import Path
import sys

from adapters import HERE, install, phase
from fixed_neck import hash_file


def run(suite,name,out,mapped):
    install(mapped)
    if suite=='body':
        import phase_rollout
        import expansion
        import cases
        choices={**expansion.CASES,**{case.name:case for case in cases.HELD}}
        weight=phase.study.HERE/'PHASE/FIT/epoch015.pt'
        phase_rollout.rollout(choices[name],'evaluate',weight,out)
    elif suite=='eon':
        sys.path.insert(0,str(HERE.parent/'20261117T064507Z-eon-overlap'))
        import overlap
        key,mode=name.rsplit('__',1)
        overlap.rollout(overlap.HERE/'BRAIN'/key,overlap.HERE/'CALIBRATION.json',mode,out)
    else:
        old=HERE.parent/'20261117T071007Z-ground-dn-response'
        sys.path.insert(0,str(old))
        import ground_response
        target,variant,mode=name.split('__')
        calibration=None
        case=ground_response.case_for(target,variant)
        if target.startswith('stride'):
            sys.path.insert(0,str(old/'REFINE'))
            import refine
            case=refine.trial(target,variant)
            calibration=old/'REFINE/CALIBRATION.json'
        ground_response.run(case,mode,out,calibration)
    from fixed_neck import write
    write(out/'RUNTIME.json',dict(parameters_sha256=hash_file(HERE/'PARAMETERS.json'),
                 runtime_sha256=hash_file(HERE/'fixed_neck.py'),
                 wrapper_sha256=hash_file(HERE/'regress.py'),
                 adapter_sha256=hash_file(HERE/'adapters.py'),
                 readout_sha256=hash_file(HERE/'READOUT.json') if mapped else None,
                 mode='static_receiver_calibration' if mapped else 'old_readout_unchanged',
                 original_result_sha256=hash_file(out/'RESULT.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite',choices=('body','eon','ground'))
    parser.add_argument('name')
    parser.add_argument('out',type=Path)
    parser.add_argument('--mapped',action='store_true')
    a=parser.parse_args()
    run(a.suite,a.name,a.out,a.mapped)
