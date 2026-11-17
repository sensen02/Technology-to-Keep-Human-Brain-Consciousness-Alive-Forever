"""Pin dynamics, anatomy, fixed readout and archived regression inputs."""
import json
from pathlib import Path
import sys

from fixed_neck import HERE, PARENT, full_catalog, hash_file, write

ROOT=HERE.parents[2]
BRAIN=HERE.parent/'20261117T064507Z-eon-overlap'
BODY=HERE.parent/'20261117T062447Z-body-commands'
GROUND=HERE.parent/'20261117T071007Z-ground-dn-response'


def freeze(out):
    neural=json.loads((HERE/'DYNAMICS_VERIFIED.json').read_text())
    assert neural['runtime_sha256']==hash_file(HERE/'fixed_neck.py')
    assert neural['parameters_sha256']==hash_file(HERE/'PARAMETERS.json')
    paths=[HERE/name for name in ('fixed_neck.py','PARAMETERS.json','adapters.py','regress.py',
             'calibrate.py','READOUT.json','verify_dynamics.py','DYNAMICS_VERIFIED.json','DESIGN.zh-CN.md',
             'FORMAL_JOBS.txt','SELECTION.txt','run_formal.sh')]
    paths += [PARENT/'CATALOG.json',PARENT/'circuit.py',PARENT/'physical.py',
              GROUND/'ground_response.py',GROUND/'REFINE/refine.py',GROUND/'REFINE/CALIBRATION.json',
              BRAIN/'overlap.py',BRAIN/'CALIBRATION.json',BODY/'PHASE/phase.py',
              BODY/'PHASE/phase_rollout.py',BODY/'expansion.py']
    old=json.loads((BODY/'PHASE/IDENTITY.json').read_text())['sha256']
    for name,digest in old.items():
        assert hash_file(Path(name))==digest,name
    files={str(path):hash_file(path) for path in paths}
    files.update(old)
    for port in full_catalog()['ports'].values():
        path=PARENT/'geometry'/port['object']['filename']
        assert hash_file(path)==port['object']['sha256']
        files[str(path)]=hash_file(path)
    for name in json.loads((HERE/'READOUT.json').read_text())['maps'].values():
        path=Path(name['source'])
        assert hash_file(path)==name['source_sha']
        files[str(path)]=hash_file(path)
    for folder in (BRAIN/'BRAIN').iterdir():
        if folder.is_dir():
            for name in ('INPUT.json','RESULT.json','BRAIN.npz'):
                path=folder/name
                files[str(path)]=hash_file(path)
    write(out,dict(sha256=files,catalog_neurons=len(full_catalog()['ports']),
           fixed_parameters=True,plasticity=False,ground_only=True))


if __name__=='__main__':
    freeze(Path(sys.argv[1]))
