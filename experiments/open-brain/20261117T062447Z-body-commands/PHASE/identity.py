"""Record exact experiment source, verified BANC graph and frozen body artifacts."""
from importlib.metadata import distribution
from pathlib import Path
import sys

import mujoco
import torch

from phase import HERE, study
import body_run
import build_catalog
import circuit
import expansion
import phase_rollout
import physical
import run_pytorch
from flygym_demo.complex_terrain import common, cpg_controller, hybrid_controller, preprogrammed, turning_controller


def snapshot(out):
    modules = (body_run, build_catalog, circuit, expansion, phase_rollout, physical, run_pytorch,
               common, cpg_controller, hybrid_controller, preprogrammed, turning_controller)
    paths = {Path(module.__file__).resolve() for module in modules}
    paths.update(HERE/file for file in ('phase.py','verify_phase.py','PREREG.zh-CN.md','SELECTION.txt'))
    paths.update((study.MODEL_FILE, study.FIRST/'BODY.mjb', study.FIRST/'BODY.json',
                  study.PARENT/'CATALOG.json', study.PARENT/'PHYSICAL_AN_PAIR/ground_forward/RESULT.json',
                  HERE/'FIT/INPUTS.json', HERE/'FIT/epoch015.pt', study.HERE/'drive.sh',
                  Path(preprogrammed.__file__).parent/'assets/single_steps_untethered.pkl'))
    installed = distribution('flygym')
    asset_files = []
    for path in installed.files:
        if path.suffix in {'.obj','.stl','.xml','.npy','.pkl','.yaml','.json'}:
            location = Path(installed.locate_file(path))
            if location.is_file():
                asset_files.append(location)
    paths.update(asset_files)
    study.write(out,dict(python=sys.version,executable=sys.executable,torch=torch.__version__,
                         mujoco=mujoco.__version__,flygym=installed.version,
                         sha256={str(path):study.sha(path) for path in sorted(paths)}))


if __name__=='__main__':
    snapshot(Path(sys.argv[1]))
