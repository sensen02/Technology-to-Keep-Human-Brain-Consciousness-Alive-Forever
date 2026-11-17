"""Load the unchanged accepted P1 body and its published checkpoint."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = HERE.parent
FIRST = BASE/'20261117T055904Z-body-imitation'
PHASE = BASE/'20261117T062447Z-body-commands/PHASE'
CURRENT = json.loads((BASE/'BODY_CURRENT.json').read_text())
sys.path.insert(0, str(PHASE))
import phase
import body_run
import mujoco
import numpy as np
from flygym.anatomy import BodySegment


def sha(path):
    from hashlib import sha256
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def model_identity(body):
    names = [mujoco.mj_id2name(body.m, mujoco.mjtObj.mjOBJ_ACTUATOR, int(a)) for a in body.actuators]
    joint_ids = body.m.actuator_trnid[body.actuators, 0]
    return dict(nq=body.m.nq, nv=body.m.nv, nu=body.m.nu, weight_native=body.weight,
                actuator_names=names, joint_ids=joint_ids.tolist(),
                joint_names=[mujoco.mj_id2name(body.m, mujoco.mjtObj.mjOBJ_JOINT,int(j)) for j in joint_ids],
                joint_ranges_rad=body.m.jnt_range[joint_ids].tolist(),
                joint_limited=body.m.jnt_limited[joint_ids].tolist(),
                force_limited=body.m.actuator_forcelimited[body.actuators].tolist(),
                force_ranges_native=body.m.actuator_forcerange[body.actuators].tolist(),
                kp=body.m.actuator_gainprm[body.actuators,0].tolist(),
                control_limited=body.m.actuator_ctrllimited[body.actuators].tolist(),
                control_ranges=body.m.actuator_ctrlrange[body.actuators].tolist())
