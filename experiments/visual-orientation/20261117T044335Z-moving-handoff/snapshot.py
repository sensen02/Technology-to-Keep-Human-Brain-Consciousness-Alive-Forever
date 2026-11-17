"""Capture and restore the native, body-controller, and source-transport states."""
from copy import deepcopy
from hashlib import sha256
import pickle

import mujoco as mj
import numpy as np
import torch


COURSE = ("time","n","action","afferents","past","previous","rgb","measured","reward","end")
BODY = ("t","buf","_jac","_padz_prev","_pad_rate","requested_rate","_adhesion_points",
        "_adhesion_wrench","_adhesion_reference_body_com","_adhesion_refresh_time_s")
ACTUATOR = ("target_rad","initial_target_rad","reset_projection_rad","command_ticks",
            "last_input_rate_rad_s","last_limited_rate_rad_s","last_effective_rate_rad_s",
            "rate_clip_mask","angle_clip_mask")
NECK = ("time","decoded","cut","an_cut","an_step","an_values","last_dn_step")
LINK = ("step","_queue","_seen","_cut")
BRAIN = ("estimate","memory","previous","visual","original_visual","turning","settle",
         "near","parked","proposal","cue")
FORCES = ("buffer","sides","peak","counts","first_hit","frame","count")
EYES = ("released","pending")


def values(obj, fields):
    return {name:deepcopy(getattr(obj,name)) for name in fields}


def assign(dst, state):
    for name,value in state.items():
        setattr(dst,name,deepcopy(value))


def capture(env):
    """Serialization is local and written by this run, including native contact cache."""
    data = dict(native_data=env.body.d,
                course=values(env,COURSE),
                body=values(env.body,BODY),
                actuator=values(env.body.actuator,ACTUATOR),
                neck=values(env.neck,NECK),
                link=values(env.neck.link,LINK),
                brain=values(env.brain,BRAIN),
                forces=values(env.forces,FORCES),
                eyes=values(env.eyes,EYES),
                rng=env.neck.codec.rng.get_state().numpy().copy(),
                codec_index=env.neck.codec.packet_index,
                pace_phase=env.motor.pace.phase)
    if data["brain"]["memory"] is not None:
        data["brain"]["memory"] = data["brain"]["memory"].numpy().copy()
    return data


def restore(env, checkpoint, brain):
    # Preserve the existing MjData object referenced by instrumentation.
    mj.mj_copyData(env.body.d,env.body.plant,checkpoint["native_data"])
    assign(env,checkpoint["course"])
    assign(env.body,checkpoint["body"])
    assign(env.body.actuator,checkpoint["actuator"])
    assign(env.neck,checkpoint["neck"])
    assign(env.neck.link,checkpoint["link"])
    env.neck.link._journal = []
    assign(env.forces,checkpoint["forces"])
    assign(env.eyes,checkpoint["eyes"])
    env.neck.codec.packet_index = checkpoint["codec_index"]
    env.neck.codec.rng.set_state(torch.from_numpy(checkpoint["rng"].copy()))
    env.motor.pace.phase = checkpoint["pace_phase"]
    env.brain = brain
    assign(env.brain,checkpoint["brain"])
    if env.brain.memory is not None:
        env.brain.memory = torch.from_numpy(env.brain.memory.copy())
    assert env.forces.count==env.time*80 and env.neck.link.step==env.time


def write_checkpoint(env, path):
    payload = pickle.dumps(capture(env),protocol=5)
    with path.open("xb") as out:
        out.write(payload)
    return sha256(payload).hexdigest()


def read_checkpoint(path, expected_digest):
    payload = path.read_bytes()
    if sha256(payload).hexdigest()!=expected_digest:
        raise ValueError("Local checkpoint digest changed.")
    return pickle.loads(payload)
