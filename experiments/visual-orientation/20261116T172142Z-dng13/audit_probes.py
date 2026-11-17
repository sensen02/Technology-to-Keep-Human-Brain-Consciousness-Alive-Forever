"""Audit saved steering integrations, matched prefixes, and physical reload steps."""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np

from probe import HERE, LEGS, STIM_FROM, WARM, build_scene, read, stan


def load(directory):
    result = read(directory/"RESULT.json")
    for path, digest in result["trace_files"].items():
        assert stan.sha(directory/path) == digest
    return result


def physical_reload_measurements(result, trace):
    if not result["measured_reload_events"]:
        return {}
    body,_,_ = build_scene(None)
    heading_yaw = trace["yaw"][WARM+STIM_FROM]
    heading = np.array([np.cos(heading_yaw),np.sin(heading_yaw)])
    prior = {}
    steps = {leg:[] for leg in LEGS}
    for entry in result["measured_reload_events"]:
        leg = entry["leg"]
        body.d.qpos[:] = trace["qpos"][entry["ms"]]
        mj.mj_kinematics(body.plant,body.d)
        point = body.d.geom_xpos[body.pads[leg],:2].mean(axis=0)
        steps[leg].append(dict(**entry,pad_world_xy=point.tolist(),
                               stride_along_initial_heading=None if leg not in prior else float((point-prior[leg])@heading)))
        prior[leg] = point
    return steps


def check(root):
    arms = {i:root/f"FLAT_{i.upper()}" for i in ("off","dna","dng","both")}
    reference = load(arms["off"])
    audit = {}
    with np.load(arms["off"]/"TRACE.npz") as off_t, np.load(arms["off"]/"NATIVE.npz") as off_n:
        horizon = WARM+STIM_FROM+2
        states = {key:off_t[key][:horizon+1] for key in ("qpos","qvel","loads","foot")}
        native_forces = off_n["force"][:horizon*80]
        off_action = np.load(arms["off"]/"DECISIONS.npy")
        for name, directory in arms.items():
            row = load(directory)
            with np.load(directory/"TRACE.npz") as trace, np.load(directory/"NATIVE.npz") as raw:
                exact_state = {key:bool(np.array_equal(baseline,trace[key][:horizon+1])) for key,baseline in states.items()}
                exact_native = bool(np.array_equal(native_forces,raw["force"][:horizon*80]))
                actions = np.load(directory/"DECISIONS.npy")
                changed = np.flatnonzero((actions[:min(len(actions),len(off_action)),5:] != off_action[:len(actions),5:]).any(axis=1))
                first_changed = int(actions[changed[0],0]) if len(changed) else None
                delta = row["yaw_steering_window_deg"]-reference["yaw_steering_window_deg"]
                joint = row["engineering_joint"]
                steps = physical_reload_measurements(row,trace)
                audit[name] = dict(exact_state_prefix=exact_state,exact_force_prefix=exact_native,
                                  first_nonzero_release_ms=row["first_nonzero_release_ms"],first_changed_action_ms=first_changed,
                                  heading_delta_from_off_deg=delta,absolute_yaw_deg=row["yaw_steering_window_deg"],
                                  physics_completed_ms=row["actual_ms"],native_integrations=row["native_integrations"],
                                  engineering_joint=joint,positive_direction_response=bool(joint and joint["passed"] and delta>=1.),
                                  actual_world_reload_steps=steps,zero_external_forces=bool(not raw["pose"][:,7].any()))
                assert all(exact_state.values()) and exact_native
                assert name=="off" or first_changed is None or first_changed>=row["first_nonzero_release_ms"]
    stan.put(root/"STEERING_AUDIT.json",dict(scope="left-sided engineered strides, single matched flat initial condition",
                                             audit=audit,whole_neck_admitted=False,native_tuning_established=False))
    for name, row in audit.items():
        print(name,row["physics_completed_ms"],row["heading_delta_from_off_deg"],
              row["positive_direction_response"],row["first_changed_action_ms"])


if __name__=="__main__":
    check(HERE)
