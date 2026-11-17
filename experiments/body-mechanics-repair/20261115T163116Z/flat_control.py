"""Compare the new collision geometry against original pads over a frozen replay."""

import importlib.util
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from hook import HERE, PROJECT, WALKING_XML, fixture
from muscle_reference import TRACE, digest

ACTUATOR = PROJECT / "experiments/s5-ann-incremental/20261107T010000Z/incremental_actuator.py"
DERIVED = HERE / "BODY_CLAWS_EXPERIMENTAL.xml"
WINDOW = 250
FIRST = 5800


def load_actuator():
    spec = importlib.util.spec_from_file_location("frozen_incremental", ACTUATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay(module, path, initial_position, initial_velocity, targets):
    model = module.load_full_model(path)
    model.geom_friction[:, 0] = 1.5
    state = mj.MjData(model)
    state.qpos[:] = initial_position
    state.qvel[:] = initial_velocity
    mj.mj_forward(model, state)
    actuator = module.IncrementalActuator(model, state)
    claws = {i for i in range(model.ngeom) if "_claw_" in model.geom(i).name}
    trajectory, peak_force, active_steps = [], 0., 0
    force = np.zeros(6)
    for target in targets:
        actuator.target_rad = target.copy()
        for _ in range(80):
            state.ctrl[:] = actuator.torque(state)
            mj.mj_step(model, state)
            active = False
            for i in range(state.ncon):
                contact = state.contact[i]
                if contact.geom1 in claws or contact.geom2 in claws:
                    mj.mj_contactForce(model, state, i, force)
                    peak_force = max(peak_force, float(abs(force[0])))
                    active |= abs(force[0]) > 1e-10
            active_steps += int(active)
            mj.mj_forward(model, state)
            trajectory.append(state.qpos.copy())
    assert not state.warning.number.any()
    assert not np.any(state.qfrc_applied)
    return np.array(trajectory), active_steps, peak_force


def signed_ridge_reaction():
    """Independently verify sign with the slide's generalized constraint force."""
    dt = .0005
    model = fixture(dt, True, True)
    state = mj.MjData(model)
    state.qpos[1] = .9
    forces = []
    for i in range(round(.95 / dt)):
        time = i * dt
        state.ctrl[0] = .9 if time < .1 else 0.
        state.qfrc_applied[0] = min(1, (time - .4) / .15) if time >= .4 else 0.
        mj.mj_step(model, state)
        if time >= .8:
            forces.append(-float(state.qfrc_constraint[0]))
    return min(forces)


def main():
    module = load_actuator()
    sources = {str(p): digest(p) for p in (WALKING_XML, DERIVED, ACTUATOR, TRACE)}
    with np.load(TRACE, allow_pickle=False) as trace:
        arguments = [trace["qpos"][FIRST], trace["qvel"][FIRST],
                     trace["target"][FIRST+1:FIRST+1+WINDOW]]
    assert len(arguments[-1]) == WINDOW
    baseline, _, _ = replay(module, WALKING_XML, *arguments)
    candidate, count, force = replay(module, DERIVED, *arguments)
    difference = float(np.max(abs(candidate - baseline)))
    constraint_force = signed_ridge_reaction()
    result = dict(scope="matched held-target flat-ground replay, no neural control",
                  sha256=sources, physical_steps_per_arm=len(baseline),
                  maximum_qpos_difference=difference, claw_loaded_steps=count,
                  peak_claw_normal_force=force, signed_fixture_reaction_min=constraint_force,
                  pass_short_flat_control=count == 0 and difference <= 1e-7
                  and constraint_force >= .8)
    assert sources == {str(p): digest(p) for p in (WALKING_XML, DERIVED, ACTUATOR, TRACE)}
    with (HERE / "FLAT_CONTROL_RESULT.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))
    if not result["pass_short_flat_control"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
