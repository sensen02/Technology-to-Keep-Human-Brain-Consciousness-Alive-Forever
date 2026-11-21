"""Capability reference using the installed FlyGym 2.1.0 tutorial configuration."""
from pathlib import Path
import inspect
import time

import mujoco
import numpy as np
from flygym import Simulation
from flygym.anatomy import BodySegment, ContactBodiesPreset
from flygym.compose import MixedTerrainWorld
from flygym_demo.complex_terrain import (
    HybridController, HybridControllerObservation, LocomotionAction,
    PreprogrammedSteps, apply_locomotion_action, make_locomotion_fly,
)
from flygym.utils.math import Rotation3D

TRIAL = Path(__file__).resolve().parent
from arena_trial import live


def main():
    start = time.monotonic()
    out = TRIAL / "STOCK"
    out.mkdir()
    fly = make_locomotion_fly(name="hybrid_demo", add_adhesion=True, colorize=True)
    world = MixedTerrainWorld()
    world.add_fly(fly, (0, 0, 1.2), Rotation3D("quat", (1, 0, 0, 0)),
                  bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
                  add_ground_contact_sensors=False)
    sim = Simulation(world)
    steps = PreprogrammedSteps()
    dofs = fly.get_actuated_jointdofs_order("position")
    controller = HybridController(timestep=sim.timestep, preprogrammed_steps=steps, output_dof_order=dofs)
    sim.reset()
    controller.reset(seed=0)
    initial_action = LocomotionAction(joint_angles=steps.default_pose_by_dof_order(dofs),
                                      adhesion_onoff=np.ones(6, dtype=bool))
    apply_locomotion_action(sim, fly.name, initial_action)
    pre_warm = sim.mj_data.time
    sim.warmup()
    warm = sim.mj_data.time - pre_warm
    mujoco.mj_saveModel(sim.mj_model, str(out/"BODY.mjb"), None)
    thorax = sim._internal_bodyids_by_fly[fly.name][fly.get_bodysegs_order().index(BodySegment("c_thorax"))]
    qpos = []
    position = []
    upright = []
    start_x = float(sim.mj_data.xpos[thorax, 0])
    for _ in range(round(2. / sim.timestep)):
        observation = HybridControllerObservation.from_sim(sim, fly.name)
        action = controller.step(observation)
        apply_locomotion_action(sim, fly.name, action)
        sim.step()
        qpos.append(sim.mj_data.qpos.copy())
        position.append(sim.mj_data.xpos[thorax].copy())
        upright.append(float(sim.mj_data.xmat[thorax, 8]))
        if sim.mj_data.warning.number.any() or not np.isfinite(qpos[-1]).all():
            break
    position = np.asarray(position)
    trace = out/"TRACE.npz"
    np.savez_compressed(trace, qpos=np.asarray(qpos), position=position, upright=upright)
    rows = dict(steps=len(qpos), dt_s=sim.timestep, warmup_s=float(warm),
                commanded_stop=False, initial_x_mm=start_x, furthest_x_mm=float(position[:,0].max()),
                final_x_mm=float(position[-1,0]), min_upright=float(min(upright)),
                final_y_mm=float(position[-1,1]), min_height_mm=float(position[:,2].min()),
                stable_numerically=not bool(sim.mj_data.warning.number.any()),
                wall_seconds=time.monotonic()-start, trace_sha256=live.sha(trace),
                model_sha256=live.sha(out/"BODY.mjb"),
                source_sha256={str(p): live.sha(p) for p in (Path(__file__), TRIAL/"STOCK_PROTOCOL.zh-CN.md",
                        Path(inspect.getfile(type(world))), Path(inspect.getfile(type(controller))))})
    live.write(out/"RESULT.json", rows)
    print({key: val for key,val in rows.items() if not key.endswith("sha256")}, flush=True)


if __name__ == "__main__":
    main()
