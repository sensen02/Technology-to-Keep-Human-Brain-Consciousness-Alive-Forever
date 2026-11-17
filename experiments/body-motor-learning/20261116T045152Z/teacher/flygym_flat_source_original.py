"""Collect measured, named tarsus trajectories from the installed FlyGym demo."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

import flygym
from flygym.anatomy import BodySegment
from flygym.compose import ActuatorType
from flygym.compose.world import FlatGroundWorld
from flygym.simulation import Simulation
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import common, cpg_controller, hybrid_controller, preprogrammed, turning_controller
import mujoco as mj
import numpy as np


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def collect(output, adhesion, seed, duration):
    output.mkdir()
    fly = common.make_locomotion_fly(add_adhesion=adhesion)
    world = FlatGroundWorld()
    world.add_fly(fly, (0, 0, 0.5), Rotation3D("quat", (1, 0, 0, 0)))
    timestep = 0.0001
    sim = Simulation(world, timestep=timestep)
    dofs = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    controller = turning_controller.HybridTurningController(
        timestep=timestep, enable_adhesion=adhesion, output_dof_order=dofs)
    controller.reset(seed=seed)
    data = sim.mj_data
    model = sim.mj_model
    mj.mj_forward(model, data)
    mj.mj_saveModel(model, str(output / "FLYGYM_MODEL.mjb"), None)
    segments = fly.get_bodysegs_order()
    order = {leg: i for i, leg in enumerate(fly.get_legs_order())}
    force_order = [order[leg] for leg in controller.legs]
    th = segments.index(BodySegment("c_thorax"))
    th_id = sim._internal_bodyids_by_fly[fly.name][th]
    foot_ids = [segments.index(BodySegment(f"{leg}_tarsus5")) for leg in controller.legs]
    axial_segments = [BodySegment(s) for s in ("c_head", "c_abdomen6")]
    geom_by_segment = sim._internal_geomid_by_bodyseg_by_fly[fly.name]
    axial_geoms = [geom_by_segment[s] for s in axial_segments]
    body_length_mm = float(np.linalg.norm(np.diff(data.geom_xpos[axial_geoms], axis=0)))
    if body_length_mm <= 0. or any(sim._internal_bodyids_by_fly[fly.name][i] < 0 for i in [th, *foot_ids]):
        raise ValueError("Missing distinct axial geoms or tarsus5/thorax bodies in the compiled fly")
    nonfoot_segments = [
        seg for seg in segments if seg.name.split("_")[0] not in controller.legs
    ]
    samples = []
    command = np.ones(2)
    sample_decimation = 10
    for i in range(round(duration / timestep)):
        obs = hybrid_controller.HybridControllerObservation.from_sim(
            sim, fly.name, legs=controller.legs)
        action = controller.step(command, obs)
        if adhesion:
            common.apply_locomotion_action(sim, fly.name, action)
        else:
            # The controller emits six false flags even when the model has no adhesion actuators.
            sim.set_actuator_inputs(fly.name, ActuatorType.POSITION, action.joint_angles)
        sim.step()
        if (i + 1) % sample_decimation:
            continue
        xyz = sim.get_body_positions(fly.name)
        rt = data.xmat[th_id].reshape(3, 3).copy()
        feet = xyz[foot_ids]
        contact, force, _, _, normal, _ = sim.get_ground_contact_info(fly.name)
        nonfoot = sim.get_bodysegment_contact_forces(
            fly.name, nonfoot_segments, ground_only=True)
        samples.append(dict(
            time_s=float(data.time), thorax_world_mm=xyz[th].copy(), thorax_rotation=rt,
            foot_world_mm=feet.copy(), foot_body_mm=(feet - xyz[th]) @ rt,
            contact_count=contact[force_order].copy(),
            contact_force_frame_native=force[force_order].copy(),
            contact_normal_world=normal[force_order].copy(),
            phase_rad=np.mod(controller.cpg_network.curr_phases, 2 * np.pi),
            commanded_joint_rad=action.joint_angles.copy(),
            adhesion_command=action.adhesion_onoff.copy(),
            reflex_correction=controller.last_info["net_corrections"].copy(),
            nonfoot_load_native=float(np.maximum(nonfoot[:, 2], 0).sum()),
        ))
        if not all(np.isfinite(data.qpos)) or data.warning.number.any():
            raise RuntimeError(f"FlyGym physics warning at {data.time:.5f}s")
    trace = {key: np.asarray([row[key] for row in samples]) for key in samples[0]}
    legs = [leg.upper() for leg in controller.legs]
    np.savez_compressed(output / "TRACE.npz", legs=np.asarray(legs), **trace)
    settled = trace["time_s"] >= 1.
    t = trace["time_s"][settled]
    xyz = trace["thorax_world_mm"][settled]
    heading = trace["thorax_rotation"][settled][:-1, :2, 0]
    heading /= np.linalg.norm(heading, axis=1, keepdims=True)
    forward = float(np.einsum("ij,ij->", np.diff(xyz[:, :2], axis=0), heading) / (t[-1] - t[0]))
    ground = trace["contact_count"][settled] > 0
    contact_ends = np.sum(ground[:-1] & ~ground[1:], axis=0)
    source = {}
    for mod in (flygym, common, cpg_controller, hybrid_controller, preprogrammed, turning_controller, Simulation, FlatGroundWorld):
        path = Path(inspect.getfile(mod)).resolve()
        source[str(path)] = sha(path)
    manifest = dict(
        flygym_version=importlib.metadata.version("flygym"), mujoco_version=mj.__version__,
        source_sha256=source, collector_sha256=sha(__file__),
        model_sha256=sha(output / "FLYGYM_MODEL.mjb"), trace_sha256=sha(output / "TRACE.npz"),
        controller="flygym_demo.complex_terrain.turning_controller.HybridTurningController",
        robot="NeuroMechFly make_locomotion_fly", terrain="FlatGroundWorld", legs=legs,
        actuator_order=[str(dof) for dof in dofs], flat_contact_sensor_order=fly.get_legs_order(),
        seed=seed, adhesion=adhesion, timestep_s=timestep, sample_period_s=sample_decimation*timestep,
        descending_signal=command.tolist(), analyzed_window_s=[float(t[0]), float(t[-1])],
        body_length_mm=body_length_mm, length_definition="neutral-pose c_head to c_abdomen6 geom origin distance",
        foot_definition="physical Tarsus5 body origin; _body in full thorax coordinates",
        contact_definition="FlyGym ground-contact sensor includes all ground-contact leg segments",
        forward_mm_s=forward, forward_body_lengths_s=forward/body_length_mm,
        thorax_z_min_mm=float(xyz[:, 2].min()), upright_dot_min=float(trace["thorax_rotation"][settled, 2, 2].min()),
        mean_contact_fraction=dict(zip(legs, ground.mean(0).tolist())),
        contact_end_count=dict(zip(legs, contact_ends.tolist())),
        peak_nonfoot_load_native=float(trace["nonfoot_load_native"][settled].max()),
    )
    (output / "RESULT.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: val for key, val in manifest.items() if key not in {"actuator_order", "source_sha256"}}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--adhesion", action="store_true")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--duration", type=float, default=5.)
    args = p.parse_args()
    if args.duration < 2:
        p.error("At least 2 s to separate initial settling from walking")
    collect(args.output.resolve(), args.adhesion, args.seed, args.duration)
