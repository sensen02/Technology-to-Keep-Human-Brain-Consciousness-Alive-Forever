"""Verify the artificial fly sensory model against archived body data and organ probes."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from flygym.compose.fly.musculoskeletal import _load_mjcf

from engine.embodied.interoception import (
    FlyInteroception, GutState, HeartObservation, LEGS, MujocoTibiaSensors, SCHEMA,
)


ROOT = Path(__file__).resolve().parent
WALK = ROOT / "experiments/banc-triad-pad/20261110T010000Z/MODEL_TRIAD_x200.xml"
MUSCLE = ROOT / "outputs/muscles_six_legs/fruitfly_six_leg_standing.xml"
TRACE = ROOT / "experiments/banc-walking-recovery/20261110T152754Z/NATIVE_ROLE_2500/trace_1ms.npz"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def joint_probe(model):
    data = mujoco.MjData(model)
    signs = {}
    angles = {}
    for leg in LEGS:
        name = f"joint_{leg}Tibia_pitch"
        data.qpos[:] = model.qpos0
        bends = []
        for position in (.6, 1.4, 2.3):
            data.joint(name).qpos[:] = position
            mujoco.mj_fwdPosition(model, data)
            femur, tibia, tarsus = (data.body(leg + part).xpos
                                    for part in ("Femur", "Tibia", "Tarsus1"))
            proximal, distal = tibia - femur, tarsus - tibia
            cross = np.linalg.norm(np.cross(proximal, distal))
            bends.append(float(np.arctan2(cross, proximal @ distal)))
        require(bends[0] < bends[1] < bends[2], f"{leg} flexion geometry")
        signs[leg] = 1
        angles[leg] = bends

    reader = MujocoTibiaSensors(model, signs)
    encoder = FlyInteroception(reader.calibration)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.
    baseline = encoder.observe(0., reader.read(data.qpos, data.qvel)).local_inputs("body_local")
    for leg in LEGS:
        name = f"joint_{leg}Tibia_pitch"
        for position, velocity, channel in ((2.3, 5., "hook_flexing"), (.6, -5., "hook_extending")):
            data.qpos[:] = model.qpos0
            data.qvel[:] = 0.
            data.joint(name).qpos[:] = position
            data.joint(name).qvel[:] = velocity
            previous = (data.qpos.copy(), data.qvel.copy())
            joints = reader.read(data.qpos, data.qvel)
            features = encoder.observe(0., joints).local_inputs("body_local")
            require(all(features[key] == value for key, value in baseline.items()
                        if not key.startswith(leg + "/")), f"{leg} isolated stimulus")
            require(features[f"{leg}/{channel}"] == .5, f"{leg} movement direction")
            require(features[f"{leg}/claw_flexion"] > .8 if position == 2.3
                    else features[f"{leg}/claw_flexion"] < .1, f"{leg} posture")
            require(np.array_equal(previous[0], data.qpos)
                    and np.array_equal(previous[1], data.qvel), "readout changed physical state")
    return reader, encoder, angles


def verify_walk(reader, encoder, trace):
    with np.load(trace, allow_pickle=False) as archive:
        time, pos, vel = (archive[key] for key in ("time", "qpos", "qvel"))
        source = archive["control_source"]
    require(len(time) == len(pos) == len(vel) == 15400 and np.all(np.diff(time) > 0.),
            "wrong or unordered walking trace")
    autonomous = source == 3
    require(int(autonomous.sum()) == 9600, "unexpected autonomous phase")
    output = []
    raw = []
    for t, q, v in zip(time, pos, vel):
        joints = reader.read(q, v)
        output.append(tuple(encoder.observe(t, joints).local_inputs("body_local").values()))
        raw.append(tuple(joint.angle_rad for joint in joints.values()))
    y = np.asarray(output).reshape(-1, 6, 4)
    require(np.all((y >= 0.) & (y <= 1.)) and np.all(y[:, :, 2] * y[:, :, 3] == 0.),
            "sensory range or direction conflict")
    require(np.allclose(y[:, :, 0] + y[:, :, 1], 1., rtol=0., atol=1e-15), "position pairing")
    require(np.all(np.ptp(y[autonomous], axis=0) > 0.), "inactive autonomous sensory channel")
    stats = {}
    for index, leg in enumerate(LEGS):
        cal = reader.calibration[leg]
        values = np.asarray(raw)[autonomous, index]
        stats[leg] = {
            "source_angle_rad": [float(values.min()), float(values.max())],
            "claw_flexion": [float(y[autonomous, index, 0].min()), float(y[autonomous, index, 0].max())],
            "hook_flexing_peak": float(y[autonomous, index, 2].max()),
            "hook_extending_peak": float(y[autonomous, index, 3].max()),
            "outside_nominal_joint_range_samples": int(np.count_nonzero(
                (values < cal.lower_rad) | (values > cal.upper_rad))),
        }
    return {"frames": len(time), "autonomous_frames": int(autonomous.sum()), "legs": stats}


def verify_organs(encoder):
    no_inputs = encoder.observe(0., {})
    require(all(sample.value is None for sample in no_inputs.samples), "unobserved organ must be missing")
    zones = {sample.name: sample.route for sample in no_inputs.samples}
    quiet = encoder.observe(0., {}, heart=HeartObservation(.5, 0.), gut=GutState())
    expanding = encoder.observe(.1, {}, heart=HeartObservation(.7, 2.), gut=GutState())
    contracting = encoder.observe(.2, {}, heart=HeartObservation(.5, -2.), gut=GutState())
    samples = lambda frame: {row.name: row.value for row in frame.samples}
    require(samples(quiet)["heart/expanding"] == 0. and samples(quiet)["heart/contracting"] == 0.,
            "stationary heart")
    require(samples(expanding)["heart/expanding"] == .5 and samples(expanding)["heart/contracting"] == 0.
            and samples(contracting)["heart/contracting"] == .5 and samples(contracting)["heart/expanding"] == 0.,
            "heart movement direction")
    require(all(expanding.local_inputs(side) == quiet.local_inputs(side)
                for side in ("body_local", "head_local")), "diagnostic route leaked")
    require(set(quiet.local_inputs("head_local")) == {"pharynx/distension"}
            and all(key.startswith(LEGS) for key in quiet.local_inputs("body_local")),
            "unverified organ or route in a local interface")

    digestive = encoder.digestion
    closed = GutState()
    for _ in range(100):
        result = digestive.advance(closed, .01, inlet_rate=1.)
        closed = result.state
        require(result.swallowed == 0. and closed.crop == 0., "transfer despite closed swallow")
    require(closed.pharynx == digestive.capacity.pharynx, "expected full pharynx")

    current, received, exited, max_error = GutState(), 0., 0., 0.
    first_nonzero = {}
    snapshots = []
    for tick in range(1400):
        update = digestive.advance(current, .01, inlet_rate=1. if tick < 200 else 0., swallow=1.)
        current = update.state
        received += update.admitted
        exited += update.exited_hindgut
        frame = encoder.observe((tick + 1) / 100, {}, gut=current)
        for name, value in asdict(current).items():
            if value > 0.:
                first_nonzero.setdefault(name, tick)
        error = abs(received - exited - sum(asdict(current).values()))
        max_error = max(error, max_error)
        if tick in (0, 199, 599, 1399):
            snapshots.append({"time_s": frame.time_s, "samples": {
                key: value for key, value in samples(frame).items() if key.endswith("/distension")}})
    require(list(first_nonzero) == ["pharynx", "crop", "midgut", "hindgut"]
            and list(first_nonzero.values()) == [0, 1, 2, 3], "noncausal transfer order")
    require(max_error < 1e-9 and exited > 0., "digestive ledger")
    require(digestive.advance(GutState(), .01).state == GutState(), "material without a source")
    return {"routes": zones, "heart_examples": [
                {k: v for k, v in samples(frame).items() if k.startswith("heart/")}
                for frame in (quiet, expanding, contracting)
            ], "swallow_closed": asdict(closed), "first_nonzero_tick_zero_based": first_nonzero,
            "admitted": received, "exited": exited, "max_volume_error": max_error,
            "digestive_samples": snapshots}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sources = (WALK, MUSCLE, TRACE, Path(__file__).resolve(),
               ROOT / "engine/embodied/interoception.py")
    before = {str(p): sha(p) for p in sources}
    models = [_load_mjcf(path).compile() for path in (WALK, MUSCLE)]
    reader, encoder, angles = joint_probe(models[0])
    muscle_reader, _, muscle_angles = joint_probe(models[1])
    require(reader.calibration == muscle_reader.calibration, "six-joint calibration differs")
    report = {"status": "PASS_ENGINEERING_INTEROCEPTION_PROBES", "schema": SCHEMA,
              "mujoco_version": mujoco.__version__, "source_sha256": before,
              "whole_body_geometry_probe": {"walking_bend_rad": angles,
                                           "muscle_bend_rad": muscle_angles},
              "frozen_walking_trace": verify_walk(reader, encoder, TRACE),
              "organ_inputs_from_engineering_probes": verify_organs(encoder),
              "training_updates": 0, "physics_steps": 0, "neck_ports_admitted": 0}
    require(before == {str(p): sha(p) for p in sources}, "input file changed")
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as file:
            file.write(rendered)
    print(f"{report['status']}: {report['frozen_walking_trace']['frames']} archived frames, "
          f"max volume error {report['organ_inputs_from_engineering_probes']['max_volume_error']:.3g}")


if __name__ == "__main__":
    main()
