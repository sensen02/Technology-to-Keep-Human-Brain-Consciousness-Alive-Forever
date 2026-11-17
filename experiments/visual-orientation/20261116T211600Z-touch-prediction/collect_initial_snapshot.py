"""Collect paired physical trials and actual arriving data from scoped DN/AN queues."""
import argparse
import inspect
from pathlib import Path

import mujoco as mj
import numpy as np
import torch

from touch_runtime import (BODY, BINDING, GRAPH, POSITION, RADIUS, START, W, TOUCH_W,
                           Fixture, Recording, TouchLink, build, load_body_policy, sha, turn)

DURATION = 1700
SAMPLE_MS = 5
GROUPS = 20
LEGS = tuple(BODY.LEG_ORDER)
FRONT = [LEGS.index(leg) for leg in ("LF", "RF")]
STATE_KEYS = ("buf", "_padz_prev", "_pad_rate", "requested_rate")
ACTUATOR_KEYS = ("target_rad", "last_input_rate_rad_s", "last_limited_rate_rad_s",
                 "last_effective_rate_rad_s", "rate_clip_mask", "angle_clip_mask")


class Apparatus:
    def __init__(self, folder):
        self.body = build(folder)
        self.motor = load_body_policy()
        self.flags = mj.mjtState.mjSTATE_INTEGRATION
        body = self.body
        self.state = np.empty(mj.mj_stateSize(body.plant, self.flags))
        mj.mj_getState(body.plant, body.d, self.state, self.flags)
        self.core = {key: getattr(body, key).copy() for key in STATE_KEYS}
        self.commands = {key: getattr(body.actuator, key).copy() for key in ACTUATOR_KEYS}
        self.ticks = body.t, body.actuator.command_ticks
        self.qpos, self.qvel = body.d.qpos.copy(), body.d.qvel.copy()
        joints = [body.plant.joint(f"joint_{leg}Tibia_pitch") for leg in LEGS]
        self.angles = [int(joint.qposadr[0]) for joint in joints]
        self.velocities = [int(joint.dofadr[0]) for joint in joints]
        self.pad_ids = [*body.pads["LF"], *body.pads["RF"]]

    def reset(self):
        body = self.body
        mj.mj_setState(body.plant, body.d, self.state, self.flags)
        mj.mj_forward(body.plant, body.d)
        body.t, body.actuator.command_ticks = self.ticks
        for key, value in self.core.items():
            setattr(body, key, value.copy())
        for key, value in self.commands.items():
            setattr(body.actuator, key, value.copy())
        np.testing.assert_array_equal(body.d.qpos, self.qpos)
        np.testing.assert_array_equal(body.d.qvel, self.qvel)

    def run(self, path, case, mode, center=None):
        self.reset()
        path.mkdir()
        body = self.body
        recorder = Recording(body, DURATION)
        body.recorder = recorder
        fixture = Fixture(body, case["side"], case["probe_ms"],
                          case["depth"] if mode == "external" else -.009, case["probe_length"])
        primary = TouchLink()
        mixed = TouchLink()
        received, mix_received, availability, provided, body_state, pad_path = [], [], [], [], [], []
        action = np.zeros(42)
        measure = body.measure()
        for ms in range(DURATION):
            if ms % SAMPLE_MS == 0:
                intended = int(case["start"] <= ms < case["stop"])
                primary.send(intended)
                effective = primary.action()
                lhs, rhs = primary.observed(), mixed.observed()
                received.append(np.zeros(2) if lhs is None else lhs)
                mix_received.append(np.zeros(2) if rhs is None else rhs)
                availability.append((lhs is not None, rhs is not None))
                provided.append((ms, intended, effective,
                                 -1 if primary.an_time is None else primary.an_time,
                                 -1 if mixed.an_time is None else mixed.an_time))
                load = measure["loads"] / W
                heading = body.d.xmat[body.th].reshape(3, 3)[:, 0]
                body_state.append(np.r_[body.d.qpos[self.angles], body.d.qvel[self.velocities],
                                       load, body.com_velocity() @ heading, load>=.02,
                                       body.d.xmat[body.th, 8]])
                action = self.motor(turn.features(body, measure), effective, 0)
            pad_path.append(body.d.geom_xpos[self.pad_ids].copy())
            if mode == "fixed":
                body.d.mocap_pos[0] = center
            elif mode in ("external", "sham"):
                fixture.move(ms)
            else:
                body.d.mocap_pos[0] = POSITION
            touch = np.asarray(recorder.activity())
            primary.sense(touch.tolist())
            front_load = np.clip(measure["loads"][FRONT] / (.32*W), 0., 1.)
            mixed.sense((.70*touch + .30*front_load).tolist())
            measure = body.tick(action)
            if body.d.xfrc_applied.any() or body.d.qfrc_applied.any():
                raise RuntimeError("Unexpected external wrench")
            primary.advance()
            mixed.advance()
        assert recorder.count == DURATION * 80
        np.testing.assert_allclose(np.diff(recorder.raw[:,5]), 12.5e-6, rtol=0., atol=1e-10)
        raw = recorder.raw
        # Truth is the physical fixture pair from native samples, never the stimulus schedule.
        above = raw[:,2:4] >= TOUCH_W*W
        touch_bins = (above.reshape(-1,SAMPLE_MS*80,2).sum(axis=1) >= 8)
        answer = path / "SAMPLES.npz"
        np.savez_compressed(answer, activity=np.asarray(received), mixed=np.asarray(mix_received),
                            valid=np.asarray(availability), commands=np.asarray(provided), state=np.asarray(body_state),
                            touch=touch_bins, native=raw, pads=np.asarray(pad_path))
        result = dict(case=case, mode=mode, source_native_steps=recorder.count,
                      planned_fixture_center=None if center is None else center.tolist(),
                      touched_bins=touch_bins.sum(axis=0).tolist(),
                      first_contact_ms=[float(np.flatnonzero(side)[0]/80.) if side.any() else None for side in above.T],
                      nonfront_peak=float(raw[:,4].max()/W), sha256=sha(answer),
                      route_touch=primary.transport.verify_final(), route_mixed=mixed.transport.verify_final())
        turn.stand.put(path / "RESULT.json", result)
        body.recorder = None
        print(path.name, result["touched_bins"], round(result["nonfront_peak"], 6), flush=True)
        return np.asarray(pad_path), result


def case_for(group):
    if not 0 <= group < GROUPS:
        raise ValueError("group out of bounds")
    rng = np.random.default_rng(20261116+group)
    return dict(group=group, split="train" if group<12 else "calibration" if group<16 else "held",
                side=group%2, start=int(rng.integers(125,500)//5*5),
                stop=int(rng.integers(1280,1600)//5*5),
                target_ms=int(rng.integers(800,1240)), probe_ms=int(rng.integers(550,1320)),
                depth=float(rng.uniform(.003,.007)), probe_length=int(rng.integers(40,82)))


def fixed_center(bench, track, case):
    body = bench.body
    side = case["side"]
    ids = bench.pad_ids[3*side:3*side+3]
    radius = body.plant.geom_size[ids,0]
    outward = 1-2*side
    earlier = track[0, 3*side:3*side+3]
    options = []
    # Select a stationary near-grazing location in the unobstructed physical path.
    for stamp in range(case["target_ms"]-170, case["target_ms"]+171, 10):
        for index in range(3):
            center = track[stamp, 3*side+index].copy()
            center[1] += outward*(radius[index]+RADIUS-.006)
            gap = np.min(np.linalg.norm(earlier-center, axis=1)-radius-RADIUS)
            motion = np.linalg.norm(track[stamp,3*side+index]-track[stamp-25,3*side+index])
            if gap>.035:
                options.append((motion/(1+abs(stamp-case["target_ms"])/150), center))
    if not options:
        raise RuntimeError(f"No motion-separated fixed placement for {case}")
    return max(options, key=lambda item: item[0])[1]


def main(root, groups):
    torch.set_num_threads(1)
    worker = root / f"worker_{min(groups):02d}"
    worker.mkdir()
    bench = Apparatus(worker)
    inputs = (Path(__file__), BINDING, GRAPH, START, BODY.PAD_XML,
              Path(inspect.getfile(TouchLink)), Path(inspect.getfile(load_body_policy)),
              Path(inspect.getfile(TouchLink)).with_name("probe_contact.so"))
    identity = {str(path):sha(path) for path in inputs}
    turn.stand.put(worker / "INPUT_IDENTITY.json", identity)
    for index in groups:
        case = case_for(index)
        family = root / f"group_{index:02d}"
        family.mkdir()
        reference, _ = bench.run(family / "clear", case, "clear")
        center = fixed_center(bench, reference, case)
        bench.run(family / "fixed", case, "fixed", center)
        bench.run(family / "external", case, "external")
        bench.run(family / "sham", case, "sham")
    assert identity == {str(path):sha(path) for path in inputs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("groups", nargs="+", type=int)
    args = parser.parse_args()
    main(args.root, args.groups)
