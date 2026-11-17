"""Body-side stateless standing and recovery bench using archived physical dynamics."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
BODY = PROJECT / "experiments/banc-walking-recovery/20261110T152714Z"
sys.path.insert(0, str(BODY))
import identity
import physics

FORMAT = "STAND_CURRENT_OBSERVATION_MLP_V1"
SIZE = 183
LENGTH = 2.9024054918026523
UPDATES = 180
SEED = 20261115
HORIZON = 240
TRAIN = [
    (((0, .4), (2, -.4)), (0., 0.)),
    (((1, -.4), (3, .35)), (0., 0.)),
    (((4, .35), (5, -.35)), (0., 0.)),
    (((0, -.35), (4, .45)), (0., 0.)),
    (((3, -.35), (5, .4)), (0., 0.)),
    (((2, .45), (4, -.3)), (0., 0.)),
    (((1, .3), (5, .3)), (0., 0.)),
    (((0, -.3), (3, -.3)), (0., 0.)),
    ((), (0., .40)),
    ((), (-.35, 0.)),
]
HOLDOUT = {
    "quiet": ((), (0., 0.)),
    "unseen_leg_pair": (((1, .50), (4, -.35)), (0., 0.)),
    "push_left": ((), (0., .55)),
    "push_right": ((), (0., -.55)),
    "push_forward": ((), (.55, 0.)),
    "combined": (((2, -.45), (5, .45)), (.20, -.35)),
}


class StandPolicy(nn.Module):
    def __init__(self, center, scale):
        super().__init__()
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)
        self.layers = nn.Sequential(nn.Linear(SIZE, 64), nn.Tanh(), nn.Linear(64, 42))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, observation):
        x = ((observation - self.center) / self.scale).clamp(-6, 6)
        return self.layers(x).tanh()


def put(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def observe(body):
    # obs238 prefix is a pure current measurement, with the load layout retained.
    return body.obs238(np.zeros(42), target_speed=0.)[:SIZE].copy()


class Bench:
    def __init__(self, output):
        self.body = physics.Core(adhesion=False)
        actuators = self.body.actuator
        names = actuators.state_dict()["joint_names"]
        expected = ["LF", "LH", "LM", "RF", "RH", "RM"]
        if len(names) != 42 or any(not str(j).startswith("joint_" + leg)
                for leg, group in zip(expected, np.reshape(names, (6, 7))) for j in group):
            raise ValueError("Unexpected actuator layout")
        self.manifest = identity.capture(self.body,
            dict(scope="stateless physical standing body bench", actuator_names=names),
            extra_paths=[HERE / "PREREG.zh-CN.md"])
        put(output / "IDENTITY.json", self.manifest)
        self.zero = np.zeros(42)
        for _ in range(1000):
            self.body.tick(self.zero)
        # A full MuJoCo integration state and the wrapper's physical/measurement
        # state are restored only before distinct trials.
        import mujoco as mj
        self.mj = mj
        self.spec = mj.mjtState.mjSTATE_INTEGRATION
        self.state = np.empty(mj.mj_stateSize(self.body.plant, self.spec))
        mj.mj_getState(self.body.plant, self.body.d, self.state, self.spec)
        self.fields = {key: getattr(self.body, key).copy() for key in
                       ("buf", "_padz_prev", "_pad_rate", "requested_rate")}
        self.actuator_fields = {key: getattr(actuators, key).copy() for key in
                                ("target_rad", "last_input_rate_rad_s", "last_limited_rate_rad_s",
                                 "last_effective_rate_rad_s", "rate_clip_mask", "angle_clip_mask")}
        self.t = self.body.t
        self.command_ticks = actuators.command_ticks
        self.center = observe(self.body)
        self.target = actuators.target_rad.copy()
        self.q = self.body.d.qpos[self.body.a_qpos].copy()
        self.com = self.body.d.subtree_com[self.body.th].copy()

    def reset(self):
        c = self.body
        self.mj.mj_setState(c.plant, c.d, self.state, self.spec)
        self.mj.mj_forward(c.plant, c.d)
        c.t = self.t
        c.actuator.command_ticks = self.command_ticks
        for key, value in self.fields.items():
            setattr(c, key, value.copy())
        for key, value in self.actuator_fields.items():
            setattr(c.actuator, key, value.copy())
        if not np.array_equal(observe(c), self.center):
            raise RuntimeError("Paired starting observations differ")

    def trial(self, settings, policy, path):
        self.reset()
        c = self.body
        entries, force_w = settings
        disturbance = self.zero.copy()
        for l, amplitude in entries:
            # Bounded changes to current motor targets, through the normal layer.
            disturbance[7*l + 1:7*l + 4] = amplitude * np.array([1., -.5, .75])
        force = np.array([*force_w, 0.]) * physics.W
        for _ in range(30):
            c.tick_rate(physics.MAX_RATE * disturbance)
            # Core clears external wrenches each half millisecond, so forces are
            # applied by the disturbance hook below in trials that require them.
        if np.any(force):
            original = c.apply_adhesion
            def apply_push(loads):
                original(loads)
                c.d.xfrc_applied[c.th, :3] = force
            c.apply_adhesion = apply_push
            try:
                for _ in range(25):
                    c.tick(self.zero)
            finally:
                c.apply_adhesion = original
                c.d.xfrc_applied[:] = 0.
        prefix = np.r_[observe(c), c.d.qpos, c.d.qvel]
        rx, ra, times, q, targets, com, support, nonfoot, force_peak = ([] for _ in range(9))
        for _ in range(HORIZON):
            x = observe(c)
            a = np.asarray(policy(x, c.actuator.target_rad), dtype=np.float64)
            if a.shape != (42,) or not np.isfinite(a).all() or np.abs(a).max() > 1.:
                raise RuntimeError("Invalid control")
            rx.append(x)
            ra.append(a.copy())
            for _ in range(5):
                m = c.tick(a)
                applied = max(np.abs(c.d.xfrc_applied).max(), np.abs(c.d.qfrc_applied).max())
                if applied:
                    raise RuntimeError("Unexpected external force during recovery")
                times.append(float(c.d.time))
                q.append(c.d.qpos.copy())
                targets.append(c.actuator.target_rad.copy())
                com.append(c.d.subtree_com[c.th].copy())
                support.append(m["loads"].sum() / physics.W)
                nonfoot.append(m["nonfoot"])
                force_peak.append(float(applied))
        x, a, times, q, targets, com, support, nonfoot, force_peak = (
            np.asarray(items) for items in (rx, ra, times, q, targets, com, support, nonfoot, force_peak))
        drift = float(np.linalg.norm(com[:, :2] - self.com[:2], axis=-1).max() / LENGTH)
        speed = np.diff(com[-201:, :2], axis=0) / .001 / LENGTH
        speed_rms = float(np.sqrt(np.square(speed).sum(-1).mean()))
        target_rms = float(np.sqrt(np.square(targets[-200:] - self.target).mean()))
        joint_rms = float(np.sqrt(np.square(q[-200:, c.a_qpos] - self.q).mean()))
        gates = dict(com=float(com[:, 2].min()) >= 1.2, support=float(support.mean()) >= .9,
                     nonfoot=float(nonfoot.max()) == 0., drift=drift <= .1,
                     steady_speed=speed_rms <= .01, target_recovery=target_rms <= .002)
        np.savez_compressed(path, x=x, action=a, prefix=prefix, time_s=times, qpos=q, target=targets,
                            com=com, support=support, nonfoot=nonfoot, recovery_force_max=force_peak)
        return dict(gates=gates, passed=all(gates.values()), com_min=float(com[:, 2].min()),
                    initial_target_error_rms=float(np.sqrt(np.square(targets[0] - self.target).mean())),
                    target_error_rms=target_rms, actual_joint_error_rms=joint_rms,
                    foot_support_mean=float(support.mean()), nonfoot_peak=float(nonfoot.max()),
                    drift_body_lengths=drift, end_speed_rms_body_lengths_s=speed_rms,
                    disturbance_rate_normalized=disturbance.tolist(), disturbance_force_native=force.tolist(),
                    disturbance_motor_ms=30, disturbance_force_ms=25 if np.any(force) else 0,
                    recovery_ms=len(q), native_integrations=(30+25*bool(np.any(force))+len(q))*80,
                    saved_sha256=sha(path))


def teacher(bench):
    return lambda x, target: np.clip(12. * (bench.target - target) / physics.MAX_RATE, -.5, .5)


def validate(output):
    results = json.loads((output / "RESULT.json").read_text())
    path = output / "TRAIN" / f"step_{UPDATES:04}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if sha(path) != results["checkpoint_sha256"] or checkpoint["format"] != FORMAT:
        raise ValueError("Checkpoint identity changed")
    saved = checkpoint["model"]
    model = StandPolicy(saved["center"], saved["scale"]).double()
    model.load_state_dict(saved)
    model.eval()
    numerical = {}
    for name, pair in results["evaluation"].items():
        files = [output / f"{name}_{arm}.npz" for arm in ("model", "zero")]
        if any(sha(p) != pair[arm]["saved_sha256"] for p, arm in zip(files, ("model", "zero"))):
            raise ValueError("Trace identity changed")
        with np.load(files[0], allow_pickle=False) as a, np.load(files[1], allow_pickle=False) as b:
            np.testing.assert_array_equal(a["prefix"], b["prefix"])
            np.testing.assert_array_equal(a["x"][0], b["x"][0])
            with torch.no_grad():
                prediction = model(torch.from_numpy(a["x"])).numpy()
                shuffled = model(torch.from_numpy(a["x"][::-1].copy())).numpy()[::-1]
            np.testing.assert_allclose(prediction, a["action"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(prediction, shuffled, rtol=0, atol=1e-12)
            assert not np.any(a["recovery_force_max"]) and not np.any(b["recovery_force_max"])
            numerical[name] = float(np.abs(prediction - a["action"]).max())
    put(output / "REPLAY.json", dict(all_verified=True, action_max_error=numerical,
                                     reordered_batches_independent=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--replay", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.replay:
        validate(args.output)
        return
    args.output.mkdir(exist_ok=False)
    started = time.monotonic()
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    bench = Bench(args.output)
    output = args.output
    teacher_fn = teacher(bench)
    training = {}
    samples = []
    for i, settings in enumerate(TRAIN):
        path = output / f"teacher_{i:02}.npz"
        row = bench.trial(settings, teacher_fn, path)
        training[str(i)] = row
        print("teacher", i, row["passed"], row["gates"], flush=True)
        if row["passed"]:
            with np.load(path, allow_pickle=False) as z:
                samples.append((z["x"], z["action"]))
    put(output / "TEACHERS.json", training)
    if len(samples) < 4:
        raise RuntimeError("Insufficient qualified teacher trajectories")
    x, y = (torch.from_numpy(np.concatenate(a)).double() for a in zip(*samples))
    center = torch.from_numpy(bench.center).double()
    scale = x.std(0).clamp_min(.004)
    model = StandPolicy(center, scale).double()
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    train_dir = output / "TRAIN"
    train_dir.mkdir()
    initial_mse = float(y.square().mean())
    with (train_dir / "journal.jsonl").open("x", buffering=1) as journal:
        for step in range(1, UPDATES + 1):
            ids = torch.randint(len(x), (192,), generator=generator)
            loss = (model(x[ids]) - y[ids]).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            checkpoint = train_dir / f"step_{step:04}.pt"
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(dict(format=FORMAT, model=model.state_dict(), optimizer=opt.state_dict(),
                            step=step, rng=generator.get_state()), temporary)
            temporary.rename(checkpoint)
            journal.write(json.dumps(dict(step=step, loss=float(loss.detach()),
                                          sha256=sha(checkpoint)), allow_nan=False) + "\n")
        # Keep the SHA of the final declared checkpoint before physical evaluation.
    digest = sha(checkpoint)
    result = {}
    model.eval()

    def inferred(x, target):
        with torch.no_grad():
            return model(torch.from_numpy(x)).numpy()

    for name, settings in HOLDOUT.items():
        row = {}
        for arm, control in (("model", inferred), ("zero", lambda x, target: bench.zero)):
            row[arm] = bench.trial(settings, control, output / f"{name}_{arm}.npz")
            print(name, arm, row[arm]["passed"], row[arm]["gates"], flush=True)
        result[name] = row
    identity.verify_runtime(bench.manifest, bench.body)
    with torch.no_grad():
        mse = float((model(x) - y).square().mean())
    put(output / "RESULT.json", dict(checkpoint_sha256=digest, updates=UPDATES,
        qualified_teacher_episodes=len(samples), initial_mse=initial_mse, final_mse=mse,
        train_rows=len(x), evaluation=result, settle_native_integrations=80000,
        wall_s=time.monotonic()-started, native_contact_audit=False, neck_in_use=False))


if __name__ == "__main__":
    main()
