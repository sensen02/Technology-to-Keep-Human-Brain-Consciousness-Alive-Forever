"""A scoped visual start/walk/stop pilot through delivered DNg100 activity."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
VISION = HERE.parent / "20261116T023521Z"
sys.path.insert(0, str(VISION))
from bench_runtime import arrays, check_forces, prepare, sample, stand_window, turn
from brain import Brain
from optics import Eyes
from verify_native import Record
from walking_bench import metrics
from forward_link import ForwardLink, BINDING, GRAPH

VISION_WEIGHT = VISION / "VISION_CAL.pt"


class WalkHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = torch.nn.Linear(2, 1)

    def forward(self, features):
        return self.classifier(features).squeeze(-1)


def visual_encoder():
    encoder = Brain().eval()
    encoder.load_state_dict(torch.load(VISION_WEIGHT, map_location="cpu", weights_only=True))
    return encoder


def qualification(trace, record, commands, finish, mode):
    np.testing.assert_array_equal(record.endpoints, trace["qpos"])
    np.testing.assert_allclose(record.poses[:, 6], trace["time"][0] + np.arange(record.count)*12.5e-6, atol=5e-8, rtol=0)
    assert record.count == finish*80 and not record.poses[:, 7].any()
    moving = np.flatnonzero(commands[:, 2])
    if mode in {"off", "cut"}:
        final = stand_window(trace, finish)
        return dict(passed=bool(final["passed"] and not len(moving) and not record.forces[:, 12].any()
                                and bool(commands[:, 1].any()) == (mode == "cut")),
                    terminal_stand=final, walk_start_ms=None, walk_stop_ms=None)
    if not len(moving):
        return dict(passed=False, reason="no_delivered_motion")
    start = int(commands[moving[0], 0])
    stop = int(commands[moving[-1], 0]) + 5
    if stop > finish or stop - start < 8000:
        return dict(passed=False, reason="short_or_unfinished_motion", walk_start_ms=start, walk_stop_ms=stop)
    bench = metrics(trace, start, stop-start, finish)
    force = record.forces[start*80:stop*80]
    totals = force[:, :6].sum(0)
    absolute = force[:, 6:12].sum(0) / np.maximum(1e-10, totals)
    displacement = trace["com"][stop, :2] - trace["com"][start, :2]
    elapsed = (stop-start)*.001
    slip = absolute.max() / max(1e-10, np.linalg.norm(displacement)/elapsed)
    forward = displacement[0]/elapsed
    com_height = record.poses[start*80:stop*80, 2].min()
    raw_load = totals.sum() / len(force) / turn.W
    native_endpoints = [record.forces[(t-200)*80:t*80] for t in (start, finish)]
    window_support = all(v[:, :6].sum(1).mean()/turn.W >= .90 and
                         ((v[:, :6]/turn.W >= .02).sum(1) >= 3).mean() >= .95 for v in native_endpoints)
    contiguous = len(moving) == moving[-1]-moving[0]+1
    settle = next((t-stop for t in range(stop+200, min(stop+1200, finish)+1, 5)
                   if stand_window(trace, t)["passed"]), None)
    gates = dict(forward=forward >= .05, slip=slip <= .5, support=raw_load >= .90,
                 no_nonfoot=not record.forces[:, 12].any(), com_height=com_height >= 1.2,
                 cycles=all(bench["cycles"].values()), start=bench["initial_stand"]["passed"],
                 stop=bench["terminal_stand"]["passed"], settle=settle is not None,
                 native_window_support=window_support, contiguous=contiguous)
    return dict(passed=all(gates.values()), native_checks={k: bool(v) for k, v in gates.items()},
                walk_start_ms=start, walk_stop_ms=stop, settle_after_stop_ms=settle,
                native_signed_forward=float(forward), native_slip_ratio=float(slip),
                native_support_W=float(raw_load), native_min_com=float(com_height),
                cycles=bench["cycles"], initial_stand=bench["initial_stand"], terminal_stand=bench["terminal_stand"])


def trial(args):
    if args.onset % 20 or args.walk_ms % 20 or args.onset < 500:
        raise ValueError("Aligned 20 ms scene schedule and at least 500 ms initial standing required.")
    if args.mode in {"cut", "restore", "off"} and args.checkpoint is None:
        raise ValueError("Use frozen fitted visual policy for a held-out control.")
    brain = None
    if args.checkpoint:
        brain = WalkHead().eval()
        brain.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    case = vars(args).copy()
    case.update(output=str(args.output), checkpoint=str(args.checkpoint) if args.checkpoint else None)
    body, motor = prepare(args.output, args.start)
    neck = ForwardLink()
    encoder = visual_encoder()
    end = args.onset + args.walk_ms
    finish = end + 1300
    restore_at = args.onset + 400
    refs = [Path(__file__), HERE/"forward_link.py", BINDING, GRAPH, VISION_WEIGHT,
            turn.STARTS/f"{args.start}.npz", Path(turn.straight_policy.__code__.co_filename)]
    if args.checkpoint:
        refs.append(args.checkpoint)
    identity = turn.stand.identity.capture(body, dict(scope="visual straight walk/stop via DNg100", case=case), extra_paths=refs)
    turn.stand.put(args.output/"RUN_IDENTITY.json", identity)
    record = Record(body, finish)
    body.recorder = record
    eyes = Eyes(body)
    measured = body.measure()
    rows = [sample(body, measured)]
    commands, pictures, labels, features, decisions, observations, actions = [], [], [], [], [], [], []
    visible = False
    target = None
    intent, capture_id = 0, -1
    background = (eyes.pose()[0].copy(), args.shift, args.gray)
    try:
        for tick in range(finish):
            if tick == 0 and args.mode in {"cut", "restore"}:
                neck.switch(True)
            if tick == restore_at and args.mode == "restore":
                neck.switch(False)
            if tick == args.onset and args.mode != "off":
                target = eyes.place(args.bearing, args.distance)
            if tick == end:
                target = None
            if tick % 20 == 0:
                exposed = eyes.capture(tick, target, args.width, (1., .045, .035, 1.), background)
                published_truth = visible
                visible = target is not None
                if exposed is not None:
                    capture_id, rgb = exposed
                    with torch.inference_mode():
                        feature = encoder(torch.as_tensor(rgb[None]))[0]
                        intent = int(published_truth) if brain is None else int(brain(feature).item() > 0)
                    pictures.append(rgb.copy())
                    labels.append(int(published_truth))
                    features.append(feature.numpy())
                    decisions.append((tick, capture_id, intent))
                neck.send(intent)
            if tick % 5 == 0:
                delivered = neck.action()
                observations.append(turn.features(body, measured))
                action = motor(observations[-1], delivered, 0)
                actions.append(action.copy())
                commands.append((tick, intent, delivered, capture_id,
                                 neck.delivered_step if neck.delivered_step is not None else -1))
            measured = body.tick(action)
            check_forces(body)
            rows.append(sample(body, measured))
            neck.advance()
    finally:
        eyes.close()
    trace = arrays(rows)
    commands = np.asarray(commands, dtype=np.int64)
    assert np.all(commands[:, 0] >= commands[:, 4])
    assert np.all(commands[commands[:, 3] >= 0, 0] >= commands[commands[:, 3] >= 0, 3] + 20)
    turn.stand.identity.verify_runtime(identity, body)
    transport = neck.transport.verify_final()
    assert transport["actual_steps"] == finish
    result = qualification(trace, record, commands, finish, args.mode)
    if args.mode == "restore":
        result["passed"] &= bool(result["walk_start_ms"] is not None and
                                 result["walk_start_ms"] >= restore_at and commands[commands[:, 0]<restore_at, 1].any() and
                                 not commands[commands[:, 0]<restore_at, 2].any())
    paths = {
        "TRACE.npz": trace, "NATIVE.npz": dict(forces=record.forces, pose=record.poses),
        "WALK_EYES.npz": dict(rgb=np.asarray(pictures), features=np.asarray(features), labels=np.asarray(labels), decisions=np.asarray(decisions)),
        "WALK_COMMANDS.npz": dict(commands=commands, observations=np.asarray(observations), actions=np.asarray(actions)),
    }
    for name, fields in paths.items():
        np.savez_compressed(args.output/name, **fields)
    turn.stand.put(args.output/"WALK_TRANSPORT.json", neck.transport.journal())
    result.update(scope="actual released images and official-source DNg100 projection; single low-speed level; flat",
                  case=case, frames=len(pictures), model="teacher" if brain is None else "trained",
                  visual_accuracy=float((np.asarray(decisions)[:, 2] == labels).mean()),
                  native_samples=record.count, contact_cross_checks=record.cross_checks,
                  graph_sha256=neck.graph_sha, visual_encoder_sha256=turn.stand.sha(VISION_WEIGHT),
                  files={name: turn.stand.sha(args.output/name) for name in (*paths, "WALK_TRANSPORT.json")})
    turn.stand.put(args.output/"WALK_RESULT.json", result)
    print(result, flush=True)


def fit(args):
    features, labels, proofs = [], [], []
    for folder in args.teachers:
        result = turn.read(folder/"WALK_RESULT.json")
        path = folder/"WALK_EYES.npz"
        assert result["model"] == "teacher" and result["passed"] and turn.stand.sha(path) == result["files"][path.name]
        with np.load(path, allow_pickle=False) as evidence:
            features.append(evidence["features"].copy())
            labels.append(evidence["labels"].copy())
        proofs.append(dict(path=str(path.resolve()), sha256=turn.stand.sha(path)))
    x = torch.tensor(np.concatenate(features), dtype=torch.float32)
    y = torch.tensor(np.concatenate(labels), dtype=torch.float32)
    torch.manual_seed(412)
    model = WalkHead()
    optimizer = torch.optim.Adam(model.parameters(), lr=.04)
    weight = (1-y).sum()/y.sum()
    history = []
    for _ in range(240):
        optimizer.zero_grad()
        pred = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, y, pos_weight=weight)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    with args.output.open("xb") as file:
        torch.save(model.state_dict(), file)
    result = dict(checkpoint_sha256=turn.stand.sha(args.output), sources=proofs,
                  samples=len(x), absent=int((y == 0).sum()), present=int(y.sum()),
                  train_accuracy=float(((model(x)>0)==y.bool()).float().mean().detach()),
                  optimizer="Adam", steps=len(history), losses=history)
    turn.stand.put(args.output.with_suffix(".json"), result)
    print({k:v for k,v in result.items() if k != "losses"}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(required=True)
    run = commands.add_parser("run")
    run.add_argument("output", type=Path)
    run.add_argument("--mode", choices=("normal", "cut", "restore", "off"), default="normal")
    run.add_argument("--onset", type=int, default=500)
    run.add_argument("--walk-ms", type=int, default=9600)
    run.add_argument("--start", default="nominal")
    run.add_argument("--width", type=float, default=.17)
    run.add_argument("--distance", type=float, default=3.3)
    run.add_argument("--bearing", type=float, default=0.)
    run.add_argument("--shift", type=float, default=0.)
    run.add_argument("--gray", type=float, default=.285)
    run.add_argument("--checkpoint", type=Path)
    run.set_defaults(call=trial)
    train = commands.add_parser("fit")
    train.add_argument("output", type=Path)
    train.add_argument("teachers", type=Path, nargs="+")
    train.set_defaults(call=fit)
    torch.set_num_threads(1)
    options = p.parse_args()
    options.call(options)
