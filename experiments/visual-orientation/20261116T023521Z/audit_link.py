"""Independently replay saved optics, neural delivery and frozen body actions."""
import argparse
from pathlib import Path

import numpy as np
import torch

from bench_runtime import HERE, load_body_policy, turn
from brain import VisualPolicy
from neck_runtime import BINDING
from sequence import RESTORE_DELAY, score
from trained_link import TrainedLink


def check_identity(path, digest):
    assert turn.stand.sha(path) == digest, path


def replay(path, motor, brain_path):
    result = turn.read(path / "RESULT.json")
    files = {key: path / name for key, name in {
        "eyes": "EYES.npz", "trace": "TRACE.npz", "receiver": "RECEIVER.npz",
        "transport": "TRANSPORT.json",
    }.items()}
    for key, item in files.items():
        check_identity(item, result[key + "_sha256"])
    codec_path = HERE / "CODEC.pt"
    check_identity(brain_path, result["checkpoint_sha256"])
    check_identity(codec_path, result["codec_sha256"])
    check_identity(BINDING, result["binding_sha256"])
    brain = VisualPolicy(brain_path)
    neck = TrainedLink(codec_path, result["profile"], result["receiver_seed"])
    assert result["graph_sha256"] == neck.graph_sha
    case = result["case"]
    mode = case["test"]
    onset = case["onset_ms"]
    actual = result["actual_ms"]
    records = turn.read(files["transport"])
    send_events = [event for event in records if event["event"] == "enqueue"]
    release_events = [event for event in records if event["event"] == "release"]
    count = 0
    with np.load(files["eyes"]) as z, np.load(files["receiver"]) as received:
        frames = z["rgb"]
        controls = z["commands"]
        labels = z["labels"]
        observations = received["observations"]
        actions = received["actions"]
        assert len(frames) == len(labels)
        intent, frame = 0, 0
        for tick in range(actual):
            if tick == 0 and mode in ("cut","restore"):
                neck.switch(True)
            if mode == "restore" and tick == onset + RESTORE_DELAY:
                neck.switch(False)
            if tick % 20 == 0:
                if tick >= 20:
                    assert labels[frame,0] == tick-20
                    intent, estimate, presence = brain.action(frames[frame])
                    assert labels[frame,5] == intent
                    np.testing.assert_allclose(controls[tick//5,5:7],[estimate,presence],atol=0.,rtol=0.)
                    frame += 1
                neck.send(intent)
            if tick % 5 == 0:
                ix = tick//5
                row = controls[ix]
                assert row[0]==tick and row[2]==intent and row[3]==neck.action()
                np.testing.assert_array_equal(row[7:9], neck.latest)
                np.testing.assert_array_equal(actions[ix], motor(observations[ix],0,neck.action()))
                count += 1
            neck.advance()
        assert frame==len(frames)
        steps = [value[0] for value in neck.measurements]
        decodes = [value[1] if value[1] is not None else -2 for value in neck.measurements]
        measurement = np.asarray([item[2] for item in neck.measurements if item[2] is not None],dtype=np.float32)
        np.testing.assert_array_equal(steps, received["release_steps"])
        np.testing.assert_array_equal(decodes, received["decoded"])
        np.testing.assert_array_equal(measurement, received["measurements"])
        with np.load(files["trace"]) as t:
            scores = score(case, t, controls, z["bearing"], onset, actual)
            assert all(result[k] == v for k,v in scores.items() if k != "passed")
    assert neck.transport.journal() == records
    assert all(x["release_step"]-x["capture_step"]==2 and x["root_ids"]==neck.ids for x in release_events)
    assert len(send_events)==len(release_events)
    neck.transport.verify_final()
    return dict(case=str(path), physical_commands_bitwise=count, eyes_bitwise=frame,
                packets_replayed=len(release_events), receiver_dropped=int(sum(d==-2 for d in decodes)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("results", type=Path, nargs="+")
    p.add_argument("--checkpoint", type=Path, default=HERE / "VISION.pt")
    args = p.parse_args()
    torch.set_num_threads(1)
    body = load_body_policy()
    audit = [replay(s.resolve(),body,args.checkpoint) for s in args.results]
    turn.stand.put(args.output, audit)
    print(len(audit), "complete replays")
