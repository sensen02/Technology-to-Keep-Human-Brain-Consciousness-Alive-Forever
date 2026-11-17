"""Recompute held brain, transport and body commands from archived observations."""
import argparse
from pathlib import Path

import numpy as np
import torch
from walk_pilot import ForwardLink, WalkHead, turn, visual_encoder


def check(run):
    result = turn.read(run/"WALK_RESULT.json")
    case = result["case"]
    checkpoint = Path(case["checkpoint"])
    assert turn.stand.sha(checkpoint) == turn.read(checkpoint.with_suffix(".json"))["checkpoint_sha256"]
    for name in ("WALK_EYES.npz", "WALK_COMMANDS.npz"):
        assert turn.stand.sha(run/name) == result["files"][name]
    with np.load(run/"WALK_EYES.npz", allow_pickle=False) as archive:
        images, features, decisions = [archive[key] for key in ("rgb","features","decisions")]
    with np.load(run/"WALK_COMMANDS.npz", allow_pickle=False) as archive:
        commands, observations, motor_actions = [archive[key] for key in ("commands","observations","actions")]
    model = WalkHead().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    encoder = visual_encoder()
    motor = turn.straight_policy()
    link = ForwardLink()
    assert link.graph_sha == result["graph_sha256"]
    current, frame_end = 0, 0
    restore_at = case["onset"] + 400
    predicted_neck, predicted_motor = [], []
    max_feature_error = 0.
    with torch.inference_mode():
        for tick in range(result["native_samples"]//80):
            if tick == 0 and case["mode"] in ("cut","restore"):
                link.switch(True)
            if tick == restore_at and case["mode"] == "restore":
                link.switch(False)
            if tick % 20 == 0:
                if frame_end < len(decisions) and decisions[frame_end,0] == tick:
                    feature = encoder(torch.as_tensor(images[frame_end:frame_end+1]))[0]
                    max_feature_error = max(max_feature_error,
                                            float(np.max(np.abs(feature.numpy()-features[frame_end]))))
                    current = int(model(feature).item() > 0)
                    assert current == decisions[frame_end,2]
                    frame_end += 1
                link.send(current)
            if tick % 5 == 0:
                delivered = link.action()
                predicted_neck.append((tick,current,delivered,link.delivered_step if link.delivered_step is not None else -1))
                predicted_motor.append(motor(observations[tick//5],delivered).copy())
            link.advance()
    assert frame_end == len(decisions) and max_feature_error == 0.
    np.testing.assert_array_equal(np.asarray(predicted_neck),commands[:,[0,1,2,4]])
    np.testing.assert_array_equal(np.asarray(predicted_motor),motor_actions)
    assert link.transport.journal() == turn.read(run/"WALK_TRANSPORT.json")
    verified = link.transport.verify_final()
    return dict(run=str(run.resolve()),eye_frames=frame_end,receiver_ticks=len(commands),
                feature_max_abs_error=max_feature_error,eye_intent_exact=True,epoch_messages_exact=True,
                motor_actions_exact=True,source_verified=verified["source_files_unchanged"],
                added_body_integration_steps=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("runs",type=Path,nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(1)
    payload = {"passed": True, "runs": [check(run) for run in args.runs]}
    turn.stand.put(args.output, payload)
    print(payload,flush=True)
