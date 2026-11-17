"""Fit binocular perception and steering from admitted recorded teacher episodes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bench_runtime import HERE, turn
from brain import Brain


def recorded(folder, sources):
    frames = []
    labels = []
    for where in sorted(folder.glob("*/RESULT.json")):
        if not turn.read(where)["passed"]:
            continue
        file = where.parent / "EYES.npz"
        assert turn.stand.sha(file) == turn.read(where)["eyes_sha256"]
        with np.load(file, allow_pickle=False) as z:
            tag = z["labels"][:, 1:5]
            frames.extend(z["rgb"])
            labels.extend(tag)
        sources[str(file)] = turn.stand.sha(file)
    if not frames:
        raise ValueError("No admitted physical teacher sequences")
    return np.asarray(frames), np.asarray(labels)


def static(name, sources, rng):
    path = HERE / f"{name}.npz"
    sources[str(path)] = turn.stand.sha(path)
    with np.load(path, allow_pickle=False) as z:
        labels = z["labels"]
        prev = rng.integers(-1, 2, len(labels))
        angle = labels[:, 1]
        active = np.where(prev == 0, abs(angle)>10.5, prev*angle > -4.)
        selected = np.where(active & (labels[:,0]==1), np.where(prev==0, np.sign(angle), prev), 0.)
        return z["rgb"], np.column_stack([labels, prev, selected])


def tensors(sample):
    a, b = sample
    return torch.from_numpy(a), torch.from_numpy(b.astype(np.float32))


def evaluate(model, dataset):
    with torch.inference_mode():
        estimate = torch.cat([model(x) for x in dataset[0].split(256)])
        labels = dataset[1]
        visible = labels[:, 0] > 0
        estimated = estimate[:, 1] > 0.
        actions = model.action_logits(estimate, labels[:,2]).argmax(dim=1)-1
        actions[~estimated] = 0
        abs_error = torch.abs(estimate[:,0]*90.-labels[:,1])
        return dict(presence_accuracy=float((visible==estimated).float().mean()),
                    visible_angle_mae_deg=float(abs_error[visible].mean()),
                    visible_angle_p95_deg=float(abs_error[visible].quantile(.95)),
                    decision_accuracy=float((actions==labels[:,3]).float().mean()))


def train(folder, output, reserved, calibrated):
    torch.set_num_threads(1)
    torch.manual_seed(6142)
    rand = torch.Generator().manual_seed(27145)
    rng = np.random.default_rng(982530)
    sources = {}
    archive = recorded(folder, sources)
    optic = static("OPTICAL_TRAIN", sources, rng)
    dev = tensors(static("OPTICAL_DEV", sources, rng))
    for extra in reserved:
        x, y = recorded(extra, sources)
        archive = (np.concatenate((archive[0], x)), np.concatenate((archive[1], y)))
    raw = tensors(optic)
    physical = tensors(archive)
    model = Brain()
    opt = torch.optim.Adam(model.parameters(), lr=.0025)
    journal = []
    for step in range(720):
        p = torch.randint(len(physical[0]), (48,), generator=rand)
        q = torch.randint(len(raw[0]), (80,), generator=rand)
        rgb = torch.cat((physical[0][p], raw[0][q]))
        labels = torch.cat((physical[1][p], raw[1][q]))
        estimate = model(rgb)
        visible = labels[:, 0]
        angle = labels[:, 1] / 90.
        opt.zero_grad(set_to_none=True)
        loss_angle = (F.smooth_l1_loss(estimate[:, 0], angle, beta=.07, reduction="none") * (visible+.08)).mean()
        loss_seen = F.binary_cross_entropy_with_logits(estimate[:,1], visible)
        supervision = F.cross_entropy(model.action_logits(estimate, labels[:,2]),
                                      (labels[:,3]+1).long(), reduction="none")
        if calibrated:
            loss_intent = (supervision * visible).sum() / visible.sum().clamp_min(1.)
            anchor = .012*(model.stop*90.+4.)**2 + .001*(model.start*90.-10.5)**2
        else:
            loss_intent = supervision.mean()
            anchor = 0.
        loss = 4.2*loss_angle + .3*loss_seen + .075*loss_intent + anchor
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
        journal.append(float(loss.detach()))
    with output.open("xb") as file:
        torch.save(model.state_dict(), file)
    reloaded = Brain().eval()
    reloaded.load_state_dict(torch.load(output, weights_only=True))
    for p,q in zip(model.parameters(), reloaded.parameters(), strict=True):
        torch.testing.assert_close(p, q, rtol=0., atol=0.)
    result = dict(checkpoint_sha256=turn.stand.sha(output), inputs=sources,
                  physical_frames=len(archive[0]), optical_train=len(raw[0]), dev_frames=len(dev[0]),
                  updates=len(journal), curve=journal, calibrated=calibrated, fitted_start=float(reloaded.start.item()*90.),
                  fitted_stop=float(reloaded.stop.item()*90.),
                  train_physical=evaluate(reloaded, physical), held_optical=evaluate(reloaded,dev))
    target = output.with_suffix(".json")
    with target.open("x") as file:
        json.dump(result,file,indent=2)
        file.write("\n")
    print({k:v for k,v in result.items() if k not in ("curve","inputs")},flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--more", nargs="*", type=Path, default=[])
    parser.add_argument("--calibrated", action="store_true")
    args = parser.parse_args()
    train(args.teacher, args.output, args.more, args.calibrated)
