"""Extend the red landmark optical calibration for a target beyond the obstacle."""
from pathlib import Path

import mujoco as mj
import numpy as np
import torch

from approach_runtime import Eyes, FORWARD, prepare, sha, turn
from approach_vision import COLOR, WIDTH, features, network

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "FAR_EYES.pt"


def collect():
    body, _ = prepare(HERE / "OPTICAL_START", "nominal")
    with np.load(FORWARD / "WALK_HELD_normal/TRACE.npz", allow_pickle=False) as saved:
        poses = np.vstack((body.d.qpos, saved["qpos"][1000:9000:1000]))
    rng = np.random.default_rng(81043)
    images, labels = [], []
    eyes = Eyes(body)
    try:
        for i in range(1600):
            body.d.qpos[:] = poses[i % len(poses)]
            mj.mj_forward(body.plant, body.d)
            angle = float(rng.uniform(-64., 64.))
            distance = float(rng.uniform(1.05, 6.40))
            seen = i % 10 != 0
            target = eyes.place(angle, distance) if seen else None
            background = (eyes.pose()[0].copy(), rng.uniform(-.6,.6), .255)
            image = eyes.image(target, WIDTH, COLOR, background)
            red = (image[...,0] > 45) & (image[...,1] < .7 * image[...,0]) & (image[...,2] < .7 * image[...,0])
            visible = bool(seen and np.count_nonzero(red) >= 3)
            images.append(image)
            labels.append((angle / 40., distance - 2.5, float(visible)))
    finally:
        eyes.close()
    np.savez_compressed(HERE / "FAR_OPTICAL.npz", images=images, labels=labels)
    return np.asarray(images), np.asarray(labels)


def fit(images, labels):
    torch.manual_seed(1821)
    x = features(torch.from_numpy(images))
    y = torch.tensor(labels, dtype=torch.float32)
    model = network()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    for _ in range(2200):
        batch = torch.randint(0, 1360, (100,))
        out, target = model(x[batch]), y[batch]
        seen = target[:,2] > 0
        loss = (torch.nn.functional.smooth_l1_loss(out[seen,:2], target[seen,:2], beta=.10)
                + .28 * torch.nn.functional.binary_cross_entropy_with_logits(out[:,2], target[:,2]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), WEIGHTS)
    reloaded = network().eval()
    reloaded.load_state_dict(torch.load(WEIGHTS, weights_only=True))
    with torch.inference_mode():
        prediction = reloaded(x[1360:])
    target = y[1360:]
    seen = target[:,2] > 0
    errors = (prediction[seen,:2] - target[seen,:2]).abs()
    result = dict(samples=1600, held=240, optimizer_updates=2200,
                  held_visible=int(seen.sum()), presence=float(((prediction[:,2] > 0) == seen).float().mean()),
                  bearing_mae_deg=float(errors[:,0].mean() * 40.),
                  distance_mae=float(errors[:,1].mean()), evaluated_distances=[1.05,6.40],
                  dataset_sha256=sha(HERE / "FAR_OPTICAL.npz"), weights_sha256=sha(WEIGHTS))
    turn.stand.put(HERE / "FAR_OPTICAL_RESULT.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    fit(*collect())
