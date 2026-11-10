"""Supervised base network trained on the filtered measured motion (no physics, no teacher).

Architecture and feature contract are the frozen ones: input = own-target history at seven
lags (7 x 42 = 294), hidden 128 tanh, 128 tanh, output 42 normalised targets, with
centre/scale taken from the frozen base model so the numbers stay comparable.
Validation splits by SEGMENT, never by neighbouring frames.
"""
from pathlib import Path
import hashlib, json, time
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T111149Z/MODEL_VELOCITY.npz')
LAGS = np.array([0, 8, 24, 32, 36, 38, 39])
K = 40
HID = 128
STEPS = 4000
BATCH = 96
LR = 2e-3


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_samples(targets, offsets):
    xs, ys = [], []
    for s in range(len(offsets) - 1):
        q = targets[int(offsets[s]):int(offsets[s + 1])]
        n = q.shape[0]
        if n <= K:
            continue
        for t in range(K, n):
            buf = q[t - K:t]
            xs.append(buf[LAGS].reshape(-1)); ys.append(q[t])
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)


def main():
    ts = np.load(HERE / 'TRAINING_SET_STAND.npz')
    targets = ts['targets_rad']; offsets = ts['offsets']; names = ts['joint_names']
    base = np.load(BASE)
    center = base['center'].astype(np.float32); scale = base['scale'].astype(np.float32)
    # normalise each segment independently by the frozen centre/scale
    qn = (targets - center) / scale
    seg_start = offsets[:-1]
    # segment-level split: every 5th segment held out
    train_mask = np.array([(s % 5) != 4 for s in range(len(offsets) - 1)])
    def gather(mask):
        xs, ys = [], []
        for s in range(len(offsets) - 1):
            if bool(mask[s]) is False:
                continue
            q = qn[int(offsets[s]):int(offsets[s + 1])]
            if q.shape[0] <= K:
                continue
            for t in range(K, q.shape[0]):
                xs.append(q[t - K:t][LAGS].reshape(-1)); ys.append(q[t])
        return np.asarray(xs, np.float32), np.asarray(ys, np.float32)
    Xtr, Ytr = gather(train_mask)
    Xva, Yva = gather(~train_mask)
    rng = np.random.default_rng(20261108)
    P = dict(w0=(rng.standard_normal((HID, 294)) * np.sqrt(1 / 294)).astype(np.float32), b0=np.zeros(HID, np.float32),
             w1=(rng.standard_normal((HID, HID)) * np.sqrt(1 / HID)).astype(np.float32), b1=np.zeros(HID, np.float32),
             w2=(rng.standard_normal((42, HID)) * np.sqrt(1 / HID)).astype(np.float32), b2=np.zeros(42, np.float32))
    M = {k: np.zeros_like(v) for k, v in P.items()}; V = {k: np.zeros_like(v) for k, v in P.items()}
    def forward(X):
        h0 = np.tanh(X @ P['w0'].T + P['b0'])
        h1 = np.tanh(h0 @ P['w1'].T + P['b1'])
        return h0, h1, h1 @ P['w2'].T + P['b2']
    t0 = time.time(); log = []
    for step in range(1, STEPS + 1):
        idx = rng.integers(0, Xtr.shape[0], BATCH)
        x, y = Xtr[idx], Ytr[idx]
        h0, h1, out = forward(x)
        err = out - y
        loss = float(np.mean(err ** 2))
        g = 2 * err / err.size
        g_w0 = (g @ P['w2'] * (1 - h1 ** 2)) @ P['w1'] * (1 - h0 ** 2)
        grads = dict(w2=g.T @ h1, b2=g.sum(0),
                     w1=((g @ P['w2'] * (1 - h1 ** 2)).T @ h0), b1=(g @ P['w2'] * (1 - h1 ** 2)).sum(0),
                     w0=g_w0.T @ x, b0=g_w0.sum(0))
        for k in P:
            M[k] = .9 * M[k] + .1 * grads[k]; V[k] = .999 * V[k] + .001 * grads[k] ** 2
            mhat = M[k] / (1 - .9 ** step); vhat = V[k] / (1 - .999 ** step)
            P[k] -= (LR * mhat / (np.sqrt(vhat) + 1e-8)).astype(np.float32)
        if step % 500 == 0:
            _, _, ov = forward(Xva)
            vl = float(np.mean((ov - Yva) ** 2))
            log.append(dict(step=step, train_mse=loss, val_mse=vl))
            print(json.dumps(log[-1]), flush=True)
    _, _, otr = forward(Xtr); _, _, ova = forward(Xva)
    out = dict(status='BASE_NETWORK_TRAINED_ON_STAND_RETARGET', samples_train=int(Xtr.shape[0]), samples_val=int(Xva.shape[0]),
               segments_train=int(train_mask.sum()), segments_val=int((~train_mask).sum()),
               final_train_mse=float(np.mean((otr - Ytr) ** 2)), final_val_mse=float(np.mean((ova - Yva) ** 2)),
               log=log, steps=STEPS, batch=BATCH, lr=LR, architecture='294->128tanh->128tanh->42', lags=LAGS.tolist(), K=K,
               centre_scale_from=str(BASE), centre_scale_sha256=sha(BASE), training_set=str(HERE / 'TRAINING_SET_STAND.npz'),
               wall_s=time.time() - t0, physics_steps=0, neural_steps=0,
               note='supervised only on filtered retargeted measured motion; no teacher, no synthetic periodic curve, no physics in training')
    np.savez_compressed(HERE / 'MODEL_MEASURED_STAND.npz', center=center, scale=scale, lags=LAGS, K=K,
                        w0=P['w0'].T.astype(np.float32), b0=P['b0'], w1=P['w1'].T.astype(np.float32), b1=P['b1'],
                        w2=P['w2'].T.astype(np.float32), b2=P['b2'],
                        joint_names=names, metadata_json=np.array(json.dumps({k: out[k] for k in ('status', 'final_train_mse', 'final_val_mse')})))
    (HERE / 'MODEL_MEASURED_STAND.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: out[k] for k in ('status', 'samples_train', 'samples_val', 'final_train_mse', 'final_val_mse', 'wall_s')}))


if __name__ == '__main__':
    main()
