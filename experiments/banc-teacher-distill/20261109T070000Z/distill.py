"""Map the recorded teacher joint angles onto our 42-joint body and train the base network by imitation."""
from pathlib import Path
import json, sys, importlib.util
import numpy as np
import mujoco as mj
HERE = Path(__file__).resolve().parent
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
SRC = ROOT / 'experiments/banc-stand-retarget/20261108T190000Z'
LEGS_OUT = ['LF', 'LH', 'LM', 'RF', 'RH', 'RM']          # order used by the base network file
LEG_MAP = {'T1_left': 'LF', 'T2_left': 'LM', 'T3_left': 'LH', 'T1_right': 'RF', 'T2_right': 'RM', 'T3_right': 'RH'}
JOINT_MAP = {'coxa_abduct': 'Coxa_roll', 'coxa_twist': 'Coxa_yaw', 'coxa': 'Coxa_pitch',
             'femur_twist': 'Trochanter_yaw', 'femur': 'Trochanter_pitch', 'tibia': 'Tibia_pitch'}
LAGS = np.array([0, 8, 24, 32, 36, 38, 39]); K = 40
STEPS, BATCH, LR, HID = 4000, 96, 2e-3, 128


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


def main():
    act = load('act', ROOT / 'experiments/s5-ann-incremental/20261107T010000Z/incremental_actuator.py')
    tf = load('tf', SRC / 'train_full42_stand.py')
    plant = act.load_full_model()
    jids = plant.actuator_trnid[:, 0]
    jname = [plant.joint(i).name for i in jids]
    lo = plant.jnt_range[jids][:, 0]; hi = plant.jnt_range[jids][:, 1]
    z = np.load(HERE / 'TEACHER_ROLLOUT.npz')
    tq = z['qpos'].astype(np.float64); names = [str(s) for s in z['joint_names']]
    col = {n: i for i, n in enumerate(names)}
    stand = np.asarray(tf.STAND_Q, float)
    targets = np.tile(stand, (tq.shape[0], 1))
    offsets, used, clip_frac = {}, [], {}
    for teach_joint in JOINT_MAP:
        for tleg, oleg in LEG_MAP.items():
            src_col = col['%s_%s' % (teach_joint, tleg)]
            ours = 'joint_%s%s' % (oleg, JOINT_MAP[teach_joint])
            if ours not in jname:
                continue
            i = jname.index(ours)
            raw = tq[:, src_col]
            off = float(stand[i] - raw.mean())
            offsets[ours] = round(off, 5)
            targets[:, i] = raw + off
            used.append(ours)
    # our Trochanter_roll has no teacher counterpart: keep the stand value (declared)
    unmapped = [j for j in jname if j not in used]
    before = targets.copy()
    targets = np.clip(targets, (lo + .01)[None], (hi - .01)[None])
    clip_frac = {jname[i]: float(np.mean(before[:, i] != targets[:, i])) for i in range(42)}
    np.savez_compressed(HERE / 'TRAINING_SET_TEACHER.npz', targets_rad=targets.astype(np.float32),
                        offsets=np.array([0, targets.shape[0]]), source='teacher_rollout',
                        joint_names=np.array(jname))
    print('samples', targets.shape, 'mapped joints', len(used), 'unmapped', len(unmapped))
    rng = np.random.default_rng(0)
    center = targets.mean(0); scale = np.maximum(targets.std(0), 1e-3)
    Y = (targets - center) / scale
    N = targets.shape[0]; split = int(.85 * N)
    tr = np.arange(split); va = np.arange(split, N)
    X = np.zeros((N, len(LAGS) * 42))
    for t in range(N):
        idxs = np.clip(t - K + 1 + LAGS, 0, N - 1)
        X[t] = targets[idxs].reshape(-1)
    P = dict(w0=rng.normal(0, .3, (X.shape[1], HID)), b0=np.zeros(HID),
             w1=rng.normal(0, .3, (HID, HID)), b1=np.zeros(HID),
             w2=rng.normal(0, .3, (HID, 42)), b2=np.zeros(42))
    m = {k: np.zeros_like(v) for k, v in P.items()}; v = {k: np.zeros_like(v) for k, v in P.items()}
    b1, b2, eps = .9, .999, 1e-8
    def forward(xb):
        h0 = np.tanh(xb @ P['w0'] + P['b0']); h1 = np.tanh(h0 @ P['w1'] + P['b1'])
        return h0, h1, h1 @ P['w2'] + P['b2']
    for step in range(1, STEPS + 1):
        idx = rng.integers(0, len(tr), BATCH)
        xb = X[tr[idx]]; yb = Y[tr[idx]]
        h0, h1, out = forward(xb)
        d = (out - yb) * 2 / yb.size
        g = dict(w2=h1.T @ d, b2=d.sum(0))
        dh1 = (d @ P['w2'].T) * (1 - h1 ** 2)
        g['w1'] = h0.T @ dh1; g['b1'] = dh1.sum(0)
        dh0 = (dh1 @ P['w1'].T) * (1 - h0 ** 2)
        g['w0'] = xb.T @ dh0; g['b0'] = dh0.sum(0)
        for k in P:
            m[k] = b1 * m[k] + (1 - b1) * g[k]; v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
            P[k] -= LR * (m[k] / (1 - b1 ** step)) / (np.sqrt(v[k] / (1 - b2 ** step)) + eps)
        if step % 1000 == 0:
            _, _, o = forward(X[tr]); tv = float(np.mean((o - Y[tr]) ** 2))
            _, _, o2 = forward(X[va]); vv = float(np.mean((o2 - Y[va]) ** 2))
            print('step %d train %.5f val %.5f' % (step, tv, vv), flush=True)
    final = dict(final_train_mse=tv * 42 * np.mean(scale ** 2), final_val_mse=vv * 42 * np.mean(scale ** 2))
    np.savez_compressed(HERE / 'MODEL_MEASURED_TEACHER.npz', center=center, scale=scale, lags=LAGS, K=K,
                        w0=P['w0'], b0=P['b0'], w1=P['w1'], b1=P['b1'], w2=P['w2'], b2=P['b2'],
                        joint_names=np.array(jname),
                        metadata_json=json.dumps(dict(status='BASE_NETWORK_TRAINED_ON_TEACHER_IMITATION', **final)))
    (HERE / 'DISTILL_MAP.json').write_text(json.dumps(dict(
        teacher_samples=int(tq.shape[0]), mapped_joints=used, unmapped_joints=unmapped,
        per_joint_offset=offsets, clip_fraction=clip_frac,
        legs=LEG_MAP, joints=JOINT_MAP,
        note='additive offsets align the teacher time-mean to our solved stand pose; joint signs assumed identical and not verified; tarsus/tarsus2/adhere_claw have no DOF in our body',
        training=dict(steps=STEPS, batch=BATCH, lr=LR, hidden=HID, lags=LAGS.tolist(), **final)), indent=2) + '\n')
    print(json.dumps(final))


if __name__ == '__main__':
    main()
