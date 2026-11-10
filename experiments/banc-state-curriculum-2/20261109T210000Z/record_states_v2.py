"""Record simulator states during the release phase of an assist-then-release rollout (our own simulation, no external generator)."""
from pathlib import Path
import json, sys, hashlib, importlib.util
import numpy as np
import mujoco as mj
HERE = Path(__file__).resolve().parent
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
SRC = ROOT / 'experiments/banc-stand-retarget/20261108T190000Z'
PAD_XML = ROOT / 'experiments/banc-tarsus-pad/20261109T010000Z/MODEL_TARSUS_PAD_R02.xml'
MODEL = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-state-curriculum/20261109T190000Z/MODEL_sc.npz')
BANK = Path(__file__).resolve().parent / 'STATE_BANK_V2.npz'
N_ROLLOUTS = 2
WEIGHT = 24.446354755878005; LEGS = ['LF', 'LH', 'LM', 'RF', 'RH', 'RM']
PDT = .001; BASE_EVERY = 5; CAP = 30.; C = .02; A_ADH = .25 * WEIGHT; STANCE = .1 * WEIGHT
ASSIST, RELEASE = 1000, 3000


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


def main():
    act = load('act', ROOT / 'experiments/s5-ann-incremental/20261107T010000Z/incremental_actuator.py')
    sens = load('sens', ROOT / 'experiments/s5-ann-walk/20261107T091243Z/proprioception.py')
    tfm = load('tfm', SRC / 'train_full42_stand.py')
    plant = act.load_full_model(PAD_XML); plant.geom_friction[:, 0] = 1.5
    d = mj.MjData(plant)
    base = np.load(SRC / 'MODEL_MEASURED_STAND.npz'); net = np.load(MODEL)
    jids = plant.actuator_trnid[:, 0]; th = plant.body('Thorax').id
    names = [plant.geom(i).name or '' for i in range(plant.ngeom)]
    pads = {L: next(i for i, nm in enumerate(names) if nm == L + 'Tarsus5_pad') for L in LEGS}
    planes = plant.geom_type == mj.mjtGeom.mjGEOM_PLANE

    class Shim:
        def __init__(s, p): s.m = p; s.jids = jids; s.qa = p.jnt_qposadr[jids]; s.va = p.jnt_dofadr[jids]; s.q0 = p.key_qpos[0]; s.thorax = th
    shim = Shim(plant)
    mj.mj_resetDataKeyframe(plant, d, 0)
    d.qpos[plant.jnt_qposadr[jids]] = np.asarray(tfm.STAND_Q, float); d.qpos[2] = tfm.STAND_ROOT_Z
    mj.mj_forward(plant, d)
    actuator = act.IncrementalActuator(plant, d); actuator.reset(d)
    K = int(base['K']); lags = np.asarray(base['lags']); q0 = np.asarray(tfm.STAND_Q, float)
    buf = np.repeat(q0[None], K, axis=0)
    ts = np.load(SRC / 'TRAINING_SET_STAND.npz')
    off = np.asarray(ts['offsets'], dtype=int); tar = ts['targets_rad'].astype(np.float64)

    def measure():
        loads = np.zeros(6); nonfoot = 0.
        for ci in range(d.ncon):
            c = d.contact[ci]; ga, gb = int(c.geom1), int(c.geom2)
            if not (planes[ga] or planes[gb]):
                continue
            g = gb if planes[ga] else ga
            f = np.zeros(6); mj.mj_contactForce(plant, d, ci, f)
            n = max(0., float((c.frame.reshape(3, 3).T @ f[:3])[2]) * (1 if planes[ga] else -1))
            leg = next((L for L, gi in pads.items() if gi == g), None)
            if leg is None: nonfoot += n
            else: loads[LEGS.index(leg)] += n
        return loads, nonfoot

    def adhesion():
        d.xfrc_applied[:] = 0.
        loads, _ = measure()
        jac = np.zeros((3, plant.nv))
        for L in LEGS:
            gi = pads[L]
            mj.mj_jac(plant, d, jac, None, np.asarray(d.geom_xpos[gi]), int(plant.geom_bodyid[gi]))
            if loads[LEGS.index(L)] <= STANCE or float((jac @ d.qvel)[2]) > 0.:
                continue
            body = int(plant.geom_bodyid[gi]); f = np.array([0., 0., -A_ADH])
            d.xfrc_applied[body, 0:3] += f
            d.xfrc_applied[body, 3:6] += np.cross(d.geom_xpos[gi] - d.xpos[body], f)

    def net_target(t):
        idx = np.clip(t - (K - 1) + lags, 0, K - 1)
        x = buf[idx].reshape(-1)
        h0 = np.tanh(x @ base['w0'] + base['b0']); h1 = np.tanh(h0 @ base['w1'] + base['b1'])
        return base['center'] + base['scale'] * (h1 @ base['w2'] + base['b2'])

    def policy_mean(obs):
        h0 = np.tanh(obs @ net['actor_w0'] + net['actor_b0']); h1 = np.tanh(h0 @ net['actor_w1'] + net['actor_b1'])
        return np.tanh(h1 @ net['actor_w2'] + net['actor_b2'])

    tgt = None; qpos_bank = []; qvel_bank = []; stats = []; sup_bank = []; com_bank = []
    for r in range(N_ROLLOUTS):
      a = int(off[r]); b = int(off[r + 1])
      seg_r = tar[a:b]
      while seg_r.shape[0] < 200:
          seg_r = np.vstack([seg_r, tar[int(off[r + 2 + len(qpos_bank) % 3]):int(off[r + 3 + len(qpos_bank) % 3])]])
      mj.mj_resetDataKeyframe(plant, d, 0)
      d.qpos[plant.jnt_qposadr[jids]] = np.asarray(tfm.STAND_Q, float); d.qpos[2] = tfm.STAND_ROOT_Z
      mj.mj_forward(plant, d)
      actuator.reset(d)
      buf = np.repeat(q0[None], K, axis=0)
      for t in range(ASSIST + RELEASE):
        if t % BASE_EVERY == 0:
            tgt = net_target(t); buf = np.vstack([buf[1:], tgt[None]])
        if t < ASSIST:
            rate = np.clip((seg_r[min(t // 5, len(seg_r) - 1)] - actuator.target_rad) / PDT, -CAP, CAP)
        else:
            loads_now, _ = measure()
            proprio = sens.features(shim, d, tgt, loads_now)
            obs = np.concatenate([proprio, buf[lags].reshape(-1)])
            rate = (tgt - actuator.target_rad) / PDT + policy_mean(obs) * C / PDT
        actuator.update(np.clip(rate, -CAP, CAP), .001)
        for k in range(80):
            if k % 40 == 0:
                adhesion()
            d.ctrl[:] = actuator.torque(d); mj.mj_step(plant, d); mj.mj_forward(plant, d)
        if t >= ASSIST and (t - ASSIST) % 10 == 0:
            qpos_bank.append(d.qpos.copy()); qvel_bank.append(d.qvel.copy())
            loads, nonfoot = measure()
            sup = float(loads.sum() / WEIGHT); com = float(d.subtree_com[th][2])
            sup_bank.append(sup); com_bank.append(com)
            stats.append(dict(tick=t, com=com, support=sup, nonfoot=float(nonfoot), rollout=r))
    np.savez_compressed(BANK, qpos=np.array(qpos_bank), qvel=np.array(qvel_bank),
                        support=np.array(sup_bank), com=np.array(com_bank))
    (BANK.with_suffix('.json')).write_text(json.dumps(dict(
        source_model=str(MODEL), source_model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        pad_xml=str(PAD_XML), pad_xml_sha256=hashlib.sha256(PAD_XML.read_bytes()).hexdigest(),
        states=len(qpos_bank), sampled_every_ticks=10, assist_ticks=ASSIST, release_ticks=RELEASE,
        com_range=[min(s['com'] for s in stats), max(s['com'] for s in stats)],
        support_range=[min(s['support'] for s in stats), max(s['support'] for s in stats)],
        nonfoot_max=max(s['nonfoot'] for s in stats),
        rollouts=N_ROLLOUTS,
        bank_support_range=[min(sup_bank), max(sup_bank)], bank_com_range=[min(com_bank), max(com_bank)],
        fraction_support_below_090=float(np.mean(np.array(sup_bank) < .90)),
        note='states recorded from our own simulation during the release phase; no external generator, no replay of commands'), indent=2) + '\n')
    print(json.dumps(dict(states=len(qpos_bank), com_min=min(s['com'] for s in stats),
                          support_min=min(s['support'] for s in stats), nonfoot_max=max(s['nonfoot'] for s in stats),
                          support_below_090=float(np.mean(np.array(sup_bank) < .90)))))


if __name__ == '__main__':
    main()
