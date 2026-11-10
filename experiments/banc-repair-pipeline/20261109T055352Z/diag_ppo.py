"""Why did every previous PPO update report kl_last ~ 0?

The KL early-stop compares each minibatch against the ROLLOUT log-probs. This script runs one
full PPO iteration with instrumentation and prints the KL sequence, how many minibatches
actually ran per epoch, and how much the parameters moved. It does not change the trainer.
"""
from pathlib import Path
import json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc
import train_single_leg as T

ROLL = int(__import__('os').environ.get('ROLL', '512'))


def collect(core, net, rng):
    O, A, LOGP, V, R, TERM, TRUNC = [], [], [], [], [], [], []
    steps = 0
    while steps < ROLL:
        leg = int(rng.integers(0, rc.NL)); prompt = T.onehot(leg)
        core.reset()
        st = dict(swing=False, start_x=0., max_lift=0., best_adv=0., placed=False, prev_x=0.)
        for k in range(T.T_EP):
            obs = core.obs(prompt)
            a, lat, logp, value, reward, term, info = T.episode_step(core, net, rng, leg, k, obs, st)
            O.append(obs); A.append(lat); LOGP.append(logp); V.append(value); R.append(reward)
            TERM.append(term); TRUNC.append((k == T.T_EP - 1) and not term)
            steps += 1
            if term or k == T.T_EP - 1:
                break
            if steps >= ROLL:
                TRUNC[-1] = True
                break
    return (np.asarray(O), np.asarray(A), np.asarray(LOGP), np.asarray(V), np.asarray(R),
            np.asarray(TERM), np.asarray(TRUNC))


def main():
    rng = np.random.default_rng(20261109)
    core = rc.Core(); net = T.Net(rng)
    T.warm_start(core, net, rng, ticks=T.WS_TICKS, steps=T.WS_STEPS)
    before = {k: v.copy() for k, v in net.p.items()}
    O, A, LOGP, V, R, TERM, TRUNC = collect(core, net, rng)
    adv = np.zeros_like(R); last = 0.
    for i in range(len(R) - 1, -1, -1):
        nxt = V[i + 1] if (i + 1 < len(V) and not TERM[i] and not TRUNC[i]) else 0.
        if TRUNC[i]:
            nxt = V[i]
        td = R[i] + T.CFG['gamma'] * nxt - V[i]
        last = td + T.CFG['gamma'] * T.CFG['gae'] * (0. if (TERM[i] or TRUNC[i]) else 1.) * last
        adv[i] = last
    ret = adv + V
    adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
    epochs = []
    opt = T.optimize(net, O, A, LOGP, adv_n, ret, rng, T.CFG['lr'])
    epochs.append(dict(optimizer=opt))
    delta = {k: float(np.abs(net.p[k] - before[k]).max()) for k in net.p}
    kl_after = float(np.mean(LOGP - np.sum(
        -0.5 * ((A - net.forward(O)[2]) / net.std()) ** 2 - np.log(net.std()) - 0.5 * np.log(2 * np.pi),
        axis=1)))
    out = dict(rollout=int(len(R)), epochs=epochs,
               total_minibatch_updates=int(opt['minibatch_updates']),
               nominal_updates_per_iteration=int(opt['nominal_updates']),
               param_max_abs_delta=delta, mean_kl_after_iteration=kl_after,
               target_kl=T.CFG['target_kl'], final_lr=opt['lr'],
               note='kl after the iteration is against the rollout policy, so it is expected to '
                    'exceed target_kl once many controlled steps are taken; per-step KL is in epochs')
    (HERE / 'PPO_KL_DIAGNOSIS_FIXED.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: out[k] for k in ('rollout', 'total_minibatch_updates',
                                          'nominal_updates_per_iteration', 'mean_kl_after_iteration',
                                          'final_lr')}, indent=2))
    print(json.dumps(opt, indent=2))
    print(json.dumps(delta, indent=2))


if __name__ == '__main__':
    main()
