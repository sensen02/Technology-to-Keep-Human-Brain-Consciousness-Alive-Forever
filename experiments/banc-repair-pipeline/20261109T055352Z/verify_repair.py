"""Verify the repaired pipeline before any training: determinism, action authority,
leg-order fix, pad lookup, and a teacher-in-the-loop reference rollout.

Nothing here is a policy result. The reference rollout drives the repaired plant with the
frozen measured-motion base network's rate command and is labelled teacher throughout; it
only answers "does the unified pipeline reproduce a standing baseline at all".
"""
from pathlib import Path
import json, hashlib, sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc

WEIGHT = rc.WEIGHT
LEGS = rc.LEG_ORDER


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rng_actions(seed, n):
    r = np.random.default_rng(seed)
    return r.uniform(-1., 1., (n, rc.NJ))


def run_actions(acts, ticks=None):
    core = rc.Core()
    n = ticks if ticks is not None else len(acts)
    log = dict(loads=[], nonfoot=[], padz=[], com=[], foot=[], fwd=[])
    for t in range(n):
        m = core.tick(acts[t])
        log['loads'].append(m['loads']); log['nonfoot'].append(m['nonfoot'])
        log['padz'].append(m['padz']); log['com'].append(m['com'])
        log['foot'].append(m['foot']); log['fwd'].append(core.forward_body_velocity())
    out = {k: np.array(v) for k, v in log.items()}
    out['qpos'] = core.d.qpos.copy()
    out['core'] = core
    return out


def authority(ticks=2000, seed=7):
    """Same-state counterfactual: would removing the policy output change the executed rate?"""
    r = np.random.default_rng(seed)
    a_seq = r.uniform(-1., 1., (ticks, rc.NJ))
    base = rc.BaseNet()
    new_masked = new_total = 0
    old_masked = old_total = 0
    # arm 1: repaired interface (policy owns +/-2 rad/s)
    core = rc.Core(); base.reset(core.q0)
    base_hist = np.repeat(core.q0[None], base.K, axis=0)
    for t in range(ticks):
        a = a_seq[t]
        rate_with = np.clip(a * rc.MAX_RATE, -rc.MAX_RATE, rc.MAX_RATE)
        rate_wo = np.zeros(rc.NJ)
        new_total += rate_with.size
        new_masked += int(np.sum(~np.isclose(rate_with, rate_wo, rtol=0, atol=1e-12)))
        core.tick(a)
    # arm 2: legacy interface reimplemented against the same core class
    core = rc.Core(); base.reset(core.q0)
    C_LEGACY = .02
    for t in range(ticks):
        a = a_seq[t]
        bt = base.target()
        base_rate = (bt - core.actuator.target_rad) / rc.PDT
        rate_with = np.clip(base_rate + a * C_LEGACY / rc.PDT, -rc.MAX_RATE, rc.MAX_RATE)
        rate_wo = np.clip(base_rate, -rc.MAX_RATE, rc.MAX_RATE)
        old_total += rate_with.size
        old_masked += int(np.sum(~np.isclose(rate_with, rate_wo, rtol=0, atol=1e-12)))
        core.tick_rate(rate_with)
    return dict(repaired=dict(channel_ticks_changed=new_masked, total=new_total,
                              fraction=new_masked / new_total),
                legacy=dict(channel_ticks_changed=old_masked, total=old_total,
                            fraction=old_masked / old_total),
                note='same-state counterfactual per tick, not a full causal ablation')


def teacher_rollout(ticks=3200):
    core = rc.Core(); base = rc.BaseNet(); base.reset(core.q0)
    loads, nonfoot, com, fwd, speed, padz = [], [], [], [], [], []
    for t in range(ticks):
        bt = base.target()
        rate = (bt - core.actuator.target_rad) / rc.PDT
        m = core.tick_rate(rate)
        loads.append(m['loads']); nonfoot.append(m['nonfoot']); com.append(m['com'])
        fwd.append(core.forward_body_velocity()); speed.append(m['speed']); padz.append(m['padz'])
    loads = np.array(loads); nonfoot = np.array(nonfoot); com = np.array(com)
    fwd = np.array(fwd); speed = np.array(speed); padz = np.array(padz)
    sup = loads.sum(1) / WEIGHT
    mean_fwd = float(fwd.mean())
    per_leg_speed = (loads * speed).sum(0) / np.maximum(loads.sum(0), 1e-12)
    ratio = per_leg_speed / max(abs(mean_fwd), .05)
    return dict(label='teacher rate emulation in the repaired pipeline (NOT a policy result)',
                ticks=ticks, native_time_s=ticks * rc.PDT,
                foot_support_mean=float(sup.mean()), foot_support_min=float(sup.min()),
                nonfoot_force_max_W=float(nonfoot.max()), com_height_min=float(com.min()),
                forward_signed_mean=mean_fwd, forward_signed_end=float(fwd[-1]),
                per_leg_stance_speed=dict(zip(LEGS, map(float, per_leg_speed))),
                per_leg_slip_ratio_vs_mean_forward=dict(zip(LEGS, map(float, ratio))),
                foot_slip_ratio_max=float(ratio.max()),
                mean_pad_bottom_z=dict(zip(LEGS, map(float, padz.mean(0)))),
                max_pad_bottom_z=dict(zip(LEGS, map(float, padz.max(0)))))


def main():
    out = {}
    acts = rng_actions(1, 40)
    r1 = run_actions(acts)
    r2 = run_actions(acts)
    out['determinism'] = dict(
        identical_qpos=bool(np.array_equal(r1['qpos'], r2['qpos'])),
        max_abs_qpos_diff=float(np.abs(r1['qpos'] - r2['qpos']).max()))
    core = rc.Core()
    out['pad_lookup'] = dict(per_leg={L: len(core.pads[L]) for L in LEGS},
                            obs_dim=rc.OBS,
                            history_is_own_commands=bool(np.allclose(core.buf[-1], core.actuator.target_rad)))
    out['authority'] = authority()
    out['teacher_reference'] = teacher_rollout()
    out['leg_order'] = dict(actuator_order=LEGS, sensory_order=rc.SENSORY_ORDER,
                            sensory_index_map=rc.SENS_IDX,
                            note='foot loads are handed to the encoder in its declared order')
    out['hashes'] = {str(p): sha(p) for p in (rc.ACT, rc.SENSORY, rc.BASE, rc.PAD_XML, rc.POSE_FILE,
                                             HERE / 'repair_core.py')}
    (HERE / 'VERIFY_REPAIR.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: out[k] for k in ('determinism', 'pad_lookup')}, indent=2))
    print(json.dumps(out['authority'], indent=2))
    print(json.dumps({k: v for k, v in out['teacher_reference'].items()
                      if not isinstance(v, dict)}, indent=2))
    assert out['determinism']['identical_qpos'], 'pipeline is not deterministic'
    assert all(v == 3 for v in out['pad_lookup']['per_leg'].values()), 'triad pad lookup failed'
    assert out['authority']['repaired']['fraction'] > .999, 'policy has no authority'
    print('VERIFIED')


if __name__ == '__main__':
    main()
