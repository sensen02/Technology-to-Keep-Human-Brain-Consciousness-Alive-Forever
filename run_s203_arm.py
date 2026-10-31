"""S203 one arm with full instrumentation.

Dumps per command interval: ctrl (nu), act (nu), qpos (nq), per-leg injected current
(min/max/sum over that leg's channel row set), spike count, per-leg tactile quantities.
Arms: A no tactile | B raw force | C force/weight_per_leg | D sign-flipped force.
Optional initial-value perturbation on one joint (delta added to qpos AND to the encoder
reference pose, so every sensory deviation signal is exactly 0 at t=0).

No CPG, no PD, no per-step pose correction: inside the loop only d.ctrl[:] is written.
"""
from pathlib import Path
import json, importlib, sys, hashlib, time, argparse
import numpy as np

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = Path('/home/sensen/Desktop/engine修复验收')
S154 = DESK / 's154-activation-rate/20261024T215751775489Z'
DT = 2.5e-5
CMD_EVERY = 200
NT_MS = 0.5
NT_STEPS = CMD_EVERY // 20
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(out, arm, seconds, delta, delta_joint, dump_current):
    out = Path(out)
    assert not out.exists(), out
    out.mkdir(parents=True)
    steps = int(round(seconds / DT))
    ncmd = steps // CMD_EVERY
    import mujoco
    sys.path.insert(0, str(ROOT))
    L = importlib.import_module('run_flybody_loop')
    from engine.embodied import adapters as ad
    from engine.embodied.motor_output import MotorOutput, MotorOutputConfig, build_motor_manifest
    from engine.embodied.sensory_channels import (ChannelSensoryEncoder, ChannelConfig,
                                                  ChannelQuantities, channel_index_map,
                                                  LEGS as SENS_LEGS)
    assert tuple(SENS_LEGS) == tuple(l.lower() for l in LEGS), (SENS_LEGS, LEGS)

    body = L.build_body('canonical')
    model = body['model']
    model.opt.timestep = DT
    WPL = float(body['weight_per_leg'])
    floor = int(body['floor'])

    z = np.load(S154 / 'init80ms/TRAJECTORY.npz')
    q0 = z['qpos'][0].copy()
    act0 = z['act'][0].copy()
    dq_adr = None
    if delta:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, delta_joint)
        assert jid >= 0, ('delta joint not found', delta_joint)
        dq_adr = int(model.jnt_qposadr[jid])
        q0[dq_adr] += float(delta)
    # reference pose uses the SAME q0, so every sensory deviation is exactly 0 at t=0
    bname = lambda b: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(b)) or ''
    gname = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or ''

    tars = {}
    for g in range(model.ngeom):
        bn = bname(int(model.geom_bodyid[g]))
        for leg in LEGS:
            if bn.upper().startswith(leg) and 'TARS' in bn.upper():
                tars[g] = leg
                break
    legs_with = sorted({v for v in tars.values()})
    assert legs_with == sorted(LEGS), ('tarsus geom missing for some leg', legs_with)
    per_leg_geoms = {l: sorted(g for g, v in tars.items() if v == l) for l in LEGS}
    assert all(len(v) == 5 for v in per_leg_geoms.values()), per_leg_geoms

    joints = {}
    for leg in LEGS:
        ids = [j for j in range(model.njnt)
               if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or '').lower().startswith('joint_' + leg.lower())]
        tib = [j for j in ids if 'tibia' in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or '').lower()]
        assert len(ids) == 7 and len(tib) == 1, (leg, len(ids), len(tib))
        joints[leg] = dict(q=[int(model.jnt_qposadr[j]) for j in ids],
                           tib_q=int(model.jnt_qposadr[tib[0]]), tib_v=int(model.jnt_dofadr[tib[0]]))

    tier_cfg = ad.TierConfig(background_mean_nA=0, background_sd_nA=0)
    built = ad.build_neural_tier(tier_cfg)
    ledger = ad.OwnershipLedger()
    tier = ad.NeuralTier(built, tier_cfg, ledger)
    n = int(built['n'])
    groups, _ = channel_index_map(n, built['tier']['root_ids'])
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    pools = json.loads((ROOT / 'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json').read_text())
    manifest = build_motor_manifest(pools, built['tier']['root_ids'], names)
    dec = MotorOutput(manifest, MotorOutputConfig())
    tier.reset()
    enc = ChannelSensoryEncoder(groups, n, ChannelConfig())
    enc.set_reference_pose(q0)
    for leg in LEGS:
        j = joints[leg]
        enc.assign(leg, j['tib_q'], j['tib_q'], j['q'])

    # ---- in-run guard: every leg must receive nonzero current when fed a nonzero quantity ----
    # NOTE: the probe keys MUST be lowercase; uppercase keys are silently ignored by encode()
    # (the defect that cost two earlier rounds). The assertion below is what catches that.
    probe = {leg.lower(): ChannelQuantities(tibia_joint_dev_rad=0.03, tibia_joint_vel_rad_s=0.0,
                                            multi_joint_dev_rad=0.0, contact_force_norm=0.0, foot_contact=0.0)
             for leg in LEGS}
    probe_cur, _ = enc.encode(probe)
    leg_rows = {leg: np.concatenate([np.asarray(groups[leg.lower()][ch], int)
                                     for ch in groups[leg.lower()]
                                     if len(groups[leg.lower()][ch])]) for leg in LEGS}
    leg_nonzero = {leg: float(np.max(np.abs(probe_cur[leg_rows[leg]]))) for leg in LEGS}
    assert all(v > 0 for v in leg_nonzero.values()), ('a leg got zero current', leg_nonzero)

    d = mujoco.MjData(model)
    d.qpos[:] = q0
    d.qvel[:] = 0.0
    d.act[:] = act0
    d.ctrl[:] = 0.0
    mujoco.mj_forward(model, d)

    divisor = 1.0 if arm in ('B', 'D') else (WPL if arm == 'C' else 1.0)  # arm T: force stays 0
    sgn = -1.0 if arm == 'D' else 1.0

    unmatched = {}
    def observe(dv):
        force = {l: 0.0 for l in LEGS}
        touch = {l: 0.0 for l in LEGS}
        for ci in range(dv.ncon):
            c = dv.contact[ci]
            if floor not in (int(c.geom1), int(c.geom2)):
                continue
            other = int(c.geom2) if int(c.geom1) == floor else int(c.geom1)
            leg = tars.get(other)
            if leg is None:
                nm = gname(other)
                unmatched[nm] = unmatched.get(nm, 0) + 1
                continue
            f = np.zeros(6)
            mujoco.mj_contactForce(model, dv, ci, f)
            mag = float(np.linalg.norm(f[:3]))
            if arm == 'T':
                touch[leg] = 1.0        # touch-only control: bristle channel on, force channel exactly 0
            elif arm != 'A':
                force[leg] += sgn * mag / divisor
                touch[leg] = 1.0
        return {leg: ChannelQuantities(
            tibia_joint_dev_rad=float(dv.qpos[joints[leg]['tib_q']] - q0[joints[leg]['tib_q']]),
            tibia_joint_vel_rad_s=float(dv.qvel[joints[leg]['tib_v']]),
            multi_joint_dev_rad=float(np.mean(dv.qpos[joints[leg]['q']] - q0[joints[leg]['q']])),
            contact_force_norm=force[leg], foot_contact=touch[leg]) for leg in LEGS}

    tr = dict(t=[], rootz=[], rootxy=[], nfeet=[], bodyfloor=[], ctrl_max=[], act_max=[],
              spikes=[], force=[])
    ctrl_hist = np.zeros((ncmd, model.nu))
    act_hist = np.zeros((ncmd, model.nu))
    qpos_hist = np.zeros((ncmd, model.nq))
    cur_hist = np.zeros((ncmd, n)) if dump_current else None
    leg_cur = np.zeros((ncmd, len(LEGS), 3))   # min, max, sum over that leg's rows
    pool_hist = None
    spike_total = 0
    cur_abs_max = 0.0
    t0 = time.time()
    for k in range(ncmd):
        q = observe(d)
        current, _rec = enc.encode({leg.lower(): q[leg] for leg in LEGS})
        cur_abs_max = max(cur_abs_max, float(np.max(np.abs(current))))
        if dump_current:
            cur_hist[k] = current
        for i, leg in enumerate(LEGS):
            v = current[leg_rows[leg]]
            leg_cur[k, i] = (float(v.min()), float(v.max()), float(v.sum()))
        sp = tier.advance(NT_STEPS, current)
        spike_total += int(sp.sum())
        t_s = k * CMD_EVERY * DT
        t_e = (k + 1) * CMD_EVERY * DT
        pend, drec = dec.consume(sp, spike_dt_s=NT_MS / 1000.0, start_s=t_s, end_s=t_e)
        if pool_hist is None:
            pool_hist = np.zeros((ncmd, len(drec['pool_spike_counts'])))
        pool_hist[k] = drec['pool_spike_counts']
        ctrl = np.asarray(pend, float).copy()
        d.ctrl[:] = ctrl
        ctrl_hist[k] = ctrl
        for j in range(CMD_EVERY):
            mujoco.mj_step(model, d)
        act_hist[k] = d.act.copy()
        qpos_hist[k] = d.qpos.copy()
        feet, bf = set(), set()
        for c in d.contact:
            if c.efc_address < 0:
                continue
            if floor not in (int(c.geom1), int(c.geom2)):
                continue
            other = int(c.geom2) if int(c.geom1) == floor else int(c.geom1)
            leg = tars.get(other)
            if leg is not None:
                feet.add(leg)
            else:
                bf.add(gname(other))
        tr['t'].append(float(d.time))
        tr['rootz'].append(float(d.qpos[2]))
        tr['rootxy'].append([float(d.qpos[0]), float(d.qpos[1])])
        tr['nfeet'].append(len(feet))
        tr['bodyfloor'].append(len(bf))
        tr['ctrl_max'].append(float(np.max(np.abs(ctrl))))
        tr['act_max'].append(float(np.max(np.abs(d.act))))
        tr['spikes'].append(spike_total)
        tr['force'].append([float(q[leg].contact_force_norm) for leg in LEGS])

    tt = np.asarray(tr['t'])
    rz = np.asarray(tr['rootz'])
    last = rz[tt >= tt[-1] - 1.0]
    pre = tt <= 0.020 + 1e-12
    post = ~pre
    def frac_zero(h, mask):
        sub = h[mask]
        return float(np.mean(sub <= 1e-12)) if sub.size else None
    def frac_one(h, mask):
        sub = h[mask]
        return float(np.mean(sub >= 1.0 - 1e-12)) if sub.size else None
    summary = dict(arm=arm, steps=steps, ncmd=ncmd, duration_s=steps * DT, dt=DT,
                   delta=delta, delta_joint=delta_joint if delta else None,
                   delta_qpos_adr=dq_adr, weight_per_leg=WPL, divisor=divisor, sign=sgn,
                   tactile_mapping=per_leg_geoms, n_tarsus_geoms=len(tars),
                   leg_nonzero_current_probe=leg_nonzero,
                   unmatched_floor_geoms=dict(sorted(unmatched.items(), key=lambda kv: -kv[1])),
                   first_body_floor_time=next((tr['t'][i] for i, v in enumerate(tr['bodyfloor']) if v > 0), None),
                   mean_bodyfloor_after_1s=float(np.mean(np.asarray(tr['bodyfloor'])[tt > 1.0])),
                   rootz_last1s_mean=float(np.mean(last)), rootz_last1s_std=float(np.std(last)),
                   final_rootz=float(d.qpos[2]),
                   final_xy=[float(d.qpos[0]), float(d.qpos[1])],
                   horizontal_disp=float(np.hypot(d.qpos[0] - q0[0], d.qpos[1] - q0[1])),
                   min_nfeet=int(min(tr['nfeet'])), total_spikes=int(spike_total),
                   current_abs_max_nA=float(cur_abs_max),
                   ctrl_max=float(ctrl_hist.max()), act_max=float(act_hist.max()),
                   ctrl_zero_frac_pre=frac_zero(ctrl_hist, pre), ctrl_one_frac_pre=frac_one(ctrl_hist, pre),
                   ctrl_zero_frac_post=frac_zero(ctrl_hist, post), ctrl_one_frac_post=frac_one(ctrl_hist, post),
                   act_one_frac_pre=frac_one(act_hist, pre), act_one_frac_post=frac_one(act_hist, post),
                   n_pre_commands=int(pre.sum()), n_post_commands=int(post.sum()),
                   actuator_names=names, joint_names=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)],
                   wall_s=float(time.time() - t0),
                   note='mj_step authorised; only d.ctrl written; no training; no XML change')
    save = dict(t=np.asarray(tr['t']), rootz=np.asarray(tr['rootz']), rootxy=np.asarray(tr['rootxy']),
                nfeet=np.asarray(tr['nfeet']), bodyfloor=np.asarray(tr['bodyfloor']),
                ctrl_max=np.asarray(tr['ctrl_max']), act_max=np.asarray(tr['act_max']),
                spikes=np.asarray(tr['spikes']), force=np.asarray(tr['force']),
                ctrl=ctrl_hist, act=act_hist, qpos=qpos_hist, leg_current=leg_cur,
                pool_spikes=pool_hist)
    if dump_current:
        save['current'] = cur_hist
    np.savez_compressed(out / 'TRACE.npz', **save)
    (out / 'SUMMARY.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ('arm', 'delta', 'first_body_floor_time', 'rootz_last1s_mean',
                                             'rootz_last1s_std', 'horizontal_disp', 'ctrl_max', 'act_max',
                                             'ctrl_zero_frac_pre', 'ctrl_one_frac_pre', 'total_spikes', 'wall_s')}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--arm', required=True, choices=list('ABCDT'))
    ap.add_argument('--seconds', type=float, default=5.0)
    ap.add_argument('--delta', type=float, default=0.0)
    ap.add_argument('--delta-joint', default='joint_LFCoxa_pitch')
    ap.add_argument('--dump-current', action='store_true')
    a = ap.parse_args()
    main(a.output, a.arm, a.seconds, a.delta, a.delta_joint, a.dump_current)
