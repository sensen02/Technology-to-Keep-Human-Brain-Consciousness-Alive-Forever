"""S204 calibration arm: same frozen start / tier / physics / manifest, only rate_for_full_control_hz differs.

Stores per command window: ctrl, act, qpos, pool spike counts AND the raw per-sample spike matrix
packed with np.packbits, so the decoder ctrl can be recomputed bitwise afterwards.
No tactile channel (same configuration as the S203 no-tactile arm under audit).
"""
from pathlib import Path
import json, importlib, sys, hashlib, time, argparse
import numpy as np

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
S154 = Path('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z')
DT = 2.5e-5
CMD_EVERY = 200
NT_MS = 0.5
NT_STEPS = CMD_EVERY // 20
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')


def main(out, seconds, rate_full):
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
                                                  ChannelQuantities, channel_index_map, LEGS as SENS_LEGS)
    assert tuple(SENS_LEGS) == tuple(l.lower() for l in LEGS)

    body = L.build_body('canonical')
    model = body['model']
    model.opt.timestep = DT
    floor = int(body['floor'])
    z = np.load(S154 / 'init80ms/TRAJECTORY.npz')
    q0 = z['qpos'][0].copy()
    act0 = z['act'][0].copy()
    bname = lambda b: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(b)) or ''

    tars = {}
    for g in range(model.ngeom):
        bn = bname(int(model.geom_bodyid[g]))
        for leg in LEGS:
            if bn.upper().startswith(leg) and 'TARS' in bn.upper():
                tars[g] = leg
                break
    assert sorted({v for v in tars.values()}) == sorted(LEGS)

    joints = {}
    for leg in LEGS:
        ids = [j for j in range(model.njnt)
               if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or '').lower().startswith('joint_' + leg.lower())]
        tib = [j for j in ids if 'tibia' in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or '').lower()]
        assert len(ids) == 7 and len(tib) == 1
        joints[leg] = dict(q=[int(model.jnt_qposadr[j]) for j in ids],
                           tib_q=int(model.jnt_qposadr[tib[0]]), tib_v=int(model.jnt_dofadr[tib[0]]))

    tier_cfg = ad.TierConfig(background_mean_nA=0, background_sd_nA=0)
    built = ad.build_neural_tier(tier_cfg)
    tier = ad.NeuralTier(built, tier_cfg, ad.OwnershipLedger())
    n = int(built['n'])
    assert n == 12000
    groups, _ = channel_index_map(n, built['tier']['root_ids'])
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    pools = json.loads((ROOT / 'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json').read_text())
    manifest = build_motor_manifest(pools, built['tier']['root_ids'], names)
    import dataclasses
    dcfg = dataclasses.replace(MotorOutputConfig(), rate_for_full_control_hz=float(rate_full))
    dec = MotorOutput(manifest, dcfg)
    tier.reset()
    enc = ChannelSensoryEncoder(groups, n, ChannelConfig())
    enc.set_reference_pose(q0)
    for leg in LEGS:
        j = joints[leg]
        enc.assign(leg, j['tib_q'], j['tib_q'], j['q'])
    # in-run guard: every leg must receive nonzero current when fed a nonzero quantity
    probe = {leg.lower(): ChannelQuantities(tibia_joint_dev_rad=0.03, tibia_joint_vel_rad_s=0.0,
                                            multi_joint_dev_rad=0.0, contact_force_norm=0.0, foot_contact=0.0)
             for leg in LEGS}
    pcur, _ = enc.encode(probe)
    leg_rows = {leg: np.concatenate([np.asarray(groups[leg.lower()][ch], int) for ch in groups[leg.lower()]
                                     if len(groups[leg.lower()][ch])]) for leg in LEGS}
    assert all(float(np.max(np.abs(pcur[leg_rows[leg]]))) > 0 for leg in LEGS)

    d = mujoco.MjData(model)
    d.qpos[:] = q0
    d.qvel[:] = 0.0
    d.act[:] = act0
    d.ctrl[:] = 0.0
    mujoco.mj_forward(model, d)

    def observe(dv):
        res = {}
        for leg in LEGS:
            j = joints[leg]
            res[leg.lower()] = ChannelQuantities(
                tibia_joint_dev_rad=float(dv.qpos[j['tib_q']] - q0[j['tib_q']]),
                tibia_joint_vel_rad_s=float(dv.qvel[j['tib_v']]),
                multi_joint_dev_rad=float(np.mean(dv.qpos[j['q']] - q0[j['q']])),
                contact_force_norm=0.0, foot_contact=0.0)   # no tactile: the S203 arm-A configuration
        return res

    ctrl_hist = np.zeros((ncmd, model.nu))
    act_hist = np.zeros((ncmd, model.nu))
    qpos_hist = np.zeros((ncmd, model.nq))
    pool_hist = np.zeros((ncmd, model.nu))
    nbits = NT_STEPS * n
    nbytes = (nbits + 7) // 8
    spikes_packed = np.zeros((ncmd, nbytes), np.uint8)
    for k in range(ncmd):
        current, _ = enc.encode(observe(d))
        sp = tier.advance(NT_STEPS, current)
        spikes_packed[k] = np.packbits(sp.reshape(-1))
        t_s = k * CMD_EVERY * DT
        t_e = (k + 1) * CMD_EVERY * DT
        pend, drec = dec.consume(sp, spike_dt_s=NT_MS / 1000.0, start_s=t_s, end_s=t_e)
        ctrl = np.asarray(pend, float).copy()
        d.ctrl[:] = ctrl
        ctrl_hist[k] = ctrl
        pool_hist[k] = drec['pool_spike_counts']
        for _j in range(CMD_EVERY):
            mujoco.mj_step(model, d)
        act_hist[k] = d.act.copy()
        qpos_hist[k] = d.qpos.copy()

    np.savez_compressed(out / 'TRACE.npz', ctrl=ctrl_hist, act=act_hist, qpos=qpos_hist,
                        pool_spikes=pool_hist, spikes_packed=spikes_packed, n_neurons=n)
    # per-cell rates from the packed spikes, per command window
    bits = np.unpackbits(spikes_packed, axis=1)[:, :nbits].reshape(ncmd, NT_STEPS, n)
    per_cell_counts = bits.sum(axis=(0, 1))
    (out / 'SUMMARY.json').write_text(json.dumps(dict(
        arm=out.name, steps=steps, ncmd=ncmd, duration_s=steps * DT, dt=DT,
        rate_for_full_control_hz=float(rate_full), filter_tau_s=float(dcfg.filter_tau_s),
        max_control=float(dcfg.max_control), spikes_per_window_total=bits.sum(axis=2).sum(axis=1).tolist(),
        total_spikes=int(bits.sum()), cells_that_fired=int((per_cell_counts > 0).sum()),
        wall_s=float(time.time()) if False else None), indent=2))
    print(json.dumps(dict(arm=out.name, rate_full=float(rate_full), ncmd=ncmd,
                          spikes=[int(x) for x in bits.sum(axis=2).sum(axis=1)],
                          cells_fired=int((per_cell_counts > 0).sum()))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seconds', type=float, default=0.05)
    ap.add_argument('--rate-full', type=float, required=True)
    a = ap.parse_args()
    main(a.output, a.seconds, a.rate_full)
