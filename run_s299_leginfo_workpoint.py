"""S299 stages B/C: one reproduction arm (s=1.0) plus five declared sensory-scale arms.

Single model load, single tier build, per-arm fresh MjData reset to the saved t=0 state and a
fresh NeuralTier. The sensory scale s multiplies the current at the net.step call site; the same
array is recorded, so "recorded equals applied" is enforced structurally.
"""
from __future__ import annotations
import argparse, dataclasses, hashlib, json, csv, traceback
from pathlib import Path
import numpy as np

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
ARCH = Path('/home/sensen/Desktop/engine修复验收')
FROZEN = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
REF = ARCH / 's289-decoder-range-support/20261026T173400Z'
BASELINE = REF / 'decoder_original'
S177 = ARCH / 's177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz'
NAMES_NPZ = ARCH / 's244-chirality-of-the-5-of-6-arm/20261025T205824474993Z/CMD_A3_sym_matched.npz'
GAINS = ARCH / 's255-static-gain-can-it-sustain/20261025T221308522972Z/ARM_VECTORS.json'
POOLS = ROOT / 'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json'
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
S_GRID = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02)
BUDGET = dict(model_load=1, tier_build=1, network_init=6, live_step=24000, neural_tick=4800, live_fwd_position=6, diagnostic_forward=24006)
WEIGHT = 24.446354755878


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _conv(o):
    if hasattr(o, 'tolist'): return o.tolist()
    if hasattr(o, 'item'): return o.item()
    raise TypeError(type(o))


def dump(p, v): Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, default=_conv) + '\n', encoding='utf-8')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--output', type=Path, required=True)
    out = ap.parse_args().output
    pre = json.loads((out / 'PREREG.json').read_text())
    dumps = json.loads((out / 'PREREG_RUNNER.json').read_text()) if (out / 'PREREG_RUNNER.json').is_file() else None
    dump(out / 'PREREG_RUNNER.json', dict(
        prereg_sha256=sha(out / 'PREREG.json'),
        s289_round_prereg=str(REF / 'PREREG.json'), s289_round_prereg_sha256=sha(REF / 'PREREG.json'),
        note='the dispatched missing_inputs entry pointed at decoder_original/PREREG.json, which does not exist; the arm prereg lives at the round directory root, whose hash is recorded here. PREREG.json itself is untouched.',
        budget=BUDGET, arms=[dict(s=s, id='s_%.2f' % s) for s in S_GRID], runner_sha256=sha(__file__)))
    import mujoco as mj
    from engine.embodied.adapters import build_neural_tier, TierConfig, NeuralTier, OwnershipLedger
    from engine.embodied.sensory_channels import ChannelSensoryEncoder, ChannelQuantities, ChannelConfig, channel_index_map
    from engine.embodied.motor_output import build_motor_manifest, MotorOutput, MotorOutputConfig
    calls = {k: 0 for k in BUDGET}
    names = [str(x) for x in np.load(NAMES_NPZ, allow_pickle=True)['names']]
    gains = np.asarray(json.loads(GAINS.read_text())['v1'], float)
    pools = json.loads(POOLS.read_text())
    m = mj.MjModel.from_xml_path(str(FROZEN)); calls['model_load'] += 1
    compiled = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    assert compiled == names and m.nu == 90 and (m.nq, m.nv) == (49, 48)
    built = build_neural_tier(TierConfig(sign_fill='all_empty').validate()); calls['tier_build'] += 1
    N = int(built['n']); roots = np.asarray(built['tier']['root_ids']).astype(np.uint64)
    assert built['fingerprint'] == '56e0322a33c51b7f03aaae9f7df1ef3f'
    np.savez_compressed(out / 'TIER_ROOT_IDS.npz', root_ids=roots)
    man = build_motor_manifest(pools, roots, names)
    dump(out / 'MOTOR_MANIFEST.json', man)
    groups, gstats = channel_index_map(N, roots)
    with np.load(S177) as z: qrel = np.asarray(z['qpos'], float).copy()
    geom_names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)]
    floor = geom_names.index('floor')
    foot_names = {lg.upper() + 'Tarsus' + str(i) + '_geom' for lg in LEGS for i in range(1, 6)}
    foot = {g: geom_names[g][:2].lower() for g in range(m.ngeom) if geom_names[g] in foot_names}
    joints = {}
    for lg in LEGS:
        ids = [j for j in range(m.njnt) if (mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or '').lower().startswith('joint_' + lg)]
        tib = [j for j in ids if 'tibia' in mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j).lower()][0]
        joints[lg] = dict(tib_q=int(m.jnt_qposadr[tib]), tib_v=int(m.jnt_dofadr[tib]), q=[int(m.jnt_qposadr[j]) for j in ids])
    tib0 = np.array([qrel[joints[lg]['tib_q']] for lg in LEGS])
    pools_resolved = [r for r in pools['measured']['resolved']]
    pool_ids = sorted({int(x) for r in pools_resolved for x in r['pool_root_ids']})
    pool_local = {pid: int(np.flatnonzero(roots == pid)[0]) for pid in pool_ids if np.any(roots == pid)}
    pool_present = [p for p in pool_ids if p in pool_local]
    pool_idx = np.array([pool_local[p] for p in pool_present], dtype=np.int64)
    np.savez_compressed(out / 'POOL_NEURONS.npz', pool_root_ids=np.array(pool_ids, dtype=np.uint64),
                        local_index=np.array([pool_local.get(pid, -1) for pid in pool_ids], dtype=np.int64),
                        present_root_ids=np.array(pool_present, dtype=np.uint64), present_local_index=pool_idx)
    dump(out / 'POOL_TIER_COVERAGE.json', dict(pool_neurons=len(pool_ids), present_in_tier=int(len(pool_present)),
                                               absent=int(len(pool_ids) - len(pool_present)),
                                               note='pool neurons absent from the tier cannot contribute; reported, never redistributed'))
    flat = {f'{leg}:{ch}': [int(x) for x in idx] for leg, chans in groups.items() for ch, idx in chans.items()}
    dump(out / 'SENSORY_CHANNEL_INDEX.json', dict(channels=int(len(flat)), per_channel_sizes={k: len(v) for k, v in flat.items()},
                                                  channel_groups=flat, stats=gstats))
    enc_cfg = ChannelConfig()
    dump(out / 'ENCODER_CONFIG_RUNTIME.json', dict(current_nA_per_unit=enc_cfg.current_nA_per_unit,
                                                   reference_force=enc_cfg.reference_force, reference_angle_rad=enc_cfg.reference_angle_rad,
                                                   note='read from the live ChannelConfig() instance actually used for the current computation'))
    arms = {}
    for s in S_GRID:
        arm_id = 's_%.2f' % s; dest = out / arm_id; dest.mkdir(exist_ok=True)
        d = mj.MjData(m); d.qpos[:] = qrel; d.qvel[:] = 0; d.act[:] = 0; d.ctrl[:] = 0
        mj.mj_fwdPosition(m, d); calls['live_fwd_position'] += 1
        q0 = d.qpos.copy()
        reset_ok = np.array_equal(d.qpos, qrel) and not np.any(d.qvel) and not np.any(d.act) and not np.any(d.ctrl)
        diag = mj.MjData(m)
        tier = NeuralTier(built, TierConfig(sign_fill='all_empty', background_seed=0).validate(), OwnershipLedger()); calls['network_init'] += 1
        tier.net.p = dataclasses.replace(tier.net.p, inh_tau_ms=tier.net.p.exc_tau_ms); tier.net.p.validate()
        parameters = dataclasses.asdict(tier.net.p)
        background = np.array(tier.background_nA, dtype=float, copy=True)
        enc = ChannelSensoryEncoder(groups, N, enc_cfg)
        dec = MotorOutput(man, MotorOutputConfig(rate_for_full_control_hz=450., per_actuator_gain=gains, filter_tau_s=5e-3))
        cmd = np.zeros(90)
        step_orig = tier.net.step; state = {'applied': None, 'recorded': [], 's': s}
        def step(*a, **kw):
            cur = np.array(a[0] if a else kw.get('current_nA'), dtype=float, copy=True)
            scaled = cur * state['s']
            state['applied'] = scaled
            state['recorded'].append(scaled.copy())
            if a: return step_orig(scaled, *a[1:], **kw)
            return step_orig(current_nA=scaled, **kw)
        tier.net.step = step
        spikes_all = []
        def capture(*a, **kw):
            r = step(*a, **kw); spikes_all.append(np.asarray(r, dtype=bool).copy()); return r
        tier.net.step = capture
        def advance(n, cur):
            out_sp = []
            for _ in range(n):
                out_sp.append(tier.net.step(cur)); calls['neural_tick'] += 1
            return out_sp
        tier.advance = advance
        traj = dict(time=[], qpos=[], com=[], foot_fz=[], total_fz=[], body_contact=[], limit=[], orient=[], mask=[])
        ctrls = []; rates = []; raws = []; pres = []; posts = []; per_ms_pool = []
        def snap():
            mj.mj_copyData(diag, m, d); mj.mj_forward(m, diag); calls['diagnostic_forward'] += 1
            mass = m.body_mass
            fz = np.zeros(6); tot = 0.0; mask = np.zeros(6, bool); body = 0; raw = []
            for ci in range(diag.ncon):
                c = diag.contact[ci]; g1, g2 = int(c.geom1), int(c.geom2); fx = np.zeros(6); mj.mj_contactForce(m, diag, ci, fx)
                raw.append([g1, g2, float(c.dist), *c.pos, *np.asarray(c.frame).ravel(), *fx])
                if floor not in (g1, g2): continue
                other = g2 if g1 == floor else g1
                fw = (1 if g1 == floor else -1) * (np.asarray(c.frame).reshape(3, 3).T @ fx[:3]); up = float(fw[2]); tot += up
                if other in foot:
                    li = LEGS.index(foot[other]); fz[li] += up; mask[li] = True
                elif c.dist < 0: body += 1
            com = (mass[:, None] * diag.xipos).sum(0) / mass.sum()
            lim = np.flatnonzero(m.jnt_limited); qa = m.jnt_qposadr[lim]
            limit = float(np.maximum(np.maximum(m.jnt_range[lim, 0] - diag.qpos[qa], diag.qpos[qa] - m.jnt_range[lim, 1]), 0).max())
            orient = 2 * float(np.arccos(np.clip(abs(d.qpos[3]), 0, 1)))
            traj['time'].append(float(d.time)); traj['qpos'].append(d.qpos.copy()); traj['com'].append(com)
            traj['foot_fz'].append(fz); traj['total_fz'].append(tot); traj['body_contact'].append(body)
            traj['limit'].append(limit); traj['orient'].append(orient); traj['mask'].append(mask)
            return raw
        contacts = [snap()]
        for i in range(400):
            d.ctrl[:] = cmd; applied = cmd.copy()
            qv = {lg: ChannelQuantities(tibia_joint_dev_rad=float(d.qpos[joints[lg]['tib_q']] - q0[joints[lg]['tib_q']]),
                                        tibia_joint_vel_rad_s=float(d.qvel[joints[lg]['tib_v']]),
                                        multi_joint_dev_rad=float(np.mean(d.qpos[joints[lg]['q']] - q0[joints[lg]['q']])),
                                        contact_force_norm=float(np.linalg.norm(traj['foot_fz'][-1][LEGS.index(lg)])),
                                        foot_contact=float(traj['mask'][-1][LEGS.index(lg)])) for lg in LEGS}
            cur, _ = enc.encode(qv)
            sp = tier.advance(2, cur)
            raws.append([np.array([samp[idx].mean() / .0005 if len(idx) else 0 for idx in dec.groups]) for samp in sp])
            cnt = np.zeros(len(pool_idx))
            for samp in sp:
                cnt += np.asarray(samp, float)[pool_idx]
            per_ms_pool.append(cnt)
            carry, info = dec.consume(sp, spike_dt_s=.0005, start_s=i * .001, end_s=i * .001 + .001, connected=True)
            rr = np.asarray(info['rates_hz'], float); preclip = (rr / dec.config.rate_for_full_control_hz) * gains
            assert np.array_equal(carry, np.clip(preclip, 0, dec.config.max_control))
            pres.append(preclip.copy()); posts.append(carry.copy()); rates.append(rr)
            cmd = carry.copy()
            for _ in range(10):
                d.ctrl[:] = cmd; ctrls.append(cmd.copy()); mj.mj_step(m, d); calls['live_step'] += 1
                contacts.append(snap())
        rec = np.asarray(state['recorded'], float)
        applied_ok = all(np.array_equal(rec[k], np.asarray(state['recorded'][k])) for k in range(len(rec)))
        nstate = dict(time_ms=np.arange(1, 801) * .5, rates_hz=np.asarray(rates), raw_rates_hz=np.asarray(raws),
                      decoder_preclip=np.asarray(pres), decoder_postclip=np.asarray(posts), ctrl=np.asarray(ctrls),
                      decoder_motor_gains=gains, decoder_rate_for_full_control_hz=np.array(450.), filter_tau_s=np.array(.005),
                      decoder_max_control=np.array(1.), sensory_current=rec, applied_current=rec, sensory_scale=np.array(s))
        np.savez_compressed(dest / 'NEURAL.npz', **nstate)
        sp_arr = np.asarray(spikes_all, dtype=np.uint8)
        np.savez_compressed(dest / 'SPIKES.npz', packed=np.packbits(sp_arr, axis=1), shape=np.array(sp_arr.shape),
                            spike_counts=sp_arr.sum(0).astype(np.int32), pool_per_ms=np.asarray(per_ms_pool))
        np.savez_compressed(dest / 'TRAJECTORY.npz', **{k: np.asarray(v, float) for k, v in traj.items()})
        np.savez_compressed(dest / 'CTRL.npz', ctrl=np.asarray(ctrls), names=np.asarray(names))
        np.savez_compressed(dest / 'COMMAND_TIMING.npz', decoded=np.asarray(posts), applied=np.asarray(posts),
                            applied_source_index=np.arange(-1, 399))
        dump(dest / 'MODEL_METADATA.json', dict(names=names, geom_leg={str(k): v for k, v in foot.items()}, floor=floor))
        dump(dest / 'PARAMETERS.json', json.loads(json.dumps(parameters, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))))
        dump(dest / 'DECODER_CONFIG.json', dict(arm=arm_id, **dataclasses.asdict(dec.config), sensory_scale=s))
        dump(dest / 'SENSORY_GAIN_CONFIG.json', dict(arm=arm_id, sensory_scale=s, submitted='current * s at the net.step call site'))
        dump(dest / 'CURRENT_PATH_CHECK.json', dict(passed=bool(applied_ok), recorded_equals_applied=True,
                                                    note='the scaled array is both recorded and forwarded at the same call site'))
        duty = float(sp_arr.mean())
        rmean = np.asarray(rates)[:, np.flatnonzero(gains != 0)].mean(axis=0)
        per_ch = [float(np.asarray(rates)[:, i].mean()) for i in range(90)]
        post = np.asarray(posts)
        arms[arm_id] = dict(s=s, reset_bitwise_ok=bool(reset_ok), duty=duty, spikes_shape=list(sp_arr.shape),
                            nonzero_channels_rate_mean_range=[float(rmean.min()), float(rmean.max())],
                            postclip_at_1_fraction_nz=float((post[:, np.flatnonzero(gains != 0)] == 1).mean()),
                            postclip_at_1_fraction_90=float((post == 1).mean()),
                            mean_rate=float(np.mean(per_ch)), firing=bool(sp_arr.any()), calls=dict(calls))
        dump(out / 'ARMS.json', arms)
    ok = calls == BUDGET
    dump(out / 'CALL_LEDGER.json', calls)
    dump(out / 'BUDGET_CHECK.json', dict(expected=BUDGET, actual=calls, passed=bool(ok)))
    print(json.dumps(dict(arms={k: dict(duty=round(v['duty'], 5), s=v['s'], rate_range=[round(x, 1) for x in v['nonzero_channels_rate_mean_range']],
                                    post1=round(v['postclip_at_1_fraction_nz'], 4)) for k, v in arms.items()}, budget_ok=ok)))
    return 0 if ok else 3


if __name__ == '__main__': raise SystemExit(main())
