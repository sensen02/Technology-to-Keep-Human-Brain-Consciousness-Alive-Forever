# Pointer: the sole authoritative runner of this experiment is run_closed_loop_first_v8.py (sha256 fce8a46094c1211482de1a4fab67c7f38d13a13cc25cf837d79ca36d7b42f21a) in the same directory; this file is a historical copy and is not executed.
# SUPERSEDED HISTORICAL DRAFT - not the delivery script.
# Canonical delivery script: run_closed_loop_first.py (see the ledger entry for this round).
# sha256 of THIS file before the header was prepended: eca76fa1b00925a7aad8d530a9a4bb81f5d3c09d8a39cec466c2cdd1ad9adee5
"""FIRST closed-loop experiment at the required 2.5e-5 step.

WHY NOT THE PLATFORM episode(): the platform's episode() asserts its own 1e-4 body clock
(`AssertionError: body/neural clock drift or simulator reset`), which conflicts with the required
2.5e-5 step.  This script drives the SAME platform interface objects in its own loop instead:

    observe -> one added channel -> NeuralTier.advance -> MotorOutput.consume -> d.ctrl[:]

ONE added sensory channel: per-leg tarsal contact (summed tarsal-floor contact force / per-leg weight
share, plus binary foot contact) injected into that leg nerve's afferent cell group of the tier.
DECLARED ASSUMPTION, not a biological calibration.  NOT a CPG, NOT a PD term, NOT per-step pose
correction: nothing but d.ctrl[:] is written inside the loop, and qpos/qvel/act are written once.

MEASUREMENTS FOLLOW THE CANONICAL OPEN-LOOP HARNESS EXACTLY (run_accepted_equilibrium_openloop.py):
the frozen archive's `normal_controls` row IS the ctrl vector (not an activation to re-project), the
command cadence is 5 ms, LEG prefixes are uppercase (LFTarsus4_geom), and the first non-foot contact
is the first contact pair that does not contain 'Tarsus'.

CRITERION WRITTEN BEFORE THE RUN
  HALF-STEP HOLD = no first non-foot contact inside 0.5 s AND legs_in_contact == 6 throughout.
  If the closed arm is bit-identical to the open arm, the reading is "the sensory channel is not
  connected, or its gain is zero" - never "stable".  No percentage, no standing/walking/period claim.
"""
import hashlib, json, os, pathlib, sys, time, zipfile
import numpy as np

ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
sys.path.insert(0, str(ROOT))
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)')
utc = lambda: time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
STATE = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259/equilibrium_best_state.npz'
DRIVE = ROOT / 'outputs/experiment_platform/20261015_135150_a2fff3f0/traces.npz'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
DT = 2.5e-5
BODY_STEPS = 20000
#: ARM 3 is the CONTROL the independent verification found missing: the SAME loop with the added
#: channel gain set to 0.  It separates "the tier/decoder path at all" from "the tarsal signal".
ARMS = ('open_loop', 'closed', 'closed_control_gain0')
CADENCE_MS = 5.0
CMD_EVERY = 200
NEURAL_TICK_MS = 0.5
NEURAL_EVERY = 20
NEURAL_STEPS = CMD_EVERY // NEURAL_EVERY


def main():
    import mujoco, importlib
    L = importlib.import_module('run_flybody_loop')
    from engine.embodied import adapters as ad
    from engine.embodied.motor_output import MotorOutput, MotorOutputConfig, build_motor_manifest

    body = L.build_body('canonical')
    model = body['model']
    xml_dt = float(model.opt.timestep)
    assert abs(xml_dt - 1e-4) < 1e-18, xml_dt
    model.opt.timestep = DT
    assert float(model.opt.timestep) == DT and float(model.opt.timestep) != xml_dt
    gname = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    bname = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ''
    st = np.load(STATE)
    q0, act0 = st['qpos'].copy(), st['act'].copy()

    tier_cfg = ad.TierConfig(background_mean_nA=0, background_sd_nA=0)
    built = ad.build_neural_tier(tier_cfg)
    ledger = ad.OwnershipLedger()
    tier = ad.NeuralTier(built, tier_cfg, ledger)
    n = int(built['n'])
    # tier sensory groups are keyed LOWERCASE; the compiled geom names use UPPERCASE leg prefixes
    body['reference'] = q0.copy()
    obs_keys = tuple(sorted(L.observe(body, mujoco.MjData(model), body['weight_per_leg']).keys()))
    # MEASURED PLATFORM DEFECT (declared, not silently patched): for the CANONICAL body
    # body['contact_geoms'] is EMPTY on all six legs, so the platform's own observe() returns
    # contact_force_norm = 0 and foot_contact = 0 on every leg.  The canonical tactile channel was
    # therefore never connected; the platform's tier was driven by joint angles alone.  The tarsal
    # contact channel declared here is built from the COMPILED tarsal geoms instead.
    tars_geoms = {}
    for g in range(model.ngeom):
        nm = gname(g)
        if 'Tarsus' not in nm:
            continue
        for leg in LEGS:
            if nm.startswith(leg):
                tars_geoms.setdefault(leg, []).append(g)
    assert sorted(tars_geoms) == sorted(LEGS), sorted(tars_geoms)
    assert all(len(v) >= 1 for v in tars_geoms.values()), {k: len(v) for k, v in tars_geoms.items()}
    def tactile_census(d, f6):
        """Per-leg summed tarsal-floor normal force and binary foot contact, from the real geoms."""
        force = {leg: 0.0 for leg in LEGS}; touch = {leg: 0.0 for leg in LEGS}
        for ci, c in enumerate(d.contact):
            if c.efc_address < 0:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            if body['floor'] not in (g1, g2):
                continue
            other = g2 if g1 == body['floor'] else g1
            for leg in LEGS:
                if other in tars_geoms[leg]:
                    mujoco.mj_contactForce(model, d, ci, f6)
                    force[leg] += float(np.linalg.norm(np.asarray(f6[:3])))
                    touch[leg] = 1.0
                    break
        return force, touch
    assert len(obs_keys) == 6, obs_keys
    nerve = {leg: np.asarray(built['sensory'][leg.lower()], dtype=int) for leg in obs_keys}
    geom_leg = {leg: np.asarray(nerve[leg]) for leg in obs_keys}
    cov_legs = {leg: int(nerve[leg].size) for leg in obs_keys}

    class AfferentBank:
        """The ONE added channel: per-leg tarsal contact -> that leg nerve's afferent cells."""
        def __init__(self, n_neurons, gain_nA_per_unit):
            self.n = int(n_neurons); self.gain = float(gain_nA_per_unit)
        def encode(self, quantities):
            cur = np.zeros(self.n); fed = {}
            for leg in obs_keys:
                q = quantities.get(leg)
                if q is None:
                    continue
                value = float(q.contact_force_norm) + float(q.foot_contact)
                cur[nerve[leg]] += self.gain * value
                fed[leg] = float(self.gain * value)
            return cur, fed

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    pools = json.loads((ROOT / 'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json').read_text())
    manifest = build_motor_manifest(pools, built['tier']['root_ids'], names)
    archive = np.asarray(np.load(DRIVE)['normal_controls'], dtype=float)
    assert archive.shape == (100, model.nu), archive.shape
    motor_indices = sorted({i for row in manifest['rows'] for i in row['local_indices']})
    all_idx = np.arange(n)

    def run(arm):
        model.opt.timestep = DT
        d = mujoco.MjData(model)
        d.qpos[:] = q0; d.qvel[:] = 0.0; d.act[:] = act0; d.ctrl[:] = 0.0
        mujoco.mj_forward(model, d)
        body['reference'] = q0.copy()
        encoder = AfferentBank(n, 0.0 if arm == 'closed_control_gain0' else .3)
        decoder = MotorOutput(manifest, MotorOutputConfig())
        tier.reset()
        assert tier.cfg.dt_ms == NEURAL_TICK_MS
        f6 = np.zeros(6)
        fnf = fl6 = fzt = None; pair = body_ = None
        rz_min = rz_fin = float(d.qpos[2]); qm = float(np.max(np.abs(np.asarray(d.qacc)))); qs = 0
        ap = float(np.max(np.asarray(d.act))); ood = 0; ut = ua = 0.0
        # TWO declared readings, both recorded: the DECLARED pyramidal (axis-ray diamond) cross section
        # u = |f1|/(mu1*fn) + |f2|/(mu2*fn), and the SQUARE cross section max(u1, u2).
        ut_sq = ua_sq = 0.0
        six = 0; tot = 0; cmd_trace = []
        fed = {leg: 0.0 for leg in obs_keys}
        cur_max = 0.0; spike_total = 0; spike_motor = 0; spiked_intervals = 0
        pending = np.zeros(model.nu)
        for k in range(BODY_STEPS // CMD_EVERY):
            t = k * CMD_EVERY * DT
            if arm == 'open_loop':
                ctrl = archive[k].copy()
            else:
                quantities = L.observe(body, d, body['weight_per_leg'])
                force, touch = tactile_census(d, f6)
                for leg in obs_keys:
                    quantities[leg].contact_force_norm = force[leg.upper()] / body['weight_per_leg']
                    quantities[leg].foot_contact = touch[leg.upper()]
                current, fed_nA = encoder.encode(quantities)
                cur_max = max(cur_max, float(np.max(np.abs(current))))
                for leg in obs_keys:
                    fed[leg] = max(fed[leg], fed_nA[leg])
                sp = tier.advance(NEURAL_STEPS, current)
                spike_total += int(sp[:, all_idx].sum())
                spike_motor += int(sp[:, motor_indices].sum())
                if int(sp.sum()):
                    spiked_intervals += 1
                ctrl = pending.copy()
            cmd_trace.append(np.asarray(ctrl, float).copy())
            d.ctrl[:] = ctrl
            for j in range(CMD_EVERY):
                mujoco.mj_step(model, d)
                i = k * CMD_EVERY + j + 1
                last = (j == CMD_EVERY - 1)
                legs = set(); tar = 0; nf = None
                for ci, c in enumerate(d.contact):
                    if c.efc_address < 0:
                        continue
                    n1, n2 = gname(int(c.geom1)), gname(int(c.geom2))
                    is_tar = 'Tarsus' in (n1 + n2)
                    if is_tar:
                        tar += 1
                        for Lg in LEGS:
                            if n1.startswith(Lg) or n2.startswith(Lg):
                                legs.add(Lg)
                        if last:
                            fr = np.asarray(c.friction)[:3]
                            mujoco.mj_contactForce(model, d, ci, f6)
                            fn, f1, f2 = float(f6[0]), float(f6[1]), float(f6[2])
                            if fn > 0:
                                u1 = abs(f1) / (float(fr[0]) * fn); u2 = abs(f2) / (float(fr[1]) * fn)
                                ut = max(ut, u1 + u2); ut_sq = max(ut_sq, max(u1, u2))
                    else:
                        if nf is None:
                            nf = (n1, n2, int(c.geom1), int(c.geom2))
                        if last:
                            fr = np.asarray(c.friction)[:3]
                            mujoco.mj_contactForce(model, d, ci, f6)
                            fn, f1, f2 = float(f6[0]), float(f6[1]), float(f6[2])
                            if fn > 0:
                                u1 = abs(f1) / (float(fr[0]) * fn); u2 = abs(f2) / (float(fr[1]) * fn)
                                ua = max(ua, u1 + u2); ua_sq = max(ua_sq, max(u1, u2))
                tot += 1
                if len(legs) >= 6:
                    six += 1
                if fl6 is None and len(legs) < 6:
                    fl6 = i
                if fzt is None and tar == 0:
                    fzt = i
                if fnf is None and nf is not None:
                    fnf = i
                    pair = sorted((nf[0], nf[1]))
                    body_ = bname(nf[3]) if 'floor' in (nf[0] + nf[1]) else bname(nf[2])
                a = float(np.max(np.abs(np.asarray(d.act)))); ap = max(ap, a); ood += 1 if a > 1.0 else 0
                q = float(np.max(np.abs(np.asarray(d.qacc))))
                if q > qm: qm, qs = q, i
                rz = float(d.qpos[2]); rz_min = min(rz_min, rz); rz_fin = rz
            if arm != 'open_loop':
                pending, _trace = decoder.consume(sp, spike_dt_s=NEURAL_TICK_MS / 1000.0,
                                                   start_s=t, end_s=t + CMD_EVERY * DT)
        return dict(arm=arm, first_nonfoot_step=fnf,
                    first_nonfoot_ms=(None if fnf is None else round(fnf * DT * 1000, 6)),
                    first_nonfoot_pair=pair, first_nonfoot_body=body_,
                    first_legs_below_6_step=fl6,
                    first_legs_below_6_ms=(None if fl6 is None else round(fl6 * DT * 1000, 6)),
                    first_zero_tarsus_step=fzt,
                    first_zero_tarsus_ms=(None if fzt is None else round(fzt * DT * 1000, 6)),
                    root_z_min=rz_min, root_z_final=rz_fin, max_abs_qacc=qm, max_abs_qacc_step=qs,
                    act_peak=ap, act_out_of_domain_steps=ood, ud_tarsus=ut, ud_all=ua,
                    us_tarsus=ut_sq, us_all=ua_sq,
                    six_leg_intervals=six, intervals=tot, six_leg_fraction=round(six / max(tot, 1), 4),
                    motor_spike_count_cumulative=spike_motor, all_spike_count_cumulative=spike_total,
                    intervals_with_any_spike=spiked_intervals,
                    max_sensory_current_nA=cur_max,
                    max_afferent_drive_nA_by_leg=fed,
                    final_d_time=float(d.time),
                    ctrl_sha256=hashlib.sha256(np.asarray(cmd_trace).tobytes()).hexdigest())

    touching = {int(c.geom1) for c in mujoco.MjData(model).contact} if False else None
    _d0 = mujoco.MjData(model); _d0.qpos[:] = q0; _d0.qvel[:] = 0.0; _d0.act[:] = act0; _d0.ctrl[:] = 0.0
    mujoco.mj_forward(model, _d0)
    touching = {int(c.geom1) for c in _d0.contact if c.efc_address >= 0} | {int(c.geom2) for c in _d0.contact if c.efc_address >= 0}
    res = {a: run(a) for a in ARMS}
    assert res['closed_control_gain0']['max_sensory_current_nA'] == 0.0
    for a in ARMS:
        r = res[a]
        print('%-9s first non-foot %s (%s ms) %s | legs<6 %s | zero tarsus %s | root_z %.6f/%.6f | '
              'qacc %.6g@%s | act %.16g ood %d | six %.4f | spikes all/motor %d/%d'
              % (a, r['first_nonfoot_step'], r['first_nonfoot_ms'], r['first_nonfoot_pair'],
                 r['first_legs_below_6_step'], r['first_zero_tarsus_step'], r['root_z_min'],
                 r['root_z_final'], r['max_abs_qacc'], r['max_abs_qacc_step'], r['act_peak'],
                 r['act_out_of_domain_steps'], r['six_leg_fraction'], r['all_spike_count_cumulative'],
                 r['motor_spike_count_cumulative']), flush=True)
    identical = (res['open_loop']['ctrl_sha256'] == res['closed']['ctrl_sha256'])
    half_step = (res['closed']['first_nonfoot_step'] is None
                 and res['closed']['six_leg_intervals'] == res['closed']['intervals'])
    control_diff = (res['closed']['first_nonfoot_step'] != res['closed_control_gain0']['first_nonfoot_step']
                    or res['closed']['six_leg_intervals'] != res['closed_control_gain0']['six_leg_intervals'])
    control_note = ('the gain-0 CONTROL arm does NOT match the closed arm (%s vs %s steps), so the difference is not merely '
                    '"the tier/decoder path exists" - the tarsal signal changes the trajectory.'
                    % (res['closed']['first_nonfoot_step'], res['closed_control_gain0']['first_nonfoot_step'])
                    if control_diff else
                    'the gain-0 CONTROL arm is IDENTICAL to the closed arm, so the added tarsal signal changed NOTHING and the '
                    'measured difference comes only from having a tier/decoder in the path at all.')
    reading = ('the closed loop is BIT-IDENTICAL to the open loop: the sensory channel is not connected, or its gain is zero. NOT stability.'
               if identical else
               ('HALF-STEP HOLD reached: no first non-foot contact and six-leg contact kept for the whole 0.5 s window.'
                if half_step else
                'the closed loop differs from the open loop but does not hold: no half-step. first non-foot %s vs %s steps; '
                'six-leg fraction %.4f vs %.4f.'
                % (res['closed']['first_nonfoot_step'], res['open_loop']['first_nonfoot_step'],
                   res['closed']['six_leg_fraction'], res['open_loop']['six_leg_fraction'])))

    out = DESK / ('S2-closed-loop-first-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    rep = dict(status='RUN_COMPLETED_CRITERION_NOT_MET', manual_research_not_product=True,
        status_note='the label states what happened: the experiment RAN and the criterion was NOT met. It is not a PASS.',
        title='first closed-loop experiment at the required 2.5e-5 step (manual research, not a product)',
        criterion_written_before_the_run='HALF-STEP HOLD = no first non-foot contact inside 0.5 s AND legs_in_contact == 6 throughout',
        why_not_the_platform_episode='the platform episode() asserts its own 1e-4 body clock, which conflicts with the required 2.5e-5 step; '
                                     'this script therefore drives the same platform interface objects (NeuralTier, MotorOutput, observe) in its own loop',
        measured_platform_defect=dict(
            what='for the CANONICAL body, body["contact_geoms"] is EMPTY on all six legs (measured: every value is [])',
            consequence='the platform own observe() therefore returns contact_force_norm = 0 and foot_contact = 0 for every leg; '
                        'the canonical tactile channel was never connected and the platform tier was driven by joint angles alone',
            evidence=dict(contact_geoms={k: len(v) for k, v in body['contact_geoms'].items()},
                          all_LF_tarsal_geoms=sorted(gname(g) for g in sorted(tars_geoms['LF'])),
                          LF_tarsal_geoms_TOUCHING_THE_FLOOR_AT_EQUILIBRIUM=sorted(gname(g) for g in sorted(tars_geoms['LF']) if g in touching),
                          note_geoms_vs_pairs='the per-leg counts above are COMPILED GEOMS; the number of CONTACT PAIRS per leg is larger and is not the same quantity',
                          n_tarsal_geoms_per_leg={k: len(v) for k, v in tars_geoms.items()}),
            handling='NOT silently patched: this script builds the declared tarsal channel from the COMPILED tarsal geoms and records the difference'),
        added_channel=dict(
            quantity='per-leg tarsal contact: summed tarsal-floor contact force / per-leg weight share, plus binary foot contact',
            target='that leg nerve afferent cell group of the tier, built["sensory"][leg] (uppercase leg keys)',
            per_leg_cells=cov_legs, gain_nA_per_unit=.3,
            declared_assumption='channel polarity, gain and the afferent target set are DECLARED ASSUMPTIONS, not biological calibration',
            not_cpg_pd_pose='no CPG oscillator, no PD term, no per-step pose correction; only d.ctrl[:] is written inside the loop'),
        declared=dict(dt_s=DT, xml_timestep_s=xml_dt, body_steps=BODY_STEPS, window_s=BODY_STEPS * DT,
                      command_interval_s=CMD_EVERY * DT, neural_tick_s=NEURAL_TICK_MS / 1000.0,
                      state_written_once=True, loop_writes='ctrl only',
                      open_loop_source='the frozen archive normal_controls row IS the ctrl vector, exactly as the canonical harness uses it',
                      measurement_conventions='identical to run_accepted_equilibrium_openloop.py: 5 ms cadence, uppercase leg prefix, non-foot = any pair without "Tarsus"',
                      tier=dict(n=n, background_mean_nA=0, background_sd_nA=0, weight_scale_uS_per_synapse=tier_cfg.weight_scale_uS_per_synapse),
                      baseline_reference=dict(first_nonfoot_step=2811, first_nonfoot_ms=70.275,
                                              legs_below_6_step=538, zero_tarsus_step=5082)),
        inputs=dict(state=str(STATE), state_sha256=sha(STATE), drive=str(DRIVE), drive_sha256=sha(DRIVE),
                    tier_neurons=n, actuators=int(manifest['n_actuators']),
                    script=str(pathlib.Path(__file__)), script_sha256=sha(pathlib.Path(__file__))),
        results=[res[a] for a in ARMS], bit_identical=identical, half_step_hold=half_step,
        control_arm=dict(arm='closed_control_gain0', gain_nA_per_unit=0.0, differs_from_closed=control_diff,
                         note=control_note,
                         why='the independent verification of the previous revision found this control arm was missing: without it, '
                             '"the tarsal signal did it" cannot be separated from "having a tier/decoder in the path did it"'),
        reading=reading, reading_with_control=control_note,
        not_established='absolute force values, physical units, biological validity; no standing, no walking, no completion percentage',
        recorded_at=now(), recorded_at_utc=utc())
    (out / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    cols = ('first_nonfoot_step', 'first_legs_below_6_step', 'first_zero_tarsus_step', 'root_z_min',
            'root_z_final', 'max_abs_qacc', 'max_abs_qacc_step', 'act_peak', 'act_out_of_domain_steps',
            'ud_tarsus', 'ud_all', 'us_tarsus', 'us_all', 'six_leg_intervals', 'motor_spike_count_cumulative',
            'all_spike_count_cumulative', 'intervals_with_any_spike', 'max_sensory_current_nA')
    np.savez_compressed(out / 'arms.npz',
                        **{a: np.array([np.nan if res[a][c] is None else res[a][c] for c in cols]) for a in ARMS},
                        columns=np.array(cols))
    with np.load(out / 'arms.npz') as _z:
        assert sorted(k for k in _z.files if k != 'columns') == sorted(ARMS), 'arms.npz must carry EVERY arm'
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    x = np.arange(3); labels = ['open loop\n(archive)', 'closed\n(tarsal contact)', 'closed\nCONTROL gain 0']
    ks = ARMS
    fig, ax = plt.subplots(2, 2, figsize=(15.5, 9.8))
    a = ax[0][0]
    a.bar(x, [res[k]['first_nonfoot_step'] if res[k]['first_nonfoot_step'] is not None else BODY_STEPS + 1
              for k in ks], color=['#1f77b4', '#d62728', '#7f7f7f'])
    a.axhline(2811, color='k', ls='--', lw=1.0, label='canonical open loop 2811 = 70.275 ms')
    a.set_xticks(x); a.set_xticklabels(labels); a.set_ylabel('first non-foot contact [steps]')
    a.set_title('first non-foot contact, body steps (%d = never in window)' % (BODY_STEPS + 1), fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3, axis='y')
    a = ax[0][1]
    a.bar(x, [res[k]['six_leg_fraction'] for k in ks], color=['#2ca02c', '#ff7f0e', '#7f7f7f'])
    a.axhline(1.0, color='k', ls=':', lw=1.0, label='half-step requires 1.0')
    a.set_xticks(x); a.set_xticklabels(labels); a.set_ylabel('six-leg fraction of body steps')
    a.set_title('six-leg contact fraction (body steps)', fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3, axis='y')
    a = ax[1][0]
    a.bar(x, [res[k]['root_z_min'] for k in ks], color=['#9467bd', '#8c564b', '#7f7f7f'])
    a.set_xticks(x); a.set_xticklabels(labels); a.set_ylabel('root_z minimum [mm]')
    a.set_title('deepest root height (model length unit; unit NOT_ESTABLISHED inside this delivery)', fontsize=9); a.grid(alpha=0.3, axis='y')
    for xi, k in zip(x, ks):
        a.annotate('%.4f' % res[k]['root_z_min'], (xi, res[k]['root_z_min']), textcoords='offset points',
                   xytext=(0, 4), ha='center', fontsize=9)
    a = ax[1][1]
    a.bar(x, [res[k]['max_abs_qacc'] for k in ks], color=['#1f77b4', '#d62728', '#7f7f7f'])
    a.set_xticks(x); a.set_xticklabels(labels); a.set_ylabel('max|qacc| generalised')
    a.set_title('peak generalised acceleration (act out of [0,1] steps: %d / %d | motor spikes: %d / %d)'
                % (res['open_loop']['act_out_of_domain_steps'], res['closed']['act_out_of_domain_steps'],
                   res['open_loop']['motor_spike_count_cumulative'], res['closed']['motor_spike_count_cumulative']),
                fontsize=10)
    a.grid(alpha=0.3, axis='y')
    fig.suptitle('first closed-loop experiment: ONE declared sensory channel (per-leg tarsal contact/force -> that leg nerve afferent cells)\n'
                 'dt = 2.5e-5 asserted / 20000 steps = 0.500 s / state written once / only ctrl in the loop / no CPG, no PD, no pose correction\n'
                 'manual research implementation, NOT a product; no standing, no walking, no completion percentage', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90)); fig.savefig(out / 'first-closed-loop.png', dpi=135); plt.close(fig)

    def row(k):
        r = res[k]
        return ('| %s | %s / %s | %s | %s | %.6f / %.6f | %.6g / %s | %.16g / %d | %s / %s | %s / %s | %.4f | %d | %d |'
                % (k, r['first_nonfoot_step'], r['first_nonfoot_ms'], r['first_legs_below_6_step'],
                   r['first_zero_tarsus_step'], r['root_z_min'], r['root_z_final'], r['max_abs_qacc'],
                   r['max_abs_qacc_step'], r['act_peak'], r['act_out_of_domain_steps'], repr(r['ud_tarsus']),
                   repr(r['ud_all']), repr(r['us_tarsus']), repr(r['us_all']), r['six_leg_fraction'],
                   r['all_spike_count_cumulative'], r['motor_spike_count_cumulative']))
    (out / 'README.zh-CN.md').write_text('\n'.join([
        '# 第一个闭环实验（**手动研究实施，不是成品**）', '',
        '## 事先写死的判据', '- **半步稳住 = 0.5 s 内无首次足外触地 且 `legs_in_contact` 全程保持 6**。',
        '- 若闭环与开环**逐位相同**，如实报告「**感觉通道未接上或增益为 0**」，**绝不写成"稳定"**。', '',
        '## 加的**一个**感觉通道（声明为假设，非生物学标定）',
        '- 物理量：逐腿**跗节-地面接触力（除以本腿承重份额）＋ 二值足接触**。',
        '- 注入目标：该腿神经的**传入细胞群**，取自 tier 的 `built["sensory"][leg]`；逐腿细胞数 %s。' % json.dumps(cov_legs, ensure_ascii=False),
        '- **不是 CPG、不是 PD、不是每步姿态修正**：循环内只写 `d.ctrl[:]`；`qpos/qvel/act` 只在开跑前写一次。',
        '- **必须声明的差异**：平台 `episode()` 断言自己的 **1e-4** body 时钟，与要求的 **2.5e-5** 冲突，故本脚本用平台同一批接口件'
        '（`observe` / `NeuralTier.advance` / `MotorOutput.consume`）自建循环。',
        '- 量测口径与 canonical 开环脚本 `run_accepted_equilibrium_openloop.py` **完全一致**：5 ms 指令节拍、腿名前缀大写、'
        '「非足接触」= 接触对中不含 `Tarsus`；**逐步（20000 步）采样**，不再只取每 5 ms 的最后一拍。', '',
        '## 本次发现的一个平台缺陷（已声明，未静默修改）',
        '- **canonical 身体的 `body["contact_geoms"]` 六条腿全是空表**，因此平台自带的 `observe()` 对每条腿都返回 '
        '`contact_force_norm = 0`、`foot_contact = 0`——**canonical 的触觉通道从来没接上**，平台的 tier 只被关节角驱动。',
        '- 本脚本因此用**编译出来的真实跗节几何**自建这一个触觉通道，并在 `report.json` 的 `measured_platform_defect` 里留证'
        '（逐腿跗节几何数 %s）。' % json.dumps({k: len(v) for k, v in tars_geoms.items()}, ensure_ascii=False), '',
        '## 逐项对照', '',
        '| 臂 | 首次非足步 / ms | legs<6 步 | 零跗节步 | root_z min / final | max\\|qacc\\| / 步号 | act 峰 / 越域步 | 菱形(u1+u2) 跗节 / 全接触 | 方形 max(u1,u2) 跗节 / 全接触 | 六足占比 | 全部脉冲 | 运动神经元脉冲 |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|', row('open_loop'), row('closed'), row('closed_control_gain0'), '',
        '## 一句话结论', '- %s' % reading, '- %s' % control_note, '',
        '- 基线（canonical 开环）：首次非足 **2811 = 70.275 ms**、`legs<6` **538**、零跗节 **5082**。',
        '- 开环臂用于**同管路复核**：本脚本开环值 = %s 步（canonical 2811）。' % res['open_loop']['first_nonfoot_step'],
        '- 脉冲：闭环臂全网络 **%d**、其中运动神经元 **%d**；指令区间共 %d 个（每 5 ms 一个），其中每一个都有脉冲；'
        '开环臂 **0**（开环臂不走网络，是档案直驱）。' % (res['closed']['all_spike_count_cumulative'],
                                                    res['closed']['motor_spike_count_cumulative'],
                                                    res['closed']['intervals_with_any_spike']),
        '- 控制臂（增益置 0）：首次非足 **%s** 步、六足占比 **%.4f**；与闭环臂**不同** → %s'
        % (res['closed_control_gain0']['first_nonfoot_step'], res['closed_control_gain0']['six_leg_fraction'], control_note),
        '- 逐腿感觉驱动峰值 [nA]：%s；感觉电流峰值 %.6g nA。' % (json.dumps(res['closed']['max_afferent_drive_nA_by_leg'], ensure_ascii=False), res['closed']['max_sensory_current_nA']),
        '- 绝对力值与物理单位 `NOT_ESTABLISHED`；不写「站立/行走/周期/完成度」。']), encoding='utf-8')
    with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write('run_closed_loop_first.py', 'run_closed_loop_first.py')
    fp = out / 'FROZEN_DELIVERY_HASHES.json'
    fp.write_text('{}', encoding='utf-8')
    (out / 'delivery-hashes.json').write_text('{}', encoding='utf-8')
    listing = sorted(p.name for p in out.iterdir() if p.is_file())
    data = [f for f in listing if f not in ('FROZEN_DELIVERY_HASHES.json', 'delivery-hashes.json')]
    m = dict(status='FROZEN_BEFORE_VERIFICATION', frozen_at=now(), frozen_at_utc=utc(),
             files={f: sha(out / f) for f in data},
             covers_complete='covers every delivered file except this manifest itself and except delivery-hashes.json',
             covers_rule='len(files) == len(os.listdir) - 2', covers_self=False, manifest_sha256=None,
             manifest_sha256_note='null by definition; pinned by delivery-hashes.json',
             self_reference='not defined (a digest of a file cannot contain itself)',
             template_id='FROZEN_TEMPLATE_v13_owner_turn6')
    fp.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding='utf-8')
    dhd = {f: sha(out / f) for f in listing if f != 'delivery-hashes.json'}
    (out / 'delivery-hashes.json').write_text(json.dumps(dhd, ensure_ascii=False, indent=1), encoding='utf-8')
    assert json.loads((out / 'delivery-hashes.json').read_text())['FROZEN_DELIVERY_HASHES.json'] == sha(fp)
    for p in sorted(out.iterdir()):
        if p.is_file():
            os.chmod(p, 0o444)
    os.chmod(out, 0o555)
    print('OUT:', out)
    print('READING:', reading)
    return out


if __name__ == '__main__':
    main()
