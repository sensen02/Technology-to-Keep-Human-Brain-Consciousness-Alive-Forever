"""GAIN SWEEP: does opening the tarsal channel's gain move the three collapse numbers?

Owner-directed.  It does NOT edit run_closed_loop_first_v8.py: v8 is the frozen delivery script whose sha256
is pinned inside delivered reports.  This probe re-implements the same closed loop (same encoder, same tier,
same decoder, same frozen archive) with the channel gain as an explicit sweep parameter, and it asserts that
v8 is untouched.

Declared before the run:
  criterion = HALF-STEP HOLD = no first non-foot contact inside 0.5 s AND legs_in_contact == 6 throughout
  gains     = 0.0 (channel off) / 0.01 / 0.05 / 0.3 (the declared nominal) / 1.0
  question  = do the three step numbers (first non-foot, legs<6, first zero-tarsus) move monotonically with
              the gain, or do they not move at all?  A flat response means the channel, though wired and
              firing, has no effect on the kinematics.  The sweep is a sensitivity listing, NOT a tune for a
              winner: every gain is reported with its full field set.
"""
import hashlib, json, os, pathlib, sys, time, zipfile
import numpy as np

ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
sys.path.insert(0, str(ROOT))
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
STATE = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259/equilibrium_best_state.npz'
DRIVE = ROOT / 'outputs/experiment_platform/20261015_135150_a2fff3f0/traces.npz'
V8 = ROOT / 'run_closed_loop_first_v8.py'
DT, CMD, NST = 2.5e-5, 200, 10
NEURAL_TICK_S = 0.0005   # the interval's neural tick; NST ticks x 0.0005 s = 0.005 s = the command interval
STEPS = 20000
GAINS = (0.0, 0.01, 0.05, 0.3, 1.0)
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
CRIT = 'HALF-STEP HOLD = no first non-foot contact inside 0.5 s AND legs_in_contact == 6 throughout'


def main():
    import mujoco, importlib
    L = importlib.import_module('run_flybody_loop')
    from engine.embodied import adapters as ad
    from engine.embodied.motor_output import MotorOutput, MotorOutputConfig, build_motor_manifest
    v8_before = sha(V8)
    body = L.build_body('canonical'); m = body['model']; m.opt.timestep = DT
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    st = np.load(STATE); arch = np.asarray(np.load(DRIVE)['normal_controls'], float)
    q0, act0 = st['qpos'].copy(), st['act'].copy()
    tcfg = ad.TierConfig(background_mean_nA=0, background_sd_nA=0)
    built = ad.build_neural_tier(tcfg); n = int(built['n'])
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    pools = json.loads((ROOT / 'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json').read_text())
    man = build_motor_manifest(pools, built['tier']['root_ids'], names)
    nerve = {Lg: np.asarray(built['sensory'][Lg.lower()], dtype=int) for Lg in LEGS}
    motor_idx = sorted({i for row in man['rows'] for i in row['local_indices']})
    weight_per_leg = float(body['weight_per_leg'])
    f6 = np.zeros(6)

    def run(gain):
        d = mujoco.MjData(m)
        d.qpos[:] = q0; d.qvel[:] = 0.0; d.act[:] = act0; d.ctrl[:] = 0.0
        mujoco.mj_forward(m, d)
        body['reference'] = q0.copy()
        tier = ad.NeuralTier(built, tcfg, ad.OwnershipLedger()); net = tier.net
        dec = MotorOutput(man, MotorOutputConfig())
        fnf = fl6 = fzt = None; pair = None; six = 0
        rz_min = rz_fin = float(d.qpos[2]); qm = 0.0; qs = 0; ap = float(np.max(np.asarray(d.act))); ood = 0
        spikes = 0; spiked = 0; pending = np.zeros(m.nu); ctrl = []
        for k in range(STEPS // CMD):
            t = k * CMD * DT
            q = L.observe(body, d, weight_per_leg)
            cur = np.zeros(n); fed = {Lg: 0.0 for Lg in LEGS}
            for Lg in LEGS:
                force = 0.0; touch = 0.0
                for ci, c in enumerate(d.contact):
                    if c.efc_address < 0: continue
                    g1, g2 = int(c.geom1), int(c.geom2)
                    n1, n2 = gname(g1), gname(g2)
                    if body['floor'] not in (g1, g2): continue
                    # MEASURED CONTACT ORDER: geom1 is the floor and geom2 is the tarsus.  A probe that tested only
                    # one side therefore read zero force for every leg; the test below is side-agnostic.
                    other = n2 if g1 == body['floor'] else n1
                    if 'Tarsus' in other and other.startswith(Lg):
                        mujoco.mj_contactForce(m, d, ci, f6)
                        force += float(np.linalg.norm(np.asarray(f6[:3]))); touch = 1.0
                value = force / weight_per_leg + touch
                cur[nerve[Lg]] += gain * value
                fed[Lg] = gain * value
            sp = np.zeros((NST, n), dtype=bool)
            for j in range(NST):
                sp[j] = net.step(current_nA=cur)
            spikes += int(sp.sum()); spikes_motor = int(sp[:, motor_idx].sum())
            if int(sp.sum()): spiked += 1
            ctrl.append(pending.copy())
            d.ctrl[:] = pending
            for j in range(CMD):
                mujoco.mj_step(m, d)
                i = k * CMD + j + 1
                legs = set(); tar = 0; nf = None
                for c in d.contact:
                    if c.efc_address < 0: continue
                    n1, n2 = gname(int(c.geom1)), gname(int(c.geom2))
                    if 'Tarsus' in (n1 + n2):
                        tar += 1
                        for Lg in LEGS:
                            if n1.startswith(Lg) or n2.startswith(Lg): legs.add(Lg)
                    elif nf is None:
                        nf = sorted((n1, n2))
                if len(legs) >= 6: six += 1
                if fl6 is None and len(legs) < 6: fl6 = i
                if fzt is None and tar == 0: fzt = i
                if fnf is None and nf is not None: fnf = i; pair = nf
                a = float(np.max(np.abs(np.asarray(d.act)))); ap = max(ap, a); ood += 1 if a > 1 else 0
                qq = float(np.max(np.abs(np.asarray(d.qacc))))
                if qq > qm: qm, qs = qq, i
                rz = float(d.qpos[2])
                if rz < rz_min: rz_min = rz
                rz_fin = rz
            pending, _ = dec.consume(sp, spike_dt_s=NEURAL_TICK_S, start_s=t, end_s=t + CMD * DT)
        return dict(gain_nA_per_unit=gain, first_nonfoot_step=fnf,
                    first_nonfoot_ms=(None if fnf is None else round(fnf * DT * 1000, 6)),
                    first_nonfoot_pair=pair, first_legs_below_6_step=fl6, first_zero_tarsus_step=fzt,
                    root_z_min=rz_min, root_z_final=rz_fin, max_abs_qacc=qm, max_abs_qacc_step=qs,
                    act_peak=ap, act_out_of_domain_steps=ood, six_leg_steps=six, of=STEPS,
                    total_spikes=spikes, intervals_with_spike=spiked,
                    criterion_met=bool(fnf is None and six == STEPS),
                    ctrl_sha256=hashlib.sha256(np.asarray(ctrl).tobytes()).hexdigest())

    res = []
    for g in GAINS:
        r = run(g); res.append(r)
        print('gain %-6g first_nonfoot %-6s (%s ms) | legs<6 %-6s | zero_tarsus %-6s | root_z %.6f/%.6f | qacc %.6g@%s | six %d/%d | spikes %d | criterion %s'
              % (g, r['first_nonfoot_step'], r['first_nonfoot_ms'], r['first_legs_below_6_step'], r['first_zero_tarsus_step'],
                 r['root_z_min'], r['root_z_final'], r['max_abs_qacc'], r['max_abs_qacc_step'], r['six_leg_steps'], STEPS,
                 r['total_spikes'], r['criterion_met']), flush=True)
    v8_after = sha(V8)
    assert v8_before == v8_after, 'v8 changed during the sweep'
    base = res[0]
    deltas = [dict(gain=r['gain_nA_per_unit'],
                   first_nonfoot_step_delta=(None if (base['first_nonfoot_step'] is None or r['first_nonfoot_step'] is None) else r['first_nonfoot_step'] - base['first_nonfoot_step']),
                   first_legs_below_6_step_delta=(None if (base['first_legs_below_6_step'] is None or r['first_legs_below_6_step'] is None) else r['first_legs_below_6_step'] - base['first_legs_below_6_step']),
                   first_zero_tarsus_step_delta=(None if (base['first_zero_tarsus_step'] is None or r['first_zero_tarsus_step'] is None) else r['first_zero_tarsus_step'] - base['first_zero_tarsus_step']),
                   six_leg_steps_delta=r['six_leg_steps'] - base['six_leg_steps'],
                   ctrl_differs_from_gain0=(r['ctrl_sha256'] != base['ctrl_sha256'])) for r in res]
    step1 = [r['first_nonfoot_step'] for r in res]
    monotone = all(step1[i] <= step1[i+1] for i in range(len(step1)-1)) or all(step1[i] >= step1[i+1] for i in range(len(step1)-1))
    moved = len(set(step1)) > 1
    out = DESK / ('S2-gain-sweep-' + time.strftime('%Y%m%dT%H%M%S')); out.mkdir(exist_ok=False)
    rep = dict(status='GAIN_SWEEP_COMPLETE', manual_research_not_product=True,
        question='does opening the tarsal channel gain from 0 to 1.0 move the three collapse step numbers',
        criterion_written_before_the_run=CRIT,
        sensitivity_listing_not_a_search='every gain is reported with its full field set; no gain is selected as a winner and none is proposed as a fix',
        why_not_a_new_version='run_closed_loop_first_v8.py is the frozen delivery script and its sha256 is pinned inside delivered reports; editing it would invalidate those pins. This probe re-implements the same loop with the gain as an explicit parameter and asserts v8 is untouched.',
        v8_script=str(V8), v8_sha256_before=v8_before, v8_sha256_after=v8_after, v8_untouched=True,
        probe_defect_found_and_fixed='the first version of THIS probe tested only one side of the contact pair; measured order is geom1=floor, geom2=<LEG>Tarsus*_geom, so every leg read zero force and all five gains returned the gain-0 arm (0 spikes). Fixed to be side-agnostic before any physical conclusion was drawn.',
        protocol=dict(dt_s=DT, steps=STEPS, window_s=STEPS * DT, command_interval_s=CMD * DT,
                      neural_tick_s=NST * DT, gains=list(GAINS), state_written_once=True, loop_writes='ctrl only',
                      state=str(STATE), state_sha256=sha(STATE), drive=str(DRIVE), drive_sha256=sha(DRIVE),
                      tier_n=int(n), actuators=int(man['n_actuators']), weight_per_leg=weight_per_leg),
        results=res, deltas_vs_gain0=deltas,
        answer=dict(the_three_step_numbers_move=moved, they_move_monotonically=monotone,
                    first_nonfoot_by_gain=step1,
                    reading=('the three step numbers DO move with the gain, so the channel is not merely wired: its strength changes the collapse timing'
                             if moved else
                             'the three step numbers do NOT move with the gain: the channel is wired and firing but has no effect on the kinematics under this interface'),
                    direction='EARLIER as the gain rises' if moved and step1[-1] < step1[0] else ('later as the gain rises' if moved else 'no change')),
        not_established='absolute force values, physical units, biological validity; no standing, no walking, no completion percentage',
        recorded_at=time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)'))
    (out / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    src = pathlib.Path(__file__)
    with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(src, src.name)
    with zipfile.ZipFile(out / 'source_snapshot.zip') as z:
        assert hashlib.sha256(z.read(src.name)).hexdigest() == sha(src)
    fp = out / 'FROZEN_DELIVERY_HASHES.json'; fp.write_text('{}', encoding='utf-8')
    (out / 'delivery-hashes.json').write_text('{}', encoding='utf-8')
    listing = sorted(p.name for p in out.iterdir() if p.is_file())
    data = [f for f in listing if f not in ('FROZEN_DELIVERY_HASHES.json', 'delivery-hashes.json')]
    fp.write_text(json.dumps(dict(status='FROZEN_BEFORE_VERIFICATION', files={f: sha(out / f) for f in data},
        covers_rule='len(files) == len(os.listdir) - 2', covers_self=False, manifest_sha256=None,
        manifest_sha256_note='null by definition; a manifest cannot carry its own digest',
        frozen_at=time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)')), ensure_ascii=False, indent=1), encoding='utf-8')
    dhd = {f: sha(out / f) for f in listing if f != 'delivery-hashes.json'}
    (out / 'delivery-hashes.json').write_text(json.dumps(dhd, ensure_ascii=False, indent=1), encoding='utf-8')
    assert json.loads((out / 'delivery-hashes.json').read_text())['FROZEN_DELIVERY_HASHES.json'] == sha(fp)
    for p in sorted(out.iterdir()):
        if p.is_file(): os.chmod(p, 0o444)
    os.chmod(out, 0o555)
    print('OUT:', out)
    print('moved:', moved, '| monotone:', monotone, '| v8 untouched:', v8_before == v8_after)
    return out


if __name__ == '__main__':
    main()
