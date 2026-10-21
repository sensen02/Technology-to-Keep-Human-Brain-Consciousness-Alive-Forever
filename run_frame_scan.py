"""FRAME SCAN: per-frame zeroing of the early command block (frames 0..14).

Owner-directed.  It does NOT touch run_closed_loop_first_v8.py: v8 is the frozen delivery script and its
sha256 is pinned inside delivered report.json files, so editing it would invalidate those pins.  This probe
only re-reads the frozen archive and perturbs ONE command frame at a time.

Declared before the run:
  criterion = HALF-STEP HOLD = no first non-foot contact inside 0.5 s AND legs_in_contact == 6 throughout
  baseline  = open loop, archive frame k as ctrl: first non-foot step 2811 = 70.275 ms (A6_geom|floor),
              legs_in_contact<6 at 538, zero tarsal at 5082
  arms      = baseline + one arm per frame index 0..14, that frame replaced by zeros
  all arms  = dt 2.5e-5 s, 20000 steps, state written once, only ctrl written inside the loop
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
DT, CMD, STEPS = 2.5e-5, 200, 20000
FRAMES = tuple(range(0, 15))
CRIT = 'HALF-STEP HOLD = no first non-foot contact inside 0.5 s AND legs_in_contact == 6 throughout'


def main():
    import mujoco, importlib
    L = importlib.import_module('run_flybody_loop')
    v8_sha_before = sha(V8)
    body = L.build_body('canonical'); m = body['model']; m.opt.timestep = DT
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    st = np.load(STATE); arch = np.asarray(np.load(DRIVE)['normal_controls'], float)
    q0, act0 = st['qpos'].copy(), st['act'].copy()
    LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')

    def run(frame):
        a = arch.copy()
        if frame is not None:
            a[frame] = 0.0
        d = mujoco.MjData(m)
        d.qpos[:] = q0; d.qvel[:] = 0.0; d.act[:] = act0; d.ctrl[:] = 0.0
        mujoco.mj_forward(m, d)
        fnf = fl6 = fzt = None; pair = None; six = 0
        for k in range(STEPS // CMD):
            d.ctrl[:] = a[k]
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
        return dict(frame=frame, first_nonfoot_step=fnf,
                    first_nonfoot_ms=(None if fnf is None else round(fnf * DT * 1000, 6)),
                    first_nonfoot_pair=pair, first_legs_below_6_step=fl6, first_zero_tarsus_step=fzt,
                    six_leg_steps=six, of=STEPS, criterion_met=bool(fnf is None and six == STEPS))

    res = {'baseline': run(None)}
    print('%-10s first_nonfoot %-6s (%s ms) | legs<6 %-6s | zero_tarsus %-6s | six %d/%d'
          % ('baseline', res['baseline']['first_nonfoot_step'], res['baseline']['first_nonfoot_ms'],
             res['baseline']['first_legs_below_6_step'], res['baseline']['first_zero_tarsus_step'],
             res['baseline']['six_leg_steps'], STEPS), flush=True)
    for f in FRAMES:
        res['frame%02d_zero' % f] = run(f)
        r = res['frame%02d_zero' % f]
        db = r['first_nonfoot_step'] - res['baseline']['first_nonfoot_step'] if r['first_nonfoot_step'] else None
        dl = (r['first_legs_below_6_step'] - res['baseline']['first_legs_below_6_step']) if r['first_legs_below_6_step'] and res['baseline']['first_legs_below_6_step'] else None
        print('%-10s first_nonfoot %-6s (%s ms) delta %-6s | legs<6 %-6s delta %-5s | zero_tarsus %-6s | six %d'
              % ('frame%02d' % f, r['first_nonfoot_step'], r['first_nonfoot_ms'], db, r['first_legs_below_6_step'], dl,
                 r['first_zero_tarsus_step'], r['six_leg_steps']), flush=True)
    v8_sha_after = sha(V8)
    assert v8_sha_before == v8_sha_after, 'v8 changed during the scan'

    base = res['baseline']
    deltas = {k: dict(first_nonfoot_step_delta=(v['first_nonfoot_step'] - base['first_nonfoot_step']) if (v['first_nonfoot_step'] and base['first_nonfoot_step']) else None,
                      first_legs_below_6_step_delta=(v['first_legs_below_6_step'] - base['first_legs_below_6_step']) if (v['first_legs_below_6_step'] and base['first_legs_below_6_step']) else None,
                      six_leg_steps_delta=v['six_leg_steps'] - base['six_leg_steps'])
              for k, v in res.items() if k != 'baseline'}
    out = DESK / ('S2-frame-scan-' + time.strftime('%Y%m%dT%H%M%S')); out.mkdir(exist_ok=False)
    rep = dict(status='FRAME_SCAN_COMPLETE', manual_research_not_product=True,
        question='which single command frame, zeroed on its own, moves the collapse time',
        criterion_written_before_the_run=CRIT,
        why_not_a_new_version='run_closed_loop_first_v8.py is the frozen delivery script and its sha256 is pinned inside delivered reports; editing it would invalidate those pins. This probe reads the same frozen archive and perturbs one frame at a time, and it asserts that v8 is untouched (before/after sha256 identical).',
        v8_script=str(V8), v8_sha256_before=v8_sha_before, v8_sha256_after=v8_sha_after, v8_untouched=True,
        protocol=dict(dt_s=DT, steps=STEPS, window_s=STEPS * DT, command_interval_s=CMD * DT,
                      state_written_once=True, loop_writes='ctrl only', frames_scanned=list(FRAMES),
                      state=str(STATE), state_sha256=sha(STATE), drive=str(DRIVE), drive_sha256=sha(DRIVE)),
        baseline=base, arms=res, deltas_vs_baseline=deltas,
        baseline_reference=dict(first_nonfoot_step=2811, first_nonfoot_ms=70.275, legs_below_6_step=538, zero_tarsus_step=5082),
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
    print('v8 untouched:', v8_sha_before == v8_sha_after)
    return out


if __name__ == '__main__':
    main()
