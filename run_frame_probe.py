"""FRAME PROBE: which command frame triggers the collapse?

Owner-directed next step.  It does NOT change the closed-loop script and it is NOT a new model or a new
sensory channel: it re-steps the SAME frozen archive with single-frame perturbations.

Declared beforehand:
  criterion  = HALF-STEP HOLD = no first non-foot contact inside 0.5 s AND legs_in_contact == 6 throughout
  baseline   = open loop, archive row k as ctrl, first non-foot at step 2811 = 70.275 ms (A6_geom|floor),
               legs<6 at 538, zero tarsal at 5082
  perturbations, all on the OPEN (archive-driven) arm:
    frame2_zero   : frame 2 replaced by zeros
    frame14_zero  : frame 14 replaced by zeros
    frame2_swap   : frames 2 and 3 exchanged
    frame14_swap  : frames 14 and 15 exchanged
    frame0to20_zero : frames 0..20 zeroed (a wider bracket, for reference only)
  every arm runs dt = 2.5e-5 s, 20000 steps, state written once, only ctrl written inside the loop.
"""
import hashlib, json, os, pathlib, sys, time, zipfile
import numpy as np

ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
sys.path.insert(0, str(ROOT))
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
STATE = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259/equilibrium_best_state.npz'
DRIVE = ROOT / 'outputs/experiment_platform/20261015_135150_a2fff3f0/traces.npz'
DT = 2.5e-5
CMD = 200
STEPS = 20000
ARMS = ('baseline', 'frame2_zero', 'frame14_zero', 'frame2_swap', 'frame14_swap', 'frame0to20_zero')
CRIT = 'HALF-STEP HOLD = no first non-foot contact inside 0.5 s AND legs_in_contact == 6 throughout'


def main():
    import mujoco, importlib
    L = importlib.import_module('run_flybody_loop')
    body = L.build_body('canonical'); m = body['model']
    m.opt.timestep = DT
    assert m.opt.timestep == DT and abs(float(L.build_body('canonical')['model'].opt.timestep) - DT) > 0
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    st = np.load(STATE); arch = np.asarray(np.load(DRIVE)['normal_controls'], float)
    q0, act0 = st['qpos'].copy(), st['act'].copy()
    LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')

    def perturbed(arm):
        a = arch.copy()
        if arm == 'baseline': pass
        elif arm == 'frame2_zero': a[2] = 0.0
        elif arm == 'frame14_zero': a[14] = 0.0
        elif arm == 'frame2_swap': a[[2, 3]] = a[[3, 2]]
        elif arm == 'frame14_swap': a[[14, 15]] = a[[15, 14]]
        elif arm == 'frame0to20_zero': a[0:21] = 0.0
        return a

    def run(arm):
        a = perturbed(arm)
        d = mujoco.MjData(m)
        d.qpos[:] = q0; d.qvel[:] = 0.0; d.act[:] = act0; d.ctrl[:] = 0.0
        mujoco.mj_forward(m, d)
        body['reference'] = q0.copy()
        fnf = fl6 = fzt = None; pair = None
        zg = np.zeros(STEPS + 1); qg = np.zeros(STEPS + 1)
        tg = np.zeros(STEPS + 1); zg[0] = float(d.qpos[2])
        six = 0; rz_min = float(d.qpos[2]); rz_fin = rz_min; qm = 0.0; qs = 0
        ap = float(np.max(np.asarray(d.act))); ood = 0
        ctrl = []
        for k in range(STEPS // CMD):
            d.ctrl[:] = a[k]; ctrl.append(a[k].copy())
            for j in range(CMD):
                mujoco.mj_step(m, d)
                i = k * CMD + j + 1
                last = (j == CMD - 1)
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
                q = float(np.max(np.abs(np.asarray(d.qacc))))
                if q > qm: qm, qs = q, i
                a2 = float(np.max(np.abs(np.asarray(d.act)))); ap = max(ap, a2); ood += 1 if a2 > 1 else 0
                rz = float(d.qpos[2])
                if rz < rz_min: rz_min = rz
                rz_fin = rz
                tg[i] = float(d.time); zg[i] = rz; qg[i] = q
        return dict(arm=arm, first_nonfoot_step=fnf, first_nonfoot_ms=(None if fnf is None else round(fnf * DT * 1000, 6)),
                    first_nonfoot_pair=pair, first_legs_below_6_step=fl6, first_zero_tarsus_step=fzt,
                    root_z_min=rz_min, root_z_final=rz_fin, max_abs_qacc=qm, max_abs_qacc_step=qs,
                    act_peak=ap, act_out_of_domain_steps=ood, six_leg_steps=six, of=STEPS,
                    six_leg_fraction=round(six / STEPS, 4),
                    criterion_met=bool(fnf is None and six == STEPS),
                    ctrl_sha256=hashlib.sha256(np.asarray(ctrl).tobytes()).hexdigest()), tg, zg, qg

    res = {}; traces = {}
    for a in ARMS:
        r, tg, zg, qg = run(a); res[a] = r; traces[a] = (tg, zg, qg)
        print('%-16s first_nonfoot %-6s (%s ms) | legs<6 %-6s | zero_tarsus %-6s | root_z %.6f/%.6f | qacc %.6g@%s | six %d/%d | criterion %s'
              % (a, r['first_nonfoot_step'], r['first_nonfoot_ms'], r['first_legs_below_6_step'], r['first_zero_tarsus_step'],
                 r['root_z_min'], r['root_z_final'], r['max_abs_qacc'], r['max_abs_qacc_step'], r['six_leg_steps'], STEPS, r['criterion_met']), flush=True)

    base = res['baseline']
    deltas = {a: dict(
        first_nonfoot_step_delta=(None if (base['first_nonfoot_step'] is None or res[a]['first_nonfoot_step'] is None)
                                  else res[a]['first_nonfoot_step'] - base['first_nonfoot_step']),
        first_nonfoot_ms_delta=(None if (base['first_nonfoot_ms'] is None or res[a]['first_nonfoot_ms'] is None)
                                else round(res[a]['first_nonfoot_ms'] - base['first_nonfoot_ms'], 6)),
        legs_below_6_step_delta=(None if (base['first_legs_below_6_step'] is None or res[a]['first_legs_below_6_step'] is None)
                                 else res[a]['first_legs_below_6_step'] - base['first_legs_below_6_step']),
        zero_tarsus_step_delta=(None if (base['first_zero_tarsus_step'] is None or res[a]['first_zero_tarsus_step'] is None)
                                else res[a]['first_zero_tarsus_step'] - base['first_zero_tarsus_step']),
        six_leg_steps_delta=res[a]['six_leg_steps'] - base['six_leg_steps'],
        ctrl_differs_from_baseline=(res[a]['ctrl_sha256'] != base['ctrl_sha256'])) for a in ARMS if a != 'baseline'}
    for a, d_ in deltas.items():
        print('%-16s Δfirst_nonfoot %-8s (Δms %-10s) Δlegs<6 %-8s Δzero_tarsus %-8s Δsix %-7d ctrl differs %s'
              % (a, d_['first_nonfoot_step_delta'], d_['first_nonfoot_ms_delta'], d_['legs_below_6_step_delta'],
                 d_['zero_tarsus_step_delta'], d_['six_leg_steps_delta'], d_['ctrl_differs_from_baseline']), flush=True)

    out = DESK / ('S2-frame-probe-' + time.strftime('%Y%m%dT%H%M%S')); out.mkdir(exist_ok=False)
    rep = dict(status='FRAME_PROBE_COMPLETE', manual_research_not_product=True,
        question='which command frame triggers the collapse, by single-frame perturbation of the frozen archive',
        criterion_written_before_the_run=CRIT,
        why_not_a_new_version='this is not a new model, not a new sensory channel and not a new closed-loop script: it re-steps the SAME frozen archive with single-frame perturbations, as the owner directed',
        protocol=dict(dt_s=DT, steps=STEPS, window_s=STEPS * DT, command_interval_s=CMD * DT,
                      state_written_once=True, loop_writes='ctrl only',
                      state=str(STATE), state_sha256=sha(STATE), drive=str(DRIVE), drive_sha256=sha(DRIVE)),
        arms=[res[a] for a in ARMS], deltas_vs_baseline=deltas,
        baseline_reference=dict(first_nonfoot_step=2811, first_nonfoot_ms=70.275, legs_below_6_step=538, zero_tarsus_step=5082),
        frame_index_of_the_two_events=dict(legs_below_6_frame=base['first_legs_below_6_step'] // CMD if base['first_legs_below_6_step'] else None,
                                           first_nonfoot_frame=base['first_nonfoot_step'] // CMD if base['first_nonfoot_step'] else None,
                                           note='frame index = step // 200, so 538 -> frame 2 and 2811 -> frame 14'),
        not_established='absolute force values, physical units, biological validity; no standing, no walking, no completion percentage',
        recorded_at=time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)'))
    (out / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    np.savez_compressed(out / 'frame_probe_traces.npz',
                        **{a + '_t_step_s': traces[a][0] for a in ARMS},
                        **{a + '_z_step': traces[a][1] for a in ARMS},
                        **{a + '_qacc_step': traces[a][2] for a in ARMS})
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 5.6))
    for a in ARMS:
        ax[0].plot(traces[a][0], traces[a][1], lw=1.2, label=a)
        ax[1].plot(traces[a][0], traces[a][2], lw=1.0, label=a)
    for a, col in (('baseline', 'k'),):
        ax[0].axvline(base['first_nonfoot_step'] * DT, color=col, ls='--', lw=1.0)
        ax[0].axvline(base['first_legs_below_6_step'] * DT, color=col, ls=':', lw=1.0)
    ax[0].set_xlabel('measured d.time [s]'); ax[0].set_ylabel('root z [model unit; NOT_ESTABLISHED]')
    ax[0].set_title('root z: baseline vs single-frame perturbations', fontsize=10)
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7)
    ax[1].set_yscale('log'); ax[1].set_xlabel('measured d.time [s]'); ax[1].set_ylabel('|qacc| generalised, log')
    ax[1].set_title('|qacc| (semi-log): where the perturbed runs separate', fontsize=10)
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=7)
    fig.suptitle('MANUAL RESEARCH IMPLEMENTATION, NOT A PRODUCT  |  single-frame perturbation of the frozen archive\n'
                 'dt 2.5e-5 s asserted  |  window 0.500 s = 20000 steps  |  x-axis = measured d.time  |  no standing, no walking, no completion percentage',
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(out / 'frame-probe.png', dpi=130); plt.close(fig)
    src = pathlib.Path(__file__)
    with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(src, src.name)
    with zipfile.ZipFile(out / 'source_snapshot.zip') as z:
        assert z.namelist() == [src.name] and hashlib.sha256(z.read(src.name)).hexdigest() == sha(src)
    fp = out / 'FROZEN_DELIVERY_HASHES.json'; fp.write_text('{}', encoding='utf-8')
    (out / 'delivery-hashes.json').write_text('{}', encoding='utf-8')
    listing = sorted(p.name for p in out.iterdir() if p.is_file())
    data = [f for f in listing if f not in ('FROZEN_DELIVERY_HASHES.json', 'delivery-hashes.json')]
    fp.write_text(json.dumps(dict(status='FROZEN_BEFORE_VERIFICATION', frozen_at=time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)'),
        files={f: sha(out / f) for f in data}, covers_rule='len(files) == len(os.listdir) - 2',
        covers_self=False, manifest_sha256=None,
        manifest_sha256_note='null by definition; a manifest cannot carry its own digest',
        template_id='FROZEN_TEMPLATE_v13_owner_turn6'), ensure_ascii=False, indent=1), encoding='utf-8')
    dhd = {f: sha(out / f) for f in listing if f != 'delivery-hashes.json'}
    (out / 'delivery-hashes.json').write_text(json.dumps(dhd, ensure_ascii=False, indent=1), encoding='utf-8')
    assert json.loads((out / 'delivery-hashes.json').read_text())['FROZEN_DELIVERY_HASHES.json'] == sha(fp)
    for p in sorted(out.iterdir()):
        if p.is_file(): os.chmod(p, 0o444)
    os.chmod(out, 0o555)
    print('OUT:', out)
    return out


if __name__ == '__main__':
    main()
