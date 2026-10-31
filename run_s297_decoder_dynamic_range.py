"""S297/P3 decoder dynamic-range recalibration: preserve each channel's mean command, restore AC.

Stage 1 (offline, zero dynamic calls): quantify clipping and modulation delivery from the saved
S289/S292 arrays and derive the fixed k table.
Stage 2 (2 declared arms, 400 ms each): original_mask reproduction gate, then decoder_ac_gain
which applies cmd = clip(mean_post + k*(post - mean_post), 0, 1) per channel.
Nothing else changes: 450 Hz, 5 ms filter, sensory gain 1, causal carry, V1 gains, XML, gear,
mass, gravity, timestep and limits stay untouched; the 26 zero-gain channels stay exactly zero.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import types

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
ARCH = Path('/home/sensen/Desktop/engine修复验收')
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
S289 = ARCH / 's289-decoder-range-support/20261026T173400Z/decoder_original'
S292 = ARCH / 's292-current-sample-two-channel-calibration/20261026T181800Z/original_mask'
S295 = ARCH / 's295-phased-lift-probe/20261026T192901Z'
BASE = ROOT / 'run_s291_two_channel_recalibration.py'
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
K_MAX = 6.0
TARGET_SPAN = 0.90
WINDOW = (20, 400)
ARMS = ('original_mask', 'decoder_ac_gain')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, v): Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, default=float) + '\n', encoding='utf-8')


def spectrum(x, dt=.001):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x); f = np.fft.rfftfreq(n, d=dt); p = np.abs(np.fft.rfft(x)) ** 2
    tot = p[1:].sum()
    lo = p[(f > 0) & (f < 8)].sum(); hi = p[(f >= 8) & (f <= 25)].sum()
    band = (f >= 8) & (f <= 25)
    return dict(resolution_hz=float(f[1] - f[0]), band_bins=int(band.sum()),
                peak_hz=float(f[band][int(np.argmax(p[band]))]),
                band_power_fraction=float(hi / tot) if tot > 0 else None,
                sub8_fraction=float(lo / tot) if tot > 0 else None,
                prominence_8_25=float(p[band].max() / max(np.median(p[band]), 1e-30)))


def offline(out):
    ne = np.load(S289 / 'NEURAL.npz')
    pre = np.asarray(ne['decoder_preclip'], float); post = np.asarray(ne['decoder_postclip'], float)
    V = np.asarray(ne['decoder_motor_gains'], float); rates = np.asarray(ne['rates_hz'], float)
    assert pre.shape == post.shape == rates.shape == (400, 90)
    nz = np.flatnonzero(V != 0)
    assert len(nz) == 64
    t = np.arange(1, 401)
    m20 = (t >= WINDOW[0]) & (t < WINDOW[1])
    rows = {}
    for i in range(90):
        if V[i] == 0:
            rows[i] = dict(index=i, gain=0.0, zero_gain=True, mean_preclip_20_400=0.0, std_preclip_20_400=0.0,
                           mean_postclip_20_400=0.0, std_postclip_20_400=0.0, clip_fraction=0.0,
                           amplitude_lost_fraction=None, MI=None, delivered=None, k=1.0, p1=None, p99=None,
                           verified_zero=bool(not np.any(pre[:, i]) and not np.any(post[:, i])))
            continue
        p = pre[m20, i]; q = post[m20, i]
        clip_fraction = float(np.mean(p >= 1.0))
        k = 1.0; p1 = p99 = None
        if p.size >= 10:
            p1, p99 = float(np.percentile(p, 1)), float(np.percentile(p, 99))
            if p99 - p1 > 0:
                k = float(min(K_MAX, TARGET_SPAN / (p99 - p1)))
        rows[i] = dict(index=i, gain=float(V[i]), zero_gain=False,
                       mean_preclip_20_400=float(p.mean()), std_preclip_20_400=float(p.std()),
                       mean_postclip_20_400=float(q.mean()), std_postclip_20_400=float(q.std()),
                       clip_fraction=clip_fraction,
                       amplitude_lost_fraction=float(1 - q.std() / p.std()) if p.std() > 0 else None,
                       MI=float(p.std() / p.mean()) if p.mean() > 0 else None,
                       delivered=float(q.std() / p.std()) if p.std() > 0 else None,
                       k=k, p1=p1, p99=p99, verified_zero=False)
        std_cand = float(np.std(np.clip(q.mean() + k * (q - q.mean()), 0, 1)))
        rows[i]['predicted_std_candidate'] = std_cand
        rows[i]['predicted_delivered'] = float(std_cand / p.std()) if p.std() > 0 else None
        rows[i]['predicted_clip_fraction'] = float(np.mean((q.mean() + k * (q - q.mean()) >= 1) | (q.mean() + k * (q - q.mean()) <= 0)))
        rows[i]['spectrum_preclip'] = spectrum(pre[:, i])
        rows[i]['spectrum_candidate'] = spectrum(np.clip(post[:, i].mean() + k * (post[:, i] - post[:, i].mean()), 0, 1))
    zeros = [i for i in range(90) if rows[i]['zero_gain']]
    assert all(rows[i]['verified_zero'] for i in zeros) and len(zeros) == 26
    groups = json.loads((S295 / 'DIRECTION_GROUPS.json').read_text())['legs']
    grp = {}
    for leg in LEGS:
        for kind in ('minus', 'plus'):
            idx = groups[leg][kind]
            cand = np.asarray([np.clip(post[:, i].mean() + rows[i]['k'] * (post[:, i] - post[:, i].mean()), 0, 1) for i in idx])
            grp[f'{leg}_{kind}'] = dict(members=idx, mean_L_or_P_base=float(post[:, idx].mean()),
                                        std_base=float(post[:, idx].mean(axis=1).std()),
                                        mean_after=float(cand.mean()), std_after=float(cand.mean(axis=0).std()),
                                        mean_preclip=float(pre[:, idx].mean()),
                                        clip_fraction_preclip=float((pre[:, idx] >= 1).mean()))
    doc = dict(round='S297', stage='offline modulation audit',
               source_neural=str(S289 / 'NEURAL.npz'), window_ms=WINDOW, k_rule=f'k_i = min({K_MAX}, {TARGET_SPAN}/(p99_i - p1_i)) on preclip over 20-400 ms; k_i=1 if p99-p1<=0',
               command_definition='cmd_i(t) = clip(mean_post_i + k_i*(post_i(t) - mean_post_i), 0, 1); mean_post_i is the 20-400 ms baseline postclip mean, so the average drive per channel is preserved',
               channels=rows, zero_gain_channels=zeros, groups=grp,
               summary=dict(channels_touching_ceiling=int(sum(1 for i in nz if np.any(pre[:, i] >= 1))),
                            pooled_clip_fraction=float(np.mean(pre[:, nz] >= 1)),
                            median_channel_clip_fraction=float(np.median([rows[i]['clip_fraction'] for i in nz])),
                            max_channel_clip_fraction=float(max(rows[i]['clip_fraction'] for i in nz)),
                            channels_mean_preclip_above_1=int(sum(1 for i in nz if rows[i]['mean_preclip_20_400'] > 1)),
                            std_preclip_nz=float(pre[:, nz].std()), std_postclip_nz=float(post[:, nz].std()),
                            delivered_median=float(np.median([rows[i]['delivered'] for i in nz])),
                            predicted_delivered_median=float(np.median([rows[i]['predicted_delivered'] for i in nz])),
                            k_median=float(np.median([rows[i]['k'] for i in nz])),
                            k_at_max=int(sum(1 for i in nz if rows[i]['k'] >= K_MAX)),
                            predicted_pooled_clip_fraction=float(np.mean([rows[i]['predicted_clip_fraction'] for i in nz])),
                            rate_spectrum_note='400 samples at 1 ms give 2.5 Hz resolution and 7 bins in 8-25 Hz',
                            rate_band_fraction_median=float(np.median([spectrum(rates[:, i])['band_power_fraction'] for i in nz])),
                            rate_narrowband_prominence_median=float(np.median([spectrum(rates[:, i])['prominence_8_25'] for i in nz])),
                            rate_fast_over_slow_median=float(np.median([spectrum(rates[:, i])['band_power_fraction'] / max(1 - spectrum(rates[:, i])['band_power_fraction'], 1e-12) for i in nz]))))
    dump(out / 'MODULATION.json', doc)
    with (out / 'AC_GAIN_TABLE.csv').open('w', encoding='utf-8') as f:
        f.write('index,gain,p1,p99,k,mean_post,mean_preclip,clip_fraction,delivered,predicted_delivered,zero_gain\n')
        for i in range(90):
            r = rows[i]
            f.write(f"{i},{r['gain']},{r['p1']},{r['p99']},{r['k']},{r['mean_postclip_20_400']},{r['mean_preclip_20_400']},"
                    f"{r['clip_fraction']},{r['delivered']},{r.get('predicted_delivered')},{int(r['zero_gain'])}\n")
    dump(out / 'AC_TABLE.json', dict(k=[rows[i]['k'] for i in range(90)], mean_post=[rows[i]['mean_postclip_20_400'] for i in range(90)],
                                     zero_gain=zeros, k_max=K_MAX, target_span=TARGET_SPAN, window_ms=WINDOW))
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--prereg-only', action='store_true', help='write PREREG and the offline stage, no dynamics')
    args = ap.parse_args(); out = args.output
    inputs = [Path(__file__).resolve(), BASE, XML, S289 / 'NEURAL.npz', S289 / 'COMMAND_TIMING.npz',
              S292 / 'NEURAL.npz', S292 / 'COMMAND_TIMING.npz', S295 / 'DIRECTION_GROUPS.json',
              ROOT / 'engine/embodied/motor_output.py']
    prereg = dict(
        round='S297', mode='open-loop engineering diagnostic: decoder interface parameter only, not a controller, not a neural circuit change',
        arms=list(ARMS), duration_ms=400, model_loads=1,
        reset_semantics='one model load; each arm resets qpos/qvel/act to the saved S292 t=0 state',
        decoder=dict(rate_for_full_control_hz=450., filter_tau_s=.005, max_control=1., per_actuator_gain='unchanged V1 including 26 zeros',
                     base_formula='preclip=(rates_hz/450)*V1; postclip=clip(preclip,0,1)',
                     ac_formula='cmd_i(t)=clip(mean_post_i + k_i*(post_i(t)-mean_post_i),0,1); only the AC gain changes, every channel mean is preserved',
                     k_rule=f'k_i=min({K_MAX},{TARGET_SPAN}/(p99_i-p1_i)) on preclip over {WINDOW[0]}-{WINDOW[1]} ms; k_i=1 if p99-p1<=0',
                     k_source='S289 decoder_original saved arrays, identical to S292 original_mask'),
        gates=dict(original='all shared S289 arrays/JSON/CSV exact before the candidate runs',
                   fidelity='replay the saved candidate ctrl for 500 steps and require identical qpos',
                   invariants='26 zero-gain channels remain exactly zero; per-channel 20-400 ms command mean within 1e-9 of baseline'),
        criteria=dict(delivered='median std(candidate command)/std(preclip) change',
                      gait='S293 five criteria reported unchanged in wording: swing phase, tripod share, 40-80 ms period, forward displacement, height maintenance',
                      support='sum(Fz)/(m g) windows 0-150, own pre-contact 20 ms, 350-400 ms'),
        decision_rules=dict(
            rule1='delivered median >=2x AND all five gait criteria still false -> interface no longer swallows modulation; the remaining gap is that the neural layer has no rhythm',
            rule2='0-150 ms support leaves [0.9,1.1], or COM sinks more, or first non-foot contact comes clearly earlier -> the constant-max co-activation is part of load bearing; next round must preserve bearing before any rhythm work',
            rule3='k at k_max for >50% of channels AND delivered median gain <1.5x -> upstream modulation is too small relative to its own band; the interface cannot repair it',
            rule4='>=2 of the five criteria flip from false to true -> interface throughput is a partial bottleneck, with the exact numbers and windows for those two and the other three still false'),
        prohibitions=['no gain sweep or k scan', 'no change to the 26 zero gains or to indices 55/62', 'no XML, gear, mass, gravity, timestep, limit edit',
                      'no new synapse, mapping or external rhythm', 'no support/walking claim'],
        input_hashes={str(p): sha(p) for p in inputs if Path(p).is_file()},
        missing_inputs=[str(p) for p in inputs if not Path(p).is_file()],
    )
    if args.prereg_only:
        assert not out.exists(), 'fresh output required'
        out.mkdir(parents=True); dump(out / 'PREREG.json', prereg); (out / 'S297_PREREG_FROZEN').write_text('frozen before dynamics\n')
        doc = offline(out)
        s = doc['summary']
        # append the frozen offline numbers into PREREG-adjacent record
        dump(out / 'OFFLINE_SUMMARY.json', s)
        print(json.dumps(dict(prereg=str(out / 'PREREG.json'), summary=s), ensure_ascii=False)); return 0
    existing = json.loads((out / 'PREREG.json').read_text())
    assert existing['input_hashes'][str(Path(__file__).resolve())] == sha(__file__), 'runner changed after prereg'
    table = json.loads((out / 'AC_TABLE.json').read_text())
    k = np.asarray(table['k'], float); mean_post = np.asarray(table['mean_post'], float)
    source = BASE.read_text(encoding='utf-8')
    assert source.count("run('recalibrated_55_62')") == 1
    p1 = "assert np.array_equal(carry,np.clip(preclip,0,dec.config.max_control));decoder_preclips.append(preclip.copy());decoder_postclips.append(carry.copy())"
    p1n = ("raw=np.clip(preclip,0,dec.config.max_control);assert np.array_equal(carry,raw);decoder_preclips.append(preclip.copy());"
           "decoder_postclips.append(raw.copy());carry=_s297_ac(carry);decoder_ac.append(carry.copy())")
    p3 = "if not out.is_absolute() or out.exists():raise ValueError('fresh absolute --output required')"
    p3n = ("if not out.is_absolute():raise ValueError('absolute --output required')\n"
           "    if out.exists() and not (out/'S297_PREREG_FROZEN').exists():raise ValueError('fresh output required')")
    p4 = "out.mkdir(parents=True,exist_ok=False);dump(out/'PREREG.json',prereg)"
    p4n = "out.mkdir(parents=True,exist_ok=True);dump(out/'PREREG.json',prereg)"
    p2 = "decoder_postclip=np.asarray(decoder_postclips),"
    p2n = "decoder_postclip=np.asarray(decoder_postclips),decoder_ac_applied=np.asarray(decoder_ac),"
    assert source.count(p1) == 1 and source.count(p2) == 1 and source.count(p3) == 1 and source.count(p4) == 1, 'S291 anchor lines not found exactly once'
    src = source.replace('S291', 'S297').replace('recalibrated_55_62', 'decoder_ac_gain').replace(p1, p1n).replace(p2, p2n).replace(p3, p3n).replace(p4, p4n)
    runtime = types.ModuleType('s297_runtime'); runtime.__file__ = str(Path(__file__).resolve())
    exec(compile(src, str(BASE), 'exec'), runtime.__dict__)
    state = {'arm': None, 'k': k, 'mean': mean_post}
    runtime.S297_STATE = state
    runtime.decoder_ac = []

    def ac(carry):
        if state['arm'] != 'decoder_ac_gain':
            return carry
        return np.clip(state['mean'] + state['k'] * (np.asarray(carry, float) - state['mean']), 0., 1.)

    def provide(path, old, names):
        doc = json.loads((out / 'AC_TABLE.json').read_text())
        rows = {i: dict(index=i, actuator=names[i], gain_new=float(old[i]),
                        note='gain unchanged by design; the only change is the AC transform') for i in (55, 62)}
        return dict(passed=True, channels=rows, source_hashes={str(p): sha(p) for p in (out / 'AC_TABLE.json', out / 'MODULATION.json')},
                    ac_table=str(out / 'AC_TABLE.json'), note='S297 supplies no new gains'), rows, np.asarray(old, float).copy()

    runtime._s297_ac = ac
    runtime.read_derivation = provide
    original_dump = runtime.dump
    calls_seen = {}

    def dump_hook(path, value):
        path = Path(path)
        if path.name == 'DECODER_CONFIG.json':
            state['arm'] = path.parent.name; runtime.decoder_ac.clear()
            if state['arm'] == 'decoder_ac_gain':
                value['ac_gain_applied'] = True; value['ac_formula'] = 'clip(mean_post + k*(post-mean_post),0,1)'
                value['k_policy'] = 'fixed table from MODULATION.json; not swept'
            value['changed_gain_indices'] = []
        if path.name == 'DERIVATION.json':
            path = path.with_name('AC_PROVIDER.json')
        if path.name == 'PREREG.json' and existing is not None:
            # keep the S297 prereg frozen; record the runtime-shaped one separately
            path = path.with_name('PREREG_RUNTIME.json')
        original_dump(path, value)

    runtime.dump = dump_hook
    rc = runtime.main(argv[1:] if argv else ['--output', str(out)])
    dump(out / 'S297_RUNTIME_RETURN.json', dict(return_code=int(rc), arms=list(ARMS)))
    return int(rc)


if __name__ == '__main__':
    raise SystemExit(main())
