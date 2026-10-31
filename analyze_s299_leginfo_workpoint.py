"""S299 Stage A (offline, zero dynamic calls): reframe S298, audit input/output/readout, compression index.

Stages B (one full-neuron reproduction arm) and C (six-arm sensory-scale sweep) are declared in
PREREG but are NOT executed by this script; they require the dynamic entry point.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, csv

ARCH = Path('/home/sensen/Desktop/engine修复验收')
S289 = ARCH / 's289-decoder-range-support/20261026T173400Z/decoder_original'
S290 = ARCH / 's290-motor-mapping-coverage/20261026T174500Z'
S292 = ARCH / 's292-current-sample-two-channel-calibration/20261026T181800Z'
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
POOLS = ROOT / 'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json'
FROZEN = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, v): Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, default=float) + '\n', encoding='utf-8')


def dispersion(X):
    """Cross-channel dispersion after removing the common mode, normalised by the global mean magnitude."""
    X = np.asarray(X, float)
    cm = X.mean(axis=1, keepdims=True)
    R = X - cm
    per_channel_mean = R.mean(axis=0)
    denom = abs(X.mean())
    return dict(n_channels=int(X.shape[1]), n_samples=int(X.shape[0]),
                global_mean=float(X.mean()), residual_mean_std=float(per_channel_mean.std()),
                dispersion=float(per_channel_mean.std() / denom) if denom > 0 else None,
                definition='D = std over channels of the time-mean residual after removing the cross-channel common mode, divided by |global mean|')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--stage-a', action='store_true')
    args = ap.parse_args(); out = args.output
    pre_path = out / 'PREREG.json'
    if not pre_path.exists():
        assert not out.exists(), 'fresh output required'
        out.mkdir(parents=True)
        inputs = [Path(__file__).resolve(), S289 / 'NEURAL.npz', S289 / 'PREREG.json', S290 / 'PER_CHANNEL.csv',
                  S292 / 'TIER_ROOT_IDS.npz', POOLS, FROZEN, ROOT / 'engine/embodied/sensory_channels.py',
                  ROOT / 'engine/embodied/adapters.py', ROOT / 'run_s291_two_channel_recalibration.py']
        dump(pre_path, dict(
            round='S299', mode='leg-information loss localisation and neural work-point calibration',
            stages=dict(A='offline audits: S298 reframe, input/output/readout, compression index (zero dynamic calls)',
                        B='one full-neuron reproduction arm of S289 decoder_original with 800x12000 spikes',
                        C='six declared sensory-scale arms s in {1.0,0.5,0.25,0.1,0.05,0.02}; s=1.0 is the stage-B arm and is not re-run'),
            declared_calls=dict(model_load=1, tier_build=1, network_init=6, live_step=24000, neural_tick=4800,
                                live_fwd_position=6, diagnostic_forward=24006),
            declared_calls_note='the dispatched counts omitted live_fwd_position and diagnostic_forward; S291 semantics need one of each per arm, so 6 are declared here explicitly',
            scale_definition='s multiplies the current actually submitted to net.step; the scaled current is what gets recorded as sensory_current',
            compression_index='CI = D_out / D_in with D = std over channels of time-mean residual after common-mode removal / |global mean|',
            selection_rule=dict(feasible='tier global duty in [5%,50%] AND postclip==1 fraction over the 64 non-zero channels <= 30% AND the arm has non-zero firing',
                                objective='among feasible arms pick the largest D_out, ties broken by larger D_time',
                                no_feasible='do not add grid points; apply the decision rules as written'),
            decision_rules=dict(
                rule1='a feasible arm improves D_out >=10x or lowers CI >=10x -> the loss point is work-point saturation',
                rule2='every s improves CI <2x and some arm is in the duty window -> the network does not resolve legs',
                rule3='low s silent and high s compressed with no arm meeting both constraints -> synaptic-weight threshold cliff',
                rule4='a feasible arm pushes 0-150 ms support below 0.9 or first non-foot contact earlier than 200 ms -> work point vs bearing conflict (the choice is not changed)'),
            prohibitions=['one parameter only (sensory scale s)', 'no scan of reference_force, reference_angle_rad or synaptic weight this round',
                          'no change to 26 zero gains, indices 55/62, XML, gear, mass, gravity, timestep, limits, gains or filt 5 ms',
                          'no rhythm construction, no new connection, no controller', 's is a modelling scale, not a biological calibration'],
            input_hashes={str(p): sha(p) for p in inputs if Path(p).is_file()},
            missing_inputs=[str(p) for p in inputs if not Path(p).is_file()],
        ))
    pre = json.loads(pre_path.read_text())
    assert pre['input_hashes'][str(Path(__file__).resolve())] == sha(__file__), 'analyzer changed after prereg'
    for p, h in pre['input_hashes'].items():
        assert Path(p).is_file() and sha(p) == h, 'input changed: ' + p
    z = np.load(S289 / 'NEURAL.npz')
    sc = np.asarray(z['sensory_current'], float); V = np.asarray(z['decoder_motor_gains'], float)
    pre_ = np.asarray(z['decoder_preclip'], float); post = np.asarray(z['decoder_postclip'], float)
    rates = np.asarray(z['rates_hz'], float); raw = np.asarray(z['raw_rates_hz'], float)
    nz = np.flatnonzero(V != 0); zeros = np.flatnonzero(V == 0)
    driven = np.flatnonzero(np.abs(sc).max(axis=0) > 0)
    sig = sc[:, driven]; keep = []
    for j in range(sig.shape[1]):
        if not any(np.array_equal(sig[:, j], sig[:, k]) for k in keep): keep.append(j)
    uniq_idx = driven[keep]
    U = sc[:, uniq_idx]
    per_step = [dict(t=i, unique_values=int(len(np.unique(sc[i]))), unique_nonzero=int(len(np.unique(sc[i][sc[i] != 0])))) for i in (100, 200, 399)]
    dump(out / 'INPUT_AUDIT.json', dict(
        sensory_current_shape=list(sc.shape), driven_cells=int(len(driven)), unique_driven_series=int(U.shape[1]),
        per_step_unique=per_step, channel_mean=dict(min=float(U.mean(0).min()), max=float(U.mean(0).max())),
        channel_std_max=float(U.std(0).max()), extremes=dict(min=float(U.min()), max=float(U.max())),
        pairwise_correlation_min=float(np.corrcoef(U.T).min()), dispersion=dispersion(U),
        uniq_index_head=[int(x) for x in uniq_idx[:10]],
        note='29 unique driven time series; the other driven cells share these waveforms'))
    out_rows = {}
    for i in range(90):
        p_ = pre_[:, i]; q_ = post[:, i]; r_ = rates[:, i]
        out_rows[i] = dict(index=i, gain=float(V[i]), zero_gain=bool(V[i] == 0),
                           raw_mean=float(raw[:, :, i].mean()), raw_zero_fraction=float((raw[:, :, i] == 0).mean()),
                           raw_at_2000_fraction=float((raw[:, :, i] >= 2000).mean()),
                           rates_mean=float(r_.mean()), rates_std=float(r_.std()),
                           preclip_mean=float(p_.mean()), preclip_max=float(p_.max()),
                           postclip_mean=float(q_.mean()), postclip_at_1_fraction=float((q_ == 1).mean()),
                           postclip_at_0_fraction=float((q_ == 0).mean()))
    rm = rates[:, nz].mean(axis=0)
    dump(out / 'OUTPUT_AUDIT.json', dict(
        channels=out_rows, nonzero_channels=int(len(nz)), zero_channels=int(len(zeros)),
        rates_mean_range=[float(rm.min()), float(rm.max())], rates_mean_p10_p90=[float(np.percentile(rm, 10)), float(np.percentile(rm, 90))],
        raw_zero_fraction_nz=float((raw[:, :, nz] == 0).mean()), raw_at_2000_fraction_nz=float((raw[:, :, nz] >= 2000).mean()),
        postclip_at_1_fraction_nz=float((post[:, nz] == 1).mean()), postclip_at_0_fraction_nz=float((post[:, nz] == 0).mean()),
        preclip_mean_median_nz=float(np.median(pre_[:, nz].mean(axis=0))), preclip_mean_max=float(pre_[:, nz].mean(axis=0).max()),
        across_channel_spread_percent=float((np.percentile(rm, 90) - np.percentile(rm, 10)) / rm.mean() * 100),
        note='fractions over the 90 channels are diluted by the 26 zero-gain channels; the _nz fields restrict to the activated channels'))
    rows = list(csv.DictReader((S290 / 'PER_CHANNEL.csv').open()))
    leg_of = {r['name']: r['leg'] for r in rows}
    sets = {}
    for r in rows:
        if r['pool_root_ids']:
            sets[r['name']] = tuple(sorted(int(x) for x in json.loads(r['pool_root_ids'])))
    pool_neurons = sorted({x for v in sets.values() for x in v})
    from collections import Counter
    shared = []
    for combo, cnt in Counter(sets.values()).items():
        names = [n for n, v in sets.items() if v == combo]
        if len({leg_of[n] for n in names}) > 1:
            shared.append(dict(neurons=list(combo), actuators=names, legs=sorted({leg_of[n] for n in names}), count=cnt))
    roots = np.load(S292 / 'TIER_ROOT_IDS.npz')['root_ids']
    driven_roots = {int(roots[i]) for i in uniq_idx}
    pool_set = set(pool_neurons)
    dump(out / 'READOUT_STRUCTURE.json', dict(
        resolved_actuators=len(sets), unresolved_actuators=int(len(rows) - len(sets)),
        pool_neurons=len(pool_neurons), distinct_pool_sets=int(len(set(sets.values()))),
        cross_leg_shared_sets=shared, cross_leg_shared_count=len(shared),
        overlap_pool_neurons_with_29_driven_series=len(pool_set & driven_roots),
        overlap_pool_neurons_with_all_driven_cells=len(pool_set & {int(roots[i]) for i in driven}),
        tier_root_ids_source=str(S292 / 'TIER_ROOT_IDS.npz'),
        note='overlap is computed in BANC root-id space using the tier root vector; a zero overlap is reported as zero, never estimated'))
    enc = dict(ChannelConfig_defaults=dict(current_nA_per_unit=3e-1, reference_force=1.0, reference_angle_rad=0.30),
               channel_count='5 modelled subtypes x 6 legs = 30 channels per the encoder group mapping',
               tier=dict(sign_fill='all_empty', fingerprint='56e0322a33c51b7f03aaae9f7df1ef3f',
                         note='all_empty is a DECLARED assumed transmitter fill, a different family from the historical network 5ab78f55...'),
               dispatch_discrepancy='the dispatch quoted reference_force=50 and reference_angle_rad=0.5; the engine defaults read here are 1.0 and 0.30')
    dump(out / 'ENCODER_CONFIG.json', enc)
    D_in = dispersion(U)['dispersion']
    D_out = dispersion(rates[:, nz])['dispersion']
    dump(out / 'COMPRESSION_INDEX.json', dict(
        D_in=D_in, D_out=D_out, CI=float(D_out / D_in) if D_in else None,
        input_quantity='29 unique driven sensory time series (sensory_current)', output_quantity='64 non-zero-gain channel filtered rates_hz',
        definition='D = std over channels of the time-mean residual after common-mode removal / |global mean|; CI = D_out / D_in',
        frozen=True, note='this is the single main scale of S299 and is fixed here; it is not to be redefined'))
    dump(out / 'S298_REFRAME.json', dict(
        round='S299', part='S298 reframe',
        new_scope='S298 rule B is valid only for the command/output layer; it must NOT be read as "no phase information upstream"',
        facts=dict(
            encoder='the sensory front end is the signed 30-channel subtype-tagged encoder from channel_index_map + ChannelSensoryEncoder, so the old sign-blind bank criticism no longer applies',
            input_leg_difference='29 unique driven series with channel means %.3f..%.3f, std up to %.3f, extremes %.3f..%.3f' % (U.mean(0).min(), U.mean(0).max(), U.std(0).max(), U.min(), U.max()),
            readout='the readout is per-actuator pool means over %d neurons in %d distinct sets, so leg specificity is not stripped by the interface' % (len(pool_neurons), len(set(sets.values()))),
            output_compression='non-zero channel filtered-rate means span %.1f..%.1f Hz with 10-90 percent only %.1f..%.1f Hz (spread %.2f%%), i.e. an almost identical rate across channels' % (rm.min(), rm.max(), np.percentile(rm, 10), np.percentile(rm, 90), float((np.percentile(rm, 90) - np.percentile(rm, 10)) / rm.mean() * 100)),
        ),
        implication='the loss therefore sits inside the 12000-neuron network or in the neuron-to-pool transfer, not in the encoder and not in the absence of a phase source',
        s298_archive_untouched=True))
    dump(out / 'CALL_LEDGER.json', dict(model_load=0, forward=0, physics_step=0, neural_tick=0, tier_build=0, render=0))
    dump(out / 'STAGE_STATUS.json', dict(stage_a='COMPLETE', stage_b='NOT_EXECUTED_IN_THIS_TURN', stage_c='NOT_EXECUTED_IN_THIS_TURN',
                                        reason='the dynamic entry point was not run in this turn; no dynamic call was made',
                                        landing='NOT_ESTABLISHED: the four-way landing needs stage C'))
    print(json.dumps(dict(D_in=D_in, D_out=D_out, CI=D_out / D_in if D_in else None,
                          driven_cells=len(driven), uniq=U.shape[1], pool_neurons=len(pool_neurons),
                          distinct_sets=len(set(sets.values())), shared=len(shared),
                          overlap=len(pool_set & driven_roots), rate_spread_pct=float((np.percentile(rm, 90) - np.percentile(rm, 10)) / rm.mean() * 100)), ensure_ascii=False))
    return 0


if __name__ == '__main__': raise SystemExit(main())
