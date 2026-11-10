"""Intrinsic-dynamics comparison: frozen 20k control vs whole strict BANC module.

Both circuits start from rest and are driven by the SAME 556-port recorded sensory
currents for a 500 ms warm-up; the following 2000 ms are analysed with identical
metrics to rounds 11-12. No physics, no learning, no descending current.
"""
from pathlib import Path
import hashlib, importlib.util, json, sys, time
import numpy as np
HERE = Path(__file__).resolve().parent
CAND = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface/LOCAL_MODULE_CANDIDATE.npz')
INPUTS = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-descending-learning/20261107T203049Z/descending/INPUTS.npz')
POOLS = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-descending-learning/20261107T203049Z/leg-pools/POOLS_AND_REACH.npz')
LEGS = ['LF', 'LH', 'LM', 'RF', 'RH', 'RM']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def analyse(rates, leg_ids):
    series = np.stack([rates[:, leg_ids == L.lower()].mean(1) for L in LEGS], 1)
    win = series[100:] - series[100:].mean(0)
    U, S, Vt = np.linalg.svd(win, full_matrices=False)
    frac = S ** 2 / np.sum(S ** 2)
    C = np.corrcoef(series.T)
    A = series[:, [0, 5, 1]].mean(1)
    B = series[:, [3, 2, 4]].mean(1)
    return dict(per_leg_mean_hz={L: float(series[:, i].mean()) for i, L in enumerate(LEGS)},
                per_leg_cv={L: float(series[:, i].std() / series[:, i].mean()) if series[:, i].mean() > 0 else None for i, L in enumerate(LEGS)},
                tripod_alternation_score=float(-np.corrcoef(A, B)[0, 1]),
                ipsilateral_score=float(np.corrcoef(series[:, [0, 1, 2]].mean(1), series[:, [3, 4, 5]].mean(1))[0, 1]),
                lf_rh_correlation=float(C[0, 4]), rf_lh_correlation=float(C[3, 1]),
                lf_rf_correlation=float(C[0, 3]), lh_rh_correlation=float(C[1, 4]),
                group_mean_hz={'A_LF_RM_LH': float(A.mean()), 'B_RF_LM_RH': float(B.mean())},
                PC1_variance_fraction=float(frac[0]), PC2_variance_fraction=float(frac[1]),
                PC1_loadings={L: float(v) for L, v in zip(LEGS, Vt[0])},
                PC2_loadings={L: float(v) for L, v in zip(LEGS, Vt[1])},
                all_leg_mean_std_hz=float(series.mean(1).std()))


def main():
    mw = load('mw', HERE / 'module_wide.py')
    cand = np.load(CAND)
    inputs = np.load(INPUTS)
    pools = np.load(POOLS)
    sensory_cache = np.asarray(cand['cache_indices'])[np.asarray(inputs['sensory_local'])]
    mn_cache = np.asarray(cand['cache_indices'])[np.asarray(pools['mn_local'])]
    leg_names = np.asarray([str(x).lower() for x in pools['leg_names']], dtype=object)
    leg_ids = leg_names[np.asarray(pools['leg_of_mn'])]
    missing=[L for L in LEGS if not (leg_ids==L.lower()).any()]
    if missing:raise ValueError('leg mapping missing '+str(missing))
    desc_cache = np.asarray(cand['cache_indices'])[mw.frozen.LocalCircuit().descending_local]
    recorded = np.asarray(inputs['sensory_recorded_nA'])
    cache = mw.load_cache()
    n_all = cache['root_ids'].size
    results = {}
    total = 0
    for tag, cell_mask in (('frozen20k', np.isin(np.arange(n_all), cand['cache_indices'])), ('banc_strict', None)):
        t0 = time.time()
        We, Wi, stats = mw.build(cache=cache, cell_mask=cell_mask)
        circ = mw.WideCircuit(We, Wi)
        n = circ.net.n
        if cell_mask is None:
            sens_idx, mn_idx, desc_idx = sensory_cache, mn_cache, desc_cache
        else:
            remap = np.full(n_all, -1, dtype=np.int64)
            remap[cell_mask] = np.arange(n)
            sens_idx, mn_idx, desc_idx = remap[sensory_cache], remap[mn_cache], remap[desc_cache]
        if (sens_idx < 0).any() or (mn_idx < 0).any() or (desc_idx < 0).any():
            raise ValueError('port mapping outside module for ' + tag)
        circ._sensory_mask[sens_idx] = True
        circ._descending_mask[desc_idx] = True
        zero = np.zeros(n)
        # warm-up 500 ms (100 bins) then 2000 ms recorded (400 bins)
        for b in range(100):
            s = zero.copy(); s[sens_idx] = recorded[b % 100]
            circ.advance(s, zero, 5.)
        rows = []
        for b in range(400):
            s = zero.copy(); s[sens_idx] = recorded[b % 100]
            circ.advance(s, zero, 5.)
            rows.append(circ.filtered_rates_hz[mn_idx].copy())
        total += 5000
        rates = np.asarray(rows)
        assert np.isfinite(rates).all()
        res = dict(analyse(rates, leg_ids), cells=stats['cells'], strict_edges=stats['strict_edges'],
                   exc_edges=stats['exc_edges'], inh_edges=stats['inh_edges'],
                   total_conductance_uS=stats['total_conductance_uS'], row_sum_max_abs_deviation_uS=stats['row_sum_max_abs_deviation_uS'],
                   neural_steps=5000, physical_steps=0, voltage_min_mV=float(circ.net.v.min()), voltage_max_mV=float(circ.net.v.max()),
                   total_spikes=int(circ.net.spike_count.sum()), MN_spikes=int(circ.net.spike_count[mn_idx].sum()),
                   wall_s=time.time() - t0, rates_sha256=hashlib.sha256(rates.astype(np.float64).tobytes()).hexdigest())
        results[tag] = res
        np.savez_compressed(HERE / f'RATES_{tag}.npz', rates_hz=rates.astype(np.float32), leg=leg_ids, leg_order=np.asarray(LEGS))
        print(tag, json.dumps({k: res[k] for k in ('cells', 'strict_edges', 'tripod_alternation_score', 'ipsilateral_score',
                                                   'lf_rh_correlation', 'rf_lh_correlation', 'group_mean_hz', 'PC1_variance_fraction',
                                                   'voltage_min_mV', 'MN_spikes', 'wall_s')}), flush=True)
    out = dict(status='WIDE_MODULE_INTRINSIC_COMPARISON_NOT_WALKING', module=HERE.name,
               protocol='both from rest; 500 ms warm-up then 2000 ms recorded sensory replay; zero descending; identical metrics as rounds 11-12',
               results=results, total_neural_steps=total, physical_steps=0, training_updates=0, new_edges=0,
               limits='Whole-BANC strict module is an uncalibrated reconstruction: single verified transmitter per presynaptic cell, uniform per-target 0.020 uS total, inhibitory sign from GABA/glutamate label only, no measured receptor signs, no compensation for missing cell types, no anatomical validation. The 20k control here starts from rest, so it is NOT the frozen 1s canonical state used in rounds 10-12 and its numbers are not interchangeable with those rounds.',
               source_hashes={str(p): sha(p) for p in (mw.CACHE, mw.FROZEN, CAND, INPUTS, POOLS, HERE / 'module_wide.py')})
    (HERE / 'RESULT.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print('total neural', total)


if __name__ == '__main__':
    main()
