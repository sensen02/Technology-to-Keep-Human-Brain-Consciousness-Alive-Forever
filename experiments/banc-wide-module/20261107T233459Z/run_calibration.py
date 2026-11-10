"""Per-leg sensory gain calibration, then tripod test with equalised recruitment.

Round-13 follow-up: the uncalibrated interface only recruits motor activity from LF.
Here each leg's sensory ports are gain-calibrated (bounded, log steps) so that driving
that leg alone reaches at least 8 Hz in its own motor pool, then all legs are driven
together with the calibrated gains to test whether equalised recruitment produces
tripod alternation. Engineering calibration of an uncalibrated proxy encoder; no
external oscillator, no new edges, no learning, no physics.
"""
from pathlib import Path
import hashlib, importlib.util, json, sys, time
import numpy as np
HERE = Path(__file__).resolve().parent
CAND = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface/LOCAL_MODULE_CANDIDATE.npz')
INPUTS = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-descending-learning/20261107T203049Z/descending/INPUTS.npz')
POOLS = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-descending-learning/20261107T203049Z/leg-pools/POOLS_AND_REACH.npz')
LEGS = ['LF', 'LH', 'LM', 'RF', 'RH', 'RM']
GAINS = (1., 2., 4., 8., 16., 32., 64.)
TARGET_HZ = 8.


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


def analyse(series):
    win = series[100:] - series[100:].mean(0)
    U, S, Vt = np.linalg.svd(win, full_matrices=False)
    frac = S ** 2 / np.sum(S ** 2)
    C = np.corrcoef(series.T)
    A = series[:, [0, 5, 1]].mean(1); B = series[:, [3, 2, 4]].mean(1)
    return dict(per_leg_mean_hz={L: float(series[:, i].mean()) for i, L in enumerate(LEGS)},
                tripod_alternation_score=float(-np.corrcoef(A, B)[0, 1]),
                ipsilateral_score=float(np.corrcoef(series[:, [0, 1, 2]].mean(1), series[:, [3, 4, 5]].mean(1))[0, 1]),
                lf_rh_correlation=float(C[0, 4]), rf_lh_correlation=float(C[3, 1]),
                group_mean_hz={'A_LF_RM_LH': float(A.mean()), 'B_RF_LM_RH': float(B.mean())},
                PC1_variance_fraction=float(frac[0]), PC1_loadings={L: float(v) for L, v in zip(LEGS, Vt[0])})


def main():
    mw = load('mw', HERE / 'module_wide.py')
    cand = np.load(CAND); inputs = np.load(INPUTS); pools = np.load(POOLS)
    leg_names = np.asarray([str(x).lower() for x in pools['leg_names']], dtype=object)
    mn_leg = leg_names[np.asarray(pools['leg_of_mn'])]
    contract = json.loads(Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/encoder/CONTRACT.json').read_text())
    leg_of = {str(r['root_888']): str(r['leg']).lower() for r in contract['qualified_records']}
    root = np.asarray(cand['root_ids'])
    sens_local = np.asarray(inputs['sensory_local'])
    sens_leg = np.array([leg_of[str(int(root[i]))] for i in sens_local], dtype=object)
    recorded = np.asarray(inputs['sensory_recorded_nA'])
    cache = mw.load_cache(); n_all = cache['root_ids'].size
    cell_mask = np.isin(np.arange(n_all), cand['cache_indices'])
    We, Wi, stats = mw.build(cache=cache, cell_mask=cell_mask)
    remap = np.full(n_all, -1, dtype=np.int64); remap[cell_mask] = np.arange(int(cell_mask.sum()))
    sens_idx = remap[np.asarray(cand['cache_indices'])[sens_local]]
    mn_idx = remap[np.asarray(cand['cache_indices'])[np.asarray(pools['mn_local'])]]

    def run(port_mask, gain_per_port, bins=400, warm=100):
        circ = mw.WideCircuit(We, Wi)
        n = circ.net.n; circ._sensory_mask[sens_idx] = True; zero = np.zeros(n)
        drive = recorded * gain_per_port[None, :]
        for b in range(warm):
            s = zero.copy(); s[sens_idx[port_mask]] = drive[b % 100][port_mask]; circ.advance(s, zero, 5.)
        rows = []
        for b in range(bins):
            s = zero.copy(); s[sens_idx[port_mask]] = drive[b % 100][port_mask]
            circ.advance(s, zero, 5.); rows.append(circ.filtered_rates_hz[mn_idx].copy())
        return np.asarray(rows), circ

    calibrated = {}; log = []
    for L in LEGS:
        sel = sens_leg == L.lower()
        chosen = None
        for g in GAINS:
            gain_vec = np.ones(sens_idx.size)
            gain_vec[sel] = g
            rates, circ = run(sel, gain_vec, bins=200, warm=50)
            hz = float(rates[:, mn_leg == L.lower()].mean())
            log.append(dict(leg=L, gain=g, driven_leg_hz=hz, spikes=int(circ.net.spike_count[mn_idx].sum())))
            print('calibrate', L, 'gain', g, 'hz %.2f' % hz, flush=True)
            if hz >= TARGET_HZ:
                chosen = g
                break
        calibrated[L] = chosen if chosen is not None else GAINS[-1]
    gain_vec = np.ones(sens_idx.size)
    for L in LEGS:
        gain_vec[sens_leg == L.lower()] = calibrated[L]
    rates_all, circ_all = run(np.ones(sens_idx.size, dtype=bool), gain_vec)
    series = np.stack([rates_all[:, mn_leg == L.lower()].mean(1) for L in LEGS], 1)
    calibrated_result = dict(analyse(series), gains=calibrated, max_gain_applied=float(gain_vec.max()),
                             MN_spikes=int(circ_all.net.spike_count[mn_idx].sum()), voltage_min_mV=float(circ_all.net.v.min()))
    # control: same drive, all gains 1 (round-13 baseline, recomputed here for a like-for-like pair)
    rates_raw, circ_raw = run(np.ones(sens_idx.size, dtype=bool), np.ones(sens_idx.size))
    series_raw = np.stack([rates_raw[:, mn_leg == L.lower()].mean(1) for L in LEGS], 1)
    uncalibrated = dict(analyse(series_raw), gains={L: 1. for L in LEGS}, MN_spikes=int(circ_raw.net.spike_count[mn_idx].sum()),
                        voltage_min_mV=float(circ_raw.net.v.min()))
    total = 5000 * (len(log) + 2)
    out = dict(status='PER_LEG_CALIBRATION_TEST', module='frozen20k', calibrated=calibrated_result, uncalibrated=uncalibrated,
               calibration_log=log, target_hz=TARGET_HZ, total_neural_steps=total, physical_steps=0, training_updates=0, new_edges=0,
               note='Per-leg gains are an engineering calibration of an uncalibrated proxy encoder, applied at the input ports only. They do not add a rhythm and are declared, not hidden.',
               source_hashes={str(p): sha(p) for p in (mw.CACHE, mw.FROZEN, CAND, INPUTS, POOLS, HERE / 'module_wide.py')})
    (HERE / 'RESULT_CALIBRATION.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print('gains', calibrated)
    print('calibrated tripod %.3f ipsi %.3f' % (calibrated_result['tripod_alternation_score'], calibrated_result['ipsilateral_score']))
    print('uncalibrated tripod %.3f ipsi %.3f' % (uncalibrated['tripod_alternation_score'], uncalibrated['ipsilateral_score']))
    print('total neural', total)


if __name__ == '__main__':
    main()
