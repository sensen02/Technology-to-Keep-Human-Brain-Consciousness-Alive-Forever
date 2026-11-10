"""Gain control for the single-leg drive result: is a silent leg silent for lack of gain?

Repeats single-leg sensory drive at 1x, 4x, 16x for one 'silent' leg (LM) and for the
responsive leg (LF), in the 20k control module, to separate 'no pathway' from 'below
threshold under the arbitrary encoder gain'.
"""
from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np
HERE = Path(__file__).resolve().parent
CAND = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface/LOCAL_MODULE_CANDIDATE.npz')
INPUTS = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-descending-learning/20261107T203049Z/descending/INPUTS.npz')
POOLS = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-descending-learning/20261107T203049Z/leg-pools/POOLS_AND_REACH.npz')
LEGS = ['LF', 'LH', 'LM', 'RF', 'RH', 'RM']


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


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
    results = {}; total = 0
    for leg in ('LM', 'LF'):
        sel = sens_leg == leg.lower()
        for gain in (1., 4., 16.):
            circ = mw.WideCircuit(We, Wi)
            n = circ.net.n; circ._sensory_mask[sens_idx] = True; zero = np.zeros(n)
            drive = np.clip(recorded[:, sel] * gain, 0, None)
            for b in range(100):
                s = zero.copy(); s[sens_idx[sel]] = drive[b % 100]; circ.advance(s, zero, 5.)
            rows = []
            for b in range(400):
                s = zero.copy(); s[sens_idx[sel]] = drive[b % 100]
                circ.advance(s, zero, 5.); rows.append(circ.filtered_rates_hz[mn_idx].copy())
            total += 5000
            rates = np.asarray(rows)
            per_leg = {L: float(rates[:, mn_leg == L.lower()].mean()) for L in LEGS}
            key = f'{leg}_gain{gain:g}'
            results[key] = dict(driven_leg=leg, gain=gain, max_drive_nA=float(drive.max()),
                                driven_leg_hz=per_leg[leg], per_leg_mean_hz=per_leg,
                                MN_spikes=int(circ.net.spike_count[mn_idx].sum()),
                                voltage_min_mV=float(circ.net.v.min()),
                                rates_sha256=hashlib.sha256(rates.astype(np.float64).tobytes()).hexdigest())
            print(key, json.dumps({k: results[key][k] for k in ('max_drive_nA', 'driven_leg_hz', 'MN_spikes')}), flush=True)
    out = dict(status='SINGLE_LEG_GAIN_CONTROL', results=results, total_neural_steps=total, physical_steps=0, training_updates=0,
               note='Encoder gains are arbitrary. If a silent leg responds only at high gain, the silence is a gain/threshold property of this uncalibrated interface, not evidence about the animal.',
               source_hashes={str(p): sha(p) for p in (mw.CACHE, mw.FROZEN, CAND, INPUTS, POOLS, HERE / 'module_wide.py')})
    (HERE / 'RESULT_GAIN.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print('total neural', total)


if __name__ == '__main__':
    main()
