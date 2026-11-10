"""Does the whole strict module carry LEG-SPECIFIC drive? Single-leg sensory stimulation test.

Six conditions: drive only the 556-port sensory currents that belong to one leg (all
other ports zero), in the whole-BANC strict module and in the 20k control. Compares the
driven leg's motor-neuron rate against the other five, and reports the per-leg vector.
Neural only: no physics, no learning, no descending current.
"""
from pathlib import Path
import hashlib, importlib.util, json, sys, time
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
    # sensory port -> leg, via the frozen encoder's per-cell leg labels
    enc_leg = np.asarray(inputs['mn323_leg']) if 'mn323_leg' in inputs.files else None
    sens_local = np.asarray(inputs['sensory_local'])
    # the encoder contract stores per-cell leg; read it from the frozen encoder contract
    contract = json.loads(Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/encoder/CONTRACT.json').read_text())
    leg_of_root = {str(r['root_888']): str(r['leg']).lower() for r in contract['qualified_records']}
    root_ids = np.asarray(cand['root_ids'])
    sens_cache = np.asarray(cand['cache_indices'])[sens_local]
    sens_leg = np.array([leg_of_root.get(str(int(root_ids[i])), '') for i in sens_local], dtype=object)
    if (sens_leg == '').any():
        raise ValueError('sensory port without leg label')
    recorded = np.asarray(inputs['sensory_recorded_nA'])
    cache = mw.load_cache(); n_all = cache['root_ids'].size
    mn_cache = np.asarray(cand['cache_indices'])[np.asarray(pools['mn_local'])]
    out = {}
    total = 0
    for tag, cell_mask in (('frozen20k', np.isin(np.arange(n_all), cand['cache_indices'])), ('banc_strict', None)):
        t0 = time.time()
        We, Wi, stats = mw.build(cache=cache, cell_mask=cell_mask)
        if cell_mask is not None:
            remap = np.full(n_all, -1, dtype=np.int64); remap[cell_mask] = np.arange(int(cell_mask.sum()))
            sens_idx, mn_idx = remap[sens_cache], remap[mn_cache]
        else:
            sens_idx, mn_idx = sens_cache, mn_cache
        conditions = {}
        for drive_leg in [None] + LEGS:
            circ = mw.WideCircuit(We, Wi)
            n = circ.net.n
            circ._sensory_mask[sens_idx] = True
            zero = np.zeros(n)
            keep = np.ones(sens_idx.size, dtype=bool) if drive_leg is None else (sens_leg == drive_leg.lower())
            for b in range(100):
                s = zero.copy(); s[sens_idx[keep]] = recorded[b % 100][keep]; circ.advance(s, zero, 5.)
            rows = []
            for b in range(400):
                s = zero.copy(); s[sens_idx[keep]] = recorded[b % 100][keep]
                circ.advance(s, zero, 5.); rows.append(circ.filtered_rates_hz[mn_idx].copy())
            total += 5000
            rates = np.asarray(rows)
            assert np.isfinite(rates).all()
            per_leg = {L: float(rates[:, mn_leg == L.lower()].mean()) for L in LEGS}
            conditions['all_legs' if drive_leg is None else 'only_' + drive_leg] = dict(
                driven_ports=int(keep.sum()), per_leg_mean_hz=per_leg,
                driven_leg_hz=float(per_leg[drive_leg]) if drive_leg else None,
                other_legs_mean_hz=float(np.mean([v for k, v in per_leg.items() if k != drive_leg])) if drive_leg else None,
                selectivity_ratio=(float(per_leg[drive_leg]) / float(np.mean([v for k, v in per_leg.items() if k != drive_leg])))
                    if drive_leg and np.mean([v for k, v in per_leg.items() if k != drive_leg]) > 0 else None,
                MN_spikes=int(circ.net.spike_count[mn_idx].sum()), voltage_min_mV=float(circ.net.v.min()),
                rates_sha256=hashlib.sha256(rates.astype(np.float64).tobytes()).hexdigest())
            print(tag, drive_leg or 'all', json.dumps({k: conditions['all_legs' if drive_leg is None else 'only_' + drive_leg][k]
                  for k in ('driven_ports', 'driven_leg_hz', 'other_legs_mean_hz', 'selectivity_ratio')}), flush=True)
        # cross-condition: does driving one leg raise its own leg most?
        winners = {}
        for L in LEGS:
            per = conditions['only_' + L]['per_leg_mean_hz']
            winners[L] = max(LEGS, key=lambda k: per[k])
        out[tag] = dict(cells=stats['cells'], strict_edges=stats['strict_edges'], conditions=conditions,
                        argmax_leg_per_condition=winners, wall_s=time.time() - t0)
    res = dict(status='WIDE_MODULE_LEG_SPECIFICITY_TEST_NOT_WALKING', results=out, total_neural_steps=total,
               physical_steps=0, training_updates=0,
               protocol='single-leg sensory drive (one leg of 556 ports nonzero), 500 ms warm-up + 2000 ms recorded replay, zero descending, both modules from rest',
               limits='Engineering drive test on an uncalibrated reconstruction; a leg-specific response would not prove the fly uses it, and its absence would not prove the fly cannot. No physics, no behaviour, no learning.',
               source_hashes={str(p): sha(p) for p in (mw.CACHE, mw.FROZEN, CAND, INPUTS, POOLS, HERE / 'module_wide.py')})
    (HERE / 'RESULT_LATERAL.json').write_text(json.dumps(res, indent=2, allow_nan=False) + '\n')
    print('total neural', total)


if __name__ == '__main__':
    main()
