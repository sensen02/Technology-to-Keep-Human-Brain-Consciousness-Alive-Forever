"""Step 2 (cheap): one warm replay from rest per gain, then the OFFLINE part only.

For each gain in 0.3 / 0.1 / 0.03 / 0.01 at a 0.5 ms neural timestep:
  (a) build that gain's 1200 x 12000 injected-current stream (linear rescaling of the recorded
      encoder output; gain 0.3 is bitwise the recorded array),
  (b) one warm replay from rest over 1200 causal 1 ms command intervals,
  (d) rate diagnostics per super_class and the pre-registered dense-regime relief test,
  (c) ridge information R^2 for the 204 motor cells and for all 12000 cells.
No physics, no release, no teacher recording.  The result is the go/no-go the parent sees
BEFORE the heavy release job.

Gain 0.3 additionally has to reproduce the sealed numerics warm trace bitwise.
"""
import sys
import time
import copy
import pickle

import numpy as np

import common as C

sys.path[:0] = [str(C.BASE), str(C.R), str(C.R / 'tools')]
import runtime as rt
from translator import SpikeHistory
from engine.neural_cond import ConductanceNetwork, ConductanceParams

js, sha, dump, exact = C.js, C.sha, C.dump, C.exact

CAP = {}


def load_matrices():
    """The sealed excitatory / inhibitory conductance matrices, read-only."""
    from scipy import sparse
    return [sparse.load_npz(rt.b.AB / 'A' / ('ACTUAL_' + k + '.npz')) for k in ['We', 'Wi']]


def warm(gain, dt, ann, cells, poolmat):
    """One warm replay from rest at one gain and one neural timestep."""
    t0 = time.monotonic()
    folder = C.O / 'CONFIGS' / (C.gain_tag(gain) + '__' + C.dt_tag(dt))
    folder.mkdir(parents=True, exist_ok=False)
    currents = C.currents_for_gain(gain)
    n = round(C.CDT / dt)
    decay = float(np.exp(-dt / C.FILTER_TAU))
    We, Wi = load_matrices()
    net = ConductanceNetwork(12000, ConductanceParams(), dt_ms=dt * 1000)
    net.set_connectivity(We, Wi)
    history = SpikeHistory(cells, dt=dt)
    rates = np.zeros(66)
    cell = np.zeros(12000)
    packed, pools, cellrates, legrates, last204 = [], [], [], [], []
    with C.PHYSICAL.open('rb') as f:
        phys = pickle.load(f)
    captures = {round((ph - 0.2) / C.CDT): ph for ph in C.PHASES}
    legs, legcells, _ = C.leg_cells_and_rh()
    for k, current in enumerate(currents):
        for step in range(n):
            sp = net.step(current_nA=current)
            packed.append(np.packbits(sp))
            history.consume(sp)
            rates = decay * rates + (1 - decay) * np.asarray(poolmat @ sp, float) / dt
            cell = decay * cell + (1 - decay) * np.asarray(sp, float) / dt
            if step == n - 1:
                last204.append(sp[cells].copy())
        pools.append(rates.copy())
        cellrates.append(cell.copy())
        legrates.append([float(cell[ids].mean()) for ids in legcells])
        if k + 1 in captures:
            ph = captures[k + 1]
            CAP['%g' % gain][ph] = dict(
                net=copy.deepcopy(net), rates=rates.copy(), history=history.state_dict(),
                cell=cell.copy(),
                integration=np.array(phys[ph]['integration'], float).copy(),
                assist=dict(phys[ph]['assist']), duration=ph, phase_offset=0.0)
    packed = np.asarray(packed)
    pools = np.asarray(pools)
    cellrates = np.asarray(cellrates)
    legrates = np.asarray(legrates)
    np.savez_compressed(
        folder / 'WARM.npz', gain_nA_per_unit=np.array(gain), neural_dt_s=np.array(dt),
        spikes12000_packed=packed, spikes204=np.asarray(last204), cells=cells,
        pool_rates=pools, cell_filtered_rates=cellrates, leg_rates=legrates,
        time=0.2 + np.arange(1, 1201) * C.CDT,
        current_max_abs_nA=np.array(float(np.abs(currents).max())),
        current_rms_nA=np.array(float(np.sqrt((currents ** 2).mean()))),
        current_cells_nonzero=np.array(int((currents != 0).any(0).sum())))
    with (folder / 'WARM_CACHE.pkl').open('wb') as f:
        pickle.dump(CAP['%g' % gain], f, protocol=5)
    dump(folder / 'WARM_IDENTITY.json', dict(
        gain_nA_per_unit=gain, neural_dt_s=dt, neural_steps=len(packed),
        steps_per_command=n, commands=1200, window_s=[0.2, 1.4],
        current_source=str(C.CURRENTS), current_sha256=sha(C.CURRENTS),
        current_rule='TRUE_CURRENTS * (gain / 0.3): a LINEAR RESCALING of the recorded encoder '
                     'output, not a re-recording; no teacher run is made by this task',
        physical_states='17 states taken from ' + str(C.PHYSICAL) + ' (teacher cache, '
                        'gain-independent because the teacher body is driven only by the '
                        'teacher PD law)',
        physical_sha256=sha(C.PHYSICAL),
        rate_filter_tau_s=C.FILTER_TAU, wall_s=time.monotonic() - t0,
        cache_sha256=sha(folder / 'WARM_CACHE.pkl'), importance='核心必需'))
    print('WARM %s %s steps=%d wall=%.1fs' % (C.gain_tag(gain), C.dt_tag(dt), len(packed),
                                              time.monotonic() - t0), flush=True)
    return dict(folder=folder, warm=np.load(folder / 'WARM.npz'), packed=packed,
                cell=cellrates, pools=pools, wall=time.monotonic() - t0)


def verify_reference_gain(w):
    """Gain 0.3 must reproduce the sealed numerics warm trace bitwise."""
    sealed = C.arrays(C.SEALED_CONFIGS / 'dt_0p5ms' / 'WARM.npz')
    with (C.SEALED_CONFIGS / 'dt_0p5ms' / 'WARM_CACHE.pkl').open('rb') as f:
        sealed_cache = pickle.load(f)
    out = dict(
        packed_spikes_exact=bool(np.array_equal(w['packed'], sealed['spikes12000_packed'])),
        cell_rates_exact=bool(np.array_equal(w['cell'], sealed['cell_filtered_rates'])),
        pool_rates_exact=bool(np.array_equal(w['pools'], sealed['pool_rates'])),
        legs_exact=bool(np.array_equal(np.asarray(w['warm']['leg_rates']),
                                       np.asarray(sealed['leg_rates']))),
        phases={}, current_max_abs_nA=float(w['warm']['current_max_abs_nA']))
    ok = all(out[k] for k in ['packed_spikes_exact', 'cell_rates_exact', 'pool_rates_exact',
                              'legs_exact'])
    for ph in C.PHASES:
        a = CAP['%g' % C.REF_GAIN][ph]
        b = sealed_cache[ph]
        entry = dict(network_bitwise=bool(exact(vars(a['net']), vars(b['net']))),
                     rates_exact=bool(np.array_equal(a['rates'], b['rates'])),
                     history_exact=bool(exact(a['history'], b['history'])),
                     integration_exact=bool(np.array_equal(a['integration'],
                                                           b['integration'])),
                     assist_exact=bool(exact(a['assist'], b['assist'])))
        entry['passed'] = all(entry.values())
        out['phases'][str(ph)] = entry
        ok = ok and entry['passed']
    out['all_17_phases_exact'] = bool(all(v['passed'] for v in out['phases'].values()))
    out['passed'] = bool(ok)
    return out


def main():
    C.O.mkdir(parents=True, exist_ok=True)
    assert (C.O / 'PREREGISTRATION.json').exists(), 'run prereg.py first'
    assert (C.O / 'STATIC_CHECKS.json').exists(), 'run static.py first'
    assert js(C.O / 'STATIC_CHECKS.json')['all_passed'], 'the static checks did not pass'
    before = js(C.O / 'INPUT_HASHES_BEFORE.json')
    assert {p: sha(p) for p in before} == before, 'a sealed input changed before the offline part'

    ann = C.anatomy()
    cells = C.motor_cells_204()
    teacher = C.teacher_torque()
    poolmat, pools = C.pool_matrix()

    results = {}
    reference = None
    for gain in C.GAINS:
        CAP['%g' % gain] = {}
        w = warm(gain, C.DT_MAIN, ann, cells, poolmat)
        src = np.vstack([np.zeros((1, 12000)), w['cell'][:-1]])
        diag = C.rate_diagnostics(w['cell'], w['packed'], ann, cells)
        diag['leg_rates_hz'] = np.asarray(w['warm']['leg_rates']).mean(0).tolist()
        relief = C.dense_relief(diag)
        print('--- gain %g' % gain, flush=True)
        f204 = C.fit_information(src, teacher, cells, 'gain%g_204' % gain)
        f12k = C.fit_information(src, teacher, None, 'gain%g_12000' % gain)
        folder = C.O / 'CONFIGS' / (C.gain_tag(gain) + '__dt_0p5ms')
        np.savez_compressed(folder / 'READOUT_MOTOR204.npz',
                            W=f204['W'], b=f204['b'], mean_fit=f204['mean'], cells=cells,
                            alpha=np.array([f204['alpha']]), gamma=np.array([f204['gamma']]),
                            r2_fit=np.array([f204['r2_fit']]),
                            r2_held=np.array([f204['r2_held']]), gain=np.array([gain]))
        np.savez_compressed(folder / 'READOUT_ALL12000.npz',
                            W=f12k['W'], b=f12k['b'], mean_fit=f12k['mean'],
                            alpha=np.array([f12k['alpha']]), gamma=np.array([f12k['gamma']]),
                            r2_fit=np.array([f12k['r2_fit']]),
                            r2_held=np.array([f12k['r2_held']]), gain=np.array([gain]))
        entry = dict(
            gain=gain, folder=str(folder), wall_s=w['wall'],
            current=dict(max_abs_nA=float(w['warm']['current_max_abs_nA']),
                         rms_nA=float(w['warm']['current_rms_nA']),
                         cells_nonzero=int(w['warm']['current_cells_nonzero'])),
            diagnostics=diag, dense_relief=relief,
            information=dict(
                motor_204=dict(alpha=f204['alpha'], gamma=f204['gamma'],
                               cv_best=f204['cv_best'], r2_fit=f204['r2_fit'],
                               r2_held=f204['r2_held'], n_features=204,
                               offset_rms=f204['offset_rms'],
                               feature_mean_fit_hz=f204['feature_mean_fit_hz'],
                               feature_rms_fit_hz=f204['feature_rms_fit_hz'],
                               cv_curve=f204['cv_curve']),
                all_12000=dict(alpha=f12k['alpha'], gamma=f12k['gamma'],
                               cv_best=f12k['cv_best'], r2_fit=f12k['r2_fit'],
                               r2_held=f12k['r2_held'], n_features=12000,
                               offset_rms=f12k['offset_rms'],
                               feature_mean_fit_hz=f12k['feature_mean_fit_hz'],
                               feature_rms_fit_hz=f12k['feature_rms_fit_hz'],
                               cv_curve=f12k['cv_curve'])))
        if gain == C.REF_GAIN:
            reference = verify_reference_gain(w)
            entry['reference_gain_bitwise_reproduction'] = reference
        results['%g' % gain] = entry
        dump(C.O / 'OFFLINE_PROGRESS.json', results)
        print('    r204=%.4f r12k=%.4f readoutHz=%.1f dense_relieved=%s' %
              (f204['r2_held'], f12k['r2_held'], diag['readout_204']['mean_hz'],
               relief['DENSE_RELIEVED']), flush=True)

    order = sorted(C.GAINS,
                   key=lambda g: (-results['%g' % g]['information']['motor_204']['r2_held'], g))
    ranking = dict(
        rule=js(C.O / 'PREREGISTRATION.json')['second_timestep_rule']['rule'],
        order=order,
        motor_held_r2={('%g' % g): results['%g' % g]['information']['motor_204']['r2_held']
                       for g in order},
        best_information_gain=order[0],
        second_timestep_gains=order[:2],
        whole_population_release_gain=order[0],
        decided_before_any_physics=True)
    dump(C.O / 'RANKING.json', ranking)
    dump(C.O / 'GAIN_METRICS_OFFLINE.json', dict(
        gains=C.GAINS, neural_dt_s=C.DT_MAIN, results=results, ranking=ranking,
        reference_gain_bitwise=reference, importance='核心必需'))
    assert reference is not None and reference['passed'], \
        'the gain 0.3 warm replay does not reproduce the sealed warm trace'
    assert {p: sha(p) for p in before} == before, 'a sealed input changed during the offline part'
    print('OFFLINE DONE ranking', order, 'best', order[0], flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        C.O.mkdir(parents=True, exist_ok=True)
        dump(C.O / ('FAILURE_offline_' + str(time.time_ns()) + '.json'),
             dict(traceback=traceback.format_exc()))
        raise
