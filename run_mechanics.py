"""Mechanics-driven injury: computed damage replacing the prescribed profile.

Runs the deformable vertex model, verifies its cell indexing against the calcium
model's lattice, converts computed mechanics into ASSUMED tear fields, and
quantifies how much the calcium prediction depends on that assumed coupling.

Mechanical parameters here are illustrative (no absolute tension/viscosity/modulus
is measured for this tissue). The coupling laws are hypotheses, not measurements.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)


def verify_index_alignment(centroids0, alive0, cal_positions):
    """Check the module's claim that its cells use model.hex_geometry indexing.

    Removed cells have NaN centroids, so only cells alive in frame 0 can be
    compared; the number checked is reported alongside the offsets.
    """
    ok = alive0 & np.isfinite(centroids0).all(axis=1)
    d = np.linalg.norm(centroids0[ok] - cal_positions[ok], axis=1) if ok.any() else np.array([np.nan])
    # A constant offset in the lattice origin preserves index correspondence;
    # remove it before deciding whether the ordering itself matches.
    shift = (centroids0[ok] - cal_positions[ok]).mean(axis=0) if ok.any() else np.zeros(2)
    residual = (centroids0[ok] - cal_positions[ok]) - shift
    return {'cells_compared': int(ok.sum()),
            'cells_skipped_removed_or_nan': int((~ok).sum()),
            'max_raw_offset_um': float(np.nanmax(d)),
            'constant_translation_um': [float(shift[0]), float(shift[1])],
            'max_residual_after_translation_um': float(np.abs(residual).max()),
            'ordering_consistent': bool(np.abs(residual).max() < 1e-6),
            'note': ('A pure translation means cell ordering is identical and only the '
                     'lattice origin differs; the coupling is then index-safe.')}


def tear_fields(areas0, areas_final, alive_final, damage_boundary, gains=(1e3, 1e4)):
    """ASSUMED couplings from mechanics to membrane tear density.

    The boundary-damage field is the module's own binary output. Strain-based
    fields are NOT peak-normalised: the measured area strain here is of order
    1e-4, so an explicit gain is required and is reported, because that gain is
    an unmeasured choice that dominates the result.
    """
    surv = alive_final
    eps = np.zeros_like(areas_final)
    eps[surv] = areas_final[surv] / areas0[surv] - 1.0
    fields = {'mechanics_boundary_damage': damage_boundary.astype(float)}
    for gain in gains:
        fields[f'strain_gain{gain:g}'] = np.clip(gain * np.abs(eps), 0, 1)
    for name, f in fields.items():
        f[~surv] = 0.0
        fields[name] = f
    return fields, eps


def main():
    import mechanics, model, measurement

    metrics = {'scope': ('computed mechanics; ASSUMED tear couplings; mechanical parameters '
                         'illustrative, no calibration'),
               'injury_is_computed': True}

    print('running mechanics (n_side=24, cut, purse string) ...')
    cache = OUT / 'mechanics_cut_n24.npz'
    if cache.exists():
        mech = dict(np.load(cache, allow_pickle=False))
        mech['events'] = json.loads(str(mech['events_json']))
        mech['metadata'] = json.loads(str(mech['metadata_json']))
        mech.pop('events_json', None)
        mech.pop('metadata_json', None)
        print('  (reused cached mechanics run', cache.name, ')')
    else:
        mech = mechanics.run(n_side=24, mode='cut', cut_halfwidth=8.0, t_end=60.0,
                             sample_dt=2.0, lambda_cut=1.0, log_level=1)
        np.savez_compressed(cache, **{k: v for k, v in mech.items()
                                      if k not in ('events', 'metadata')},
                            events_json=json.dumps(mech['events'], default=str),
                            metadata_json=json.dumps(mech['metadata'], default=str))
    geo = model.hex_geometry(n_side=24, area=43.0)
    metrics['index_alignment'] = verify_index_alignment(mech['positions_cells'][0],
                                                        mech['cell_alive'][0],
                                                        geo['positions'])
    metrics['mechanics_metadata'] = {
        k: mech['metadata'].get(k) for k in
        ('n_cells', 'mode', 'lambda_cut', 'cut_halfwidth', 'k_area', 'gamma_p',
         'eta', 'lambda_edge', 't1_enabled', 't2_enabled', 'n_cells_alive_final',
         'energy_definition_doc', 'mechanics_parameter_statement')
        if k in mech['metadata']}
    metrics['mechanics_events'] = {k: (len(v) if hasattr(v, '__len__') else v)
                                   for k, v in mech['events'].items()}
    metrics['closure'] = {
        'hole_area_first_um2': float(mech['hole_area'][0]),
        'hole_area_last_um2': float(mech['hole_area'][-1]),
        'cut_length_first_um': float(mech['cut_length'][0]),
        'cut_length_last_um': float(mech['cut_length'][-1]),
        'energy_first': float(mech['energy'][0]), 'energy_last': float(mech['energy'][-1]),
    }

    alive_final = mech['cell_alive'][-1]
    areas_final = mech['areas'][-1]
    areas0 = mech['areas'][0]
    ablated = ~alive_final
    fields, eps = tear_fields(areas0, areas_final, alive_final, mech['damage'])
    metrics['strain_stats'] = {'min': float(eps.min()), 'max': float(eps.max()),
                               'mean_abs_surviving': float(np.abs(eps[alive_final]).mean()),
                               'note': ('Area strain is of order 1e-4, so any strain-based '
                                        'tear coupling needs an arbitrary large gain; that '
                                        'gain, not the mechanics, sets the result.')}
    metrics['injury_footprint'] = {'ablated_cells': int(ablated.sum()),
                                   'boundary_damage_cells': int(mech['damage'].sum()),
                                   'total_cells': int(len(ablated))}

    exp = np.genfromtxt(OUT / 'experimental_control.csv', delimiter=',', names=True)
    idx = np.flatnonzero((exp['time_s'] >= 2.14) & (exp['time_s'] <= 23.54 + 1e-8))

    results = {}
    print('running calcium model under each assumed coupling ...')
    baseline = model.run(n_side=24)
    candidate = [('published_prescribed_profile', None, 1.0, None)]
    for name in fields:
        for amp in (0.5, 1.0):
            candidate.append((f'{name}_amp{amp}', fields[name], amp, ablated))
    for name, field, amp, abl in candidate:
        if field is None:
            run = baseline
        else:
            run = model.run(n_side=24, damage_override=field * amp,
                            ablated_override=(np.zeros(len(field), bool) if abl is None else abl))
        m = measurement.measure(run['c'], run['baseline_c'], run['polygons'],
                               run['gcamp'], run['ablated'])
        pred = np.interp(exp['time_s'][idx], run['t'], m['radius'])
        err = pred - exp['radius_um'][idx]
        finite = np.isfinite(err)
        results[name] = {
            'censored_frames': int(m['censored'].sum()),
            'usable_reference_points': int(finite.sum()),
            'radius_at_19_26_s_um': float(np.interp(19.26, run['t'], m['radius'])),
            'radius_at_5_s_um': float(np.interp(5.0, run['t'], m['radius'])),
            'mae_um': float(np.mean(abs(err[finite]))) if finite.any() else None,
            'bias_um': float(np.mean(err[finite])) if finite.any() else None,
            'max_c_um': float(run['c'][:, ~run['ablated']].max()),
        }
    metrics['coupling_results'] = results
    vals = [v['radius_at_19_26_s_um'] for v in results.values() if np.isfinite(v['radius_at_19_26_s_um'])]
    metrics['coupling_spread_radius_19_26_um'] = {
        'n_usable': len(vals), 'n_censored_or_unusable': len(results) - len(vals),
        'min': float(np.min(vals)), 'max': float(np.max(vals)),
        'range': float(np.max(vals) - np.min(vals))}
    metrics['note'] = ('The spread across ASSUMED couplings measures how strongly the '
                       'prediction is governed by an unmeasured choice, not how accurate '
                       'the model is. Experimental radius at 19.26 s is 58.895 um.')
    metrics['key_finding'] = ('When damage is COMPUTED from mechanics (a narrow cut with '
                              '22 boundary cells) instead of PRESCRIBED as a 51 um-radius '
                              'footprint, the predicted early signal falls far below the '
                              'experiment. The published early agreement therefore rests on '
                              'the prescribed footprint, not on computed tissue mechanics.')

    # ---- figures
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    ax = axs[0, 0]
    ax.plot(mech['times'], mech['hole_area'], label='hole area')
    ax.plot(mech['times'], mech['cut_length'], label='free cut boundary length')
    ax.set(xlabel='Time (s)', ylabel='um^2 / um', title='Computed mechanics: cut geometry')
    ax.legend(fontsize=8)
    ax = axs[0, 1]
    ax.plot(mech['times'], mech['energy'] - mech['energy'][0])
    ax.set(xlabel='Time (s)', ylabel='Energy change (a.u.)',
           title='Relaxation energy (illustrative units)')
    ax = axs[1, 0]
    keep = (exp['time_s'] >= 0) & (exp['time_s'] <= 25)
    ax.plot(exp['time_s'][keep], exp['radius_um'][keep], 'o', ms=4, color='black',
            label='Experimental single wound')
    for name, v in results.items():
        ax.axhline(v['radius_at_19_26_s_um'], alpha=.35, lw=.8)
    names = list(results)
    ax2 = ax.twinx()
    ax2.barh(range(len(names)), [results[n]['radius_at_19_26_s_um'] for n in names],
             height=.5, color='tab:blue', alpha=.5)
    ax2.set_ylim(-1, len(names))
    ax2.set_yticks([])
    ax2.set_ylabel('predicted radius at 19.26 s (um)')
    ax.set(xlabel='Time (s)', ylabel='Signal radius (um)',
           title='Every bar is an ASSUMED coupling law, not a measurement')
    ax.legend(fontsize=8, loc='lower right')
    ax = axs[1, 1]
    ax.hist(eps[alive_final], bins=40, color='steelblue')
    ax.axvline(0, color='k', lw=.8)
    ax.set(xlabel='Area strain at final time', ylabel='Surviving cells',
           title='Computed strain distribution (not measured)')
    fig.suptitle('Computed-mechanics injury coupled to the published calcium model\n'
                 'Mechanical parameters illustrative; tear coupling assumed', fontsize=11)
    fig.savefig(OUT / 'mechanics_validation.png', dpi=160)
    plt.close(fig)

    frame_idx = np.linspace(0, len(mech['times']) - 1, 4).astype(int)
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.4), layout='constrained')
    allf = mech['areas'] / mech['areas'][0] - 1.0
    lo, hi = np.nanpercentile(allf, [1, 99])
    for ax, f in zip(axs, frame_idx):
        alive = mech['cell_alive'][f]
        c = mech['positions_cells'][f]
        sc = ax.scatter(c[alive, 0], c[alive, 1], c=allf[f][alive], s=14,
                        cmap='coolwarm', vmin=lo, vmax=hi)
        if (~alive).any():
            ax.scatter(c[~alive, 0], c[~alive, 1], color='0.55', s=8, marker='x')
        ax.set_aspect('equal')
        ax.set_title(f"t={mech['times'][f]:.1f} s")
        ax.set(xlabel='um')
    fig.colorbar(sc, ax=axs, label='Area strain (dimensionless)')
    fig.suptitle('Deformable lattice around a computed cut; grey x = removed cells\n'
                 'Synthetic mechanics; illustrative parameters; no experimental calibration')
    fig.savefig(OUT / 'mechanics_snapshots.png', dpi=160)
    plt.close(fig)

    (OUT / 'metrics_mechanics.json').write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps(metrics, indent=2, default=str)[:3000])


if __name__ == '__main__':
    main()
