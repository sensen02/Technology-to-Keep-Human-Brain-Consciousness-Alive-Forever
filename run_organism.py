"""Organism-scale coupling with REAL oxygen units and PER-CELL cell-state input.

Chain executed here:
  transport.run(...)   prescribed vessels, O2 diffusion + per-cell uptake  [physiological units]
  cellstate.run(...)   ATP / integrity / triage driven PER CELL by the local O2 array
  growth.run(...)      growth and division gated by a Monod factor derived from local O2
  hormone.run(...)     receptor response and a growth multiplier

Unit conventions (provenance inside transport.py):
  1 uM dissolved O2 ~= 0.71 mmHg (solubility ~1.4 uM/mmHg at 37 C)
  arterial plasma   = 140 uM (PO2 100 mmHg, dissolved only, carrier_factor = 1)
  venous reference  =  56 uM (PO2 40 mmHg) -> availability 1.0
  hypoxic threshold =  14 uM (PO2 10 mmHg)

Honest scope: vessel geometry is PRESCRIBED, per-cell O2 consumption is the transport
module's illustrative default (vmax 1 amol/cell/s, Km 5 uM), and the growth mapping is an
explicit modelling choice. Insects use tracheae and Hydra has no vasculature, so this
vascular layer is NOT their gas-exchange organ. Not a validated organism.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'

UM_PER_MMHG = 1.4
O2_ARTERIAL_UM = 140.0
O2_VENOUS_UM = 56.0
O2_HYPOXIC_UM = 14.0
K_O2_GROWTH_UM = 5.0


def um_to_mmhg(c):
    return np.asarray(c, float) / UM_PER_MMHG


def build_cells(n_side, domain_um):
    import model
    geo = model.hex_geometry(n_side, area=43.0)
    pos = geo['positions'].copy()
    pos -= pos.min(axis=0)
    pos += 0.5 * (np.asarray(domain_um, float) - (pos.max(axis=0) - pos.min(axis=0)))
    return pos


def vessels_vertical(domain_um, spacing_um, radius_um=3.0, plasma_conc_um=O2_ARTERIAL_UM):
    return [dict(id=f'v{i}', path=[[float(x), 0.0], [float(x), float(domain_um[1])]],
                 radius_um=float(radius_um), permeability_um_s=50.0,
                 exchange='renkin_crone', plasma_conc_um=float(plasma_conc_um))
            for i, x in enumerate(np.arange(spacing_um / 2, domain_um[0], spacing_um))]


def run_transport(domain_um, dx_um, positions, spacing_um, t_end=60.0, dt=0.05):
    import transport
    cells = {'centroids': positions, 'areas': np.full(len(positions), 43.0)}
    return transport.run(domain_um=domain_um, dx_um=dx_um, t_end=t_end, dt=dt,
                         substances=['o2'],
                         vessels=vessels_vertical(domain_um, spacing_um),
                         cells=cells, scheme='split', initial=O2_VENOUS_UM)


def main():
    import cellstate, growth, hormone

    metrics = {'scope': ('coupled transport + per-cell cell state + growth + hormone; '
                         'prescribed vessels; not a validated organism'),
               'units': {'uM_per_mmHg': UM_PER_MMHG, 'arterial_uM': O2_ARTERIAL_UM,
                         'venous_reference_uM': O2_VENOUS_UM, 'hypoxic_uM': O2_HYPOXIC_UM,
                         'growth_monod_K_uM': K_O2_GROWTH_UM}}
    domain, dx = (400.0, 400.0), 4.0
    positions = build_cells(n_side=61, domain_um=domain)
    metrics['tissue'] = {'n_cells': int(len(positions)), 'domain_um': list(domain), 'dx_um': dx}

    fields, spacing_results = {}, {}
    for spacing in (40.0, 80.0, 160.0, 320.0):
        print(f'transport, vessel spacing {spacing} um ...')
        r = run_transport(domain, dx, positions, spacing)
        conc = np.asarray(r['cell_concentration']['o2'])
        conc = conc[-1] if conc.ndim > 1 else conc
        budget = r['budget'].get('o2', {}) if isinstance(r['budget'], dict) else {}
        fields[f'{spacing:g}_um'] = (r, conc)
        spacing_results[f'{spacing:g}_um'] = {
            'mean_uM': float(conc.mean()), 'min_uM': float(conc.min()), 'max_uM': float(conc.max()),
            'mean_mmHg': float(um_to_mmhg(conc).mean()),
            'min_mmHg': float(um_to_mmhg(conc).min()),
            'hypoxic_fraction_below_10mmHg': float(np.mean(conc < O2_HYPOXIC_UM)),
            'budget_residual_relative': float(np.max(np.abs(np.asarray(
                budget.get('residual_relative', [0.0]), float)))),
        }
    metrics['vessel_spacing'] = spacing_results

    r80, c80 = fields['80_um']
    r_far, c_far = fields['320_um']
    tear_arr = np.full(len(positions), 0.02)

    print('cell state: per-cell oxygen vs uniform mean, at two spacings ...')
    cellstate_cases = {}
    for label, conc in (('vessel_spacing_80um', c80), ('vessel_spacing_320um', c_far)):
        avail_case = np.clip(conc / O2_VENOUS_UM, 0.0, 5.0)
        pc = cellstate.run(n_cells=len(positions), t_end=300.0, dt=0.5, sample_dt=50.0,
                           oxygen=avail_case, tear=tear_arr)
        un = cellstate.run(n_cells=len(positions), t_end=300.0, dt=0.5, sample_dt=50.0,
                           oxygen=float(avail_case.mean()), tear=tear_arr)
        dead = pc['state_final'] >= 2
        cellstate_cases[label] = {
            'min_availability': float(avail_case.min()),
            'fraction_below_venous_reference': float(np.mean(avail_case < 1.0)),
            'per_cell_death_fraction': float(dead.mean()),
            'uniform_mean_death_fraction': float(un['death_fraction'][-1]),
            'difference_in_death_fraction': float(dead.mean() - un['death_fraction'][-1]),
            'mean_atp_final_uM': float(pc['atp'][-1].mean()),
        }
    metrics['cellstate'] = cellstate_cases
    metrics['cellstate_observation'] = (
        'Where a poorly perfused subpopulation exists, the per-cell run kills it while the '
        'uniform-mean run does not; where the tissue is uniformly well oxygenated the two '
        'agree exactly, so the per-cell input matters only when the field is heterogeneous. '
        'These are model verdicts, not measurements.')
    avail = np.clip(c_far / O2_VENOUS_UM, 0.0, 5.0)
    state_arr = cellstate.run(n_cells=len(positions), t_end=300.0, dt=0.5, sample_dt=50.0,
                              oxygen=avail, tear=tear_arr)['state_final']

    print('growth runs ...')
    monod = c_far / (K_O2_GROWTH_UM + c_far)
    g_unlim = growth.run(n_cells=len(positions), positions=positions, t_end_h=24.0, dt_h=0.5,
                         substrate_mode='unlimited', contact_inhibition=False)
    g_limited = growth.run(n_cells=len(positions), positions=positions, t_end_h=24.0, dt_h=0.5,
                           substrate_mode='reservoir', nutrient=np.clip(monod, 0, 1),
                           ks_su_per_um3=1.0, contact_inhibition=False)
    g_gated = growth.run(n_cells=len(positions), positions=positions, t_end_h=24.0, dt_h=0.5,
                         substrate_mode='unlimited', state=state_arr,
                         contact_inhibition=False)

    def summary(g):
        return {'n_initial': int(g['n_cells'][0]), 'n_final': int(g['n_cells'][-1]),
                'divisions': int(g['cumulative_divisions'][-1]),
                'mean_area_first_um2': float(g['mean_area'][0]),
                'mean_area_last_um2': float(g['mean_area'][-1]),
                'substrate_consumed_cum': float(g['substrate_consumed_cum'][-1])}
    metrics['growth'] = {'unlimited': summary(g_unlim), 'o2_limited_monod': summary(g_limited),
                         'injury_gated_unlimited': summary(g_gated)}
    metrics['growth_mapping'] = ('nutrient = c/(5 uM + c), so the O2 Monod half-saturation is '
                                 '5 uM; the growth module then applies its own Monod with ks=1.0. '
                                 'This composition is a modelling choice, not a measurement.')

    doses = [0.1, 1.0, 10.0, 100.0]
    resp = hormone.dose_response(doses)
    metrics['hormone'] = {'doses_nM': doses, 'response': resp.tolist(),
                          'growth_multiplier': hormone.effect_on_growth(resp).tolist(),
                          'langmuir_self_test': hormone.self_test()['binding_equilibrium']['max_abs_difference']}

    fig, axs = plt.subplots(2, 2, figsize=(13, 10), layout='constrained')
    ax = axs[0, 0]
    im = ax.imshow(np.asarray(r_far['fields']['o2'][-1]).T, origin='lower',
                   extent=[0, domain[0], 0, domain[1]], cmap='viridis')
    for v in vessels_vertical(domain, 160.0):
        ax.plot([v['path'][0][0]] * 2, [0, domain[1]], 'r-', lw=1.0, alpha=.8)
    ax.set(title='Tissue O2 (uM); red = prescribed vessels (320 um spacing)',
           xlabel='um', ylabel='um')
    fig.colorbar(im, ax=ax, label='O2 (uM)')

    ax = axs[0, 1]
    for spacing, (_, c) in fields.items():
        ax.hist(um_to_mmhg(c), bins=40, histtype='step', label=f'{spacing}')
    ax.axvline(O2_HYPOXIC_UM / UM_PER_MMHG, color='k', ls=':', label='10 mmHg')
    ax.set(xlabel='Per-cell PO2 (mmHg)', ylabel='Cells',
           title='Per-cell oxygen at three vessel spacings')
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    ax.plot(g_unlim['times_h'], g_unlim['n_cells'], label='unlimited substrate')
    ax.plot(g_limited['times_h'], g_limited['n_cells'], '--', label='O2-limited (Monod K=5 uM)')
    ax.plot(g_gated['times_h'], g_gated['n_cells'], ':', label='injury-gated (per-cell state)')
    ax.axhline(g_unlim['n_cells'][0] * 2 ** (24 / 20), color='k', ls=':', lw=.8,
               label='analytic doubling 20 h')
    ax.set(xlabel='Time (h)', ylabel='Cell number',
           title='Growth under supply and injury gates')
    ax.legend(fontsize=8)

    ax = axs[1, 1]
    ax.scatter(um_to_mmhg(c_far), avail, s=4, alpha=.3, label='per-cell availability')
    ax.axhline(1.0, color='k', ls=':', lw=.8, label='venous reference (56 uM)')
    ax.set(xlabel='Per-cell PO2 (mmHg)', ylabel='availability passed to cellstate',
           title='The O2 field now reaches cells individually (not as a mean)')
    ax.legend(fontsize=8)
    fig.suptitle('Organism-scale chain in physiological units: vessels -> local O2 -> per-cell state -> growth\n'
                 'Prescribed vessels, illustrative consumption, model verdicts not measurements', fontsize=11)
    fig.savefig(OUT / 'organism_scale.png', dpi=160)
    plt.close(fig)

    (OUT / 'metrics_organism.json').write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps({k: v for k, v in metrics.items() if k != 'units'}, indent=2, default=str)[:2200])


if __name__ == '__main__':
    main()
