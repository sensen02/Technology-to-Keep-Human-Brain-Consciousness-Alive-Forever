"""Lineage/differentiation layer driven by the REAL per-cell oxygen field.

Chain: transport (prescribed vessels) -> per-cell O2 -> lineage.run(oxygen=...) ->
differentiated-type fraction as a function of local oxygen.

Honest scope: the lineage module's differentiation rates, thresholds and widths are
ILLUSTRATIVE (no measured source); the cell types are abstractions, not
transcriptionally defined types. A patterned outcome here tests the prescribed rule,
not real biology. The O2 field itself comes from prescribed vessel geometry with
illustrative per-cell consumption.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'

from run_organism import (build_cells, run_transport, um_to_mmhg, O2_VENOUS_UM,
                          O2_HYPOXIC_UM)


def main():
    import lineage

    metrics = {'scope': ('lineage patterning driven by a real per-cell oxygen field; '
                         'illustrative differentiation parameters; not evidence of real '
                         'differentiation')}
    domain, dx = (400.0, 400.0), 4.0
    positions = build_cells(n_side=61, domain_um=domain)
    print('transport (320 um vessel spacing) ...')
    r = run_transport(domain, dx, positions, 320.0)
    conc = np.asarray(r['cell_concentration']['o2'])
    conc = conc[-1] if conc.ndim > 1 else conc
    avail = np.clip(conc / O2_VENOUS_UM, 0.0, 5.0)
    metrics['oxygen'] = {'n_cells': int(len(positions)), 'mean_uM': float(conc.mean()),
                         'min_mmHg': float(um_to_mmhg(conc).min()),
                         'max_mmHg': float(um_to_mmhg(conc).max()),
                         'hypoxic_fraction': float(np.mean(conc < O2_HYPOXIC_UM))}

    print('lineage run driven by the O2 field ...')
    res = lineage.run(n_cells=len(positions), positions=positions, t_end_h=200.0,
                      seed=2025, rule='signal_bias', oxygen=avail)
    frac = {k: float(np.asarray(v)[-1]) for k, v in res['fractions_by_type'].items()}
    counts = {k: int(np.asarray(v)[-1]) for k, v in res['counts_by_type'].items()}
    metrics['lineage'] = {'rule': res['metadata'].get('rules_active'),
                          'final_fractions': frac, 'final_counts': counts,
                          'n_differentiated': int(np.asarray(res['n_differentiated'])[-1]),
                          'identity_residual': res['identity_residual'],
                          'event_rows': len(res['event_table']),
                          'verification': res.get('verification')}

    # patterning quality: differentiated fraction vs local oxygen, binned
    types_final = np.asarray(res['tree']['types'])
    alive = np.asarray(res['tree']['alive'], bool)
    n_nodes = types_final.shape[0]
    # Daughters inherit their parent's position, so build a per-node O2 estimate by
    # walking each node's root: roots keep the input cell order.
    parent_of = np.asarray(res['tree']['parents'], dtype=np.int64)
    root_ox = np.full(n_nodes, np.nan)
    root_ox[:len(avail)] = avail[:n_nodes]
    for i in range(n_nodes):
        j = i
        guard = 0
        while parent_of[j] >= 0 and guard < 10000:
            j = int(parent_of[j]); guard += 1
        root_ox[i] = avail[j] if j < len(avail) else np.nan
    mask = alive & np.isfinite(root_ox)
    is_myo = np.array([t == 'myocyte' for t in types_final])
    is_diff = np.array([t != 'stem' for t in types_final])
    mmhg = um_to_mmhg(root_ox[mask] * O2_VENOUS_UM)
    diff = is_diff[mask]
    myo = is_myo[mask]
    bins = np.array([0, 10, 20, 30, 40, 60, 1e9])
    idx = np.digitize(mmhg, bins) - 1
    binned = []
    for b in range(len(bins) - 1):
        sel = idx == b
        if sel.sum() > 0:
            binned.append({'po2_range_mmHg': f'{bins[b]:g}-{bins[b+1]:g}',
                           'n_cells': int(sel.sum()),
                           'differentiated_fraction': float(diff[sel].mean()),
                           'myocyte_fraction': float(myo[sel].mean())})
    metrics['patterning'] = {
        'differentiated_fraction_overall': float(diff.mean()),
        'myocyte_fraction_overall': float(myo.mean()),
        'by_local_po2': binned,
        'note': ('Root O2 is the input field for founder cells; daughters inherit the '
                 'parent position, so this shows patterning against the oxygen the lineage '
                 'started from, not a remeasured field.'),
    }

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), layout='constrained')
    ax = axs[0]
    for name, series in res['fractions_by_type'].items():
        ax.plot(res['times_h'], series, label=name)
    ax.set(xlabel='Time (h)', ylabel='Fraction of cells',
           title='Cell-type fractions over time (illustrative rates)')
    ax.legend(fontsize=8)
    ax = axs[1]
    for name, series in res['counts_by_type'].items():
        ax.plot(res['times_h'], series, label=name)
    ax.set(xlabel='Time (h)', ylabel='Cells', title='Type counts')
    ax.legend(fontsize=8)
    ax = axs[2]
    labels = [b['po2_range_mmHg'] for b in metrics['patterning']['by_local_po2']]
    values = [b['myocyte_fraction'] for b in metrics['patterning']['by_local_po2']]
    ns = [b['n_cells'] for b in metrics['patterning']['by_local_po2']]
    ax.bar(range(len(values)), values, color='tab:orange')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f'{l}\nn={n}' for l, n in zip(labels, ns)], fontsize=8)
    ax.set(xlabel='Local PO2 at the founder cell (mmHg)', ylabel='Myocyte fraction',
           title='Patterning from the real O2 field\n(rule-dependent, illustrative parameters)')
    fig.suptitle('Differentiation layer driven by a real per-cell O2 field (prescribed vessels)\n'
                 'Differentiation rules and rates are illustrative; not evidence of real differentiation',
                 fontsize=11)
    fig.savefig(OUT / 'organism_lineage.png', dpi=160)
    plt.close(fig)

    (OUT / 'metrics_lineage_coupling.json').write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps(metrics, indent=2, default=str)[:1800])


if __name__ == '__main__':
    main()
