"""Manually implemented illustrative Round3 demo, not validated physiology.

Event at 2 s changes chemical membrane permeability and a ligand barrier only:
no compartment deletion, instantaneous release, ATP, measured fly injury
threshold or viability conclusion. 'Neck lesion' is a scenario label, not neck
anatomy. Supported bath is an infinite K reservoir with finite exchange, not
oxygen/nutrition support. All numeric model inputs below are illustrative.

Electrical readout is explicitly ONE-WAY frozen-concentration multirate:
Ek at the START of each chemical interval drives an independent, additional K
leak at one CableNeuron endpoint through set_end_leak. Ordinary membrane leak
is NOT replaced or declared pure K. Conductance [uS] is not derived from the
chemical permeability [um/s]. Cable currents are NOT fed into pool amounts;
therefore this is NOT mass/charge-conserved electrodiffusion. Chemistry assumes
externally clamped electroneutral salt transport with untracked counterions.
"""
from pathlib import Path
from dataclasses import replace, asdict
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from engine.injury_tissue import InjuryPotassium, PotassiumConfig, InjuryLigand
from engine.cable import CableNeuron, Morphology
from run_injury_tissue_selftest import run_tests

OUT = Path(__file__).resolve().parent/'outputs'/'brain_isolation'


def main():
    tests = run_tests()
    OUT.mkdir(parents=True, exist_ok=True)
    base = PotassiumConfig(buffer_capacity_amount=0.)
    configs = {
        'intact_closed': base,
        'intact_glia': replace(base, buffer_capacity_amount=4000.),
        'neck_lesion_closed': base,
        'neck_lesion_glia': replace(base, buffer_capacity_amount=4000.),
        'neck_lesion_supported_bath': base,
    }
    potassium = {k: InjuryPotassium(c) for k, c in configs.items()}
    shape = (4, 1, 1)
    initial = np.array([8., 8., 0., 0.]).reshape(shape)
    capacity = np.array([0., 0., 2., 2.]).reshape(shape)
    ligand = {k: InjuryLigand(shape=shape, spacing_um=(5., 5., 5.),
        diffusion_um2_s=10., initial_nM=initial, receptor_capacity_nM=capacity,
        kon_nM_inv_s=.1, koff_s=.05, response_tau_s=2.,
        barrier_edges={(1, 2): 0.}) for k in configs}
    # Explicit parameters avoid accidentally presenting fitted cable defaults
    # as measured injury parameters. This tiny cylinder is synthetic geometry.
    cables = {k: CableNeuron(Morphology.cylinder(100., 2., 10), Cm_uF_cm2=1.,
        g_leak_S_cm2=.0001, Ra_ohm_cm=150., E_leak_mV=-65., dt_ms=.5)
        for k in configs}
    dt_s, duration_s, injury_s = .1, 20., 2.
    electrical_dt_ms = .5
    substeps = round(dt_s*1000/electrical_dt_ms)
    assert abs(substeps*electrical_dt_ms/1000-dt_s) < 1e-12
    extra_k_g_uS = .002
    traces = {k: [] for k in configs}
    max_k_errors = dict.fromkeys(configs, 0.)
    max_ligand_errors = dict.fromkeys(configs, 0.)

    def record(k):
        m, l, cable = potassium[k], ligand[k], cables[k]
        traces[k].append([m.time_s, m.inside_mM, m.outside_mM, m.buffer_amount,
            m.ek_mV, cable.v[-1], float(l.model.occupancy[2:].mean()),
            float(l.model.response[2:].mean()), m.mass_balance_error(),
            l.mass_balance_error(), m.ledger['reservoir_in']-m.ledger['reservoir_out']])

    for k in configs:
        record(k)
    for step in range(round(duration_s/dt_s)):
        if step == round(injury_s/dt_s):
            for k in configs:
                if k.startswith('neck_lesion'): 
                    potassium[k].apply_injury(.1, 80. if 'supported' in k else 0.)
                    ligand[k].switch_barrier({(1, 2): 50.})
        for k, m in potassium.items():
            cable = cables[k]
            # Additional diagnostic K conductance present throughout; no cut
            # in axial cable topology, to isolate the chemical reversal effect.
            cable.set_end_leak([cable.m.n-1], extra_k_g_uS, m.ek_mV)
            for _ in range(substeps):
                cable.step()
            m.step(dt_s)
            ligand[k].step(dt_s)
            max_k_errors[k] = max(max_k_errors[k], abs(m.mass_balance_error()))
            max_ligand_errors[k] = max(max_ligand_errors[k], abs(ligand[k].mass_balance_error()))
            record(k)
    assert potassium['neck_lesion_glia'].outside_mM < potassium['neck_lesion_closed'].outside_mM
    assert potassium['neck_lesion_supported_bath'].outside_mM < potassium['neck_lesion_closed'].outside_mM
    assert cables['neck_lesion_glia'].v[-1] < cables['neck_lesion_closed'].v[-1]
    assert ligand['intact_closed'].model.occupancy[2:].max() == 0
    assert ligand['neck_lesion_closed'].model.occupancy[2:].mean() > 0
    assert max(max_k_errors.values()) < 1e-7
    assert max(max_ligand_errors.values()) < 1e-8

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    panels = [(2, 'Free extracellular K', 'mM'), (3, 'Finite glial sequestration', 'mM um³'),
              (4, 'Diagnostic Nernst Ek', 'mV'), (5, 'Passive endpoint potential', 'mV'),
              (7, 'Ligand receptor slow response', 'dimensionless'),
              (10, 'Signed external K reservoir transfer', 'mM um³')]
    for ax, (col, title, unit) in zip(axes.flat, panels):
        for k, rows in traces.items():
            a = np.array(rows)
            ax.plot(a[:, 0], a[:, col], label=k.replace('_', ' '), lw=1.6)
        ax.axvline(injury_s, color='grey', ls='--', lw=1)
        ax.set(xlabel='time (s)', ylabel=unit, title=title)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('ILLUSTRATIVE injury chemistry → passive K-leak readout; NOT fly thresholds or viability\n'
                 'One-way clamped neutral-salt chemistry; no current feedback. Intact-glia extreme hyperpolarization is a model artifact:\n'
                 'unopposed finite-pool sequestration without pumps/homeostasis, NOT physiological glial protection.', fontsize=11)
    fig.savefig(OUT/'injury_coupling_demo.png', dpi=150)
    plt.close(fig)
    metrics = {
        'status': 'manually implemented prototype; numerical tests passed, not physiological validation',
        'provenance': 'All geometry, concentrations, kinetics, permeability, buffer, reservoir and cable inputs illustrative; R/F physical constants only.',
        'scope': __doc__,
        'units': {'K_concentration': 'mM', 'K_amount': 'mM*um^3', 'K_amount_to_mol': 1e-18,
                  'ligand_concentration': 'nM', 'ligand_amount': 'nM*um^3', 'ligand_amount_to_mol': 1e-24},
        'chemistry_dt_s': dt_s, 'electrical_dt_ms': electrical_dt_ms,
        'duration_s': duration_s, 'injury_time_s': injury_s,
        'added_K_leak_uS': extra_k_g_uS,
        'coupling': 'Start-of-interval Ek frozen for 200 passive cable substeps; electrical currents never returned to K pools; extra K conductance distinct from ordinary membrane leak.',
        'ligand_switch': 'Matched initial configurations, zero barrier to 50 um^3/s on injury; new LocalTissue public construction plus state/response transfer and cumulative segment ledger.',
        'model_insufficiency': 'Intact-glia buffer sequesters the finite extracellular pool without pumps or homeostatic replenishment. Near-depletion and extreme negative Ek/passive voltage are artifacts of this reduction, NOT physiological glial protection or calibrated resting voltages. No parameter tuning hides this limitation.',
        'comparison_note': 'All scenarios start with identical free K and zero bound K; glial capacity differs explicitly, allowing pre-event sequestration without any mass addition. Separate intact_closed and intact_glia controls expose this baseline difference. Intact K membrane permeability is zero. Ligand has no assigned molecular identity or direct electrical effect.',
        'trace_columns': ['time_s', 'Ki_mM', 'Ke_mM', 'glial_bound_mM_um3', 'Ek_mV',
                          'endpoint_mV', 'ligand_occupancy', 'ligand_slow_response',
                          'K_mass_error_mM_um3', 'ligand_mass_error_nM_um3', 'K_reservoir_net_mM_um3'],
        'selftests': tests,
        'scenarios': {k: {'config': asdict(configs[k]), 'final_Ki_mM': m.inside_mM,
            'final_Ke_mM': m.outside_mM, 'final_Ek_mV': m.ek_mV,
            'final_endpoint_mV': float(cables[k].v[-1]), 'final_buffer_amount': m.buffer_amount,
            'initial_K_amount': m.initial_amount, 'final_K_amount': m.total_amount(),
            'K_ledger': m.ledger, 'K_events': m.events,
            'max_K_mass_error': max_k_errors[k], 'max_ligand_mass_error': max_ligand_errors[k],
            'ligand_ledger': ligand[k].ledger, 'ligand_events': ligand[k].events,
            'ligand_final_response': float(ligand[k].model.response[2:].mean())}
            for k, m in potassium.items()}}
    (OUT/'injury_coupling_metrics.json').write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(OUT/'injury_coupling_traces.npz', **{k: np.array(v) for k, v in traces.items()})
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
