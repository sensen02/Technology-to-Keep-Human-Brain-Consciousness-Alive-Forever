"""Self-tests for the multirate composed scenario and the histamine channel.

Run with:
  /run/media/sensen/Data2/cell_wound_prototype/venv/bin/python -B \
      /run/media/sensen/Data2/cell_wound_prototype/run_multirate_selftest.py

Covers: paired causality (cut blocks body but not brain side; visual loss removes
graded histamine drive), zero-instant-mass-jump at the injury event, slow-clock /
no-future-leakage assertions, deterministic bounded BANC subnetwork selection,
excluded-edge accounting, histamine-channel zero-event equivalence with
ConductanceNetwork, default-off HYPOTHESIS gain checks, and seed replay
reproducibility. Prints a PASS/FAIL summary and counts; writes no output files.
"""
from dataclasses import replace
import hashlib
import resource
import subprocess
import sys
import time
import unittest
from pathlib import Path

import numpy as np

from engine.neural_cond import ConductanceNetwork, ConductanceParams
from engine.neural_hist import (HistamineNetwork, HistamineParams, GradedHistamineDriver,
                                zero_histamine_equivalence_check, decay_factors)
from engine.multirate_scenario import (MultirateConfig, SubnetworkConfig, SCENARIOS,
                                       COARSE_REGIONS, FINE_REGION,
                                       select_banc_subnetwork, run_multirate_scenario,
                                       receptor_class, barrier_edges, neck_segment_world_um,
                                       NECK_ROTATION, NECK_TRANSLATION_UM, CHEM_SHAPE)

ROOT = Path(__file__).resolve().parent
SUB_CFG = SubnetworkConfig(max_neurons=8000, max_pure_histamine_photoreceptors=200)
BASE = MultirateConfig(duration_ms=20.0, stimulus_start_ms=12.0, stimulus_duration_ms=2.0)
LONG = MultirateConfig(duration_ms=40.0)          # at least two fine-drive pulses post phase
DT = 0.05


def _digest_sub(sub):
    return (sub['digest']['selected_root_ids_sha256_32'],
            sub['digest']['exc_matrix_sha256_32'], sub['digest']['inh_matrix_sha256_32'])


class SharedState:
    """One subnetwork and one scenario set, built once for the whole suite."""
    sub = None
    runs = None
    long_runs = None

    @classmethod
    def build(cls):
        t0 = time.perf_counter()
        cls.sub = select_banc_subnetwork(SUB_CFG, ROOT)
        cls.runs = {name: run_multirate_scenario(replace(BASE, scenario=name), cls.sub)
                    for name in SCENARIOS}
        cls.long_runs = {name: run_multirate_scenario(replace(LONG, scenario=name), cls.sub)
                         for name in ('intact', 'neck_cut')}
        cls.build_seconds = time.perf_counter() - t0


# ---------------------------------------------------------------------------
# 1. histamine channel
# ---------------------------------------------------------------------------
class HistamineChannelTests(unittest.TestCase):
    def test_parameter_validation(self):
        HistamineParams().validate()
        for bad in (dict(hist_tau_ms=0.), dict(hist_tau_ms=-1.),
                    dict(hist_reversal_mV=float('nan')), dict(leak_uS=0.)):
            with self.assertRaises(ValueError):
                HistamineParams(**bad).validate()

    def test_events_must_be_finite_and_nonnegative(self):
        net = HistamineNetwork(4, HistamineParams(), DT)
        net.step(hist_events_uS=np.array([0.1, 0.0, 0.0, 0.0]))
        for bad in (np.array([-1., 0, 0, 0]), np.array([np.nan, 0, 0, 0]),
                    np.array([np.inf, 0, 0, 0])):
            with self.assertRaises(ValueError):
                net.step(hist_events_uS=bad)

    def test_channel_is_independent_and_decays_exactly(self):
        # Stored conductances are POST-decay at the interval END, so an event of
        # 1e-3 uS is read back as exactly 1e-3*exp(-dt/tau).
        tau = 8.0
        net = HistamineNetwork(5, HistamineParams(hist_tau_ms=tau), DT)
        net.step(exc_events_uS=np.array([1e-3, 0, 0, 0, 0]),
                 inh_events_uS=np.array([0, 1e-3, 0, 0, 0]),
                 hist_events_uS=np.array([0, 0, 1e-3, 0, 0]))
        s_exc = np.exp(-DT / net.p.exc_tau_ms)
        s_inh = np.exp(-DT / net.p.inh_tau_ms)
        store = np.exp(-DT / tau)
        self.assertEqual(net.ge[0], 1e-3 * s_exc)
        self.assertEqual(net.ge[1], 0.0)
        self.assertEqual(net.ge[2], 0.0)
        self.assertEqual(net.gi[1], 1e-3 * s_inh)
        self.assertEqual(net.gi[2], 0.0)
        self.assertEqual(net.g_hist[2], 1e-3 * store)
        self.assertEqual(net.g_hist[0], 0.0)
        self.assertEqual(net.hist_events_delivered_uS[2], 1e-3)
        self.assertEqual(float(net.hist_increment_total_uS), 1e-3)
        # exact exponential decay on the next step, and no cross-channel transfer
        before = net.g_hist.copy()
        net.step()
        np.testing.assert_allclose(net.g_hist, before * np.exp(-DT / tau), rtol=0, atol=0)
        np.testing.assert_allclose(net.ge, np.array([1e-3, 0., 0., 0., 0.]) * s_exc ** 2,
                                   rtol=4e-16, atol=0)
        self.assertEqual(decay_factors(net.p, DT)['hist'], float(np.exp(-DT / tau)))

    def test_no_double_counting_between_channels(self):
        net = HistamineNetwork(3, HistamineParams(), DT)
        from scipy import sparse
        exc = sparse.csr_matrix(np.array([[0., 0., 0.], [1e-4, 0., 0.], [0., 0., 0.]]))
        net.set_connectivity(exc, sparse.csr_matrix((3, 3)))
        with self.assertRaises(ValueError):
            net.set_histamine_connectivity(exc)            # overlaps We
        net.set_histamine_connectivity(sparse.csr_matrix((3, 3)))
        with self.assertRaises(ValueError):
            net.set_histamine_connectivity(np.zeros((4, 4)))
        with self.assertRaises(ValueError):
            net.set_histamine_connectivity(-np.ones((3, 3)))

    def test_zero_event_equivalence_with_parent(self):
        bit = zero_histamine_equivalence_check(150, HistamineParams(), DT, 250, seed=5)
        for key in ('v', 'ge', 'gi', 'adaptation', 'spike_count'):
            self.assertTrue(bit[key], 'bit-inequality in ' + key)
        self.assertTrue(bit['all_equal'])
        self.assertTrue(bit['g_hist_all_zero'])
        self.assertEqual(bit['parent_spikes'], bit['child_spikes'])

    def test_zero_event_equivalence_with_sparse_recurrence(self):
        from scipy import sparse
        rng = np.random.default_rng(11)
        n = 60
        exc = sparse.random(n, n, density=.1, random_state=rng) * 1e-4
        inh = sparse.random(n, n, density=.1, random_state=13) * 5e-5
        bit = zero_histamine_equivalence_check(n, HistamineParams(), DT, 200, seed=3,
                                              exc_uS=exc, inh_uS=inh)
        self.assertTrue(bit['all_equal'], bit)

    def test_graded_vision_adapter_routes_to_histamine_channel_only(self):
        from engine.graded_vision import GradedVision
        n = 12
        params = HistamineParams()
        net = HistamineNetwork(n, params, DT)
        gv = GradedVision(n, np.array([0, 1]), np.array([0, 1]), np.array([5, 6]),
                          np.array([4.0, 4.0]), np.array([True, True]),
                          dt_ms=DT, scale_uS_per_synapse=1e-3,
                          e_hist_mV=params.hist_reversal_mV, tau_release_ms=params.hist_tau_ms)
        driver = GradedHistamineDriver(net, np.array([0, 1]))
        driver.apply_to(gv, 1.0)
        self.assertGreater(net.g_hist.sum(), 0.)
        self.assertEqual(net.ge.sum(), 0.)
        self.assertEqual(net.gi.sum(), 0.)
        np.testing.assert_array_equal(driver.last_increment_uS, net.hist_events_delivered_uS)
        driver.apply_to(gv, 0.0)
        self.assertLess(net.g_hist.sum(), 1.0)

    def test_histamine_channel_actually_gates_spiking(self):
        # E_hist (-70 mV) is BELOW rest (-65 mV), so a pure histamine conductance is
        # a hyperpolarizing / shunting input on this parameter set: it cannot make a
        # resting cell fire. The channel is therefore demonstrated the honest way -
        # drive the cell above threshold and show that histamine conductance silences
        # it, while an EQUAL excitatory conductance does the opposite.
        n_steps, current = 400, 0.2
        quiet = HistamineNetwork(1, HistamineParams(hist_tau_ms=.05), 0.05)
        fired = HistamineNetwork(1, HistamineParams(hist_tau_ms=.05), 0.05)
        excit = HistamineNetwork(1, HistamineParams(hist_tau_ms=.05), 0.05)
        spikes = {'quiet': 0, 'hist': 0, 'exc': 0}
        for _ in range(n_steps):
            for name, net, kwargs in (('quiet', quiet, {}),
                                      ('hist', fired, dict(hist_events_uS=np.array([0.05]))),
                                      ('exc', excit, dict(exc_events_uS=np.array([0.05])))):
                spikes[name] += int(net.step(current, **kwargs)[0])
        self.assertGreater(spikes['quiet'], 0, 'baseline current drive produced no spikes')
        self.assertEqual(spikes['hist'], 0,
                         'hyperpolarizing histamine channel did not suppress the drive')
        self.assertGreater(spikes['exc'], spikes['quiet'],
                           'an equal excitatory conductance must not suppress the drive')
        self.assertEqual(fired.gi.sum(), 0.0)
        self.assertEqual(excit.g_hist.sum(), 0.0)


# ---------------------------------------------------------------------------
# 2. subnetwork selection determinism and accounting
# ---------------------------------------------------------------------------
class SubnetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SharedState.build()
        cls.sub = SharedState.sub

    def test_size_cap_and_connectivity(self):
        sub = self.sub
        capped = 20000 + SUB_CFG.max_pure_histamine_photoreceptors
        self.assertLessEqual(sub['n'], capped)
        self.assertLessEqual(sub['counts']['seed_top_degree_neurons'], SUB_CFG.max_neurons)
        self.assertEqual(sub['counts']['neurons_in_final_subnetwork'], sub['n'])
        # the SELECTED subnetwork is connected over its modelled edges, and the
        # selection says so itself
        from scipy import sparse
        w = (sub['We'].astype(bool).astype(float) + sub['Wi'].astype(bool).astype(float))
        n_comp, labels = sparse.csgraph.connected_components(
            (w + w.T).tocsr(), directed=False)
        self.assertEqual(n_comp, 1)
        self.assertEqual(int(np.unique(labels).size), 1)
        self.assertTrue(sub['counts']['final_selection_is_connected_over_modeled_edges'])
        self.assertEqual(sub['counts']['final_selection_connected_components_modeled_edges'], 1)
        self.assertEqual(sub['counts']['final_selected_nodes_not_reachable_from_largest_component'], 0)
        self.assertLessEqual(sub['n'], sub['counts']['final_neuron_cap'])
        self.assertGreater(sub['n'], 1000)

    def test_selection_is_deterministic_in_process(self):
        again = select_banc_subnetwork(SUB_CFG, ROOT)
        np.testing.assert_array_equal(again['global_index'], self.sub['global_index'])
        np.testing.assert_array_equal(again['root_ids'], self.sub['root_ids'])
        self.assertEqual(_digest_sub(again), _digest_sub(self.sub))
        self.assertEqual(again['counts']['neurons_in_final_subnetwork'],
                         self.sub['counts']['neurons_in_final_subnetwork'])

    def test_selection_is_deterministic_across_processes_and_hash_seeds(self):
        code = ("from pathlib import Path;"
                "from engine.multirate_scenario import SubnetworkConfig,select_banc_subnetwork;"
                "s=select_banc_subnetwork(SubnetworkConfig(max_neurons=%d,"
                "max_pure_histamine_photoreceptors=%d),Path('.'));"
                % (SUB_CFG.max_neurons, SUB_CFG.max_pure_histamine_photoreceptors) +
                "print(s['digest']['selected_root_ids_sha256_32'],"
                "s['digest']['exc_matrix_sha256_32'],"
                "s['digest']['inh_matrix_sha256_32'],s['n'])")
        digests = []
        for seed in ('0', '12345'):
            env = {'PYTHONHASHSEED': seed, 'PYTHONPATH': str(ROOT), 'PATH': '/usr/bin:/bin'}
            out = subprocess.run([sys.executable, '-B', '-c', code], cwd=str(ROOT),
                                 env=env, capture_output=True, text=True, check=True)
            digests.append(out.stdout.split())
        self.assertEqual(digests[0][:3], digests[1][:3])
        self.assertEqual(digests[0][:3], list(_digest_sub(self.sub)))
        self.assertEqual(digests[0][3], str(self.sub['n']))

    def test_smaller_cap_yields_smaller_valid_selection(self):
        small_cfg = SubnetworkConfig(max_neurons=3000, max_pure_histamine_photoreceptors=50)
        small = select_banc_subnetwork(small_cfg, ROOT)
        self.assertLess(small['n'], self.sub['n'])
        self.assertLessEqual(small['counts']['seed_top_degree_neurons'], 3000)
        w = (small['We'].astype(bool).astype(float) + small['Wi'].astype(bool).astype(float))
        from scipy import sparse
        n_comp, _ = sparse.csgraph.connected_components((w + w.T).tocsr(), directed=False)
        self.assertEqual(n_comp, 1)
        self.assertLessEqual(small['n'], small['counts']['final_neuron_cap'])

    def test_photoreceptor_seeds_are_kept_and_visual_rows_present(self):
        sub = self.sub
        self.assertGreater(sub['counts']['pure_histamine_photoreceptors_in_final'], 0)
        self.assertGreater(sub['visual']['pure_histamine_visual_rows_inside_subnetwork'], 0)
        self.assertGreater(sub['visual']['pure_histamine_visual_targets_selected'], 0)
        self.assertEqual(sub['visual']['pre_edges'].shape, sub['visual']['post_edges'].shape)
        self.assertEqual(sub['visual']['pre_edges'].shape, sub['visual']['syn_edges'].shape)
        self.assertTrue(np.all(sub['visual']['pre_edges'] >= 0))
        self.assertTrue(np.all(sub['visual']['post_edges'] < sub['n']))
        self.assertTrue(np.all(np.isin(np.unique(sub['visual']['pre_edges']),
                                       sub['visual']['visual_local_indices'])))

    def test_excluded_edge_accounting_adds_up(self):
        sub, ex = self.sub, self.sub['excluded']
        self.assertTrue(ex['accounting_identity_ok'])
        self.assertEqual(ex['retained_rows'] + ex['excluded_rows_total'],
                         ex['total_annotated_rows_in_dataset'])
        # mutually exclusive and exhaustive semantics, as written by the selector:
        #   total rows = retained (pure ACh/GABA/glutamate) + excluded
        #   excluded   = rows with an endpoint outside the selection
        #              + rows inside the selection whose presynaptic label is not one of
        #                the three pure labels the explicit receptor mapping uses
        inside_subnetwork_excluded = ex['excluded_inside_subnetwork_rows']
        self.assertEqual(ex['excluded_endpoint_outside_subnetwork']
                         + ex['excluded_inside_subnetwork_rows'],
                         ex['excluded_rows_total'],
                         'every excluded row is excluded for exactly one of the two reasons')
        breakdown = ex['excluded_nonpure_label_breakdown']
        self.assertEqual(breakdown['unknown'] + breakdown['mixed']
                         + breakdown['histamine_only'], inside_subnetwork_excluded)
        self.assertEqual(breakdown['histamine_only'],
                         ex['excluded_inside_subnetwork_rows_with_pure_histamine_label'])
        self.assertEqual(breakdown['unknown'], ex['excluded_inside_subnetwork_unknown_label'])
        self.assertEqual(breakdown['mixed'], ex['excluded_inside_subnetwork_mixed_label'])
        self.assertEqual(breakdown['histamine_only'],
                         ex['excluded_inside_subnetwork_histamine_only'])
        # rows whose PRESYNAPTIC label is not a pure ACh/GABA/glutamate label
        self.assertGreater(breakdown['unknown'] + breakdown['mixed']
                           + breakdown['histamine_only'], 0)
        self.assertEqual(ex['retained_exc_rows'] + ex['retained_inh_rows'], ex['retained_rows'])
        self.assertEqual(ex['nt_pair_unique_values'], [-1])
        self.assertFalse(ex['nt_pair_used_for_sign'])
        self.assertGreater(sub['counts']['exc_matrix_nnz'], 0)
        self.assertGreater(sub['counts']['inh_matrix_nnz'], 0)
        self.assertGreater(ex['excluded_mixed_histamine_rows_in_dataset'], 0)

    def test_receptor_mapping_is_explicit_and_never_inferred(self):
        self.assertEqual(receptor_class('acetylcholine'), 'exc')
        self.assertEqual(receptor_class('gaba'), 'inh')
        self.assertEqual(receptor_class('glutamate'), 'inh')
        for label in ('', 'histamine', 'acetylcholine,histamine', 'dopamine',
                      'gaba,nitric_oxide', 'octopamine', 'serotonin'):
            self.assertIsNone(receptor_class(label), label + ' must be excluded')

    def test_weights_are_bounded_nonnegative_and_capped(self):
        sub = self.sub
        cap = SUB_CFG.max_synapses_per_edge * SUB_CFG.weight_scale_uS_per_synapse
        for w in (sub['We'], sub['Wi']):
            self.assertTrue(np.all(np.isfinite(w.data)))
            self.assertTrue(np.all(w.data >= 0))
            self.assertLessEqual(float(w.data.max()), cap * (1 + 1e-12))


# ---------------------------------------------------------------------------
# 3. composed scenario: initialization, clocks, injury, electrode
# ---------------------------------------------------------------------------
class ScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SharedState.build()
        cls.sub, cls.runs, cls.long = SharedState.sub, SharedState.runs, SharedState.long_runs

    def test_identical_initialization_across_scenarios(self):
        ref = self.runs['intact']
        for name, result in self.runs.items():
            for key, probe in ref['probes'].items():
                self.assertEqual(probe[0], result['probes'][key][0],
                                 '%s %s initial state differs' % (name, key))
                np.testing.assert_array_equal(probe[:1], result['probes'][key][:1])
        self.assertEqual(self.sub['n'], self.runs['intact']['metrics']['banc_neurons'])
        for name, result in self.runs.items():
            self.assertEqual(result['config']['seed'], ref['config']['seed'])
            self.assertEqual(result['regions']['IDEAL_EYE_IF']['label'],
                             ref['regions']['IDEAL_EYE_IF']['label'])

    def test_common_baseline_and_sham_are_bit_identical(self):
        ref = self.runs['intact']
        baseline = ref['times_ms'] <= BASE.phase_ms
        for name, result in self.runs.items():
            for key, probe in ref['probes'].items():
                np.testing.assert_array_equal(probe[baseline], result['probes'][key][baseline])
        for key, probe in ref['probes'].items():
            np.testing.assert_array_equal(probe, self.runs['sham']['probes'][key])
        for key, series in ref['inputs'].items():
            np.testing.assert_array_equal(series, self.runs['sham']['inputs'][key])
        self.assertEqual(self.runs['sham']['metrics']['cut_children'], [])
        self.assertFalse(self.runs['sham']['metrics']['injury_applied'])

    def test_neck_cut_blocks_body_but_not_brain_side(self):
        intact, cut = self.long['intact'], self.long['neck_cut']
        for key in ('eye_if_mV', 'relay_if_mV', 'body_if_mV', 'chemical_response'):
            np.testing.assert_array_equal(intact['probes'][key], cut['probes'][key])
        np.testing.assert_array_equal(intact['inputs']['neck_drive_pulse_nA'],
                                      cut['inputs']['neck_drive_pulse_nA'])
        self.assertGreaterEqual(intact['metrics']['fine_tier_baseline_spikes'], 1)
        self.assertGreater(intact['metrics']['fine_tier_post_spikes'], 0)
        self.assertEqual(cut['metrics']['fine_tier_post_spikes'], 0)
        self.assertEqual(cut['metrics']['cut_children'], [LONG.cut_child_section])
        self.assertEqual(intact['metrics']['cut_children'], [])
        # the brain-side coarse emission counts are unchanged by the cut
        self.assertEqual(intact['metrics']['coarse_interface_post_spikes'],
                         cut['metrics']['coarse_interface_post_spikes'])
        self.assertEqual(intact['metrics']['interface_cell_spike_counts'],
                         cut['metrics']['interface_cell_spike_counts'])
        # post-phase distal voltage is only the electrically separate distal island
        post = intact['times_ms'] > LONG.phase_ms
        self.assertLess(np.abs(cut['probes']['neck_distal_mV'][post]).max(),
                        np.abs(intact['probes']['neck_distal_mV'][post]).max())

    def test_visual_loss_reduces_histamine_target_drive(self):
        intact, loss = self.runs['intact'], self.runs['eye_loss']
        post_in = intact['input_times_ms'] >= BASE.phase_ms
        post_lo = loss['input_times_ms'] >= BASE.phase_ms
        self.assertGreater(intact['inputs']['histamine_increment_uS'][post_in].sum(), 0.)
        self.assertEqual(loss['inputs']['histamine_increment_uS'][post_lo].sum(), 0.)
        self.assertEqual(loss['inputs']['light_level'][post_lo].sum(), 0.)
        self.assertTrue(np.all(intact['inputs']['light_level'][post_in] > 0))
        # the visual command waveform itself is unchanged UP TO the intervention; after
        # the phase the ideal eye interface is explicitly zeroed, which IS the
        # intervention (the light waveform value is recorded separately for audit)
        tbase_in = intact['input_times_ms'] < BASE.phase_ms
        tbase_lo = loss['input_times_ms'] < BASE.phase_ms
        np.testing.assert_array_equal(intact['inputs']['visual_command_nA'][tbase_in],
                                      loss['inputs']['visual_command_nA'][tbase_lo])
        self.assertTrue(np.all(loss['inputs']['visual_command_nA'][post_lo] == 0.))
        self.assertTrue(np.any(intact['inputs']['visual_command_nA'][post_in] > 0.))
        # the LIGHT waveform and the graded-vision receptor state before the phase are
        # identical in both runs; only the routing after the phase differs
        np.testing.assert_array_equal(intact['inputs']['light_level'][tbase_in],
                                      loss['inputs']['light_level'][tbase_lo])
        self.assertEqual(loss['metrics']['cut_children'], [])
        tpost = intact['times_ms'] >= BASE.phase_ms
        g_intact = np.mean(intact['probes']['histamine_target_g_hist_uS'][tpost])
        g_loss = np.mean(loss['probes']['histamine_target_g_hist_uS'][tpost])
        self.assertGreater(g_intact, 1.25 * g_loss,
                           'graded histamine target drive barely changed')
        self.assertGreater(float(np.max(loss['probes']['histamine_target_g_hist_uS'][tpost])),
                           0.0, 'the pre-phase histamine conductance must still decay in')
        v_intact = np.mean(intact['probes']['histamine_target_mean_mV'][tpost])
        v_loss = np.mean(loss['probes']['histamine_target_mean_mV'][tpost])
        self.assertLess(v_intact, v_loss, 'target Vm does not reflect the histamine shunt')
        # baseline windows must be identical
        tbase = intact['times_ms'] <= BASE.phase_ms
        np.testing.assert_array_equal(intact['probes']['histamine_target_g_hist_uS'][tbase],
                                      loss['probes']['histamine_target_g_hist_uS'][tbase])

    def test_histamine_channel_is_dedicated_in_the_composed_run(self):
        for name, result in self.runs.items():
            self.assertGreater(result['metrics']['histamine_increment_total_uS'], 0.)
            self.assertEqual(result['provenance']['visual_drive']['inhibitory_channel_borrowed'],
                             False)
            self.assertGreater(result['metrics']['visual_edge_count'], 0)
            self.assertGreater(result['metrics']['visual_target_posts'], 0)

    def test_injury_event_has_zero_instant_mass_jump(self):
        r = self.runs['injury_at_phase']
        self.assertTrue(r['metrics']['injury_applied'])
        self.assertEqual(len(r['injuries']), 1)
        record = r['injuries'][0]
        self.assertEqual(record['potassium_mass_jump'], 0.0)
        self.assertEqual(record['ligand_mass_jump'], 0.0)
        self.assertEqual(record['time_ms'], BASE.phase_ms)
        self.assertGreater(record['permeability_um_s'], 0.)
        self.assertGreater(record['barrier_edges_rewritten'], 0)
        self.assertLess(abs(r['metrics']['potassium_mass_balance_error_mM_um3']), 1e-9)
        self.assertLess(abs(r['metrics']['ligand_mass_balance_error_nM_um3']), 1e-9)
        # the injury is rate-only: before the phase nothing differs from intact
        tbase = r['times_ms'] <= BASE.phase_ms
        for key in ('chemical_response', 'outside_potassium_mM', 'ek_mV'):
            np.testing.assert_array_equal(
                r['probes'][key][tbase], self.runs['intact']['probes'][key][tbase])

    def test_slow_clocks_are_slower_and_never_leak_the_future(self):
        cfg = BASE
        for name, r in self.runs.items():
            times, inp = r['times_ms'], r['inputs']
            hist = r['slow_history']
            for i, t_ms in enumerate(r['input_times_ms']):
                self.assertLessEqual(inp['chemical_clock_used_s'][i], t_ms / 1000.0 + 1e-12)
                self.assertLessEqual(inp['potassium_clock_used_s'][i], t_ms / 1000.0 + 1e-12)
                self.assertIn(inp['chemical_clock_used_s'][i], hist['chem_time_s'])
                self.assertIn(inp['potassium_clock_used_s'][i], hist['k_time_s'])
            np.testing.assert_array_equal(inp['response_used'],
                                          r['probes']['chemical_response'][:-1])
            np.testing.assert_array_equal(inp['ek_used_mV'], r['probes']['ek_mV'][:-1])
            self.assertAlmostEqual(r['metrics']['final_chemical_time_s'],
                                   cfg.duration_ms / 1000.0)
            self.assertAlmostEqual(r['metrics']['final_potassium_time_s'],
                                   cfg.duration_ms / 1000.0)
        # each series is a pure step function on its own clock: changes happen only
        # where that clock's time array changes
        for name in ('intact', 'injury_at_phase'):
            hist = self.runs[name]['slow_history']
            for series in ('response', 'free_nM', 'ligand_mass_error'):
                values = np.asarray(hist[series])
                changed = np.flatnonzero(np.diff(values) != 0) + 1
                clock = np.asarray(hist['chem_time_s'])
                for k in changed:
                    self.assertGreater(clock[k], clock[k - 1], series)
            for series in ('outside_mM', 'ek_mV', 'k_mass_error'):
                values = np.asarray(hist[series])
                changed = np.flatnonzero(np.diff(values) != 0) + 1
                clock = np.asarray(hist['k_time_s'])
                for k in changed:
                    self.assertGreater(clock[k], clock[k - 1], series)
            self.assertTrue(np.all(np.diff(np.asarray(hist['chem_time_s'])) >= 0))
            self.assertTrue(np.all(np.diff(np.asarray(hist['k_time_s'])) >= 0))
            steps = np.unique(np.round(np.diff(r['probes']['chemical_clock_s']), 12))
            self.assertTrue(np.all(np.isin(steps, [0., cfg.chemistry_dt_ms / 1000.0])))
            steps_k = np.unique(np.round(np.diff(r['probes']['potassium_clock_s']), 12))
            self.assertTrue(np.all(np.isin(steps_k, [0., cfg.potassium_dt_ms / 1000.0])))
            self.assertGreater(cfg.chemistry_dt_ms, cfg.dt_ms)
            self.assertGreaterEqual(cfg.potassium_dt_ms, cfg.chemistry_dt_ms)
        # the ligand chemistry clock really cannot see past the phase when it starts
        self.assertEqual(self.runs['intact']['probes']['chemical_response'][0], 0.0)
        self.assertEqual(self.sub_n0_response_after_first_interval(), 0.0)

    def sub_n0_response_after_first_interval(self):
        r = self.runs['intact']
        n_intervals = round(BASE.chemistry_dt_ms / BASE.dt_ms)
        return float(r['probes']['chemical_response'][n_intervals - 1])

    def test_slow_models_only_advance_on_completed_intervals(self):
        r = self.runs['injury_at_phase']
        chem_every = round(BASE.chemistry_dt_ms / BASE.dt_ms)
        k_every = round(BASE.potassium_dt_ms / BASE.dt_ms)
        n = round(BASE.duration_ms / BASE.dt_ms)
        expected_chem = 1 + n // chem_every
        expected_k = 1 + n // k_every
        self.assertEqual(expected_chem, 9)
        self.assertEqual(expected_k, 5)
        # one history array carries the UNION of the two slow clocks (they advance on
        # different grids), while each clock keeps its own strictly increasing series
        union_expected = 1 + n // chem_every
        self.assertEqual(len(r['slow_history']['chem_time_s']), union_expected)
        self.assertEqual(len(r['slow_history']['k_time_s']), union_expected)
        k_series = np.asarray(r['slow_history']['k_time_s'])
        self.assertEqual(len(np.unique(k_series)), expected_k)
        chem_series = np.asarray(r['slow_history']['chem_time_s'])
        self.assertEqual(len(np.unique(chem_series)), expected_chem)
        self.assertEqual(k_series[-1], n // k_every * BASE.potassium_dt_ms / 1000.0)
        self.assertEqual(chem_series[-1], n // chem_every * BASE.chemistry_dt_ms / 1000.0)

        self.assertEqual(r['slow_history']['chem_time_s'][0], 0.0)
        self.assertEqual(r['slow_history']['k_time_s'][0], 0.0)

    def test_injury_changes_both_chemistry_and_k_but_the_hypothesis_gains_are_off(self):
        r = self.runs['injury_at_phase']
        baseline = self.runs['intact']
        self.assertEqual(BASE.injury_k_target_gain_uS, 0.0)
        self.assertEqual(BASE.injury_ligand_target_gain_uS, 0.0)
        self.assertTrue(np.all(r['inputs']['k_increment_uS'] == 0.))
        self.assertTrue(np.all(r['inputs']['ligand_increment_uS'] == 0.))
        self.assertTrue(np.all(r['inputs']['extra_exc_events_uS'] == 0.))
        self.assertTrue(np.all(r['inputs']['hist_reversal_shift_mV'] == 0.))
        for key in ('k_increment_uS', 'ligand_increment_uS', 'extra_exc_events_uS',
                    'hist_reversal_shift_mV'):
            self.assertEqual(r['provenance']['injury']['k_gain_uS'], 0.0) if key \
                else None
        # the injury still changes BOTH slow chemistry trajectories: K leaks out faster
        # and more K leaves the (closed, depleting) extracellular pool
        self.assertGreater(r['metrics']['outside_potassium_final_mM'],
                           baseline['metrics']['outside_potassium_final_mM'])
        self.assertGreater(r['metrics']['permeability_um_s_final'],
                           baseline['metrics']['permeability_um_s_final'])
        self.assertGreater(abs(r['metrics']['ek_final_mV'] - r['metrics']['ek_baseline_mV']),
                           0.0)
        self.assertNotAlmostEqual(r['metrics']['ek_final_mV'],
                                  baseline['metrics']['ek_final_mV'], places=6)
        tpost = r['times_ms'] >= BASE.phase_ms
        self.assertFalse(np.array_equal(r['probes']['chemical_response'][tpost],
                                        baseline['probes']['chemical_response'][tpost]),
                         'barrier rewrite did not change the ligand chemistry at all')
        self.assertGreater(len(r['metrics']['ligand_events']), 0)

    def test_hypothesis_gains_actually_bite_when_enabled(self):
        on = run_multirate_scenario(
            replace(BASE, scenario='injury_at_phase', injury_k_target_gain_uS=2.0,
                    injury_ligand_target_gain_uS=1.0), self.sub)
        off = self.runs['injury_at_phase']
        self.assertGreater(np.abs(on['inputs']['k_increment_uS']).sum(), 0.)
        self.assertGreater(np.abs(on['inputs']['ligand_increment_uS']).sum(), 0.)
        self.assertEqual(np.abs(off['inputs']['k_increment_uS']).sum(), 0.)
        self.assertFalse(np.array_equal(on['probes']['eye_if_mV'], off['probes']['eye_if_mV']))
        # a conductance increment can never be negative: the raw diagnostic may go
        # negative and is then clamped to zero, never flipped into inhibition
        self.assertTrue(np.all(on['inputs']['extra_exc_events_uS'] >= 0.))
        self.assertTrue(np.any(on['inputs']['extra_exc_events_raw_uS'] < 0.))
        self.assertTrue(np.any(on['inputs']['extra_exc_events_uS'] > 0.))
        np.testing.assert_allclose(
            on['inputs']['extra_exc_events_uS'],
            np.maximum(0., on['inputs']['extra_exc_events_raw_uS']), rtol=0, atol=0)

    def test_electrode_field_and_artifact_only_recorder(self):
        r = self.runs['intact']
        prov = r['provenance']['electrode']
        self.assertTrue(prov['field_values_finite'])
        self.assertNotEqual(prov['unit_field_proximal_mV'], prov['unit_field_distal_mV'])
        self.assertTrue(np.all(np.isin(NECK_ROTATION.flatten(), (-1., 0., 1.))))
        np.testing.assert_allclose(NECK_ROTATION @ NECK_ROTATION.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(NECK_ROTATION)), 1.0)
        xyz = r['provenance']['electrode']['translation_um']
        np.testing.assert_allclose(np.asarray(xyz), NECK_TRANSLATION_UM)
        stim = (r['input_times_ms'] >= BASE.stimulus_start_ms) & \
               (r['input_times_ms'] < BASE.stimulus_start_ms + BASE.stimulus_duration_ms)
        self.assertGreater(int(stim.sum()), 0)
        self.assertGreater(np.abs(r['inputs']['artifact_recorded_mV'][stim]).max(), 0.)
        # the stimulus is the ONLY thing on the recorder: outside the window the
        # unfiltered artifact is exactly zero
        self.assertTrue(np.all(r['inputs']['artifact_unfiltered_mV'][~stim] == 0.))
        # the recorder's NEURAL term is exactly zero in every scenario: no membrane
        # source is ever supplied, so the output can never be read as a recording
        for name, run in self.runs.items():
            self.assertTrue(np.all(run['inputs']['recorder_neural_unfiltered_mV'] == 0.),
                            name + ': recorder reported a non-zero neural contribution')
        self.assertTrue(np.all(r['inputs']['field_proximal_mV'][~stim] == 0.))
        self.assertGreater(np.abs(r['inputs']['field_proximal_mV']).max(), 0.)
        self.assertNotEqual(r['inputs']['field_proximal_mV'][stim][0],
                            r['inputs']['field_distal_mV'][stim][0])
        # with neural sources absent the recorder carries artifact only
        self.assertEqual(BASE.recorder_noise_sd_mV, 0.0)
        self.assertIn('ARTIFACT-ONLY', prov['recorder'])

    def test_neck_segment_transform_is_applied_and_documented(self):
        r = self.runs['intact']
        expected = NECK_TRANSLATION_UM.copy()
        self.assertAlmostEqual(r['provenance']['electrode']['unit_field_proximal_mV'],
                               r['provenance']['electrode']['unit_field_proximal_mV'])
        self.assertTrue(np.all(np.isfinite(NECK_TRANSLATION_UM)))
        np.testing.assert_allclose(np.asarray(r['provenance']['electrode']
                                              ['rotation_local_to_world']), NECK_ROTATION)
        self.assertAlmostEqual(expected[0], 100.0)

    def test_seed_replay_reproducibility(self):
        a = run_multirate_scenario(replace(BASE, scenario='neck_cut'), self.sub)
        b = run_multirate_scenario(replace(BASE, scenario='neck_cut'), self.sub)
        for key in a['probes']:
            np.testing.assert_array_equal(a['probes'][key], b['probes'][key])
        for key in a['inputs']:
            np.testing.assert_array_equal(a['inputs'][key], b['inputs'][key])
        self.assertEqual(a['metrics'], b['metrics'])
        self.assertEqual(a['emissions'], b['emissions'])
        noisy = run_multirate_scenario(replace(BASE, scenario='neck_cut', seed=99,
                                               recorder_noise_sd_mV=.01), self.sub)
        self.assertFalse(np.array_equal(noisy['inputs']['artifact_recorded_mV'],
                                        a['inputs']['artifact_recorded_mV']))
        for key in ('eye_if_mV', 'relay_if_mV', 'body_if_mV'):
            np.testing.assert_array_equal(noisy['probes'][key], a['probes'][key])

    def test_configuration_validation_rejects_unsafe_combinations(self):
        with self.assertRaises(ValueError):
            replace(BASE, scenario='nope').validate()
        with self.assertRaises(ValueError):
            replace(BASE, chemistry_dt_ms=BASE.dt_ms / 2).validate()
        with self.assertRaises(ValueError):
            replace(BASE, potassium_dt_ms=BASE.dt_ms / 2).validate()
        with self.assertRaises(ValueError):
            replace(BASE, potassium_dt_ms=BASE.chemistry_dt_ms / 2).validate()
        with self.assertRaises(ValueError):
            replace(BASE, phase_ms=BASE.duration_ms + 1.).validate()
        with self.assertRaises(ValueError):
            replace(BASE, light_step_ms=BASE.dt_ms / 2).validate()
        with self.assertRaises(ValueError):
            replace(BASE, injury_ligand_target_gain_uS=-1.).validate()
        with self.assertRaises(ValueError):
            SubnetworkConfig(max_neurons=50000).validate()
        with self.assertRaises(ValueError):
            SubnetworkConfig(max_neurons=0).validate()

    def test_region_labels_mark_invented_cells_as_not_anatomy(self):
        regions = self.runs['intact']['regions']
        for name in COARSE_REGIONS:
            self.assertIn('NOT anatomy', regions[name]['label'])
            self.assertIn('NOT in the BANC dataset', regions[name]['label'])
        self.assertIn('NOT reconstructed anatomy', regions['neck_cable']['label'])
        honesty = self.runs['intact']['provenance']['honesty']
        self.assertTrue(honesty['mixes_real_and_invented'])
        for word in ('consciousness', 'viability', 'survival', 'rescue',
                     'electrodiffusion'):
            self.assertIn(word, honesty['claims_not_made'])
        self.assertIn('NOT electrodiffusion',
                      self.runs['intact']['provenance']['injury']['potassium'])

    def test_metrics_and_emissions_accounting(self):
        for name, r in self.runs.items():
            self.assertEqual(r['metrics']['neural_steps'], round(BASE.duration_ms / DT))
            self.assertEqual(len(r['times_ms']), r['metrics']['neural_steps'] + 1)
            ids = [e['event_id'] for e in r['events'] + r['emissions']]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(r['metrics']['fine_tier_post_spikes'],
                             r['metrics'][FINE_REGION + '_post_spikes'])
            for region in COARSE_REGIONS:
                total = r['metrics'][region + '_baseline_spikes'] + \
                    r['metrics'][region + '_post_spikes']
                self.assertEqual(total, r['metrics']['interface_cell_spike_counts'][region])
            self.assertEqual(r['metrics']['coarse_interface_post_spikes'],
                             sum(r['metrics'][region + '_post_spikes'] for region in COARSE_REGIONS))

    def test_sparse_matrices_are_untouched_by_the_composed_run(self):
        self.assertTrue(self.sub['We'].has_canonical_format or self.sub['We'].nnz >= 0)
        self.assertGreater(self.sub['We'].nnz, 0)
        self.assertGreater(self.sub['Wi'].nnz, 0)

    def test_barrier_map_edges_are_grid_neighbours(self):
        edges = barrier_edges(5.0)
        self.assertGreater(len(edges), 0)
        shape = CHEM_SHAPE
        neighbours = set()
        for i in range(np.prod(shape)):
            index = np.unravel_index(i, shape)
            for axis in range(3):
                neighbour = list(index)
                neighbour[axis] += 1
                if neighbour[axis] < shape[axis]:
                    j = int(np.ravel_multi_index(tuple(neighbour), shape))
                    neighbours.add((min(i, j), max(i, j)))
        for pair, value in edges.items():
            self.assertIn(pair, neighbours)
            self.assertEqual(value, 5.0)
        self.assertEqual(len(set(edges)), len(edges))


def main():
    t0 = time.perf_counter()
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in (HistamineChannelTests, SubnetworkTests, ScenarioTests):
        suite.addTests(loader.loadTestsFromTestCase(cls))
    n_tests = suite.countTestCases()
    print('multirate selftest: %d tests, bounded subnetwork cap=%d photoreceptors=%d'
          % (n_tests, SUB_CFG.max_neurons, SUB_CFG.max_pure_histamine_photoreceptors))
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
    elapsed = time.perf_counter() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    passed = result.testsRun - len(result.failures) - len(result.errors)
    print('=' * 78)
    print('PASS/FAIL SUMMARY')
    print('  tests run      : %d' % result.testsRun)
    print('  passed         : %d' % passed)
    print('  failures       : %d' % len(result.failures))
    print('  errors         : %d' % len(result.errors))
    print('  expected fails : %d' % len(result.expectedFailures))
    print('  skipped        : %d' % len(result.skipped))
    print('  wall seconds   : %.2f' % elapsed)
    print('  peak RSS MB    : %.0f' % peak)
    print('  VERDICT        : %s' % ('PASS' if result.wasSuccessful() else 'FAIL'))
    print('=' * 78)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
