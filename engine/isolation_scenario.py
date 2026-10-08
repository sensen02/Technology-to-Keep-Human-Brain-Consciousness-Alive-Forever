"""Integrated IDEALIZED graph experiment, hand implemented, not fly physiology.

Point eye -> point brain -> classical squid-HH neck -> point body. Names are
region identities in a deliberately invented graph, NOT reconstructed anatomy.
A neck cut disconnects a cable junction, not the eye or the graph's output edge.
No controller, metabolic physiology, injury chemistry or survival model.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from .neural_active import IdealCableSpec
from .neural_cond import ConductanceParams
from .neural_hybrid import HybridNetwork, ConductanceEdge
from .local_tissue import LocalTissue, NonPhysiologicalSupportProxy
from .electrode import Contact, BipolarField, VoltageRecorder


@dataclass(frozen=True)
class IsolationConfig:
    scenario: str = 'intact'
    duration_ms: float = 120.
    dt_ms: float = 0.025
    phase_ms: float = 40.
    chemical_dt_ms: float = 1.
    seed: int = 7
    visual_mean_nA: float = 0.08
    visual_modulation_nA: float = 0.025
    visual_period_ms: float = 25.
    stimulus_start_ms: float = 80.
    stimulus_duration_ms: float = 1.
    stimulus_nA: float = 200.
    chemical_initial_nM: float = 4.
    chemical_target_gain_uS: float = 0.
    support_target_gain_nA: float = 0.
    recorder_noise_sd_mV: float = 0.
    cut_child_section: int = 1

    def validate(self):
        if self.scenario not in ('intact', 'sham', 'neck_cut', 'eye_loss', 'neck_cut_supported'):
            raise ValueError('unknown scenario')
        for name, value in asdict(self).items():
            if name != 'scenario' and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError(name + ' must be finite')
        if not 0 < self.dt_ms <= .1 or not 0 < self.phase_ms < self.duration_ms:
            raise ValueError('positive baseline, post phase and dt <= .1 required')
        for name in ('duration_ms', 'phase_ms', 'chemical_dt_ms', 'stimulus_start_ms', 'stimulus_duration_ms'):
            x = getattr(self, name) / self.dt_ms
            if x < 0 or not math.isclose(x, round(x), abs_tol=1e-8, rel_tol=0):
                raise ValueError(name + ' must align to neural clock')
        if self.chemical_dt_ms < self.dt_ms or not math.isclose(self.duration_ms / self.chemical_dt_ms, round(self.duration_ms / self.chemical_dt_ms)):
            raise ValueError('duration must align to slower chemical clock')
        if not math.isclose(self.phase_ms / self.chemical_dt_ms, round(self.phase_ms / self.chemical_dt_ms)):
            raise ValueError('phase must align to chemical clock')
        if self.visual_period_ms <= 0 or self.visual_mean_nA < abs(self.visual_modulation_nA):
            raise ValueError('nonnegative visual waveform and positive period required')
        for name in ('chemical_initial_nM', 'chemical_target_gain_uS', 'support_target_gain_nA', 'recorder_noise_sd_mV', 'stimulus_duration_ms'):
            if getattr(self, name) < 0:
                raise ValueError(name + ' must be nonnegative')
        if type(self.seed) is not int or self.seed < 0 or type(self.cut_child_section) is not int or self.cut_child_section not in (1, 2):
            raise ValueError('nonnegative integer seed and child 1 or 2 required')


def run_isolation_scenario(config=None):
    """Return arrays and JSON-compatible metadata; never write files.

    State probes include t=0 and every interval END. Inputs are held over
    [input_times_ms, input_times_ms+dt); slow state advances only at completed
    chemical intervals and is available to the NEXT neural step (no lookahead).
    Optional chemical mapping adds an excitatory brain conductance increment
    gain*response*(1-exp(-dt/tau_exc)); gain is the steady-state conductance.
    This explicit HYPOTHESIS is off by default, not an identified pathway.
    """
    c = config or IsolationConfig()
    c.validate()
    n = round(c.duration_ms / c.dt_ms)
    slow_every = round(c.chemical_dt_ms / c.dt_ms)
    phase_step = round(c.phase_ms / c.dt_ms)
    spec = IdealCableSpec(nseg_per_section=11)
    params = ConductanceParams()
    edges = (ConductanceEdge('eye', 'brain', 'exc', .012, 1.),
             ConductanceEdge('brain', 'neck', 'exc', .5, 1.),
             ConductanceEdge('neck', 'body', 'exc', .012, 1.))
    initial = np.zeros((3, 2, 2)); initial[0] = c.chemical_initial_nM
    tissue = LocalTissue((3, 2, 2), spacing_um=(10., 10., 10.), diffusion_um2_s=1000.,
                         initial_nM=initial, receptor_capacity_nM=2.,
                         kon_nM_inv_s=50., koff_s=10., response_tau_s=.02)
    support = NonPhysiologicalSupportProxy((1,), initial=.8)
    # Explicit local->world rigid transform: local +x becomes world +y.
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    translation = np.array([100., -1500., 50.])
    contacts = (Contact((140., -1200., 50.), 20.), Contact((140., 1200., 50.), 20.))
    field = BipolarField(contacts, .3, (100., 2500., 50.))
    recorder = VoltageRecorder((140., -1100., 50.), (100., 2500., 50.),
                               noise_sd_mV=c.recorder_noise_sd_mV, seed=c.seed)
    state = {name: np.zeros(n+1) for name in ('eye_mV', 'brain_mV', 'body_mV',
             'neck_proximal_mV', 'neck_distal_mV', 'chemical_response',
             'chemical_free_target_nM', 'chemical_clock_s', 'support_proxy')}
    inputs = {name: np.zeros(n) for name in ('visual_command_nA', 'eye_mapping', 'eye_input_nA',
              'stimulus_source_nA', 'field_proximal_mV', 'field_distal_mV',
              'artifact_unfiltered_mV', 'artifact_recorded_mV', 'chemical_response_used',
              'chemical_clock_used_s', 'chemical_increment_uS', 'support_current_nA')}
    events = []
    with HybridNetwork(('eye', 'brain', 'neck', 'body'), ('eye', 'brain', 'body'),
                       {'neck': spec}, edges, params, c.dt_ms) as net:
        cable = net.active.cables['neck']
        world = cable.segment_xyz_um() @ rotation.T + translation
        unit_field = field.potential_mV(world, (1., -1.))

        def sample(i):
            for gid in net.point_ids:
                state[gid + '_mV'][i] = net.point.v[net.point_index[gid]]
            voltage = cable.voltage_mV()
            state['neck_proximal_mV'][i], state['neck_distal_mV'][i] = voltage[0], voltage[-1]
            state['chemical_response'][i] = tissue.response[0, 0, 0]
            state['chemical_free_target_nM'][i] = tissue.free_nM[0, 0, 0]
            state['chemical_clock_s'][i] = tissue.time_s
            state['support_proxy'][i] = support.availability[0]

        sample(0)
        for k in range(n):
            t = k * c.dt_ms
            if k == phase_step:
                kind = 'phase_noop'
                if c.scenario in ('neck_cut', 'neck_cut_supported'):
                    cable.cut_axial(c.cut_child_section)
                    kind = 'axial_disconnect_sealed_ends'
                elif c.scenario == 'eye_loss':
                    kind = 'eye_input_mapping_zero'
                events.append({'event_id': 'phase:0', 'time_ms': t, 'kind': kind,
                               'child_section': c.cut_child_section if 'cut' in c.scenario else None})
            post = k >= phase_step
            mapping = 0. if post and c.scenario == 'eye_loss' else 1.
            visual = c.visual_mean_nA + c.visual_modulation_nA * np.sin(2*np.pi*t/c.visual_period_ms)
            stim = c.stimulus_nA if c.stimulus_start_ms <= t < c.stimulus_start_ms+c.stimulus_duration_ms else 0.
            cable.set_extracellular_voltage_mV(unit_field*stim)
            response = float(tissue.response[0, 0, 0])
            increment = c.chemical_target_gain_uS * response * (-np.expm1(-c.dt_ms/params.exc_tau_ms))
            # External state increment, not a recurrent graph edge or hidden current.
            net.point.ge[net.point_index['brain']] += increment
            support_current = c.support_target_gain_nA * (support.availability[0] - .8)
            rec = recorder.step(c.dt_ms, stimulus_field=field, stimulus_currents_nA=(stim, -stim))
            vals = (visual, mapping, visual*mapping, stim, unit_field[0]*stim, unit_field[-1]*stim,
                    rec['artifact_unfiltered_mV'], rec['measured_mV'], response, tissue.time_s,
                    increment, support_current)
            for name, value in zip(inputs, vals):
                inputs[name][k] = value
            net.step([visual*mapping, support_current, 0.])
            if (k+1) % slow_every == 0:
                dt_s = c.chemical_dt_ms / 1000.
                tissue.step(dt_s)
                supply = 0. if post and c.scenario == 'neck_cut' else 40.
                support.step(dt_s, supply_s=supply, demand_s=10.)
            sample(k+1)
        deliveries = [{'event_id': 'syn:'+str(e.event_id), **{key: value for key, value in asdict(e).items() if key != 'event_id'}} for e in net.deliveries]
        emissions = [{'event_id': 'spike:'+str(i), 'time_ms': t, 'region': gid} for i, (t, gid) in enumerate(net.emissions)]
        metrics = {}
        for gid in net.global_ids:
            for phase, lo, hi in (('baseline', 0., c.phase_ms), ('post', c.phase_ms, c.duration_ms)):
                count = sum(lo < t <= hi and g == gid for t, g in net.emissions)
                metrics[gid+'_'+phase+'_spikes'] = count
                metrics[gid+'_'+phase+'_rate_Hz'] = count*1000/(hi-lo)
        metrics['chemical_mass_balance_error_nM_um3'] = tissue.mass_balance_error()
        metrics['final_chemical_time_s'] = tissue.time_s
        metrics['final_neural_time_ms'] = net.t_ms
        metrics['cut_children'] = sorted(cable.cut_children)
    return {'config': asdict(c), 'times_ms': np.arange(n+1)*c.dt_ms,
            'input_times_ms': np.arange(n)*c.dt_ms, 'probes': state, 'inputs': inputs,
            'events': events, 'deliveries': deliveries, 'emissions': emissions, 'metrics': metrics,
            'regions': {'eye': 'ideal visual-input point', 'brain': 'ideal brain point',
                        'neck': 'ideal 3-section squid-HH cable', 'body': 'ideal body output point'},
            'units': {'times_ms': 'ms', '*_mV': 'mV (Vm except explicitly field/artifact)',
                      '*_nA': 'nA', '*_uS': 'uS', '*_nM': 'nM', '*_s': 's',
                      'chemical_response': 'dimensionless', 'support_proxy': 'dimensionless nonphysiological availability'},
            'provenance': {'status': 'HAND-IMPLEMENTED IDEALIZED small circuit, NOT fly physiology/anatomy',
                'graph': [asdict(e) for e in edges], 'point': params.provenance(), 'cable': spec.provenance(),
                'initialization': 'fixed rest -65 mV, zero synaptic state, deterministic ligand patch; seed only recorder noise',
                'chemical': '3x2x2 grid; 10 um spacing; D=1000 um2/s; capacity=2 nM; kon=50/(nM*s), koff=10/s, response tau=.02 s; deliberately FAST illustrative kinetics; no time rescaling',
                'chemical_mapping': 'HYPOTHESIS: voxel [0,0,0] response -> brain exc conductance; gain default0; constant response gives pre-decay voltage-evaluation steady g=gain*response, stored end-step g is exp(-dt/tau) smaller; held past slow state only',
                'chemical_boundaries': {'external_faces': 'no flux', 'baths': [], 'finite_pools': [], 'source_amount_s': 0., 'clearance_s': 0., 'initial_bound_nM': 0., 'initial_free_nM_grid': initial.tolist()},
                'structural_caveat': 'unidirectional ideal graph: unchanged upstream brain after neck cut is structurally guaranteed at zero support coupling, NOT evidence of isolation resilience',
                'support_mapping': 'HYPOTHESIS optional brain current gain_nA*(proxy-.8), default0; proxy supply40/s demand10/s, cut supply0, supported retains40/s; NOT oxygen/ATP or survival',
                'field': 'homogeneous .3 S/m bipolar spherical-volume contacts; prescribed field on neck segments only; point cells have no field geometry',
                'contacts': [asdict(x) for x in contacts], 'field_reference_um': [100., 2500., 50.],
                'rotation_local_to_world': rotation.tolist(), 'translation_um': translation.tolist(),
                'segment_world_um': world.tolist(),
                'recorder': 'artifact-only field sample plus 1000 Hz bandwidth/noise added after filtering; NO neural source reconstruction; never Vm',
                'recorder_geometry_um': {'position': [140., -1100., 50.], 'reference': [100., 2500., 50.]},
                'stimulus_charge_caveat': 'spatial source/return balance only; temporally monophasic pulse, no electrode electrochemistry or charge-safety claim',
                'clock': 'inputs START, spikes END; chemical ms/1000, update at completed slow interval, then next neural input',
                'cut': 'sealed axial disconnect only; no membrane damage; pending pre-cut spikes may still deliver; eye-loss zeros drive mapping only',
                'controller': 'none'}}
