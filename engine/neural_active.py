"""Bounded, idealized active NEURON cables -- NOT fly-calibrated physiology.

Classical NEURON ``hh`` (squid illustrative kinetics, 6.3 degC) with its OWN
leak/capacitance/resistivity; DN_PASSIVE is NOT mixed into this active model.
No morphology import, injury membrane chemistry, electrode field solver, or
whole-network performance claim. Only one exclusive runtime per process.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np


@dataclass(frozen=True)
class IdealCableSpec:
    length_um: float = 3000.0
    diameter_um: float = 20.0
    sections: int = 3
    nseg_per_section: int = 21
    cm_uF_cm2: float = 1.0
    ra_ohm_cm: float = 35.4

    def validate(self):
        for name in ('length_um', 'diameter_um', 'cm_uF_cm2', 'ra_ohm_cm'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + ' must be finite positive')
        for name in ('sections', 'nseg_per_section'):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(name + ' must be a positive integer')
        if self.nseg_per_section % 2 != 1:
            raise ValueError('odd nseg required')
        if self.sections * self.nseg_per_section > 1000:
            raise ValueError('bounded prototype: at most 1000 segments per cable')

    def provenance(self):
        return {
            'geometry': {'status': 'ideal uniform cylinder, NOT reconstructed',
                         'parameters': asdict(self)},
            'active': {'status': 'illustrative classical squid HH, NOT fly calibrated',
                       'mechanism': 'NEURON built-in hh', 'temperature_degC': 6.3,
                       'gnabar_S_cm2': 0.12, 'gkbar_S_cm2': 0.036,
                       'ena_mV': 50.0, 'ek_mV': -77.0},
            'passive': {'status': 'squid demonstration assumptions; NOT DN_PASSIVE fits',
                        'cm_uF_cm2': self.cm_uF_cm2, 'ra_ohm_cm': self.ra_ohm_cm,
                        'hh_leak_S_cm2': 0.0003, 'hh_leak_reversal_mV': -54.3},
            'fly_passive_reference_not_used': 'engine.cable.DN_PASSIVE: DNp01/DNp03 PMC11071487',
        }


class ActiveCable:
    """Construct via ActiveRuntime.add_cable; probe and drive at segment centers.

    Flattened segment order = section order then x order. Default input site 0,
    output spike site last segment. Output is an upward 0 mV crossing at a grid
    END timestamp (no interpolated event time). No threshold/reset in HH cable.
    """
    def __init__(self, runtime, gid, spec, receptors):
        self.runtime, self.gid, self.spec = runtime, gid, spec
        h = runtime.h
        self.sections = []
        self.cut_children = set()
        for i in range(spec.sections):
            sec = h.Section(name=f'ideal_{gid}_{i}')
            self.sections.append(sec)
            sec.L, sec.diam = spec.length_um / spec.sections, spec.diameter_um
            sec.nseg, sec.cm, sec.Ra = spec.nseg_per_section, spec.cm_uF_cm2, spec.ra_ohm_cm
            sec.insert('hh')
            sec.insert('extracellular')
            sec.ena, sec.ek = 50.0, -77.0
            for seg in sec:
                seg.hh.gnabar, seg.hh.gkbar = 0.12, 0.036
                seg.hh.gl, seg.hh.el = 0.0003, -54.3
                for layer in range(2):
                    seg.xg[layer], seg.xc[layer], seg.xraxial[layer] = 1e9, 0.0, 1e9
                seg.e_extracellular = 0.0
            if i:
                sec.connect(self.sections[i-1](1), 0)
        self.segments = [seg for sec in self.sections for seg in sec]
        self.clamps = []
        for seg in self.segments:
            clamp = h.IClamp(seg)
            clamp.delay, clamp.dur, clamp.amp = 0.0, 1e9, 0.0
            self.clamps.append(clamp)
        self.synapses = {}
        for name, (reversal, tau) in receptors.items():
            syn = h.ExpSyn(self.segments[0])
            syn.e, syn.tau = reversal, tau
            self.synapses[name] = syn
        self.spike_count = 0

    def _check(self):
        self.runtime._check()

    def voltage_mV(self):
        """Transmembrane voltage at every segment center (copy)."""
        self._check()
        return np.array([seg.v for seg in self.segments])

    def segment_xyz_um(self):
        """Ideal straight cable along +x, origin (0,0,0), segment-center sites.

        Different cells share this local frame; caller must explicitly transform
        to world coordinates for electrode coupling (no anatomical placement).
        """
        self._check()
        n = len(self.segments)
        xyz = np.zeros((n, 3))
        xyz[:, 0] = (np.arange(n) + 0.5) * self.spec.length_um / n
        return xyz

    def extracellular_mV(self):
        """Solved extracellular layer-0 voltage, distinct from command."""
        self._check()
        return np.array([seg.vext[0] for seg in self.segments])

    def _vector(self, values):
        a = np.broadcast_to(np.asarray(values, float), (len(self.segments),))
        if not np.isfinite(a).all():
            raise ValueError('input must be finite scalar or per-segment vector')
        return a

    def set_inward_current_nA(self, values):
        """Set held current per segment; positive INWARD. Scalar drives EACH segment.

        Use a zero vector with one nonzero entry for focal injection. This is
        IClamp membrane injection, NOT an extracellular electrode field model.
        """
        self._check()
        for clamp, value in zip(self.clamps, self._vector(values)):
            clamp.amp = float(value)

    def set_extracellular_voltage_mV(self, values):
        """Held prescribed field command e_extracellular per segment.

        Near-ideal grounded bath (xg=1e9 S/cm2, xc=0, xraxial=1e9 Mohm/cm).
        This voltage interface accepts a future field solver's output, but does
        NOT compute electrode impedance, tissue fields, or recording potentials.
        """
        self._check()
        for seg, value in zip(self.segments, self._vector(values)):
            seg.e_extracellular = float(value)

    def add_conductance_uS(self, receptor, increment):
        """Immediate START-of-interval ExpSyn state increment, in uS.

        All receptor synapses are at input segment 0. NEURON integrates true
        g*(v-E) currents implicitly; g decays with ExpSyn's cnexp kinetics.
        No local NetCons, recurrent graph, or parallel event queue exists here.
        """
        self._check()
        if not self.runtime.initialized:
            raise RuntimeError('reset before delivering events')
        if receptor not in self.synapses:
            raise ValueError('unknown receptor')
        if not math.isfinite(increment) or increment < 0:
            raise ValueError('conductance increment must be finite nonnegative')
        self.synapses[receptor].g += float(increment)

    def cut_axial(self, child_section):
        """Disconnect section's parent junction; new ends are sealed, irreversible.

        No deletion, huge-resistance approximation, membrane loss or wound leak.
        Can be applied at a step boundary, including before initialization.
        """
        self._check()
        if isinstance(child_section, bool) or not isinstance(child_section, int) or not 0 < child_section < len(self.sections):
            raise ValueError('child section must be a nonroot section index')
        if child_section not in self.cut_children:
            self.runtime.h.disconnect(sec=self.sections[child_section])
            self.cut_children.add(child_section)


class ActiveRuntime:
    """Exclusive fixed-dt global NEURON owner, <=8 cables / <=1000 total segments.

    Refuses pre-existing NEURON sections and concurrent runtimes rather than
    silently resetting unrelated simulations. Use context manager or close().
    reset() resets ALL owned cables/time/solver, cuts persist; close deletes ONLY
    owned sections and restores prior scalar solver settings. Not thread-safe;
    do not create external sections or manipulate hoc globals while open.
    """
    _owner = None

    def __init__(self, dt_ms=0.025, receptors=None):
        if not math.isfinite(dt_ms) or not 0 < dt_ms <= 0.1:
            raise ValueError('active dt must be in (0, 0.1] ms')
        receptor_map = dict(receptors or {'exc': (0.0, 3.0), 'inh': (-70.0, 8.0)})
        for name, (rev, tau) in receptor_map.items():
            if not isinstance(name, str) or not math.isfinite(rev) or not math.isfinite(tau) or tau <= 0:
                raise ValueError('invalid receptor reversal/tau')
        from neuron import h
        if ActiveRuntime._owner is not None or list(h.allsec()):
            raise RuntimeError('NEURON global state is occupied; use an isolated process')
        h.load_file('stdrun.hoc')
        self.h, self.dt, self.receptors = h, float(dt_ms), receptor_map
        self._saved = (h.dt, h.celsius, int(h.secondorder), int(h.CVode().active()), h.t, h.steps_per_ms)
        self.cables, self.steps = {}, 0
        self.closed, self.initialized = False, False
        ActiveRuntime._owner = self

    def _check(self):
        if self.closed or ActiveRuntime._owner is not self:
            raise RuntimeError('active runtime is closed/not owner')

    def add_cable(self, gid, spec=None):
        self._check()
        if self.initialized:
            raise RuntimeError('fixed topology: add cables before reset')
        spec = spec or IdealCableSpec()
        spec.validate()
        count = sum(len(c.segments) for c in self.cables.values())
        if gid in self.cables or len(self.cables) >= 8 or count + spec.sections * spec.nseg_per_section > 1000:
            raise ValueError('duplicate ID or active runtime capacity exceeded')
        cell = ActiveCable(self, gid, spec, self.receptors)
        self.cables[gid] = cell
        return cell

    @property
    def t_ms(self):
        return self.steps * self.dt

    def reset(self, voltage_mV=-65.0):
        self._check()
        if not math.isfinite(voltage_mV):
            raise ValueError('finite initialization voltage required')
        h = self.h
        h.CVode().active(0)
        h.dt, h.steps_per_ms, h.celsius, h.secondorder = self.dt, 1 / self.dt, 6.3, 0
        for cell in self.cables.values():
            cell.set_inward_current_nA(0.0)
            cell.set_extracellular_voltage_mV(0.0)
            cell.spike_count = 0
        h.finitialize(voltage_mV)
        self.steps, self.initialized = 0, True

    def step(self):
        self._check()
        h = self.h
        if not self.initialized:
            raise RuntimeError('call reset before stepping')
        if (abs(h.t - self.t_ms) > 1e-7 or abs(h.dt - self.dt) > 1e-12 or
                h.CVode().active() or h.celsius != 6.3 or h.secondorder != 0):
            raise RuntimeError('NEURON global solver state changed outside owner')
        previous = {gid: cell.segments[-1].v for gid, cell in self.cables.items()}
        h.fcurrent()  # apply current, field and conductance discontinuities
        h.fadvance()
        self.steps += 1
        spikes = set()
        for gid, cell in self.cables.items():
            if previous[gid] < 0 <= cell.segments[-1].v:
                spikes.add(gid)
                cell.spike_count += 1
        return spikes

    def close(self):
        if self.closed:
            return
        self._check()
        for cell in self.cables.values():
            cell.synapses.clear()
            cell.clamps.clear()
            cell.segments.clear()
            for sec in cell.sections:
                self.h.delete_section(sec=sec)
            cell.sections.clear()
        h = self.h
        h.dt, h.celsius, h.secondorder, cvode, h.t, h.steps_per_ms = self._saved
        h.CVode().active(cvode)
        self.closed, self.initialized = True, False
        ActiveRuntime._owner = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
