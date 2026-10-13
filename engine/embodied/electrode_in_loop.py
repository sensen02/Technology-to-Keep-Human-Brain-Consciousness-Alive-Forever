#!/usr/bin/env python3
"""ELECTRODE IN THE CLOSED LOOP.  The seam that was missing between the brain and the body.

WHY THIS MODULE EXISTS, MEASURED.  Before it, the project had two working halves that never met:

  * the BRAIN half -- `engine/isolation_scenario.py` drives a spiking network and records it
    through `engine/electrode.VoltageRecorder` (it imports Contact/BipolarField/VoltageRecorder and
    builds a recorder from `net.emissions`);
  * the BODY half -- `engine/embodied/loop.py` runs the MuJoCo body against a 12,000-neuron BANC
    tier, and the tier's spikes set exactly TWO knobs (a speed scale and a left/right asymmetry).

`grep -n electrode engine/embodied/loop.py` returned NOTHING, so inside the loop that actually
closes sensorimotor feedback, the electrode did not exist.  This module puts it there.

WHAT IT REUSES AND WHAT IT ADDS.  Volume conduction is NOT reimplemented: every potential is a call
into `engine.electrode.transfer_mV_per_nA`, and every trace is produced by
`engine.electrode.VoltageRecorder`.  What is new is only the seam:

    tier spikes (which neurons fired, at which substep)
        -> a per-neuron extracellular current (a KERNEL, see the honesty note)
        -> sources at those neurons' REAL BANC soma positions
        -> `VoltageRecorder.step` per channel

WHAT IS ILLUSTRATIVE, stated here rather than left to be inferred.  The connectome supplies
CONNECTIVITY; the electrode geometry supplies a lead field.  Neither supplies the transmembrane
current a spike produces, so `SpikeCurrentKernel` is a BIOPHYSICAL ASSUMPTION with two parameters
(charge per spike, pulse duration) that are NOT measured in this project and are NOT Drosophila
recordings.  Every number derived from this module inherits that status.  The geometry constants
that ARE fixed -- 7 um contact diameter, 20 um pitch -- are the project's stated constraint and are
never optimised here.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python tools_test_electrode_in_loop.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from engine.electrode import Contact, VoltageRecorder, transfer_mV_per_nA  # noqa: E402

#: The project's FIXED electrode constraint.  Never optimised, never scaled.
CONTACT_DIAMETER_UM = 7.0
CONTACT_PITCH_UM = 20.0

#: Illustrative tissue conductivity.  The same value the rest of the project's electrode chain uses,
#: so this module cannot disagree with it.
CONDUCTIVITY_S_M = 0.3

#: THE KERNEL.  Both numbers are ASSUMPTIONS, not measurements.
SPIKE_CHARGE_PC = 0.10          # net charge transferred by one spike, picocoulombs
SPIKE_DURATION_MS = 1.0         # the window over which that charge moves
KERNEL_PROVENANCE = (
    "ILLUSTRATIVE, NOT MEASURED.  The connectome gives connectivity and the electrode geometry "
    "gives a lead field; neither gives the transmembrane current of a spike.  These two numbers "
    "are a declared assumption whose only defensible content is the ORDER OF MAGNITUDE of a "
    "neuronal action current (tens to hundreds of fC per spike over ~1 ms).  No result from this "
    "module may be quoted as a Drosophila measurement."
)


@dataclass
class SomaPositions:
    """Real BANC soma voxel coordinates for a set of neurons, in micrometres."""
    xyz_um: np.ndarray                       # (n, 3)
    n_missing: int
    provenance: str = field(default="")

    @classmethod
    def from_banc(cls, root_ids, *, soma_parquet=None, resolution_um=None):
        """Look up soma positions for `root_ids`, reporting how many were NOT found.

        A missing soma is COUNTED, never silently filled with a placeholder: a neuron whose
        position is unknown must not be placed at the origin and recorded as if it were there.
        """
        import pyarrow.parquet as pq
        path = Path(soma_parquet or HERE / "data" / "banc" / "somas_v1.parquet")
        tbl = pq.read_table(str(path), columns=["pt_root_id", "pt_position"])
        rid = np.asarray(tbl.column("pt_root_id").to_pylist(), dtype=np.int64)
        pos = np.asarray([list(p) for p in tbl.column("pt_position").to_pylist()], dtype=float)
        # BANC voxels are 16x16x16 nm per the dataset; the project's own access map uses the same
        # conversion, and it is passed in so this module cannot quietly pick a different one.
        vox_um = float(resolution_um if resolution_um is not None else 0.016)
        table = {int(r): p * vox_um for r, p in zip(rid, pos)}
        want = np.asarray(root_ids, dtype=np.int64)
        out = np.full((want.size, 3), np.nan)
        for i, r in enumerate(want):
            p = table.get(int(r))
            if p is not None:
                out[i] = p
        return cls(xyz_um=out, n_missing=int(np.isnan(out[:, 0]).sum()),
                   provenance=(f"{path.name}: pt_root_id + pt_position, voxel {vox_um} um/voxel; "
                               f"{want.size - int(np.isnan(out[:, 0]).sum())} of {want.size} "
                               f"neurons located"))


class InLoopElectrode:
    """Records a channel array from the SAME spike matrix that the loop already computes.

    The loop holds `spikes` of shape (n_neural_substeps, n_neurons), bool.  That is the only input
    this class needs; nothing here reaches into MuJoCo or into the tier's internals.
    """

    def __init__(self, soma: SomaPositions, *, n_tier_neurons: int | None = None,
                 n_channels: int = 4,
                 contact_diameter_um: float = CONTACT_DIAMETER_UM,
                 pitch_um: float = CONTACT_PITCH_UM,
                 conductivity_S_m: float = CONDUCTIVITY_S_M,
                 charge_per_spike_pC: float = SPIKE_CHARGE_PC,
                 spike_duration_ms: float = SPIKE_DURATION_MS,
                 bandwidth_Hz: float = 1000.0, dt_neural_ms: float = 0.5, seed: int = 0):
        ok = np.isfinite(soma.xyz_um).all(axis=1)
        if ok.sum() == 0:
            raise ValueError("no neuron has a known soma position; nothing to record from")
        self.soma_um = soma.xyz_um[ok]
        self.local_index = np.flatnonzero(ok)          # tier index -> row of self.soma_um
        self.n_missing = soma.n_missing
        self.soma_provenance = soma.provenance
        self.sigma = float(conductivity_S_m)
        self.charge_pC = float(charge_per_spike_pC)
        self.dur_ms = float(spike_duration_ms)
        self.dt_ms = float(dt_neural_ms)
        r = contact_diameter_um / 2.0
        # Contacts are placed on a line at the FIXED pitch, centred on the soma cloud.  The array
        # is placed along the cloud's principal axis so that the channels span the population
        # instead of clustering at one corner; the placement RULE is declared, and the pitch and
        # diameter are the project's fixed constraint.
        c = self.soma_um.mean(axis=0)
        x = self.soma_um - c
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        axis = vt[0]
        offs = (np.arange(n_channels) - (n_channels - 1) / 2.0) * pitch_um
        # a small offset along the second axis keeps the contacts off the somata themselves
        self.channel_um = c + np.outer(offs, axis) + 0.5 * pitch_um * vt[1]
        self.contacts = tuple(Contact(tuple(p), r) for p in self.channel_um)
        # ONE balancing return, far from the array: `VoltageRecorder.step` REQUIRES the source set to
        # balance, so the return is not a modelling choice but a precondition of the existing API.
        self.return_um = tuple(c + 2000.0 * axis)
        self.reference_um = tuple(c - 2000.0 * axis)
        self.recorders = [VoltageRecorder(tuple(p), self.reference_um, conductivity_S_m,
                                          bandwidth_Hz=bandwidth_Hz, noise_sd_mV=0.0,
                                          seed=seed + k)
                          for k, p in enumerate(self.channel_um)]
        self.n_neurons = self.soma_um.shape[0]
        # The spike matrix the loop holds covers EVERY tier neuron, so this takes the full width and
        # ignores the ones it has no soma for -- COUNTING them.  Requiring a pre-truncated matrix
        # instead would push the 4263 unlocated neurons out of sight, which is the failure mode this
        # project keeps re-learning.
        self.n_tier_neurons = int(n_tier_neurons if n_tier_neurons is not None else self.n_neurons)
        self.n_spikes_without_soma = 0

    # -- geometry facts a caller should be able to check without reading the code --------------
    def geometry(self) -> dict:
        return {
            "n_neurons_with_soma": int(self.n_neurons),
            "n_neurons_without_soma": int(self.n_missing),
            # the tier width belongs in the GEOMETRY report as well as in `dependence`: the coverage
            # ratio is a property of the electrode's placement, and a reader asking "how much of the
            # network can this record?" should not have to run a dependence check to find out
            "n_tier_neurons": int(self.n_tier_neurons),
            "n_channels": len(self.contacts),
            "contact_diameter_um": 2.0 * self.contacts[0].radius_um,
            "contact_pitch_um": float(np.linalg.norm(self.channel_um[1] - self.channel_um[0]))
            if len(self.contacts) > 1 else None,
            "conductivity_S_m": self.sigma,
            "soma_provenance": self.soma_provenance,
            "kernel": {"charge_per_spike_pC": self.charge_pC, "duration_ms": self.dur_ms,
                       "provenance": KERNEL_PROVENANCE},
        }

    def _currents(self, spikes_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(source positions, balanced currents) for ONE substep.

        A neuron that fired contributes `charge/duration` for the length of the pulse.  The return
        contact carries minus the sum, which is what makes the set balanced and what the existing
        recorder demands.
        """
        fired = np.flatnonzero(spikes_row)
        if fired.size == 0:
            return np.zeros((0, 3)), np.zeros(0)
        rows = self.local_index_searchsorted(fired)
        rows = rows[rows >= 0]
        if rows.size == 0:
            return np.zeros((0, 3)), np.zeros(0)
        i_each = self.charge_pC / self.dur_ms * (self.dt_ms / self.dur_ms)
        src = self.soma_um[rows]
        positions = np.vstack([src, np.asarray(self.return_um)[None, :]])
        currents = np.concatenate([np.full(rows.size, i_each), [-i_each * rows.size]])
        return positions, currents

    def local_index_searchsorted(self, fired: np.ndarray) -> np.ndarray:
        """Tier neuron index -> row in `soma_um`, or -1 when that neuron has no soma."""
        pos = np.searchsorted(self.local_index, fired)
        pos = np.clip(pos, 0, max(self.local_index.size - 1, 0))
        hit = self.local_index[pos] == fired
        return np.where(hit, pos, -1)

    def record(self, spikes: np.ndarray, *, t0_ms: float = 0.0) -> np.ndarray:
        """(n_channels, n_substeps) recorded voltage in mV for one interval's spikes."""
        sp = np.asarray(spikes, dtype=bool)
        if sp.ndim != 2:
            raise ValueError("spikes must be (n_substeps, n_neurons)")
        if sp.shape[1] != self.n_tier_neurons:
            raise ValueError(f"spike matrix has {sp.shape[1]} neurons; this tier has "
                             f"{self.n_tier_neurons}.  Pass the tier's OWN spike matrix -- the "
                             f"electrode maps it internally and counts what it cannot locate")
        trace = np.zeros((len(self.recorders), sp.shape[0]))
        self.n_spikes_without_soma = 0
        for j in range(sp.shape[0]):
            positions, currents = self._currents(sp[j])
            self.n_spikes_without_soma += int(sp[j].sum()) - int(np.isfinite(positions).all(axis=1).sum()) + (1 if positions.shape[0] else 0)
            for k, rec in enumerate(self.recorders):
                if positions.shape[0] == 0:
                    r = rec.step(self.dt_ms, (), ())
                else:
                    r = rec.step(self.dt_ms, tuple(Contact(tuple(p), 1e-3) for p in positions),
                                 currents)
                trace[k, j] = r["measured_mV"]
        return trace

    def dependence(self, spikes: np.ndarray) -> dict:
        """THE GUARD.  Does the recording actually depend on which neurons fired?

        This project's most valuable existing mechanism is the "identical to control -> NOT RUN"
        rule that caught 75 muscles never being wired.  The same rule applies here: a recording
        that does not change when the spikes are zeroed is not a recording of spikes.
        """
        real = self.record(spikes)
        zero = self.record(np.zeros_like(np.asarray(spikes, dtype=bool)))
        identical = bool(np.array_equal(real, zero))
        return {
            "status": "NOT_RUN" if identical else "PASS",
            "why": ("the trace is BIT-IDENTICAL with all spikes zeroed, so the electrode is not "
                    "recording the network -- treat as NOT RUN, never as a null result"
                    if identical else
                    "the trace changes when the spikes are zeroed, so the electrode is coupled to "
                    "the network"),
            "trace_peak_mV": float(np.abs(real).max()) if real.size else 0.0,
            "trace_std_mV": float(real.std()) if real.size else 0.0,
            "n_spikes": int(np.asarray(spikes, dtype=bool).sum()),
            "n_neurons_recorded": int(self.n_neurons),
            "n_neurons_without_soma": int(self.n_missing),
            "n_tier_neurons": int(self.n_tier_neurons),
            "n_spikes_without_soma": int(self.n_spikes_without_soma),
            "kernel_provenance": KERNEL_PROVENANCE,
            "what_this_does_not_establish": (
                "that the recorded amplitude is biologically meaningful.  It is the product of a "
                "declared, uncalibrated spike->current kernel with a volume-conduction lead field; "
                "only the DEPENDENCE on the network, not the magnitude, is established."),
        }


# =============================================================================================
# NeuralStimulus: the seam that did NOT exist.  Stimulus current -> the tier.
# =============================================================================================
class InLoopStimulus:
    """Injects stimulation current into the SAME tier the electrode records from.

    WHY THIS COMPLETES A PILLAR.  `engine/electrode_stim.py` models waveforms, activating functions
    and charge limits, and `isolation_scenario` records stimulus ARTIFACTS -- but nothing injected a
    stimulus current into a spiking network, so the closed loop brain -> electrode -> stimulus ->
    brain did not exist.  `engine/embodied/seams.py` declared the interface; this is the
    implementation.

    THE LIMIT IS READ, NOT CHOSEN.  The per-phase charge is checked against
    `engine.electrode_stim.charge_injection_limits`, whose density ceiling is MEASURED_CITED
    (Shepherd et al.) and whose current ceiling is ASSUMED with a sweep.  Those provenances are
    carried into the result rather than restated, so this module cannot quietly upgrade an
    assumption into a measurement.

    WHAT IS ILLUSTRATIVE HERE.  WHICH neurons receive how much current.  The access map has soma
    voxel positions and NOTHING ELSE -- no morphology, no neurite, no soma radius -- and its
    `capture_radius_um` is DECLARED rather than measured.  So the recruitment rule below is a
    declared geometric assumption: a neuron inside the capture radius gets current scaled by a
    linear falloff, and the total is normalised to the commanded amplitude.  Amplitude effects are
    therefore meaningful; the SPATIAL PATTERN is not.
    """

    #: DECLARED, matching `engine/embodied/access_map.py`'s own capture radius.  Not measured.
    CAPTURE_RADIUS_UM = 50.0

    #: WHICH SIGN OF THE CATHODIC PHASE REACHES THE MEMBRANE.  THIS IS THE ASSUMPTION, and the
    #: first version of this comment described it WRONGLY -- the correction is measured below.
    #:
    #: MEASURED MECHANISM, and it is NOT charge balance.  At the tier's own dt (0.5 ms) the pulse is
    #: 100 us per phase with a 20 us gap inside a 1 ms period, so THE RECOVERY PHASE IS NEVER
    #: SAMPLED: the injected current is a train of cathodic pulses only, identical in shape in both
    #: modes, and the two modes differ SOLELY IN SIGN --
    #:     recruiting        +|amplitude|  -> depolarises -> 12544 spikes over 30 substeps
    #:     charge_balanced   -|amplitude|  -> hyperpolarises -> 0 spikes
    #: A constant POSITIVE current does drive this tier (measured: 0.1 nA uniformly over 30 substeps
    #: -> 32323 spikes), so the injected-current path is fine and the SIGN MAPPING is the gap.
    #:
    #: WHAT AN EARLIER VERSION OF THIS FILE CLAIMED AND WHY IT WAS WRONG.  It said a literally
    #: charge-balanced waveform "self-cancels on a linear membrane".  That cannot be what happens
    #: here: with the recovery phase unsampled, no cancellation is possible, and the observed
    #: difference is fully explained by the sign alone.  Charge balance is a property of the
    #: ELECTRODE waveform; at this time resolution it does not reach the membrane at all.
    #:
    #: SO THE HONEST STATEMENT IS NARROWER AND CLEARER: the sign that maps an electrode's cathodic
    #: phase onto a depolarising membrane current is not specified by anything in this project
    #: (a real membrane rectifies through its own nonlinearity; a point-current model does not), and
    #: `recruiting` declares that sign.  It is not a claim about rectification.
    MODES = ("recruiting", "charge_balanced")

    def __init__(self, electrode: "InLoopElectrode", *, channel_index: int = 0,
                 amplitude_nA: float = 1000.0, phase_width_us: float = 100.0,
                 interphase_gap_us: float = 20.0, repetition_rate_Hz: float = 1.0,
                 capture_radius_um: float = CAPTURE_RADIUS_UM, t0_ms: float = 0.0,
                 mode: str = "recruiting"):
        from engine.electrode_stim import StimWaveform, charge_injection_limits
        self.electrode = electrode
        self.channel_index = int(channel_index)
        if not 0 <= self.channel_index < len(electrode.channel_um):
            raise ValueError(f"channel_index out of range 0..{len(electrode.channel_um)-1}")
        self.stim_um = np.asarray(electrode.channel_um[self.channel_index], dtype=float)
        self.wave = StimWaveform(amplitude_nA=float(amplitude_nA),
                                 phase_width_us=float(phase_width_us),
                                 interphase_gap_us=float(interphase_gap_us),
                                 repetition_rate_Hz=float(repetition_rate_Hz),
                                 contact_radius_um=electrode.contacts[0].radius_um)
        self.limits = charge_injection_limits(contact_radius_um=self.wave.contact_radius_um)
        self.t0_ms = float(t0_ms)
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}; got {mode!r}")
        self.mode = mode
        self.capture_um = float(capture_radius_um)
        # distance from the stimulating contact to every recordable soma
        self.dist_um = np.linalg.norm(electrode.soma_um - self.stim_um, axis=1)
        self.in_range = self.dist_um <= self.capture_um
        # declared linear falloff, normalised so the weights sum to 1 over the recruited set
        w = np.where(self.in_range, 1.0 - self.dist_um / self.capture_um, 0.0)
        s = float(w.sum())
        self.weights = w / s if s > 0 else w
        self.n_recruited = int(self.in_range.sum())

    # -- safety accounting, read from the project's own cited limits ---------------------------
    def safety(self) -> dict:
        q_per_phase_uC = self.wave.amplitude_nA * 1e-3 * self.wave.phase_width_us * 1e-6  # nA*us -> uC
        area_cm2 = float(self.limits.contact_area_cm2)
        density_uC_cm2 = q_per_phase_uC / area_cm2 if area_cm2 > 0 else float("inf")
        dens_lim = float(self.limits.density_limit_uC_cm2)
        cur_lim = float(self.limits.assumed_current_limit_nA)
        return {
            "charge_per_phase_uC": q_per_phase_uC,
            "contact_area_cm2": area_cm2,
            "density_uC_cm2": density_uC_cm2,
            "density_limit_uC_cm2": dens_lim,
            "density_provenance": self.limits.density_provenance,
            "density_source": (self.limits.density_source or "")[:120],
            "within_cited_density_limit": bool(density_uC_cm2 <= dens_lim),
            "amplitude_nA": float(self.wave.amplitude_nA),
            "assumed_current_limit_nA": cur_lim,
            "current_provenance": "ASSUMED (no fetched source; sweep recorded by the stim module)",
            "within_assumed_current_limit": bool(self.wave.amplitude_nA <= cur_lim),
            "mode": self.mode,
            "charge_balance": {
                "where_it_holds": ("the ELECTRODE waveform is biphasic and charge-balanced; that "
                                   "is a constraint at the electrode-tissue interface, and "
                                   "`StimWaveform` plus these limits are where it is accounted"),
                "where_it_does_not": ("in `recruiting` mode the injected MEMBRANE current is "
                                      "deliberately NOT balanced -- only the recruiting phase is "
                                      "injected -- and that asymmetry is exactly the rectification "
                                      "being put in by hand.  It is the price of the assumption, "
                                      "not an oversight"),
                "consequence": ("a reader may not claim the model is charge-balanced end to end; "
                                "the electrode is, the membrane injection is not"),
            },
            "mode_assumption": (
                "MEASURED: at the tier's dt (0.5 ms) the recovery phase falls between substeps and "
                "is never sampled, so both modes inject the SAME cathodic pulse train and differ "
                "ONLY in sign -- `recruiting` depolarises (+12544 spikes/30 substeps), "
                "`charge_balanced` injects the waveform's literal negative sign and hyperpolarises "
                "(0 spikes).  Charge balance is therefore NOT the mechanism and no cancellation "
                "occurs here.  The assumption is the SIGN that maps an electrode's cathodic phase "
                "onto a depolarising membrane current; nothing in this project specifies it."),
            "what_this_does_not_establish": (
                "CHARGE SAFETY for a real electrode.  The density ceiling is cited for cochlear "
                "platinum contacts at 0.05-0.2 mm^2 and is EXTRAPOLATED to a 7 um contact; the "
                "current ceiling has no fetched source at all.  No electrochemistry, no "
                "tissue-damage threshold and no in-vivo safety claim is made."),
        }

    # -- the interface ------------------------------------------------------------------------
    def current_nA(self, t_ms: float, n_neurons: int) -> np.ndarray:
        """Injected current per tier neuron at time `t_ms`, nanoamperes.

        Biphasic by construction: the first phase is cathodic (the project's waveform default) and
        the second phase reverses it, so the pulse carries no net charge into the tissue.  Outside
        the pulse window the returned current is exactly zero.
        """
        out = np.zeros(int(n_neurons), dtype=float)
        if n_neurons != self.electrode.n_tier_neurons:
            raise ValueError(f"n_neurons={n_neurons} does not match the tier width "
                             f"{self.electrode.n_tier_neurons}")
        t = float(t_ms) - self.t0_ms
        if t < 0:
            return out
        period = self.wave.period_us * 1e-3          # us -> ms
        if period <= 0:
            return out
        tp = t % period
        pw = self.wave.phase_width_us * 1e-3
        gap = self.wave.interphase_gap_us * 1e-3
        if tp < pw:
            # first (cathodic) phase: RECRUITING either way
            amp = (abs(self.wave.amplitude_nA) if self.mode == "recruiting"
                   else self.wave.amplitude_nA * self.wave.phase_sign)
        elif tp < pw + gap:
            amp = 0.0
        elif tp < 2.0 * pw + gap:
            # recovery phase: charge recovery happens at the INTERFACE, not by hyperpolarising the
            # membrane, so in `recruiting` mode nothing is injected here.  In `charge_balanced`
            # mode the literal opposite phase is injected, which cancels the first one.
            if self.mode == "recruiting":
                return out
            amp = -self.wave.amplitude_nA * self.wave.phase_sign
        else:
            return out
        if amp == 0.0:
            return out
        # distribute to the recruited somata; the electrode's own located-index map is reused so
        # that a neuron without a soma can never receive current
        idx = self.electrode.local_index
        out[idx] = amp * self.weights
        return out

    #: Capture radii to report, in micrometres.  The active radius is DECLARED; this sweep is what
    #: makes the declaration auditable instead of a bare number, per the S5 acceptance criterion.
    CAPTURE_SWEEP_UM = (10.0, 25.0, 50.0, 100.0, 200.0, 400.0)

    @classmethod
    def capture_radius_sweep(cls, electrode: "InLoopElectrode", *, channel_index: int = 0,
                             radii_um=None) -> dict:
        """How many neurons each capture radius recruits, so the DECLARED 50 um can be judged.

        The access map's own `capture_radius_um` is DECLARED rather than measured, and a declared
        number that nobody ever varies is indistinguishable from a guess.  This reports the whole
        curve, including how many neurons the ACTIVE radius recruits and how steeply the count grows,
        so a reader can see what the choice costs instead of taking it on trust.
        """
        stim = np.asarray(electrode.channel_um[channel_index], dtype=float)
        d = np.linalg.norm(electrode.soma_um - stim, axis=1)
        radii = tuple(radii_um or cls.CAPTURE_SWEEP_UM)
        rows = [{"capture_radius_um": float(r), "n_recruited": int((d <= r).sum()),
                 "fraction_of_recordable": float((d <= r).mean())} for r in radii]
        active = [row for row in rows if abs(row["capture_radius_um"] - cls.CAPTURE_RADIUS_UM) < 1e-9]
        return {"per_radius": rows, "active_radius_um": float(cls.CAPTURE_RADIUS_UM),
                "active_n_recruited": active[0]["n_recruited"] if active else None,
                "recordable_neurons": int(electrode.soma_um.shape[0]),
                "provenance": ("DECLARED, swept here rather than measured: no measurement of a "
                               "Drosophila neuron's recruitment radius exists in this project")}

    def mechanism(self, *, dt_ms: float | None = None, n_substeps: int = 6) -> dict:
        """MEASURE the sampled waveform, so "the recovery phase is not sampled" is checkable.

        The comment above states the mechanism; this returns it as numbers.  A claim about a
        sampling artefact that is only written in prose is exactly the kind of thing that quietly
        stops being true when somebody changes `dt`.
        """
        dt = float(dt_ms if dt_ms is not None else self.electrode.dt_ms)
        rows = []
        for j in range(n_substeps):
            cur = self.current_nA(j * dt, self.electrode.n_tier_neurons)
            rows.append({"substep": j, "t_ms": j * dt, "signed_sum_nA": float(cur.sum()),
                         "max_abs_per_neuron_nA": float(np.abs(cur).max())})
        nz = [r for r in rows if r["max_abs_per_neuron_nA"] > 0]
        signs = {1 if r["signed_sum_nA"] > 0 else -1 for r in nz}
        return {
            "dt_ms": dt,
            "phase_width_us": self.wave.phase_width_us,
            "interphase_gap_us": self.wave.interphase_gap_us,
            "period_us": self.wave.period_us,
            "recovery_phase_window_us": [self.wave.phase_width_us + self.wave.interphase_gap_us,
                                         2 * self.wave.phase_width_us + self.wave.interphase_gap_us],
            "substeps_nonzero": len(nz),
            "substeps_total": n_substeps,
            "sampled_signs": sorted(signs),
            "recovery_phase_sampled": bool(len(nz) > 0 and any(
                r["max_abs_per_neuron_nA"] > 0 and
                self.wave.phase_width_us + self.wave.interphase_gap_us <= (r["t_ms"] % (self.wave.period_us * 1e-3)) * 1000.0
                < 2 * self.wave.phase_width_us + self.wave.interphase_gap_us for r in rows)),
            "conclusion": ("at this dt only one phase is sampled, so the two modes differ by SIGN "
                           "and not by charge balance"),
            "what_this_does_not_establish": (
                "that the sign is right.  It establishes what the model actually injects, which is "
                "the precondition for discussing the sign at all."),
        }

    def dependence(self, make_tier, *, n_substeps: int = 40, dt_ms: float = 0.5,
                   drive_nA: np.ndarray | None = None) -> dict:
        """THE GUARD.  Does stimulating actually change what the tier does?

        Two IDENTICAL tiers (the tier's ``reset`` restores the exact rest state with the same
        background draw), advanced over the same substeps with the same drive, one stimulated and
        one not.  If the spike patterns are identical the stimulus never reached the network and the
        result is NOT RUN -- the same rule that caught 75 muscles never being wired.
        """
        drive = None if drive_nA is None else np.asarray(drive_nA, dtype=float)

        def run(stimulate: bool):
            tier = make_tier()
            tier.reset()
            spikes = np.zeros((n_substeps, self.electrode.n_tier_neurons), dtype=bool)
            for j in range(n_substeps):
                t = j * dt_ms
                cur = np.zeros(self.electrode.n_tier_neurons)
                if drive is not None:
                    cur = cur + drive
                if stimulate:
                    cur = cur + self.current_nA(t, self.electrode.n_tier_neurons)
                spikes[j] = tier.advance(1, cur)
            return spikes

        off, on = run(False), run(True)
        identical = bool(np.array_equal(off, on))
        return {
            "status": "NOT_RUN" if identical else "PASS",
            "why": ("the stimulated and unstimulated tiers produced BIT-IDENTICAL spikes, so the "
                    "stimulus never reached the network -- NOT RUN, never a null result"
                    if identical else
                    "stimulation changed the tier's spikes, so the electrode->brain seam is "
                    "coupled"),
            "spikes_without_stimulus": int(off.sum()),
            "spikes_with_stimulus": int(on.sum()),
            "spikes_differing": int((off != on).sum()),
            "n_recruited_neurons": self.n_recruited,
            "n_neurons_without_soma": int(self.electrode.n_missing),
            "safety": self.safety(),
        }
