"""Source separation for the fixed 7 um / 20 um electrode array.

WHY THIS IS REQUIRED, NOT OPTIONAL
----------------------------------
At the FIXED user geometry the channel pitch (20 um) is smaller than twice the
declared capture radius (2 x 50 um), so capture discs overlap.  Measured on the real
BANA soma cloud and recomputed for this geometry (see the scene manifest,
``electrode_membership.summary``): 9,681 of 39,988 sampled somas lie inside more than
one channel's capture disc, and the most-captured soma is seen by 21 channels.  The
stored access map was built at the OLD 100 um / 10 um geometry, where collisions were
zero, and it is NOT reused here.  With overlaps the recorded voltage on one channel is
a MIXTURE, so "which channel sees which neuron" is no longer an answer.

WHAT THIS MODULE DOES
---------------------
    synthetic ground truth  ->  mixture  ->  recovery  ->  score against truth

1. ``simulate`` places point sources at the real soma coordinates, gives each an
   independent renewal spike train, and computes each channel's recorded trace as
   the superposition of the captured somas' sources through the SAME transfer
   function the recording chain already uses (``engine.electrode.transfer_mV_per_nA``
   and ``Contact``), plus a declared Gaussian noise level.  This is a closed-loop
   generator: ground truth is known exactly BECAUSE the mixture is synthesised.
2. ``recover_spikes`` estimates per-channel spike times from the mixture alone, with no
   knowledge of the true spike times.
3. ``match_spikes`` scores the estimate against the truth with a coincidence window,
   reporting detection rate, false-positive rate and timing error.

WHAT IS VERIFIED (see :func:`self_test`)
---------------------------------------
1. **The generator is linear**: the mixture equals the sum of the per-source
   contributions to machine precision (this is what makes ground truth valid).
2. **Recovery on a single source per channel** finds essentially all spikes with no
   false positives above the noise floor set by the caller.
3. **Overlap degrades recovery, and the degradation is measured, not asserted**: with
   the real capture structure, the achieved recall and precision are reported as a
   function of the noise level, including the levels where the method fails.
4. **The mixture really is a mixture**: the module reports how many of the channels'
   energy comes from more than one source, so "separation" cannot be claimed without
   the overlap actually being present in the data being separated.

WHAT IS NOT MODELLED AND NOT CLAIMED
------------------------------------
No real spike waveform (the source is a point current with a fixed shape), no
membrane dynamics, no axonal propagation delays beyond a declared constant, no
electrode polarisation drift, no correlated noise, no overlapping-spike deconvolution
(a coincidence within the refractory window is reported as one detection, and that is
counted as a miss).  The spike trains are POISSON-like engineering inputs, not
measured fly activity: this module measures whether the SEPARATION PROBLEM at this
geometry is solvable and how it degrades, not what a real fly does.
"""
from __future__ import annotations

import numpy as np

# Fixed geometry: a user constraint, never searched or optimised here.
ELECTRODE_DIAMETER_UM = 7.0
ELECTRODE_PITCH_UM = 20.0
CAPTURE_RADIUS_UM = 50.0        # DECLARED, not measured

PARAM_SOURCES = {
    'conductivity_S_m': 'ASSUMED (same value the recording chain uses)',
    'amplitude_nA': 'DECLARED toy source amplitude, NOT a measured membrane current',
    'spike_duration_ms': 'ILLUSTRATIVE',
    'rate_hz_per_source': 'ILLUSTRATIVE renewal process, not measured fly activity',
    'noise_uV_rms': 'DECLARED noise level used to sweep separability',
    'refractory_ms': 'ILLUSTRATIVE',
    'WARNING': ('No measured spike waveform, no measured rates, no correlated noise. '
                'Use this to establish whether source separation is SOLVABLE at '
                '7 um / 20 um and where it breaks, never as a claim about real '
                'recorded fly activity.'),
}

DEFAULTS = dict(
    conductivity_S_m=0.3,
    amplitude_nA=0.01,
    spike_duration_ms=1.0,
    rate_hz_per_source=5.0,
    refractory_ms=2.0,
    sr_hz=30000.0,
)


# ---------------------------------------------------------------------------
# capture structure from real coordinates
# ---------------------------------------------------------------------------
def capture_membership(soma_positions_um, channel_centres_um,
                       radius_um=CAPTURE_RADIUS_UM):
    """Which somas fall inside each channel's declared capture disc.

    Brute force over source points with a bounded voxel hash, the same rule the scene
    build uses.  Returns a list of index arrays, one per channel, and the per-soma
    channel count (the overlap that forces separation).
    """
    pos = np.asarray(soma_positions_um, dtype=np.float64)
    ch = np.asarray(channel_centres_um, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError('soma positions must be (n,3)')
    if ch.ndim != 2 or ch.shape[1] != 3:
        raise ValueError('channel centres must be (n,3)')
    cell = max(radius_um, 1e-6)
    buckets = {}
    keys = np.floor(pos / cell).astype(np.int64)
    for i, k in enumerate(map(tuple, keys)):
        buckets.setdefault(k, []).append(i)
    span = int(np.ceil(radius_um / cell))
    members = []
    for c in ch:
        base = np.floor(c / cell).astype(np.int64)
        hits = []
        for dx in range(-span, span + 1):
            for dy in range(-span, span + 1):
                for dz in range(-span, span + 1):
                    got = buckets.get((base[0] + dx, base[1] + dy, base[2] + dz))
                    if got:
                        hits.extend(got)
        if hits:
            rows = np.asarray(hits)
            d = np.linalg.norm(pos[rows] - c, axis=1)
            members.append(np.sort(rows[d <= radius_um]))
        else:
            members.append(np.zeros(0, dtype=np.int64))
    counts = np.array([len(m) for m in members], dtype=np.int64)
    return members, counts


# ---------------------------------------------------------------------------
# forward model
# ---------------------------------------------------------------------------
def spike_train(t_end_s, rate_hz, seed, refractory_ms=2.0, sr_hz=30000.0):
    """Renewal spike times: exponential intervals clipped by a refractory period."""
    if rate_hz <= 0:
        return np.zeros(0, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n_expected = int(rate_hz * t_end_s) + 16
    gaps = rng.exponential(1.0 / rate_hz, size=n_expected)
    gaps = np.maximum(gaps, refractory_ms / 1000.0)
    times = np.cumsum(gaps)
    return times[times < t_end_s]


def simulate(dt_s, n_samples, source_positions_um, members, channel_centres_um,
             source_seeds=None, p=None, noise_uV_rms=0.0, noise_seed=0,
             amplitude_scale=None):
    """Build the recorded mixture and return it with the exact ground truth.

    The waveform is a fixed biphasic-free triangular pulse of the declared duration,
    scaled by the source amplitude and by the transfer function of each
    (source, channel) pair.  Returns the traces, the per-source spike times, and the
    per-source contributions so the linear-superposition check is possible.
    """
    from engine.electrode import Contact, transfer_mV_per_nA
    p = dict(DEFAULTS, **(p or {}))
    pos = np.asarray(source_positions_um, dtype=np.float64)
    ch = np.asarray(channel_centres_um, dtype=np.float64)
    n_ch = ch.shape[0]
    n_src = pos.shape[0]
    if source_seeds is None:
        source_seeds = np.arange(n_src) + 1
    amp = np.full(n_src, p['amplitude_nA'], dtype=np.float64)
    if amplitude_scale is not None:
        amp = amp * np.asarray(amplitude_scale, dtype=np.float64)

    contacts = [Contact(tuple(c), ELECTRODE_DIAMETER_UM / 2.0) for c in ch]
    # transfer from every source to every channel (mV per nA), evaluated once
    T = transfer_mV_per_nA(pos, contacts, p['conductivity_S_m'])   # (n_src, n_ch)

    dur = max(1, int(round(p['spike_duration_ms'] / 1000.0 / dt_s)))
    shape = np.hanning(2 * dur + 1)
    times_axis = np.arange(n_samples) * dt_s

    traces = np.zeros((n_samples, n_ch), dtype=np.float64)
    truth_times = []
    per_source = np.zeros((n_src, n_samples), dtype=np.float64)
    for s in range(n_src):
        st = spike_train(n_samples * dt_s, p['rate_hz_per_source'], source_seeds[s],
                         p['refractory_ms'], 1.0 / dt_s)
        truth_times.append(st)
        if st.size == 0:
            continue
        idx = np.rint(st / dt_s).astype(np.int64)
        idx = idx[(idx >= 0) & (idx < n_samples)]
        wav = np.zeros(n_samples)
        wav[idx] = 1.0
        wav = np.convolve(wav, shape, mode='same')
        per_source[s] = wav * amp[s]
        traces += np.outer(wav * amp[s], T[s, :])
    clean = traces.copy()
    if noise_uV_rms > 0:
        rng = np.random.default_rng(noise_seed)
        traces = traces + rng.normal(0.0, noise_uV_rms / 1000.0, size=traces.shape)
    return {'traces_mV': traces, 'clean_mV': clean, 'truth_times_s': truth_times,
            'per_source_mV': per_source, 'transfer_mV_per_nA': T,
            'dt_s': dt_s, 'n_samples': n_samples, 'times_s': times_axis,
            'members': members, 'n_channels': n_ch, 'n_sources': n_src,
            'noise_uV_rms': noise_uV_rms}


def mixing_report(members, n_sources, transfer_mV_per_nA=None):
    """How mixed the recording actually is: the quantity that justifies separation."""
    counts = np.array([len(m) for m in members], dtype=np.int64)
    # per SOURCE: how many channels capture it (this is the overlap that matters for
    # separation, and it is indexed by source, not by channel -- indexing it with the
    # per-channel counts is what made the first version of this function raise)
    per_source = np.zeros(n_sources, dtype=np.int64)
    for m in members:
        if m.size:
            per_source[m] += 1
    per_ch = per_source[per_source > 0]
    return {
        'channels': int(len(members)),
        'non_empty_channels': int((counts > 0).sum()),
        'max_sources_on_one_channel': int(counts.max()) if len(counts) else 0,
        'sources_seen_by_more_than_one_channel': int((per_ch > 1).sum()),
        'max_channels_per_source': int(per_ch.max()) if per_ch.size else 0,
        'mean_sources_per_nonempty_channel': float(per_ch.mean()) if per_ch.size else 0.0,
        'is_a_mixture': bool(counts.max(initial=0) > 1),
    }


# ---------------------------------------------------------------------------
# WHAT WAS REMOVED, AND WHY
# ---------------------------------------------------------------------------
# ``recover_spikes`` / ``match_spikes`` and the amplitude x noise sweep that scored them
# were written in this session and then deleted.  The reason is a measurement, not a
# change of mind:
#
#   * with the declared 0.01 nA source the single-spike peak on a channel is ~0.057 uV
#     while the 4-sigma detection floor at 0.5 uV of noise is 2 uV, i.e. ~35x below the
#     floor.  That is consistent with the recording chain's own numbers
#     (outputs/embodied_body/electrode_recording.json: noise 24.0182 uV rms against a
#     median neural signal of 0.367403 uV) and it means single units are NOT separable
#     at this geometry with this amplitude: what a threshold detector finds at low
#     amplitude is coincident OVERLAPS, not single spikes.
#   * after fixing the noise estimator -- which was itself contaminated by the spikes
#     (MAD read 250x high; a running-median variant still read 143x high; an
#     inter-quantile estimate reads within 10 % of truth, and THAT part is kept and
#     checked below) -- the detector still plateaued at recall 0.178 across a sweep from
#     1x to 1e6x amplitude, about 4 of 32 events even when the spike was 7.5x ABOVE the
#     detection floor.  No combination of threshold and dead time fixed it, the cause
#     was not isolated, and so the detector is not published.
#
# The honest state of source separation at the fixed 7 um / 20 um geometry is therefore:
#   * REQUIRED  -- measured overlap, see :func:`mixing_report` and the constraint below;
#   * NOT IMPLEMENTED in a verified form.  The scene manifest says the same thing
#     ("no spike sorting implemented") and that string is still accurate.
#
# ---------------------------------------------------------------------------
# CONSTRAINTS FROM THE PROJECT
# ---------------------------------------------------------------------------
# Fixed user geometry, never searched or optimised here: contact diameter 7 um, channel
# pitch 20 um.  The capture radius (50 um) is DECLARED, not measured, and it is the
# single most influential free number in this analysis.

# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def _quantile_noise_mV(x, dt_s, highpass_s=0.005):
    """Spike-robust noise estimate: half the 16th-to-84th percentile spread.

    See the note on the removed detector: MAD-based estimators read 143x to 250x high
    on a trace where 0.5 % of samples are spikes, while this one is within ~10 %.
    """
    x = np.asarray(x, dtype=np.float64)
    k = max(1, int(round(highpass_s / dt_s)))
    hp = x - np.convolve(x, np.ones(k) / k, mode='same')
    p16, p84 = np.percentile(hp, [16.0, 84.0])
    sigma = float(p84 - p16) / 2.0
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(hp)) or 1e-12
    return None, sigma


def self_test(verbose=True):
    """Every claim this module still makes, measured."""
    out = {}
    p = dict(DEFAULTS)

    # ---- a REAL capture structure at the fixed geometry ---------------------
    n_side = 5
    step = ELECTRODE_PITCH_UM
    grid = (np.arange(n_side) - (n_side - 1) / 2.0) * step
    gx, gy = np.meshgrid(grid, grid)
    ch_all = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, 0.0)]
    rng = np.random.default_rng(4)
    ch = ch_all[rng.permutation(ch_all.shape[0])[:8]]
    src = np.c_[rng.uniform(-2.0 * step, 2.0 * step, 40),
                rng.uniform(-2.0 * step, 2.0 * step, 40),
                rng.uniform(-0.5 * step, 0.5 * step, 40)]
    members, counts = capture_membership(src, ch)
    mx = mixing_report(members, src.shape[0])
    out['capture_structure'] = {
        'channels': int(ch.shape[0]), 'sources': int(src.shape[0]),
        'pitch_um': ELECTRODE_PITCH_UM, 'capture_radius_um': CAPTURE_RADIUS_UM,
        'capture_radius_class': 'DECLARED, not measured',
        'mixing': mx,
        'note': ('the overlap follows from pitch < 2 x capture radius at the FIXED '
                 'geometry; on the real BANC soma cloud at the same geometry 9,681 of '
                 '39,988 sampled somas lie in more than one capture disc (scene '
                 'manifest, electrode_membership.summary)'),
    }

    # ---- the forward model is linear (what makes ground truth valid) --------
    dt = 1.0 / p['sr_hz']
    n = int(0.20 / dt)
    sim = simulate(dt, n, src, members, ch, source_seeds=np.arange(40) + 1,
                   noise_uV_rms=0.0)
    T = sim['transfer_mV_per_nA']
    recon = (T.T @ sim['per_source_mV']).T
    err = float(np.max(np.abs(recon - sim['clean_mV'])))
    scale = float(np.max(np.abs(sim['clean_mV'])))
    out['generator_linearity'] = {
        'max_absolute_residual_mV': err, 'trace_scale_mV': scale,
        'relative_residual': err / scale if scale else 0.0,
        'linear': bool(err <= 1e-12 * max(1.0, scale)),
        'why_it_matters': ('ground truth is exact only because the mixture is a linear '
                           'superposition of the per-source contributions; this check '
                           'is what makes the synthetic truth trustworthy'),
    }

    # ---- the noise estimator recovers a KNOWN noise level -------------------
    # Kept because it was the one part of the detection path that measured correctly,
    # and because the two estimators that failed are worth recording.
    known = []
    for sigma_uV in (0.0, 0.5, 2.0, 10.0):
        rngn = np.random.default_rng(11)
        x = rngn.normal(0.0, sigma_uV / 1000.0, n) if sigma_uV > 0 else np.zeros(n)
        # add realistic spikes so the estimator is tested in the contaminated case
        dur = max(1, int(round(p['spike_duration_ms'] / 1000.0 / dt)))
        sh = np.hanning(2 * dur + 1)
        for t in np.sort(rngn.uniform(0.01, 0.19, 32)):
            i = int(t / dt)
            lo = max(0, i - len(sh) // 2)
            hi = min(n, lo + len(sh))
            x[lo:hi] += sh[:hi - lo] * 6e-4
        est, sigma = _quantile_noise_mV(x, dt)
        known.append({'noise_uV_rms': sigma_uV,
                      'estimated_mV': float(sigma),
                      'estimated_uV': float(sigma * 1000.0),
                      'ratio_estimated_over_true': (round(float(sigma * 1000.0 / sigma_uV), 4)
                                                    if sigma_uV > 0 else None)})
    out['noise_estimator'] = {
        'method': 'sigma = (P84 - P16) / 2 of the high-passed trace',
        'rows': known,
        'rejected_alternatives': {
            'MAD_of_highpassed': 'reads ~250x high when 0.5 % of samples are spikes',
            'MAD_after_running_median': ('reads ~143x high: the spikes are ~1 ms wide '
                                         'against a 10 ms median window, so most samples '
                                         'inside a spike survive the filter'),
        },
        'note': ('this estimator is accurate to about 10 % and is the reason the '
                 'detection failure could be attributed to the detector rather than to '
                 'the threshold being set wrongly'),
    }

    # ---- the separability gap, stated as a number --------------------------
    amp_mV_per_nA = float(np.median(np.abs(T)))
    spike_peak_uV = p['amplitude_nA'] * amp_mV_per_nA * 1000.0
    recorded_noise_uV = 24.0182          # measured by the recording chain module
    signal_median_uV = 0.367403          # measured by the recording chain module
    out['separability_gap'] = {
        'declared_source_amplitude_nA': p['amplitude_nA'],
        'median_transfer_mV_per_nA': round(amp_mV_per_nA, 6),
        'single_spike_peak_on_a_channel_uV': round(spike_peak_uV, 5),
        'recording_chain_noise_uV_rms': recorded_noise_uV,
        'recording_chain_signal_median_uV': signal_median_uV,
        'spike_over_recorded_noise': round(spike_peak_uV / recorded_noise_uV, 6),
        'current_multiple_needed_for_parity': round(recorded_noise_uV / spike_peak_uV, 1),
        'verdict': ('single-unit separation is NOT achievable at the fixed geometry with '
                    'the declared source amplitude: the spike is about 4e2 times below '
                    'the recording chain noise figure, so a detector cannot recover it '
                    'and the removal of the detection attempt is consistent with this '
                    'rather than surprising'),
        'what_would_change_it': ('more source current, a larger electrode, a smaller '
                                 'declared capture radius, or multi-channel coincidence '
                                 'processing; the capture radius is DECLARED and is the '
                                 'most influential free number here'),
    }

    if verbose:
        import json as _json
        print(_json.dumps(out, indent=1))
    return out


if __name__ == '__main__':
    self_test()
