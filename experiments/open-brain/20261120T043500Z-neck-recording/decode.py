"""Infer propagation direction from simultaneous head/thorax electrode voltages."""
import numpy as np


def peaks(traces_uv):
    envelope = np.abs(traces_uv).max(axis=1)
    return np.flatnonzero((envelope[1:-1] >= 0.5) &
                          (envelope[1:-1] > envelope[:-2]) &
                          (envelope[1:-1] >= envelope[2:])) + 1


def partners(ticks, others):
    options = []
    for t in ticks:
        candidates = others[others.searchsorted(t - 18):others.searchsorted(t + 19)]
        options.append(candidates[np.abs(candidates - t) >= 10])
    return options


def decode(head_uv, thorax_uv):
    """Keep only mutual unique pairs 1.0–1.8 ms apart at 10 kHz."""
    h, t = peaks(head_uv), peaks(thorax_uv)
    ht, th = partners(h, t), dict(zip(t, partners(t, h)))
    pairs = [(a, b[0]) for a, b in zip(h, ht) if len(b) == 1 and len(th[b[0]]) == 1]
    indices = np.asarray(pairs, dtype=int).reshape(-1, 2)
    return indices, dict(head_peaks=len(h), thorax_peaks=len(t),
                         unpaired_head=len(h)-len(pairs), unpaired_thorax=len(t)-len(pairs))
