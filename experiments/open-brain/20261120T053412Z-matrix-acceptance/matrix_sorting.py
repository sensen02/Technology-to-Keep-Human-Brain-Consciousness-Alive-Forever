"""Voltage-only two-bank spike sorting. Each template spans both electrode banks."""
import numpy as np

LAGS = [d for d in range(-18, 19) if 10 <= abs(d) <= 18]


def noise(voltage):
    return np.median(abs(voltage - np.median(voltage, axis=0)), axis=0) / .67449


def maxima(data):
    envelope = np.max(np.abs(data), axis=1)
    t = np.flatnonzero((envelope[1:-1] > 4.1) & (envelope[1:-1] > envelope[:-2])
                       & (envelope[1:-1] >= envelope[2:])) + 1
    return t


def learn(calibration):
    """Repeated paired voltage shapes at a signed lag, supported in both recordings."""
    scale = np.median([noise(v) for v in calibration], axis=0)
    groups = {lag: [] for lag in LAGS}
    seeds = {lag: [] for lag in LAGS}
    for session, voltage in enumerate(calibration):
        h, t = np.split(voltage / scale, 2, axis=1)
        head = maxima(h)
        thorax = maxima(t)
        for lag in LAGS:
            a = head[np.isin(head + lag, thorax)]
            b = a + lag
            shape = np.stack([np.stack((h[a], h[a+1]), axis=1),
                              np.stack((t[b], t[b+1]), axis=1)], axis=1)
            groups[lag].extend(shape)
            seeds[lag].extend([session]*len(a))
    templates, delays, support = [], [], []
    for lag in LAGS:
        if len(groups[lag]) < 12:
            continue
        v = np.array(groups[lag])
        session = np.array(seeds[lag])
        first = v[:, :, 0, :]
        norm = np.linalg.norm(first, axis=2)
        shape = first / norm[:, :, None]
        similar = np.ones((len(v), len(v)), dtype=bool)
        for bank in range(2):
            ratio = norm[:, bank, None] / norm[None, :, bank]
            similar &= (shape[:, bank] @ shape[:, bank].T > .76) & (ratio > .60) & (ratio < 1/.60)
        used = np.zeros(len(v), bool)
        while True:
            counts = np.stack([np.sum(similar[:, (~used) & (session == s)], axis=1) for s in range(2)])
            eligible = (~used) & (counts.min(axis=0) >= 4) & (counts.sum(axis=0) >= 12)
            if not eligible.any():
                break
            leader = np.argmax(np.where(eligible, counts.sum(axis=0), -1))
            members = similar[leader] & ~used
            templates.append(np.median(v[members], axis=0))
            delays.append(lag)
            support.append(counts[:, leader])
            used |= members
    if not templates:
        raise ValueError('no recurring waveform supported in both calibration runs')
    return np.array(templates), np.array(delays), scale, np.array(support)


def detect(voltage, bank, delays, scale, *, multichannel, subtract):
    residual = (voltage / scale).reshape(len(voltage), 2, voltage.shape[1] // 2).copy()
    templates = bank.copy()
    if not multichannel:
        # The most informative contact on each bank, fixed for each unit at calibration.
        sites = np.argmax((templates**2).sum(axis=2), axis=2)
        templates *= (np.arange(voltage.shape[1] // 2) == sites[:, :, None])[:, :, None, :]
    norms = np.linalg.norm(templates.reshape(len(bank), 2, -1), axis=2)
    fired = np.zeros((len(bank), len(voltage)), bool)
    events = []
    for _ in range(20 if subtract else 1):
        projections = []
        for i in range(2):
            windows = np.concatenate((residual[:-1,i], residual[1:,i]), axis=1)
            unit = templates[:,i].reshape(len(bank), -1) / norms[:,i,None]
            projections.append(windows @ unit.T)
        pool = []
        for k, lag in enumerate(delays):
            start = max(0, -lag)
            end = min(len(voltage)-1, len(voltage)-1-lag)
            h = np.arange(start, end)
            z1, z2 = projections[0][h,k], projections[1][h+lag,k]
            gain = np.stack((z1/norms[k,0], z2/norms[k,1]), axis=1)
            amp = np.clip(gain, .35, 1.25)
            # Twice the reduction in whitened squared error, at the constrained fit.
            energy = np.sum((2*gain-amp)*amp*norms[k]**2, axis=1)
            accept = ((z1 >= 2.8) & (z2 >= 2.8) & (gain.min(axis=1) >= .32)
                      & (energy >= 40) & ~fired[k,h])
            for j in np.flatnonzero(accept):
                pool.append((energy[j], k, h[j], h[j]+lag, amp[j]))
        if not pool:
            break
        busy = np.zeros((2,len(voltage)),bool)
        for _, k, h, t, amp in sorted(pool, key=lambda c: -c[0]):
            if busy[0,h] or busy[1,t]:
                continue
            events.append((h,t,k))
            fired[k, max(0,h-1):h+2] = True
            for i, tick in enumerate((h,t)):
                busy[i,max(0,tick-1):tick+2] = True
                if subtract:
                    residual[tick:tick+2,i] -= amp[i]*templates[k,i]
    return np.array(sorted(events), dtype=int).reshape(-1,3)
