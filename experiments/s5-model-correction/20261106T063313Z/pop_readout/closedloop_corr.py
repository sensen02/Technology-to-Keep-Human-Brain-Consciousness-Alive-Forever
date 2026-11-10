"""How does the closed-loop command track the teacher torque DURING the release?

Cheap: reads only the released RUNS/ arrays and the sealed teacher BODY.npz.  No new
physics, no new phase, no repeat.

For every release the applied command of interval k (covering [ph+k ms, ph+(k+1) ms)) is
compared with the torque the teacher PD rule applies at the SAME instant on ITS OWN
trajectory, teacher ctrl[round(ph*2000) + 2k] (0.5 ms teacher grid).  The correlation is
reported over the first 20 / 50 / 100 / 200 ms of the release so that "right from the
start but lost as the state drifts away" is distinguishable from "wrong from the start".

The teacher recording ends at 1.4 s, so for the late phases the 100/200 ms windows are
truncated to the available overlap; the available length is reported, never extrapolated.
"""
import numpy as np

import common as C

WINDOWS = [20, 50, 100, 200]


def safe_corr(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return None if d == 0.0 else float((a * b).sum() / d)


def main():
    teacher = C.arrays(C.TEACHER / 'BODY.npz')
    ttime, tctrl = teacher['time'], teacher['ctrl']
    assert tctrl.shape == (2801, 42)
    summary = C.js(C.O / 'SUMMARY.json')
    limits = C.arrays(C.O / 'READOUT_FIT.npz')['limits']

    rows = []
    for r in summary['row_level']:
        readout, tag, ph = r['readout'], r['dt_tag'], float(r['release_time_s'])
        folder = C.O / 'RUNS' / f'{readout}__{tag}__rel_{ph}'
        applied = C.arrays(folder / 'NEURAL.npz')['applied']          # (200, 42)
        start = int(round(ph * 2000))                                  # 0.5 ms teacher index
        # interval k is compared with teacher sample start+2k, so the number of intervals whose
        # teacher sample exists is  floor((2800-start)/2)+1  (teacher grid ends at 1.4 s)
        n_avail = max(0, (2800 - start) // 2 + 1)
        per_window = {}
        for w in WINDOWS:
            n = min(w, 200, n_avail)
            idx = start + 2 * np.arange(n)
            n_used = int(len(idx))
            if n_used == 0:
                per_window[f'{w}ms'] = dict(available_ms=0, correlation=None)
                continue
            a = applied[:n_used].ravel()
            b = tctrl[idx].ravel()
            per_window[f'{w}ms'] = dict(
                available_ms=n_used, correlation=safe_corr(a, b),
                sign_agreement=float(np.mean(np.sign(applied[:n_used]) == np.sign(tctrl[idx]))),
                rms_ratio=float(np.sqrt((applied[:n_used] ** 2).mean())
                                / max(np.sqrt((tctrl[idx] ** 2).mean()), 1e-12)),
                saturation_fraction=float(
                    (np.abs(np.abs(applied[:n_used]) - limits[None, :]) < 1e-12).mean()))
        # lag control: shift the command by the 5 ms pool-filter window (command 5 ms later
        # than the teacher instant it is compared with); a correlation that IMPROVES under
        # this shift would mean the comparison was mis-aligned rather than wrong.
        lag = {}
        for w in WINDOWS:
            n = min(w, 200 - 5, n_avail)
            if n <= 0:
                lag[f'{w}ms'] = None
                continue
            idx = start + 2 * np.arange(n)
            lag[f'{w}ms'] = safe_corr(applied[5:5 + n].ravel(), tctrl[idx].ravel())
        rows.append(dict(readout=readout, dt_tag=tag, phase=ph,
                         per_window=per_window, lag5ms_control=lag,
                         first_positive_nonfoot_s=r['first_positive_nonfoot_s'], J=r['J']))

    # per readout x timestep: median over phases (never a single phase as the answer)
    agg = {}
    for readout in C.TAGS:
        for _, tag in C.DTS:
            sel = [x for x in rows if x['readout'] == readout and x['dt_tag'] == tag]
            d = {}
            for w in WINDOWS:
                v = [x['per_window'][f'{w}ms']['correlation'] for x in sel]
                v = [float(y) for y in v if y is not None]
                d[f'{w}ms'] = dict(per_phase=[x['per_window'][f'{w}ms']['correlation']
                                              for x in sel],
                                   min=float(min(v)) if v else None,
                                   median=float(np.median(v)) if v else None,
                                   max=float(max(v)) if v else None)
            agg[f'{readout}__{tag}'] = d

    C.dump(C.O / 'CLOSEDLOOP_TRACKING.json', dict(
        windows_ms=WINDOWS,
        definition='Pearson correlation, pooled over all 42 joints and all intervals in the '
                   'window, between the applied closed-loop command and the teacher torque at the '
                   'same instant of the teacher recording. The teacher torque is a PD function of '
                   'the TEACHER state, so any decay of this correlation is the divergence of the '
                   'release trajectory from the teacher trajectory, not a readout defect.',
        truncation_note='the teacher recording stops at 1.4 s, so windows that would extend past '
                        'it are truncated to the available overlap; available_ms says how many '
                        'intervals actually entered each number',
        per_release=rows, per_condition=agg,
        importance='核心必需'))
    for k, v in agg.items():
        print('%-22s' % k, {w: (None if v[w]['median'] is None else round(v[w]['median'], 4))
                            for w in v}, flush=True)


if __name__ == '__main__':
    main()
