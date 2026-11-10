"""S5/P2 numerics_shift figure: D_shift versus the shift, next to the sealed dt values.

Reads only the stage's own SUMMARY.json (no sealed stage is written).
Axis labels are English to avoid missing CJK glyphs; the Chinese report explains every panel.
"""
import json, struct
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics_shift')


def js(p):
    return json.loads(Path(p).read_text())


def png_size(p):
    data = Path(p).read_bytes()[:33]
    assert data[:8] == b'\x89PNG\r\n\x1a\n', 'not a PNG'
    assert data[12:16] == b'IHDR'
    w, h = struct.unpack('>II', data[16:24])
    return w, h


def main():
    s = js(OUT / 'SUMMARY.json')
    v = s['verdict']
    pts = s['growth']['points']
    shift_ms = np.asarray([p['shift_ms'] for p in pts], float)
    point = np.asarray([p['pointwise_pool_nRMS'] for p in pts], float)
    agg = np.asarray([p['aggregate_window_mean_nRMS'] for p in pts], float)
    dt_fine = v['contrast']['pointwise_dt_nRMS_200ms']
    dt_coarse = s['sealed_dt_baseline']['coarse_pair_200ms']['pool_rates']
    same_dt = s['sealed_dt_baseline']['same_dt_method_pair_200ms']['pool_rates']['normalized_RMS']
    agg_fine = s['sealed_aggregate_anchors']['fine_0p03125_vs_0p015625']['window_mean']['normalized_RMS']
    agg_coarse = s['sealed_aggregate_anchors']['coarse_0p0625_vs_0p03125']['window_mean']['normalized_RMS']
    corr = [1.0] + [s['comparisons'][str(p['shift_substeps'])]['final_spike_count']['per_cell_pearson'] for p in pts[1:]]
    eng = s['growth']['engineering_threshold']
    ph = v['rule']['phase_threshold']
    it = v['rule']['integrator_threshold']

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))
    eps = 1e-4

    a = ax[0]
    a.axhspan(ph, 1.0, color='#f4b942', alpha=0.16)
    a.axhspan(eps, it, color='#4c72b0', alpha=0.16)
    a.axhline(eng, color='#2ca02c', ls='--', lw=1.4, label='engineering threshold 0.15')
    a.axhline(dt_fine, color='#d62728', ls='-', lw=1.4, label='sealed adjacent dt 0.03125 vs 0.015625 = %.4f' % dt_fine)
    a.axhline(dt_coarse, color='#d62728', ls=':', lw=1.3, label='sealed dt 0.0625 vs 0.03125 = %.4f' % dt_coarse)
    a.axhline(same_dt, color='#8c564b', ls='-.', lw=1.2,
              label='sealed same dt, other integrator = %.4f' % same_dt)
    a.plot(shift_ms, np.maximum(point, eps), 'o-', color='#1f3d7a', ms=7, lw=1.8, label='shifted input, D_shift')
    for x, y in zip(shift_ms, point):
        a.annotate('%.4f' % y, (x, max(y, eps)), textcoords='offset points', xytext=(6, -12), fontsize=8.5)
    a.set_yscale('log')
    a.set_xlabel('injected-current stream shift (ms)')
    a.set_ylabel('pointwise pool-rate nRMS over 200 ms')
    a.set_title('(a) pointwise: a 0.125 ms input shift vs the dt difference')
    a.grid(alpha=0.3, which='both')
    a.legend(fontsize=7.4, loc='lower right')
    a.set_ylim(eps * 0.7, 1.2)
    a.set_xlim(-0.004, 0.134)

    b = ax[1]
    b.axhline(eng, color='#2ca02c', ls='--', lw=1.2, label='engineering threshold 0.15')
    b.axhline(agg_fine, color='#d62728', ls='-', lw=1.4, label='sealed adjacent dt = %.4f' % agg_fine)
    b.axhline(agg_coarse, color='#d62728', ls=':', lw=1.3, label='sealed dt 0.0625 vs 0.03125 = %.4f' % agg_coarse)
    b.plot(shift_ms, np.maximum(agg, eps), 's-', color='#2a9d8f', ms=7, lw=1.8,
           label='shifted input, window-mean nRMS')
    for x, y in zip(shift_ms, agg):
        b.annotate('%.4f' % y, (x, max(y, eps)), textcoords='offset points', xytext=(6, -12), fontsize=8.5)
    b.set_yscale('log')
    b.set_xlabel('injected-current stream shift (ms)')
    b.set_ylabel('window-mean pool-rate nRMS over 200 ms')
    b.set_title('(b) aggregate: the same shift on the window-mean metric')
    b.grid(alpha=0.3, which='both')
    b.legend(fontsize=7.4, loc='lower right')
    b.set_ylim(eps * 0.7, 1.2)
    b.set_xlim(-0.004, 0.134)

    c = ax[2]
    c.plot(shift_ms, corr, 'D-', color='#7b2d8b', ms=7, lw=1.8)
    for x, y in zip(shift_ms, corr):
        c.annotate('%.6f' % y, (x, y), textcoords='offset points', xytext=(6, -12), fontsize=8.5)
    c.set_xlabel('injected-current stream shift (ms)')
    c.set_ylabel('per-cell spike-count Pearson vs control')
    c.set_title('(c) per-cell spike counts across the shift')
    c.grid(alpha=0.3, which='both')
    lo = min(corr)
    c.set_ylim(min(0.9, lo - 0.002), 1.0002)

    fig.suptitle('S5/P2 numerics_shift: dt = 0.015625 ms, 64 sub-steps/ms, 12000 cells, 200 ms; '
                 'reading = %s' % v['reading'], fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = OUT / 'SHIFT_GROWTH.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    w, h = png_size(p)
    meta = dict(figure=str(p), size_bytes=p.stat().st_size, width=w, height=h,
                panels=['(a) pointwise pool-rate nRMS at 200 ms vs shift, with sealed dt and same-dt anchors',
                        '(b) window-mean (aggregate) pool-rate nRMS at 200 ms vs shift, with sealed dt anchors',
                        '(c) per-cell spike-count Pearson correlation vs control across the shift'],
                values=dict(shift_ms=shift_ms.tolist(), pointwise=point.tolist(), aggregate=agg.tolist(),
                            per_cell_pearson=corr, sealed_dt_fine=dt_fine, sealed_dt_coarse=dt_coarse,
                            sealed_same_dt_other_integrator=same_dt, sealed_aggregate_fine=agg_fine,
                            sealed_aggregate_coarse=agg_coarse, engineering_threshold=eng,
                            phase_threshold=ph, integrator_threshold=it),
                importance='核心必需')
    (OUT / 'FIGURE.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    print('FIGURE', p, w, h, p.stat().st_size, flush=True)


if __name__ == '__main__':
    main()
