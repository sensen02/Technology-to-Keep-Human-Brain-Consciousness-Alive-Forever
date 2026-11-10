"""Inventory analysis for the measured-dataset subsample: error bands and usable duration.

Nothing is simulated here. All numbers come from the stored retarget arrays and the exact
lengths of all 16,252 snippets.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DESK = Path('/home/sensen/Desktop/engine修复验收/banc-locomotion-dataset/20261108T011500Z')
BANDS = (.02, .05, .10, .15)


def main():
    rt = np.load(HERE / 'RETARGET_SUBSAMPLE.npz')
    sub = np.load(HERE / 'SUBSAMPLE_RAW.npz')
    err = rt['normalized_error']            # (snippets, window, 6)
    inlim = rt['in_limit']                  # (snippets, window, 6)
    nfev = rt['nfev']
    all_lengths = sub['all_lengths']; tstep = float(sub['timestep_s'])
    frame_max = err.max(axis=2)             # (snippets, window)
    frame_mean = err.mean(axis=2)
    total_frames = int(all_lengths.sum()); total_s = total_frames * tstep
    per_frame = frame_max.ravel()
    bands = {}
    for b in BANDS:
        frac = float((per_frame <= b).mean())
        bands[f'error_le_{b:g}'] = dict(frame_fraction=frac,
                                        estimated_frames=int(round(total_frames * frac)),
                                        estimated_duration_s=float(total_s * frac))
    bands['error_all'] = dict(frame_fraction=1.0, estimated_frames=total_frames, estimated_duration_s=float(total_s))
    per_leg = {}
    for l, leg in enumerate(rt['leg_names'].tolist()):
        e = err[:, :, l].ravel()
        per_leg[leg] = dict(p50=float(np.percentile(e, 50)), p95=float(np.percentile(e, 95)), max=float(e.max()),
                            rms=float(np.sqrt(np.mean(e ** 2))))
    snippets = []
    for s in range(err.shape[0]):
        idx = int(np.asarray(rt['snippet_indices'])[s])
        snippets.append(dict(snippet_index=idx, full_length_frames=int(all_lengths[idx]),
                             window_frames=int(err.shape[1]),
                             error_p50=float(np.percentile(err[s], 50)), error_max=float(err[s].max()),
                             frame_fraction_le_0p1=float((frame_max[s] <= .1).mean()),
                             nfev_terminated=int(np.sum(nfev[s] == 16))))
    p50_snips = float(np.percentile([x['error_p50'] for x in snippets], 50))
    summary = dict(
        status='MEASURED_DATASET_INVENTORY_COMPLETE',
        dataset=dict(snippets=int(all_lengths.size), frames=total_frames, duration_s=float(total_s),
                     timestep_s=tstep, source='official measured walking dataset (female-only, trk-files 0-9)'),
        subsample=dict(snippets=int(err.shape[0]), window_frames=int(err.shape[1]), step=32,
                       ik_solves=int(err.size), covered_frames=int(err.shape[0] * err.shape[1]),
                       covered_fraction_of_all_frames=float(err.shape[0] * err.shape[1] / total_frames)),
        error=dict(max=float(err.max()), p50=float(np.percentile(err, 50)), p95=float(np.percentile(err, 95)),
                   rms=float(np.sqrt(np.mean(err ** 2))),
                   snippet_p50_median=p50_snips),
        bands=bands, per_leg=per_leg,
        in_limit=dict(fraction=float(inlim.mean()),
                      note='trivially 1.0: the solver bounds are the compiled joint ranges, so in-limit is enforced by construction and carries no independent information'),
        solver_terminations=dict(hit_max_nfev_subsequent=int(np.sum(nfev == 16)), total=int(nfev.size),
                                 fraction=float((nfev == 16).mean())),
        frozen_geometry_gate=dict(threshold_max=.1, threshold_rms=.05,
                                  passed=bool(err.max() <= .1 and np.sqrt(np.mean(err ** 2)) <= .05),
                                  note='the earlier single-trajectory gate also failed; this subsample fails it too, so these targets are initialisation material only'),
        interpretation=dict(
            what_it_means='roughly the central frames retarget within about 5 percent of leg length, but the worst frames reach 44 percent, so a supervised dataset built from this would contain visibly wrong leg configurations unless frames are filtered by error',
            what_it_does_not_mean='it does not show the measured motion is wrong, and it does not establish that any retargeted target produces planted-foot walking; that needs physics on the current body'),
        usable_duration_estimates=bands,
        physics_steps=0, neural_steps=0,
        limits='band durations are estimates scaled from a 508-snippet, 30-frame-per-snippet sample; they are not a frame-by-frame measurement of all 2,477,312 frames')
    (HERE / 'RESULT.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    (HERE / 'SNIPPET_STATS.json').write_text(json.dumps(dict(snippets=snippets), indent=2) + '\n')
    DESK.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6), layout='constrained')
    ax[0].hist(per_frame, bins=60, color='tab:blue')
    for b in BANDS:
        ax[0].axvline(b, ls=':', c='k')
    ax[0].set(xlabel='per-frame max normalized landmark error', ylabel='frames', title='Retarget error distribution (508 snippets x 30 frames)')
    ax[1].bar([f'<= {b:g}' for b in BANDS] + ['all'], [bands[f'error_le_{b:g}']['estimated_duration_s'] for b in BANDS] + [total_s])
    ax[1].set(ylabel='estimated usable duration (s)', title='Estimated usable measured motion by error band')
    legs = list(per_leg)
    ax[2].bar(legs, [per_leg[l]['p50'] for l in legs], color='tab:green', label='p50')
    ax[2].scatter(legs, [per_leg[l]['max'] for l in legs], color='tab:red', label='max', zorder=3)
    ax[2].axhline(.1, ls='--', c='k', lw=.8, label='frozen gate 0.1')
    ax[2].set(ylabel='normalized error', title='Per-leg retarget error'); ax[2].legend(fontsize=8)
    fig.suptitle('Measured walking dataset -> current 42-joint body: frozen-solver retarget inventory (no physics)')
    fig.savefig(DESK / 'DATASET_INVENTORY.png', dpi=150); plt.close(fig)
    print(json.dumps(dict(status=summary['status'], error_max=summary['error']['max'], error_p50=summary['error']['p50'],
                          rms=summary['error']['rms'], gate_passed=summary['frozen_geometry_gate']['passed'],
                          usable_le_0p1_s=bands['error_le_0.1']['estimated_duration_s'],
                          usable_le_0p05_s=bands['error_le_0.05']['estimated_duration_s'],
                          total_s=total_s)))


if __name__ == '__main__':
    main()
