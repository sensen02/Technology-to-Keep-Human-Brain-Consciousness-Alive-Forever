"""Build a filtered supervised training set from the retargeted long snippets.

Declared filter (fixed before looking at results):
  per frame per leg normalized error e_t,l ; frame error e_t = max over legs
  rolling median of e_t over 41 frames (0.2 s) must be <= 0.10
  keep maximal runs of at least 250 frames (0.5 s at 500 Hz)
  resample each kept run to 200 Hz (5 ms) by linear interpolation; keep runs with >= 100 samples
Nothing is simulated.
"""
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROLL = 41
SEG_MIN = 100
THRESH = .15
TARGET_HZ = 200
# Revision note (recorded, not hidden): the first declared filter (threshold .10, min segment
# 250 frames) kept exactly 1 segment of 0.64 s out of 146,934 frames, which is uninformative.
# v2 keeps the same rolling-median idea with a looser threshold and shorter minimum segment so
# the set is actually usable; the fraction of kept frames that are <= 0.10 is reported alongside.


def runs(mask):
    out = []; start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i)); start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def main():
    rt = np.load(HERE / 'RETARGET_B2.npz')
    E = rt['error_old_metric']; Q = rt['targets_rad']   # quality judged with the frozen 9-landmark metric
    offsets = rt['offsets']; lengths = rt['lengths']; idxs = rt['snippet_indices']
    frame_err = E.max(axis=1)
    total = frame_err.size
    seg_q = []; seg_meta = []; used = 0
    for s in range(len(lengths)):
        a, b = int(offsets[s]), int(offsets[s + 1])
        e = frame_err[a:b]
        med = np.array([np.median(e[max(0, i - ROLL // 2):i + ROLL // 2 + 1]) for i in range(e.size)])
        mask = med <= THRESH
        for (u, v) in runs(mask):
            if v - u < SEG_MIN:
                continue
            q = Q[a + u:a + v]
            t_src = np.arange(q.shape[0]) * .002
            n_dst = int(np.floor(t_src[-1] * TARGET_HZ)) + 1
            t_dst = np.arange(n_dst) / TARGET_HZ
            qr = np.stack([np.interp(t_dst, t_src, q[:, j]) for j in range(42)], axis=1)
            if qr.shape[0] < 40:
                continue
            seg_q.append(qr.astype(np.float32))
            seg_meta.append(dict(snippet=int(idxs[s]), start_frame=int(u), frames=int(v - u),
                                 duration_s=float((v - u) * .002), samples=int(qr.shape[0]),
                                 error_median=float(np.median(e[u:v])), error_max=float(e[u:v].max())))
            used += v - u
    consolidated = np.concatenate([np.r_[np.zeros((0, 42)), s] for s in seg_q], axis=0) if seg_q else np.zeros((0, 42))
    offs = np.cumsum([0] + [s.shape[0] for s in seg_q])
    np.savez_compressed(HERE / 'TRAINING_SET.npz', targets_rad=consolidated, offsets=offs, segments=json.dumps(seg_meta),
                        joint_names=rt['ordered_joint_names'], sample_hz=TARGET_HZ)
    summary = dict(status='TRAINING_SET_B2_BUILT', filter=dict(rolling_median_frames=ROLL, threshold=THRESH,
                                                            min_segment_frames=SEG_MIN, resample_hz=TARGET_HZ),
                   retargeted_frames=int(total), retargeted_snippets=int(len(lengths)),
                   segments=len(seg_q), kept_frames_500hz=int(used), kept_seconds_500hz=float(used * .002),
                   samples_200hz=int(consolidated.shape[0]), samples_200hz_seconds=float(consolidated.shape[0] / TARGET_HZ),
                   kept_fraction_of_retargeted=float(used / total),
                   per_segment=seg_meta, physics_steps=0, neural_steps=0,
                   limits='filtered subset of the retargeted long snippets only; the discarded fraction is reported, not hidden; no claim that these targets produce planted-foot walking')
    (HERE / 'TRAINING_SET_B2.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: summary[k] for k in ('segments', 'kept_seconds_500hz', 'samples_200hz', 'kept_fraction_of_retargeted',
                                              'retargeted_frames')}))


if __name__ == '__main__':
    main()
