"""Extract declared subsample windows from the official measured dataset (teacher venv, h5py).

Writes only frames and lengths; no retargeting here.
"""
from pathlib import Path
import hashlib, json
import h5py
import numpy as np

HDF = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T180844Z/dataset/walking-dataset_female-only_snippets-16252_trk-files-0-9.hdf5')
OUT = Path(__file__).resolve().parent
STEP = 32
WINDOW = 30


def main():
    with h5py.File(HDF, 'r') as f:
        lengths = np.asarray(f['trajectory_lengths'][:], dtype=np.int64)
        tstep = float(f['timestep_seconds'][()])
        keys = sorted(f['trajectories'].keys())
        idx = list(range(0, len(keys), STEP))
        roots, joints, lens, order = [], [], [], []
        for i in idx:
            k = keys[i]
            g = f['trajectories'][k]
            n = int(lengths[i])
            take = min(WINDOW, n)
            group_keys = list(g.keys())
            root_key = next(kk for kk in group_keys if 'root' in kk.lower())
            joint_key = next(kk for kk in group_keys if 'joint' in kk.lower() or 'mocap' in kk.lower())
            roots.append(np.asarray(g[root_key][:take], dtype=np.float64))
            joints.append(np.asarray(g[joint_key][:take], dtype=np.float64))
            lens.append(n)
            order.append(k)
    lengths_all = lengths
    np.savez_compressed(OUT / 'SUBSAMPLE_RAW.npz',
                        root=np.stack(roots), joints=np.stack(joints), sampled_lengths=np.asarray(lens),
                        sampled_keys=np.asarray(order), all_lengths=lengths_all, timestep_s=tstep, step=STEP, window=WINDOW)
    summary = dict(hdf5=str(HDF), hdf5_sha256=hashlib.sha256(HDF.read_bytes()).hexdigest(),
                   timestep_s=tstep, snippets=int(lengths_all.size), total_frames=int(lengths_all.sum()),
                   total_duration_s=float(lengths_all.sum() * tstep),
                   frames_min=int(lengths_all.min()), frames_max=int(lengths_all.max()),
                   frames_median=float(np.median(lengths_all)),
                   subsample_step=STEP, subsample_snippets=len(order), window=WINDOW,
                   subsample_frames=int(sum(min(WINDOW, n) for n in lens)),
                   subsample_duration_s=float(sum(min(WINDOW, n) for n in lens) * tstep),
                   group_keys=str(group_keys), root_key=root_key, joint_key=joint_key,
                   note='lengths and durations are exact for all snippets; frames are only the declared subsample windows')
    (OUT / 'SUBSAMPLE.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('snippets', 'total_frames', 'total_duration_s', 'frames_median',
                                              'subsample_snippets', 'subsample_frames', 'root_key', 'joint_key')}))


if __name__ == '__main__':
    main()
