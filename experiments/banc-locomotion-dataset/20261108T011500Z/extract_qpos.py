"""Read qpos for the declared snippet windows from the official measured dataset.

Runs in the isolated teacher environment (dm_control + official flybody); writes qpos in
the official 109-dim layout (7 root + 102 joint angles) exactly as the official task builds
it, so no hand-made quaternion conversion is used.
"""
from pathlib import Path
import hashlib, json
import numpy as np
HERE = Path(__file__).resolve().parent
DS = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T180844Z/dataset/walking-dataset_female-only_snippets-16252_trk-files-0-9.hdf5')
STEP, WINDOW = 32, 30


def main():
    from flybody.fly_envs import walk_imitation
    prereg = json.loads((HERE / 'PREREG.json').read_text())
    assert prereg['coverage_plan']['subsample_step'] == STEP and prereg['coverage_plan']['window'] == WINDOW
    env = walk_imitation(ref_path=str(DS), traj_indices=[], random_state=np.random.RandomState(41))
    raw = np.load(HERE / 'SUBSAMPLE_RAW.npz')
    keys = raw['sampled_keys']
    qpos, lengths = [], []
    for n, key in enumerate(keys):
        idx = int(key)
        env.task.set_next_trajectory_index(idx)
        env.reset()
        q = np.asarray(env.task._snippet['qpos'], dtype=np.float64)
        take = min(WINDOW, q.shape[0])
        qpos.append(q[:take]); lengths.append(q.shape[0])
        if n % 50 == 0:
            print('snippet', n, idx, q.shape, flush=True)
    out = dict(qpos=np.stack(qpos), snippet_indices=np.asarray([int(k) for k in keys]),
               full_lengths=np.asarray(lengths), timestep_s=float(raw['timestep_s']), window=WINDOW, step=STEP)
    np.savez_compressed(HERE / 'SUBSAMPLE_QPOS.npz', **out)
    summary = dict(source=str(DS), source_sha256=hashlib.sha256(DS.read_bytes()).hexdigest(),
                   snippets=len(keys), window=WINDOW, step=STEP, qpos_shape=list(out['qpos'].shape),
                   layout='7 root + 102 official joint angles, official task layout',
                   runtime='isolated teacher environment (dm_control + official flybody)',
                   physics_steps=0, note='qpos read from the official snippet object; no quaternion conversion done by hand')
    (HERE / 'SUBSAMPLE_QPOS.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
