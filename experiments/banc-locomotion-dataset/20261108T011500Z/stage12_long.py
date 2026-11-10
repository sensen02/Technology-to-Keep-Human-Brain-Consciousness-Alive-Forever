"""Stage 1+2 for the declared long-snippet set: qpos from the official task, then official FK landmarks.

Runs in the isolated teacher environment. No stepping, no policy, no retargeting here.
"""
from pathlib import Path
import hashlib, json
import numpy as np
import mujoco as mj

HERE = Path(__file__).resolve().parent
DS = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T180844Z/dataset/walking-dataset_female-only_snippets-16252_trk-files-0-9.hdf5')
SRC = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/kinematics')
MJB = Path('/home/sensen/Desktop/engine修复验收/flybody-teacher/20261107T174640Z/official_straight_10s_v2/MODEL.mjb')
IDENT = Path('/home/sensen/Desktop/engine修复验收/flybody-teacher/20261107T180844Z/measured_longest_07718/MODEL_IDENTITY.json')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def landmarks(m, d, ids, local, th):
    rr = d.xmat[th].reshape(3, 3)
    xyz = d.xpos[ids] + np.einsum('...ij,...j->...i', d.xmat[ids].reshape(6, 9, 3, 3), local)
    return (xyz - d.xpos[th]) @ rr


def main():
    assert mj.__version__ == '3.1.6'
    sel = json.loads((HERE / 'SELECTION_LONG.json').read_text())['indices']
    from flybody.fly_envs import walk_imitation
    env = walk_imitation(ref_path=str(DS), traj_indices=[], random_state=np.random.RandomState(41))
    src = np.load(SRC / 'SOURCE_LANDMARKS.npz')
    ids = np.asarray(src['landmark_body_ids']); local = np.asarray(src['landmark_body_local'])
    identity = json.loads(IDENT.read_text()); jn = identity['mocap_joint_names']
    m = mj.MjModel.from_binary_path(str(MJB)); d = mj.MjData(m)
    mj.mj_resetDataKeyframe(m, d, 0); mj.mj_forward(m, d)
    th = m.body('walker/thorax').id
    qadd = [m.joint(n).qposadr[0] for n in jn[1:]]
    qpos_list, lm_list, lengths = [], [], []
    for n, idx in enumerate(sel):
        env.task.set_next_trajectory_index(int(idx)); env.reset()
        q = np.asarray(env.task._snippet['qpos'], dtype=np.float64)
        lm = np.zeros((q.shape[0], 6, 9, 3))
        for t in range(q.shape[0]):
            d.qpos[:] = m.qpos0
            d.qpos[:7] = q[t, :7]; d.qpos[qadd] = q[t, 7:]
            mj.mj_forward(m, d)
            lm[t] = landmarks(m, d, ids, local, th)
        qpos_list.append(q); lm_list.append(lm); lengths.append(q.shape[0])
        if n % 20 == 0:
            print('snippet', n, '/', len(sel), 'idx', idx, q.shape, flush=True)
    np.savez_compressed(HERE / 'LONG_SNIPPETS.npz',
                        qpos=np.concatenate(qpos_list, axis=0), landmarks=np.concatenate(lm_list, axis=0),
                        offsets=np.cumsum([0] + lengths), snippet_indices=np.asarray(sel, dtype=np.int64),
                        lengths=np.asarray(lengths), anatomical_frame=src['anatomical_frame'])
    summary = dict(runtime=mj.__version__, model=str(MJB), model_sha256=sha(MJB),
                   selection=str(HERE / 'SELECTION_LONG.json'), snippets=len(sel),
                   frames=int(sum(lengths)), mj_forward_calls=int(sum(lengths)),
                   landmarks_sha256_definitions=sha(SRC / 'SOURCE_LANDMARKS.npz'),
                   physics_steps=0,
                   note='official FK per frame with the frozen landmark definitions; no hand-made quaternion conversion')
    (HERE / 'LONG_SNIPPETS.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('snippets', 'frames', 'physics_steps')}))


if __name__ == '__main__':
    main()
