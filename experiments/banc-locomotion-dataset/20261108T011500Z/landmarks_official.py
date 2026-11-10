"""Official-runtime landmark extraction for the declared measured subsample.

Runs in the isolated teacher environment (MuJoCo 3.1.6 + the saved official binary model) and
reproduces exactly the landmark definitions used by the frozen retarget round: same six legs,
same nine landmark bodies, same local offsets, same anatomical frame. No stepping, no policy.
"""
from pathlib import Path
import hashlib, json
import numpy as np
import mujoco as mj

HERE = Path(__file__).resolve().parent
SRC = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/kinematics')
MJB = Path('/home/sensen/Desktop/engine修复验收/flybody-teacher/20261107T174640Z/official_straight_10s_v2/MODEL.mjb')
IDENT = Path('/home/sensen/Desktop/engine修复验收/flybody-teacher/20261107T180844Z/measured_longest_07718/MODEL_IDENTITY.json')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rot(q):
    a = np.zeros(9); mj.mju_quat2Mat(a, np.asarray(q)); return a.reshape(3, 3)


def landmarks(m, d, ids, local, th):
    rr = d.xmat[th].reshape(3, 3)
    xyz = d.xpos[ids] + np.einsum('...ij,...j->...i', d.xmat[ids].reshape(6, 9, 3, 3), local)
    return (xyz - d.xpos[th]) @ rr


def main():
    assert mj.__version__ == '3.1.6', mj.__version__
    sub = np.load(HERE / 'SUBSAMPLE_QPOS.npz')
    qpos = sub['qpos']
    src = np.load(SRC / 'SOURCE_LANDMARKS.npz')
    ids = np.asarray(src['landmark_body_ids'])
    local = np.asarray(src['landmark_body_local'])
    identity = json.loads(IDENT.read_text())
    jn = identity['mocap_joint_names']
    assert len(jn) == 103
    m = mj.MjModel.from_binary_path(str(MJB)); d = mj.MjData(m)
    mj.mj_resetDataKeyframe(m, d, 0); mj.mj_forward(m, d)
    th = m.body('walker/thorax').id
    qadd = [m.joint(n).qposadr[0] for n in jn[1:]]
    assert qpos.shape[2] == 109
    out = np.zeros(qpos.shape[:2] + (6, 9, 3)); roots = np.zeros(qpos.shape[:2] + (7,))
    for s in range(qpos.shape[0]):
        for t in range(qpos.shape[1]):
            q = qpos[s, t]
            d.qpos[:] = m.qpos0
            d.qpos[:7] = q[:7]; d.qpos[qadd] = q[7:]
            mj.mj_forward(m, d)
            out[s, t] = landmarks(m, d, ids, local, th)
            roots[s, t] = d.qpos[:7]
    np.savez_compressed(HERE / 'SUBSAMPLE_LANDMARKS.npz', landmarks=out, root=roots,
                        leg_names=src['leg_names'], landmark_names=src['landmark_names'],
                        anatomical_frame=src['anatomical_frame'])
    summary = dict(runtime=mj.__version__, model=str(MJB), model_sha256=sha(MJB),
                   qpos_source=str(HERE / 'SUBSAMPLE_QPOS.npz'), qpos_sha256=sha(HERE / 'SUBSAMPLE_QPOS.npz'),
                   landmark_definitions_sha256=sha(SRC / 'SOURCE_LANDMARKS.npz'),
                   shape=list(out.shape), snippets=int(qpos.shape[0]), window=int(qpos.shape[1]),
                   physics_steps=0, mj_forward_calls=int(out.shape[0] * out.shape[1]),
                   note='official FK only; landmark ids/local offsets reused byte-for-byte from the frozen retarget round')
    (HERE / 'SUBSAMPLE_LANDMARKS.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('runtime', 'shape', 'physics_steps', 'mj_forward_calls')}))


if __name__ == '__main__':
    main()
