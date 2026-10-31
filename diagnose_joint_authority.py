"""Per-joint two-sided torque authority at the operating pose (B-path step 1).

For every one of the 42 leg hinges, compute the achievable generalized-force range
along that DOF from BOUNDED muscle activation (0..1), separately for the 66 trusted
pool and for all 90 actuators. Because activation bounds are independent, the exact
extremes are the sums of positive / negative entries of that row of the actuator
force matrix, so no optimiser is needed and the numbers are exact.

Also reports, for each joint, whether both directions have non-trivial authority and
how the range compares with the torque the algorithm was applying at handover.

Static only: one model load, a handful of forwards, no integration, no neural steps.
"""
from pathlib import Path
import json, hashlib, time
import numpy as np
import mujoco
from flygym.compose.fly.musculoskeletal import _load_mjcf

S = Path('/home/sensen/Desktop/engine修复验收/s5-neural-second-half/20261028T107000Z')
O = Path('/home/sensen/Desktop/engine修复验收/s5-joint-authority/20261028T109000Z')
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray)
                                  else v.item() if isinstance(v, np.generic) else str(v)) + '\n')


def main():
    O.mkdir(parents=True, exist_ok=False)
    started = time.time(); counts = dict(model_load=0, forward=0, jac=0, body_step=0, neural_step=0)
    h = np.load(S / 'HANDOVER_BEFORE.npz')
    b = np.load(S / 'BODY.npz')
    mapping = json.loads((S / 'MAPPING.json').read_text())
    groups = {}
    for row in mapping['rows']:
        if row['local_indices']:
            groups.setdefault(tuple(row['pool_root_ids']), []).append(row['actuator_index'])
    G = np.zeros((90, len(groups)))
    for j, ids in enumerate(groups.values()):
        G[ids, j] = 1.
    dump(O / 'PREREGISTRATION.json', dict(
        scope='Exact bounded two-sided torque authority per leg joint at the handover pose. Static only.',
        method='min/max of A[row,:]@x with 0<=x<=1 equals sum of negative / positive entries; no optimiser',
        budget=dict(model_load=1, forward=10, body_step=0, neural_step=0),
        source_hashes={str(p): sha(p) for p in [Path(__file__), S / 'MODEL.xml', S / 'HANDOVER_BEFORE.npz',
                                                S / 'MAPPING.json', S / 'GAINS.json']}))
    m = _load_mjcf(str(S / 'MODEL.xml')).compile(); counts['model_load'] += 1
    d = mujoco.MjData(m)
    q = h['qpos'].copy()
    d.qpos[:] = q; d.qvel[:] = 0; d.ctrl[:] = 0
    d.qfrc_applied[:] = 0; d.xfrc_applied[:] = 0
    d.act[:] = .0001; mujoco.mj_forward(m, d); counts['forward'] += 1
    f0 = d.actuator_force.copy()
    M = np.zeros((m.nv, m.nu))
    if d.actuator_moment.size == m.nv * m.nu:
        M = d.actuator_moment.reshape(m.nv, m.nu).copy()
    else:
        for j in range(m.nu):
            adr = int(d.moment_rowadr[j]); nnz = int(d.moment_rownnz[j])
            M[d.moment_colind[adr:adr + nnz], j] = d.actuator_moment[adr:adr + nnz]
    d.act[:] = 1.; mujoco.mj_forward(m, d); counts['forward'] += 1
    f1 = d.actuator_force.copy()
    A = M * (f1 - f0)[None, :]
    jid = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    names = [m.joint(j).name for j in jid]
    vadr = np.array([m.jnt_dofadr[j] for j in jid])
    A66 = A @ G
    i = int(np.argmin(abs(b['time'] - float(h['time'])))) - 1
    pd = b['assist'][i, 6:].copy()
    rows = []
    for k, name in enumerate(names):
        r66 = A66[vadr[k], :]; r90 = A[vadr[k], :]
        rows.append(dict(joint=name, leg=name.split('_')[1][:2] if name.startswith('joint_') else None,
                         neg66=float(r66[r66 < 0].sum()), pos66=float(r66[r66 > 0].sum()),
                         neg90=float(r90[r90 < 0].sum()), pos90=float(r90[r90 > 0].sum()),
                         pd_torque=float(pd[k]),
                         two_sided66=bool((r66 < -1e-3).any() and (r66 > 1e-3).any()),
                         two_sided90=bool((r90 < -1e-3).any() and (r90 > 1e-3).any())))
    dump(O / 'JOINT_AUTHORITY.json', rows)
    onesided66 = [r['joint'] for r in rows if not r['two_sided66']]
    onesided90 = [r['joint'] for r in rows if not r['two_sided90']]
    weak = [r['joint'] for r in rows if min(-r['neg90'], r['pos90']) < 0.2]
    summary = dict(joints=len(rows), one_sided_66=onesided66, one_sided_90=onesided90,
                   weak_side_below_0p2_90=weak,
                   two_sided_count_66=int(sum(r['two_sided66'] for r in rows)),
                   two_sided_count_90=int(sum(r['two_sided90'] for r in rows)),
                   note='A joint whose achievable range does not cover the demand in one direction cannot be held by ANY bounded muscle activation; that is a plant/geometry defect, not a network defect.')
    dump(O / 'SUMMARY.json', summary)
    dump(O / 'EXECUTION.json', dict(counts=counts, runtime_s=time.time() - started, exit_code=0))
    (O / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
