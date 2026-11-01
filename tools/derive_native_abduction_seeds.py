"""Pure-arithmetic derivation of the preregistered coxa abduction-contraction seeds.

One model compile is used only to read compiled joint names and jnt_range; there is
no mj_forward, no mj_step, no contact evaluation and no optimisation here. The seed
table it writes is pinned by sha256 inside PREREGISTRATION.json before the search
script is allowed to run.
"""
from pathlib import Path
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import sys, json, hashlib, time
import numpy as np
import mujoco
from flygym.compose.fly.musculoskeletal import _load_mjcf

S = Path('/home/sensen/Desktop/engine修复验收/s5-detailed-handover-measurement/20261028T121000Z')
O = Path('/home/sensen/Desktop/engine修复验收/s5-equilibrium-abduction-seeds/20261101T201309Z')
EXPECTED = '3470c72f7e68f181ec1a80a5dc5fbff182b750091e7a1ddc7ccfbfb590b80cb6'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
FACTORS = (0.0, 0.25, 0.5, 0.75)
CONTRACT_AXES = ('roll', 'yaw')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False,
                            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()) + '\n')


def main():
    started = time.time()
    assert not (O / 'SEED_TABLE.json').exists(), 'SEED_TABLE.json already derived; never re-derive'
    assert not (O / 'EXECUTION.json').exists(), 'Search already executed'
    counts = dict(model_load_attempt=0, model_load=0, forward=0, mj_step=0,
                  geom_distance=0, contact_force=0, optimization=0)
    assert sha(S / 'MODEL.xml') == EXPECTED
    counts['model_load_attempt'] += 1
    m = _load_mjcf(str(S / 'MODEL.xml')).compile()
    counts['model_load'] += 1
    assert (m.nq, m.nv, m.nu, m.na) == (49, 48, 90, 90)
    hinges = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    assert len(hinges) == 42 and np.all(m.jnt_limited[hinges])
    qa = m.jnt_qposadr[hinges].astype(int)
    assert np.array_equal(np.sort(qa), np.arange(7, 49))
    names = [m.joint(int(j)).name for j in hinges]
    lower = m.jnt_range[hinges, 0].copy()
    upper = m.jnt_range[hinges, 1].copy()
    with np.load(S / 'BODY.npz') as z:
        seed_index = int(np.argmin(abs(z['time'] - .15)))
        seed_time = float(z['time'][seed_index])
        qseed = z['qpos'][seed_index].copy()
        body_sha = hashlib.sha256(z['qpos'].tobytes()).hexdigest()
    assert abs(seed_time - .15) < 1e-10

    contract = [f'joint_{L}Coxa_{a}' for L in LEGS for a in CONTRACT_AXES]
    assert len(contract) == 12 and len(set(contract)) == 12
    assert all(n in names for n in contract), sorted(set(contract) - set(names))
    idx = np.array([names.index(n) for n in contract], int)
    base = qseed[qa].copy()
    root_z = float(qseed[2])

    rows = []
    for k in FACTORS:
        raw = base.copy()
        # k is the retained fraction: k=0 collapses abduction/yaw onto the zero neutral.
        raw[idx] = k * base[idx]
        projected = np.clip(raw, lower, upper)
        changes = [dict(variable=int(i), joint=names[i], role=('contracted' if i in idx else 'unchanged'),
                        raw=float(raw[i]), projected=float(projected[i]),
                        delta=float(projected[i] - raw[i]),
                        bound=('lower' if projected[i] == lower[i] else 'upper') if projected[i] != raw[i] else None)
                   for i in np.flatnonzero(projected != raw)]
        x0 = np.r_[projected, root_z, np.full(66, .1)]
        assert len(x0) == 109
        pooled = np.ascontiguousarray(np.r_[x0, np.float64(k)])
        rows.append(dict(
            retained_fraction_k=float(k),
            definition='all 30 hinges other than the 12 {leg}Coxa_{roll,yaw} keep the archived t=0.15 value; '
                       'those 12 coxa roll/yaw hinges are multiplied by the retained fraction k toward the zero '
                       'neutral (k=0 => abduction and yaw fully zeroed, k=1 => archived seed unchanged); root z keeps '
                       'the archived t=0.15 value; root attitude fixed; the 66 pool activations start at 0.1',
            contracted_values={n: float(base[i]) for n, i in zip(contract, idx)},
            contracted_values_after_factor={n: float(k * base[i]) for n, i in zip(contract, idx)},
            raw_contracted_42=raw, projected_42=projected, x0_109=x0,
            projections=changes, n_projections=len(changes),
            total_projection_abs=float(np.sum(np.abs(projected - raw))),
            sum_abs_abduction_roll_yaw_seed=float(np.sum(np.abs(base[idx]))),
            sum_abs_abduction_roll_yaw_x0=float(np.sum(np.abs(projected[idx]))),
        ))
    table = dict(
        scope='Pure array arithmetic plus one model compile for compiled joint names and jnt_range; '
              'zero forward, zero step, zero contact evaluation, zero optimisation',
        model=str(S / 'MODEL.xml'), model_sha256=EXPECTED,
        body_npz=str(S / 'BODY.npz'), body_qpos_sha256=body_sha,
        seed_index=seed_index, seed_time=seed_time, archived_root_z=root_z,
        archived_hinge_values=base, hinge_names=names,
        joint_lower=lower, joint_upper=upper,
        root_z_range=[root_z - .25, root_z + .25], pool_range=[1e-4, 1.0], pool_initial=0.1,
        contraction_rule='x[j] = k*x_seed[j] for the 12 coxa roll/yaw hinges (k = retained fraction, so k=0 fully '
                         'zeroes abduction and yaw); the other 30 hinges, root z and pool initial values are '
                         'unchanged; then the declared minimal projection onto compiled native bounds (clip only, no '
                         'other change). Because the compiled left coxa_roll lower bound is positive, the zero '
                         'neutral is unreachable for the three left legs and the clip result is recorded per joint.',
        contraction_factors=list(FACTORS), contract_order=contract,
        seeds=rows,
        n_projections_total=int(sum(r['n_projections'] for r in rows)),
        derived_counts=counts, wall_seconds=time.time() - started,
    )
    canonical = json.dumps([dict(k=r['retained_fraction_k'], x0=np.asarray(r['x0_109']).tolist()) for r in rows],
                           sort_keys=True).encode()
    table['canonical_seed_vector_sha256'] = hashlib.sha256(canonical).hexdigest()
    dump(O / 'SEED_TABLE.json', table)
    dump(O / 'SEED_DERIVATION_EXECUTION.json',
         dict(counts=counts, wall_seconds=time.time() - started, exit_code=0,
              note='No physical evaluation of any kind; only compiled names/ranges were read.'))
    print(json.dumps({str(r['retained_fraction_k']): dict(n_proj=r['n_projections'],
                                                         sum_abs_x0=r['sum_abs_abduction_roll_yaw_x0'],
                                                         proj=r['projections']) for r in rows},
                     ensure_ascii=False, indent=2), flush=True)
    print('canonical_seed_vector_sha256', table['canonical_seed_vector_sha256'], flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
