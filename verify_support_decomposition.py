"""S167/P1-S2 saved-only decomposition verifier; no engine or producer imports.

Only the two declared rigid-contact relaxations are checked. No primal LP is
solved here: linprog is used exclusively to search a separating certificate.
Native units, SI NOT_ESTABLISHED; neither standing nor neural attainability.
Run after the producer finishes, with --output /absolute/new/run/root.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import ast
import csv
import hashlib
import json
import numpy as np
from scipy.optimize import linprog

S165 = Path('/home/sensen/Desktop/engine修复验收/s165-static-support-corrected/20261024T231400000000Z')
S166 = Path('/home/sensen/Desktop/engine修复验收/s166-mapped-support/20261024T231418078692647Z')
CSV = Path('/home/sensen/Desktop/engine修复验收/s163-sensory-readout-audit/20261024T224712953805523Z/GRAPH_90_MUSCLE_MAPPING.csv')
HELPERS = Path(__file__).with_name('newverify_static_support_corrected.py')
INDICES = (0, 15, 83)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def close(a, b, label, tol=1e-9):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (label, a.shape, b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all(), label
    error = float(np.max(np.abs(a-b), initial=0.))
    assert error <= tol, (label, error, tol)
    return error


def helpers():
    # Same audited numerical functions used by verify_mapped_support.py;
    # compile only pure definitions, never import or execute its module/main.
    tree = ast.parse(HELPERS.read_text())
    names = {'exact_dot', 'separation', 'scalar_rate', 'muscle'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for f in nodes for n in ast.walk(f))
    namespace = {'np': np, 'Fraction': Fraction, 'linprog': linprog}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(HELPERS), 'exec'), namespace)
    return namespace


def mapping():
    with CSV.open(newline='') as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r['actuator_index']))
    assert [int(r['actuator_index']) for r in rows] == list(range(90))
    members = [tuple(sorted(map(int, r['local_indices'].split(';')))) if r['local_indices'] else () for r in rows]
    assert all(len(t) == len(set(t)) for t in members)
    unique = sorted(set(t for t in members if t))
    assert len(unique) == 66
    index = {t: j for j, t in enumerate(unique)}
    pool = np.array([index[t] if t else -1 for t in members], dtype=int)
    assert np.array_equal(pool, [int(r['pool_id']) for r in rows])
    mapped, missing = np.flatnonzero(pool >= 0), np.flatnonzero(pool < 0)
    assert len(mapped) == 74 and len(missing) == 16
    assert all(r['mapping_status'] == ('MAPPED' if pool[j] >= 0 else 'UNMAPPED_NO_TIER_CELLS') for j, r in enumerate(rows))
    original = np.zeros((90, 66))
    original[mapped, pool[mapped]] = .5
    unbind = np.zeros((90, 74))
    unbind[mapped, np.arange(74)] = .5
    fill = np.column_stack((original, np.zeros((90, 16))))
    fill[missing, 66+np.arange(16)] = .5
    return original, pool, unique, {'unbind': unbind, 'fill': fill}


def contact_check(s, edges, meta):
    """Decode native four-edge pyramids, real geom IDs, and signed world forces.

    Saved signed distance is reported, not silently replaced with a gap or
    margin. Penetration/compliance are deliberately relaxed, not certified.
    """
    floors = set(map(int, meta['floor_geom_ids']))
    assert len(floors) == 1, 'actual metadata must identify one floor'
    legs = list(meta['foot_order'])
    geomleg = {int(k): legs.index(v) for k, v in meta['geom_map'].items()}
    assert len(legs) == 6
    assert s['efc_J_dense'].shape == (72, 48)
    assert s['efc_type'].shape == (72,) and np.all(s['efc_type'] == 6)
    assert s['contact_dim'].shape == (18,) and np.all(s['contact_dim'] == 3)
    assert np.all(s['contact_row_count'] == 4)
    assert int(s['contact_ncon']) == 18 and np.all(s['contact_valid_mask'])
    addresses = s['contact_efc_address'].astype(int)
    assert np.array_equal(np.sort(np.concatenate([np.arange(a, a+4) for a in addresses])), np.arange(72))
    close(s['contact_dist'], s['contact_dist'], 'finite signed native distance')
    force = np.zeros((6, 3)); contacts = []
    for ci, addr in enumerate(addresses):
        assert np.array_equal(np.flatnonzero(s['efc_id'] == ci), np.arange(addr, addr+4))
        mu = s['contact_friction'][ci, :2]
        assert np.isfinite(mu).all() and np.all(mu > 0)
        frame = s['contact_frame'][ci]
        close(frame@frame.T, np.eye(3), 'native contact frame', 1e-9)
        g1, g2 = map(int, s['contact_geom_ids'][ci])
        assert (g1 in floors) != (g2 in floors), 'exactly one actual floor side'
        foot = g2 if g1 in floors else g1
        assert foot in geomleg
        item = dict(contact=ci, leg=legs[geomleg[foot]], geom1=g1, geom2=g2,
                    signed_native_distance=float(s['contact_dist'][ci]),
                    penetration_native=max(0., -float(s['contact_dist'][ci])),
                    friction=mu.tolist())
        if edges is not None:
            e = edges[addr:addr+4]
            assert e.shape == (4,) and np.all(e >= 0.)
            local = np.array([e.sum(), mu[0]*(e[0]-e[1]), mu[1]*(e[2]-e[3])])
            margin = float(local[0]-np.sum(np.abs(local[1:]/mu)))
            assert margin >= -1e-8, ('native pyramid friction', margin)
            world = (1 if g1 in floors else -1)*(frame.T@local)
            force[geomleg[foot]] += world
            item.update(local_force=local.tolist(), world_force=world.tolist(), pyramid_margin=margin)
        contacts.append(item)
    return dict(contacts=contacts, foot_order=legs,
                leg_force_world=None if edges is None else force.tolist(),
                contact_scope='All original 72 edges retained; native signed distances recorded; no complementarity or compliant-force claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    requested = parser.parse_args().output
    assert requested.is_absolute(), '--output must be an absolute run root'
    out = requested.resolve()
    assert out.is_dir() and out != S165 and out != S166
    products = [out/'INDEPENDENT_CHECK.json', out/'CERTIFICATES.npz']
    assert not any(p.exists() for p in products), 'Refuse to overwrite verification evidence'
    prereg = json.loads((S165/'PREREG.json').read_text())
    source = Path(prereg['source'])/'init80ms'
    inputs = [Path(__file__).resolve(), HELPERS, CSV, S165/'PREREG.json',
              S165/'AFFINITY_PROOF.json', S166/'MAPPING.npz',
              source/'NATIVE_MODEL_ARRAYS.npz', source/'TRAJECTORY.npz', source/'COM_CONTACT_METADATA.json']
    for i in INDICES:
        inputs += [S165/f'STATIC_AFFINE_{i:03d}.npz', S165/f'ZERO_V_{i:03d}.npz']
        inputs += [out/f'CASE_{i:03d}_{condition}.npz' for condition in ('unbind', 'fill')]
    # Freeze identities of all existing S165/S166 evidence, not only arrays used.
    inputs += [p for root in (S165, S166) for p in root.rglob('*') if p.is_file()]
    inputs = sorted(set(inputs), key=str)
    before = {str(p): sha(p) for p in inputs}
    h = helpers()
    original, pool, unique, transforms = mapping()
    old = load(S166/'MAPPING.npz')
    close(old['T'], original, 'S166 CSV mapping', 0.)
    assert np.array_equal(old['actuator_pool'], pool)
    native = load(source/'NATIVE_MODEL_ARRAYS.npz')
    trajectory = load(source/'TRAJECTORY.npz')
    assert np.array_equal(trajectory['actuator_pool'], pool)
    meta = json.loads((source/'COM_CONTACT_METADATA.json').read_text())
    proof = json.loads((S165/'AFFINITY_PROOF.json').read_text())
    for k in ('compiled_gain_muscle', 'compiled_bias_muscle', 'compiled_dyn_muscle', 'no_force_limits'):
        assert proof[k] is True
    for k in ('gainprm', 'biasprm'):
        close(proof[k], native['actuator_'+k], 'native '+k, 0.)
    dyn = np.asarray(proof['dynprm'])
    expected_dyn = native['actuator_dynprm'].copy(); expected_dyn[:, 0] = .001
    close(dyn, expected_dyn, 'tau-only patch', 0.)
    assert np.all(dyn[:, 1] == .0004)
    assert np.array_equal(proof['actadr'], np.arange(90))
    assert np.array_equal(native['actuator_actadr'], np.arange(90))
    assert np.all(native['actuator_actnum'] == 1)
    for key, value in [('gaintype', 2), ('biastype', 2), ('dyntype', 4)]:
        assert np.all(native['actuator_'+key] == value)
    ranges = np.asarray(proof['ctrlrange'])
    assert ranges.shape == (90, 2) and np.all(ranges[:, 0] == 0) and np.all(ranges[:, 1] == 1)
    assert np.all(proof['ctrllimited'])
    certificates = {'actuator_pool': pool, **{'T_'+k: v for k, v in transforms.items()}}
    results = []
    for i in INDICES:
        s = load(S165/f'ZERO_V_{i:03d}.npz')
        saved = load(S165/f'STATIC_AFFINE_{i:03d}.npz')
        for k in ('qpos', 'act', 'ctrl', 'qfrc_applied', 'xfrc_applied'):
            close(s[k], trajectory[k][i], 'S154 '+k, 0.)
        assert not s['qvel'].any() and not s['actuator_velocity'].any()
        assert not s['qfrc_applied'].any() and not s['xfrc_applied'].any()
        gain, bias = np.array([h['muscle'](s['actuator_length'][j], 0., 0., native['actuator_gainprm'][j], native['actuator_biasprm'][j], native['actuator_lengthrange'][j])[1:] for j in range(90)]).T
        moment, J = s['actuator_moment'], s['efc_J_dense']
        assert moment.shape == (90, 48)
        B = moment.T*gain[None, :]
        base = moment.T@bias+s['qfrc_passive']-s['qfrc_bias']
        for k, value in dict(B=B, base=base, gain=gain, bias=bias, J=J, moment=moment).items():
            close(saved[k], value, 'independent affine '+k)
        close(gain*s['act']+bias, s['actuator_force'], 'actual native muscle force')
        for condition, T in transforms.items():
            label = f'CASE_{i:03d}_{condition}'
            z = load(out/(label+'.npz'))
            assert {'T', 'Aeq', 'beq', 'bounds', 'x', 'activation', 'residual'} <= z.keys()
            nfinite = T.shape[1]
            assert nfinite == (74 if condition == 'unbind' else 82)
            A = np.column_stack((B@T, J.T)); b = -base
            assert A.shape == (48, nfinite+72)
            bounds = np.c_[np.zeros(nfinite+72), np.r_[np.ones(nfinite), np.full(72, np.inf)]]
            checks = {k: close(z[k], v, label+' independent '+k, 0. if k=='T' else 1e-12) for k, v in dict(T=T, Aeq=A, beq=b).items()}
            # Certify the exact frozen binary64 problem after independent scalar reconstruction.
            # BLAS/order differences at ~1e-17 are not physics changes; never certify a subtly different b.
            close(z['Aeq'], np.column_stack((saved['B']@T,saved['J'].T)), label+' frozen A',0.)
            close(z['beq'], -saved['base'], label+' frozen b',0.)
            A=z['Aeq']; b=z['beq']
            assert np.array_equal(z['bounds'], bounds)
            for k, shape in [('x', (nfinite+72,)), ('activation', (90,)), ('residual', (48,))]:
                assert z[k].shape == shape
            entry = dict(sample_index=i, condition=condition, finite_variable_count=nfinite, checks=checks)
            if np.isfinite(z['x']).all():
                x = z['x']
                assert np.all(x >= 0.) and np.all(x[:nfinite] <= 1.), label+' strict finite/edge bounds'
                activation = T@x[:nfinite]
                checks['activation'] = close(z['activation'], activation, label+' activation', 0.)
                assert np.all(activation >= 0.) and np.all(activation <= .5)
                if condition == 'unbind':
                    assert not activation[pool < 0].any()
                else:
                    for p in range(66):
                        assert np.all(activation[pool == p] == activation[pool == p][0])
                    close(activation[pool < 0], .5*x[66:82], 'independent missing outputs', 0.)
                residual = A@x-b
                checks['residual_saved'] = close(z['residual'], residual, label+' saved residual', 0.)
                checks['residual48_maxabs'] = close(residual, np.zeros(48), label+' all48 feasibility', 1e-8)
                ctrl = np.clip(activation, ranges[:, 0], ranges[:, 1])
                assert np.array_equal(ctrl, activation)
                rates = np.array([h['scalar_rate'](float(activation[j]), float(ctrl[j]), dyn[j]) for j in range(90)])
                assert np.array_equal(rates, np.zeros(90))
                certificates[label+'_primal'] = x
                certificates[label+'_offline_ctrl_NOT_APPLIED'] = ctrl
                certificates[label+'_offline_act_dot'] = rates
                entry.update(result='feasible', activation_rate_exact_zero=True,
                             contact=contact_check(s, x[nfinite:], meta))
            else:
                assert all(np.isnan(z[k]).all() for k in ('x', 'activation', 'residual')), label+' invalid failure schema'
                # generalized separation infers all finite columns, not a fixed66.
                y, cert = h['separation'](A, b, bounds[:, 0], bounds[:, 1])
                if y is not None:
                    certificates[label+'_separator'] = y
                entry.update(result='certified' if cert['validated'] else 'numeric', certificate=cert,
                             contact=contact_check(s, None, meta))
            results.append(entry)
    after = {str(p): sha(p) for p in inputs}
    assert before == after, 'S165/S166 or other input changed during verification'
    report = dict(status='SAVED_ONLY_INDEPENDENT_CHECK_COMPLETE', results=results,
                  scope='S167/P1-S2 two declared fixed-posture rigid-contact relaxations only',
                  new_engine_model_forward_step_neural_calls=0, primal_LP_solves=0,
                  units='native; SI NOT_ESTABLISHED', pool_tuples=[list(t) for t in unique],
                  source_hashes_preserved=True, input_sha256=before, input_sha256_after=after,
                  limitations='No solver status proves infeasibility: exact-rational separator required. Feasibility is saved all48 residual <=1e-8 with strict bounds; no dynamic standing, native-soft feasibility, neural/FIR attainability, or biological validation.')
    with products[1].open('xb') as f:
        np.savez_compressed(f, **certificates)
    report['certificates_sha256'] = sha(products[1])
    with products[0].open('x') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print(json.dumps({'output': str(out), 'results': [{k: r[k] for k in ('sample_index', 'condition', 'result')} for r in results]}, indent=2))


if __name__ == '__main__':
    main()
