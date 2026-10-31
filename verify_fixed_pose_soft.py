"""S171/P1-S2 independent saved-only fixed-pose native SOFT verifier.

No producer import, optimizer, engine, model load, forward, step or neural calls.
A positive separator proves only the exact saved binary64 48-row box system.
Phase-I minimax points in infeasible cases are NOT equilibrium witnesses.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import csv
import hashlib
import json
import numpy as np

BASE = Path('/home/sensen/Desktop/engine修复验收')
S169 = BASE/'s169-near-initial-support/20261024T233428554259Z'
BEST = S169/'BEST_DIAGNOSTIC_NOT_CANDIDATE.npz'
S166 = BASE/'s166-mapped-support/20261024T231418078692647Z/MAPPING.npz'
CSV = BASE/'s163-sensory-readout-audit/20261024T224712953805523Z/GRAPH_90_MUSCLE_MAPPING.csv'
S154 = BASE/'s154-activation-rate/20261024T215751775489Z/init80ms'


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, label, tol=1e-12):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (label, a.shape, b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all(), label
    err = float(np.max(abs(a-b), initial=0.))
    assert err <= tol, (label, err, tol)
    return err


def bitwise(a, b, label):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype, label
    assert a.tobytes(order='C') == b.tobytes(order='C'), label


def muscle(length, gain, bias, lengthrange):
    """Independent scalar native zero-velocity gain and passive muscle bias."""
    def curve(l, low, high):
        if l < low or l > high:
            return 0.
        ml, mh = .5*(low+1), .5*(high+1)
        if l <= ml:
            return .5*((l-low)/(ml-low))**2
        if l <= 1:
            return 1-.5*((1-l)/(1-ml))**2
        if l <= mh:
            return 1-.5*((l-1)/(mh-1))**2
        return .5*((high-l)/(high-mh))**2
    g, p = gain, bias
    assert g[2] > 0 and p[2] > 0
    l0 = (lengthrange[1]-lengthrange[0])/(g[1]-g[0])
    l = g[0]+(length-lengthrange[0])/l0
    gv = -g[2]*curve(l, g[4], g[5])
    l0b = (lengthrange[1]-lengthrange[0])/(p[1]-p[0])
    lb = p[0]+(length-lengthrange[0])/l0b
    mid = .5*(1+p[5])
    bv = 0. if lb <= 1 else -p[2]*p[7]*.5*((lb-1)/(mid-1))**2 if lb <= mid else -p[2]*p[7]*(.5+(lb-mid)/(mid-1))
    return gv, bv


def mapping():
    with CSV.open(newline='') as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r['actuator_index']))
    assert [int(r['actuator_index']) for r in rows] == list(range(90))
    members = [tuple(sorted(map(int, r['local_indices'].split(';')))) if r['local_indices'] else () for r in rows]
    unique = sorted(set(m for m in members if m))
    assert len(unique) == 66 and all(len(m) == len(set(m)) for m in members)
    pool = np.array([unique.index(m) if m else -1 for m in members])
    assert np.array_equal(pool, [int(r['pool_id']) for r in rows])
    missing = np.flatnonzero(pool < 0)
    assert len(missing) == 16
    T = np.zeros((90, 66))
    mapped = np.flatnonzero(pool >= 0)
    T[mapped, pool[mapped]] = .5
    old = load(S166)
    bitwise(old['T'], T, 'S166 actual T')
    assert np.array_equal(old['actuator_pool'], pool)
    return T, missing


def contacts(s, f0, meta, depthcap):
    types, J = s['efc_type'], s['efc_J_dense']
    assert J.shape == (len(f0), 48)
    assert np.all(np.isin(types, [0, 3, 4, 5, 6]))
    assert not np.any(s['efc_frictionloss'])
    assert int(s['ncon']) == len(s['contact_dist'])
    assert np.all(s['contact_valid_mask']) and np.all(np.isin(s['contact_dim'], [1, 3]))
    floors = set(map(int, meta['floor_geom_ids']))
    legs = meta['foot_order']
    geomleg = {int(k): legs.index(v) for k, v in meta['geom_map'].items()}
    assert len(legs) == 6
    used, details = [], []
    forces = np.zeros((6, 3))
    nonfootdepth = 0.
    footdepth = 0.
    for ci, addr in enumerate(s['contact_efc_address'].astype(int)):
        count = int(s['contact_row_count'][ci])
        assert count == (0 if addr < 0 else 1 if s['contact_dim'][ci] == 1 else 4)
        ids = np.arange(addr, addr+count) if addr >= 0 else np.zeros(0, int)
        used.extend(ids.tolist())
        assert np.array_equal(np.flatnonzero((s['efc_id'] == ci)&np.isin(types, [5, 6])), ids)
        edges = f0[ids]
        assert np.all(edges >= 0)
        mu = s['contact_friction'][ci, :2]
        frame = s['contact_frame'][ci]
        close(frame@frame.T, np.eye(3), 'contact frame', 1e-9)
        if count == 4:
            assert np.all(types[ids] == 6) and np.all(mu > 0)
            local = np.array([sum(edges), mu[0]*(edges[0]-edges[1]), mu[1]*(edges[2]-edges[3])])
            margin = local[0]-sum(abs(local[1:]/mu))
        else:
            if count:
                assert types[ids[0]] == 5
            local = np.array([sum(edges), 0., 0.])
            margin = local[0]
        assert margin >= -1e-12
        g1, g2 = map(int, s['contact_geom'][ci])
        assert g1 == s['contact_geom1'][ci] and g2 == s['contact_geom2'][ci]
        isfloor = (g1 in floors) != (g2 in floors)
        other = g2 if g1 in floors else g1
        foot = isfloor and other in geomleg
        dist = float(s['contact_dist'][ci])
        assert np.isfinite(dist)
        penetration = max(0., -dist)
        world = (1 if g1 in floors else -1)*(frame.T@local)
        if foot:
            forces[geomleg[other]] += world
            footdepth = max(footdepth, penetration)
        else:
            nonfootdepth = max(nonfootdepth, penetration)
        details.append(dict(contact=ci, foot=bool(foot), signed_dist=dist,
                            pyramid_margin=float(margin), force_world=world.tolist()))
    assert np.array_equal(np.sort(used), np.flatnonzero(np.isin(types, [5, 6])))
    assert footdepth <= depthcap+1e-12, ('foot depth', footdepth, depthcap)
    assert nonfootdepth == 0., ('NONFOOT penetration', nonfootdepth)
    bearing = int(np.count_nonzero(forces[:, 2] > 0))
    assert bearing >= 3, ('bearing legs', bearing)
    return dict(foot_order=legs, leg_force_world=forces.tolist(), bearing_legs=bearing,
                foot_depth=footdepth, frozen_depth_cap=depthcap,
                nonfoot_depth=nonfootdepth, contacts=details)


def certificate(A, b, y):
    """Validate supplied y, never search for or modify it."""
    assert y.shape == (48,) and np.isfinite(y).all()
    ld = np.longdouble
    aa, bb, yy = A.astype(ld), b.astype(ld), y.astype(ld)
    dots = np.sum(aa*yy[:, None], axis=0, dtype=ld)
    target = np.sum(bb*yy, dtype=ld)
    support = np.sum(np.maximum(dots, 0), dtype=ld)
    gap = target-support
    magnitude = np.sum(abs(bb*yy), dtype=ld)+np.sum(abs(aa*yy[:, None]), dtype=ld)
    budget = ld('1e-12')*max(ld(1), magnitude)
    assert gap-budget > ld('1e-8'), ('separator conservative gap', gap, budget)
    # An exact rational confirmation removes any platform longdouble caveat.
    def dot(u, v):
        return sum((Fraction.from_float(float(a))*Fraction.from_float(float(c)) for a, c in zip(u, v)), Fraction(0))
    exactgap = dot(b, y)-sum((max(dot(A[:, j], y), 0) for j in range(A.shape[1])), Fraction(0))
    assert exactgap > Fraction.from_float(1e-8)
    return dict(validated=True, gap=float(gap), roundoff_budget=float(budget),
                conservative_lower_gap=float(gap-budget), exact_gap=float(exactgap),
                exact_numerator=str(exactgap.numerator), exact_denominator=str(exactgap.denominator),
                arithmetic='longdouble with conservative budget plus exact rational binary64 confirmation')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert args.output.is_absolute()
    out = args.output.resolve()
    dest = out/'S171_SAVED_INDEPENDENT_CHECK.json'
    assert not dest.exists(), 'Refuse to overwrite independent evidence'
    input_hashes = json.loads((out/'INPUT_HASHES.json').read_text())
    assert input_hashes == json.loads((out/'INPUT_HASHES_AFTER.json').read_text())
    assert all(sha(path) == digest for path, digest in input_hashes.items())
    initial_summary = json.loads((out/'SUMMARY.json').read_text())
    summary = json.loads((out/'SUMMARY_FINAL.json').read_text())
    assert initial_summary['calls']['phaseI_LPs'] == 3
    assert all(initial_summary['calls'][k] == 0 for k in ('model_load', 'forward', 'step', 'neural'))
    s, frozen = load(BEST), load(S169/'FROZEN_SEARCH.npz')
    native = load(S154/'NATIVE_MODEL_ARRAYS.npz')
    meta = json.loads((S154/'COM_CONTACT_METADATA.json').read_text())
    T, missing = mapping()
    bitwise(frozen['T'], T, 'S169 frozen T')
    indices = frozen['indices'].astype(int)
    assert np.array_equal(indices, np.r_[2, np.arange(7, 49)])
    assert np.all(s['qpos'][indices] >= frozen['lower'])
    assert np.all(s['qpos'][indices] <= frozen['upper'])
    fixed = np.setdiff1d(np.arange(49), indices)
    bitwise(s['qpos'][fixed], frozen['qpos0'][fixed], 'fixed root xy/orientation')
    assert not np.any(s['qvel']) and not np.any(s['actuator_velocity'])
    assert not np.any(s['qfrc_applied']) and not np.any(s['xfrc_applied'])
    assert np.all(native['actuator_gaintype'] == 2)
    assert np.all(native['actuator_biastype'] == 2)
    assert np.all(native['actuator_dyntype'] == 4)
    assert np.array_equal(native['actuator_actadr'], np.arange(90))
    assert np.all(native['actuator_actnum'] == 1)
    dyn = native['actuator_dynprm']
    assert np.all(dyn[:, 0] == .001) and np.all(dyn[:, 1] == .0004)
    gain, bias = np.array([muscle(s['actuator_length'][j], native['actuator_gainprm'][j],
                                 native['actuator_biasprm'][j], native['actuator_lengthrange'][j]) for j in range(90)]).T
    M = s['actuator_moment']
    assert M.shape == (90, 48)
    B = M.T*gain[None, :]
    base = M.T@bias+s['qfrc_passive']-s['qfrc_bias']
    close(B, s['B'], 'independent scalar B')
    close(base, s['base'], 'independent scalar base')
    close(gain, s['gain'], 'gain'); close(bias, s['bias'], 'bias')
    assert np.all(np.isfinite(s['efc_D'])) and np.all(s['efc_D'] > 0)
    assert np.all(np.isfinite(s['efc_aref']))
    raw = s['efc_D']*s['efc_aref']
    f0 = np.where(s['efc_type'] == 0, raw, np.maximum(raw, 0.))
    bitwise(f0, s['f0'], 'all native type0/3/4/5/6 f0')
    bitwise(s['J'], s['efc_J_dense'], 'full native J')
    contact = contacts(s, f0, meta, float(frozen['original_foot_depth']))
    transforms = [T, np.concatenate((T, .5*np.eye(90)[:, missing]), axis=1), .5*np.eye(90)]
    results = []
    for i, transform in enumerate(transforms, 1):
        case = load(out/f'CASE_{i}.npz')
        bitwise(case['T'], transform, f'case{i} T')
        # Exact identity targets are saved, independently validated S169 arrays.
        A = s['B']@transform
        b = -s['base']-s['J'].T@f0
        assert A.shape == (48, transform.shape[1]) and b.shape == (48,)
        bitwise(case['A'], A, f'case{i} A all48')
        bitwise(case['b'], b, f'case{i} b all48')
        x = case['x']; t = float(case['t'])
        assert x.shape == (transform.shape[1],) and np.isfinite(x).all()
        assert np.isfinite(t) and t >= 0
        assert np.all(x >= -1e-8) and np.all(x <= 1+1e-8)
        activation = transform@x
        close(case['activation'], activation, 'activation', 1e-12)
        bitwise(case['ctrl'], case['activation'], 'saved ctrl=act')
        close(case['act_dot'], np.zeros(90), 'saved analytic actdot', 0.)
        assert np.all(activation >= -1e-8) and np.all(activation <= .5+1e-8)
        residual = A@x-b
        close(case['residual'], residual, 'all48 residual', 1e-10)
        maxres = float(max(abs(residual)))
        assert maxres <= t+1e-8, ('phase I epigraph', i, maxres, t)
        feasible = t <= 1e-8
        if feasible:
            assert maxres <= 1e-8
            # Native clipped act/ctrl are equal in [0,1], hence exact delta=0;
            # positive native tau (including smoothing) cannot change actdot0.
            assert np.all(activation >= 0) and np.all(activation <= 1)
            actdot = (activation-activation)/np.maximum(dyn[:, 1]/(.5+1.5*activation), 1e-15)
            assert np.all(actdot == 0)
            sep = None
            classification = 'FEASIBLE'
        else:
            corrected = load(out/f'CERTIFICATE_{i}.npz')
            dual = case['dual_marginals']
            assert dual.shape == (96,) and np.all(np.isfinite(dual))
            assert np.all(dual <= 1e-10)
            bitwise(corrected['y'], dual[:48]-dual[48:], 'corrected top-bottom dual orientation')
            bitwise(case['y'], dual[48:]-dual[:48], 'preserved initial wrong-sign y')
            assert float(case['separator_gap']) < 0
            sep = certificate(A, b, corrected['y'])
            close(corrected['separator_gap'], sep['gap'], 'corrected saved separator gap', 1e-10)
            classification = 'INFEASIBLE'
        results.append(dict(case=i, classification=classification, variables=len(x), t=t,
                            residual_maxabs=maxres, separator=sep,
                            minimax_point_is_witness=bool(feasible),
                            activation_stationarity='ctrl=act implies analytic actdot0; no engine evaluation'))
    # Producer SUMMARY comparison is adapted only to declared schema, never inferred from residual plot.
    declared = summary.get('cases', summary.get('results'))
    assert declared is not None, 'SUMMARY must declare case classifications'
    if isinstance(declared, dict):
        declared = list(declared.values())
    assert len(declared) == 3
    for actual, saved in zip(results, declared):
        label = str(saved.get('classification', saved.get('status', ''))).upper()
        assert actual['classification'] in label, ('SUMMARY classification', actual, saved)
    result = dict(status='VERIFIED_SAVED_ONLY', cases=results, contact=contact,
                  matrix_identity='bitwise all48 A,b against validated exact S169 saved arrays',
                  source_hashes_verified=len(input_hashes),
                  calls=dict(model_load=0, forward=0, step=0, neural=0, optimization=0),
                  scope='fixed S169 best pose native SOFT system only; not neural/FIR attainability or dynamic standing; SI NOT_ESTABLISHED')
    dest.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
